"""
Final generation step + grounding verification pass (design doc §7.1).

`predict_outcome()` is the only function `pipeline.py` needs to call: it
builds the prompt, calls the LLM, parses the required JSON schema, and runs
a verification pass that strips any citation the model invented outside the
set of law provisions it was actually shown (hallucination guard).

legalrag_adjustments.md §5: this now takes a pre-built `case_digest` string
(see generation/case_digest.py) instead of raw case-evidence hits — the
caller (pipeline.py) is responsible for building the digest, since the same
case-evidence hits are also needed un-digested for the submission's
`case_evidence` field.

legalrag_adjustments.md §7 ("Neurosymbolic — rẻ, dễ implement"): after
parsing, a small rule-based check downgrades confidence when the model's
*self-reported* confidence isn't backed by any surviving grounded citation.
The system prompt already asks the model to self-report low confidence when
context is thin (rule #3), but nothing previously enforced that — a model
can claim high confidence with zero valid citations. This does not change
the predicted label (that would require case-specific legal judgement this
rule-based layer doesn't have); it only caps the confidence score so
downstream consumers of `confidence` aren't misled.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from backend import config
from backend.generation.prompt_builder import allowed_citation_keys, build_prediction_prompt
from backend.models import LawEvidenceItem, Prediction, RetrievedChunk, generate_text

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# legalrag_adjustments.md §7: if zero citations survive the hallucination
# guard, cap self-reported confidence at this ceiling regardless of what the
# model claimed — an ungrounded prediction should never be reported as
# high-confidence.
_UNGROUNDED_CONFIDENCE_CEILING = 0.3


@dataclass
class OutcomePrediction:
    prediction: Prediction
    law_citations: list[LawEvidenceItem] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    dropped_hallucinated_citations: int = 0


def _extract_json(raw: str) -> dict:
    """The model is instructed to return ONLY JSON, but LLMs sometimes wrap
    it in prose or a code fence anyway — extract the first {...} block
    defensively rather than trusting `json.loads(raw)` directly."""
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw[:200]!r}")
    return json.loads(match.group(0))


def _safe_default(reason: str) -> OutcomePrediction:
    """A parse/generation failure must never crash the whole submission run
    for one case — fall back to the most conservative label (B_WIN, i.e.
    claim not established) with zero confidence, flagged in reasoning."""
    return OutcomePrediction(
        prediction="B_WIN",
        law_citations=[],
        confidence=0.0,
        reasoning=f"[fallback] {reason}",
    )


def predict_outcome(
    case_query: str,
    law_chunks: list[RetrievedChunk],
    case_digest: str,
    max_new_tokens: int = config.GENERATION_MAX_NEW_TOKENS_DEFAULT,
    temperature: float = 0.2,
) -> OutcomePrediction:
    system_prompt, user_prompt = build_prediction_prompt(case_query, law_chunks, case_digest)

    try:
        raw = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        parsed = _extract_json(raw)
    except Exception as e:
        return _safe_default(f"generation/parsing failed: {e}")

    prediction = parsed.get("prediction")
    if prediction not in config.VALID_PREDICTIONS:
        return _safe_default(f"invalid prediction label from model: {prediction!r}")

    allowed = allowed_citation_keys(law_chunks)
    raw_citations = parsed.get("law_citations") or []
    kept, dropped = [], 0
    for item in raw_citations:
        try:
            key = (str(item["law_id"]), int(item["aid"]))
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if key in allowed:
            kept.append(LawEvidenceItem(law_id=key[0], aid=key[1]))
        else:
            dropped += 1  # hallucinated / outside retrieved context — drop it

    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    reasoning = str(parsed.get("reasoning", ""))
    if not kept and confidence > _UNGROUNDED_CONFIDENCE_CEILING:
        # Rule-based grounding check (legalrag_adjustments.md §7): the model
        # claimed more confidence than a zero-citation answer should get.
        reasoning = (
            f"[confidence capped: no grounded law citation survived verification] {reasoning}"
        )
        confidence = min(confidence, _UNGROUNDED_CONFIDENCE_CEILING)

    return OutcomePrediction(
        prediction=prediction,
        law_citations=kept,
        confidence=confidence,
        reasoning=reasoning,
        dropped_hallucinated_citations=dropped,
    )

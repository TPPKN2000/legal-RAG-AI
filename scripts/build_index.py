"""
Build the BM25 + Pinecone indexes for the law corpus.

legalrag_adjustments.md §6a: this script used to build LawChunk objects by
hand directly from the raw JSON (one flat "child" chunk per article, full
article text, no real Khoản/Điểm splitting) and never called
`backend.ingestion.chunker` at all — so `chunker.build_parent_lookup()`
existed in the codebase but was dead code, and the reranker never had
access to whole-article parent context.

This now routes through the actual ingestion pipeline the design doc
describes: `parser.load_law_corpus` -> `chunker.chunk_articles` (real
Chương>Mục>Điều>Khoản>Điểm structural splitting, with the §6b soft-split for
oversized clauses) -> `chunker.build_parent_lookup`. The parent lookup is
persisted to `config.PARENT_LOOKUP_PATH` so `pipeline.py` can load it at
retrieval/rerank time without re-parsing the corpus.
"""
import argparse
import logging
import pickle

from backend import config
from backend.indexing import vector_store
from backend.indexing.bm25_index import BM25Index
from backend.ingestion.chunker import build_parent_lookup, chunk_articles
from backend.ingestion.parser import load_law_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--rebuild-pinecone", action="store_true")
    args = parser.parse_args()

    logger.info(f"loading law corpus from {args.corpus}")
    law_docs = load_law_corpus(args.corpus)
    logger.info(f"loaded {len(law_docs)} law documents")

    law_chunks = []
    for doc in law_docs:
        law_chunks.extend(chunk_articles(doc.articles))

    n_parent = sum(1 for c in law_chunks if c.level == "parent")
    n_child = sum(1 for c in law_chunks if c.level == "child")
    logger.info(f"chunked into {n_parent} parent + {n_child} child chunks")

    if n_child == 0:
        logger.error("No chunks created.")
        return

    # 0. Persist the parent (whole-Điều) lookup so pipeline.py can re-attach
    #    full-article context at rerank time (legalrag_adjustments.md §6a).
    parent_lookup = build_parent_lookup(law_chunks)
    config.PARENT_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.PARENT_LOOKUP_PATH, "wb") as f:
        pickle.dump(parent_lookup, f)
    logger.info(f"parent lookup ({len(parent_lookup)} articles) saved to {config.PARENT_LOOKUP_PATH}")

    # 1. Build BM25 Index
    logger.info("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(law_chunks)
    bm25.save()  # default path: config.BM25_INDEX_PATH = data/bm25_index.pkl
    logger.info(f"BM25 index saved to {config.BM25_INDEX_PATH}")

    # 2. Build Pinecone Index
    if args.rebuild_pinecone:
        logger.info("Rebuilding Pinecone index...")
        count = vector_store.upsert_chunks(law_chunks)
        logger.info(f"Successfully upserted {count} chunks to Pinecone.")


if __name__ == "__main__":
    main()

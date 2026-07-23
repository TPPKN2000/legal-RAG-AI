# Design Experiment

> Tài liệu này gồm 2 phần: **Phần A** là nguyên văn quy định chính thức của ban tổ chức ALQAC2026 (định dạng nộp bài, nhãn dự đoán, công thức chấm điểm) — không được tự ý diễn giải lại vì đây là cơ sở chấm điểm thật. **Phần B** mô tả **cách mã nguồn `backend/` và `test/` hiện triển khai và đo lường cục bộ** theo đúng quy định ở Phần A, kèm các khoảng cách/giới hạn đã biết giữa phép đo local và phép đo thật của ban tổ chức.

---

## Phần A — Quy định chính thức (ban tổ chức)

### Test Dataset
`.\data\ALQAC2026_public_test.json`

### Submission Format
The submission must contain a list of predictions, one object per test case:
`submission_example.json`

A submission may be rejected or partially ignored if it violates the required format.

The organizers may validate the following conditions:

- Every test case has exactly one submitted prediction.
- Every `case_id` exists in the official test set.
- There are no duplicate `case_id`s.
- The `prediction` value is either A_WIN | B_WIN | PARTIAL_A_WIN | PARTIAL_B_WIN.
- `law_evidence` is a list of valid legal provision identifiers from the law corpus.
- The JSON file is valid and can be parsed automatically.

Duplicate evidence items may be automatically deduplicated before scoring.

### Required Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `case_id` | string | Yes | Public identifier of the test case. |
| `prediction` | string | Yes | Must be either `A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, or `B_WIN`. |
| `law_evidence` | list[object] | Yes | List of relevant legal provisions retrieved from the law corpus. Each object must contain `law_id` and `aid`. |

Each item in `law_evidence` must follow this format:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `law_id` | string | Yes | Identifier of the legal document in the law corpus. |
| `aid` | integer | Yes | Article/provision ID within the corresponding legal document. |

### Prediction labels
The `prediction` field must be one of the following values:

| Value | Definition |
|-------|------------|
| `A_WIN` | The court fully accepts all of the plaintiff's claims. |
| `PARTIAL_A_WIN` | The court partially accepts the plaintiff's claims, and the accepted portion is greater than 50%. |
| `PARTIAL_B_WIN` | The court partially accepts the plaintiff's claims, but the accepted portion is 50% or less. |
| `B_WIN` | The court fully rejects all of the plaintiff's claims. |

If a case contains multiple claims, teams should focus on the main claim described in the `case_query`.

### Evaluation Method
The official evaluation metrics for the Legal Case Outcome Prediction task. The final score consists of three components:

- **Outcome Accuracy**: whether the system correctly predicts the winning side.
- **Penalized Case Evidence Recall**: whether the system retrieves the correct case-content evidence, with a penalty for excessive API calls.
- **Micro Law Evidence F1**: whether the system retrieves the correct legal provisions from the law corpus.

\[
\text{FinalScore} = 0.70 \cdot \text{OutcomeAccuracy}
+ 0.20 \cdot \text{PenalizedCaseRecall}
+ 0.10 \cdot F1_{\text{micro}}^{\text{law}}
\]

(Xem `docs/evaluation.md` cho định nghĩa đầy đủ của từng thành phần, bao gồm công thức API-efficiency penalty \(E_i\).)

The metric is designed to reward systems that:

a. Predict the correct outcome.
b. Retrieve the correct case evidence.
c. Retrieve the relevant legal provisions.
d. Use the Case Content API efficiently.

---

## Phần B — Triển khai thực tế trong mã nguồn & giới hạn phép đo local

### B.1 Hai đường sinh output khác nhau, KHÔNG được nhầm lẫn

| | `test/test_all_backend.py` | `backend/submission.py` |
|---|---|---|
| Mục đích | Đánh giá nội bộ (local dev-set eval) | Sinh file nộp bài chính thức |
| Input | `ALQAC2026_public_test.json` — **có** `verdict_label`, `related_law_provisions` | Bất kỳ test set nào theo `config.TEST_SET_PATH` — chỉ cần `{case_id, case_query}` |
| Gọi pipeline qua | `backend.pipeline.process_case_with_debug()` (trả thêm debug dict) | `backend.pipeline.process_case()` (chỉ trả `SubmissionRecord`) |
| Có kiểm tra format nộp bài? | Không | Có — `_validate_submission()` |
| Output | `test/test_submission_backend.json` + log OutcomeAccuracy/F1/API calls | `submission.json` (hoặc `--out` tuỳ chỉnh), đúng schema `docs/submission_example.json` |

`verdict_label` và `related_law_provisions` **chỉ tồn tại ở Public test set**. Private test set (và input thật lúc chấm điểm) chỉ có `{case_id, case_query}` — xem comment trong `backend/pipeline.py._case_api_budget()` và `test/test_all_backend.py` phần load gold. Vì vậy `test_all_backend.py` không được dùng để sinh submission chính thức, và `backend.submission` không tự đánh giá được accuracy cục bộ (không có gold để so sánh).

### B.2 Validation nộp bài — `backend/submission.py._validate_submission()`

Hàm này chủ động mirror lại đúng các điều kiện ở Phần A trước khi ghi file, để bắt lỗi format sớm ở local thay vì chờ ban tổ chức từ chối:

```python
def _validate_submission(records: list[SubmissionRecord], test_case_ids: set[str]) -> None:
    for r in records:
        if r.case_id not in test_case_ids: raise ValueError(...)   # "case_id không có trong test set"
        if r.case_id in seen_ids: raise ValueError(...)             # "duplicate case_id"
        seen_ids.add(r.case_id)
        if r.prediction not in config.VALID_PREDICTIONS: raise ValueError(...)  # nhãn hợp lệ
    missing = test_case_ids - seen_ids
    if missing: raise ValueError(...)                                 # "mỗi case phải có đúng 1 prediction"
```

Việc **dedup `case_evidence`/`law_evidence`** ("Duplicate evidence items may be automatically deduplicated before scoring" ở Phần A) đã được xử lý *chủ động* ngay tại tầng schema, không đợi ban tổ chức lọc hộ:

```python
# backend/models.py — SubmissionRecord
@field_validator("case_evidence")
def _dedup_case_evidence(cls, v): ...   # loại trùng chunk_id, giữ thứ tự xuất hiện đầu tiên
@field_validator("law_evidence")
def _dedup_law_evidence(cls, v): ...    # loại trùng theo cặp (law_id, aid)
```

**Giới hạn còn lại (chưa được kiểm):** `_validate_submission()` KHÔNG kiểm tra độc lập rằng mỗi `(law_id, aid)` trong `law_evidence` thực sự tồn tại trong `corpus_law_pub.json` — nó tin tưởng rằng `generate.py`'s hallucination guard (`allowed_citation_keys`) đã giới hạn citation về đúng tập đã truy hồi từ corpus. Vì `allowed_citation_keys` được xây từ `law_chunks` (kết quả `collect_law_evidence`, vốn xuất phát từ index đã build từ corpus thật), điều này *nên* luôn đúng theo cấu trúc, nhưng chưa có test nào khẳng định trực tiếp theo Phần A ("valid legal provision identifiers from the law corpus").

### B.3 Ngân sách API — B.a điều đã lộ khi private test set không có `n_segments`

Công thức Phần A/`evaluation.md §2.4`: \(B_i = 2 n_i\), phạt dần đến 0 tại \(5 n_i\). Nhưng:

```python
# backend/pipeline.py._case_api_budget
def _case_api_budget(case: CaseQuery) -> int:
    if case.n_segments:
        return max(1, int(config.API_BUDGET_MULTIPLIER * case.n_segments))  # dùng n_i thật nếu có
    return config.DEFAULT_MAX_API_CALLS_PER_CASE   # <-- NHÁNH MẶC ĐỊNH khi chạy private test set thật
```

Vì input thi thật chỉ có `{case_id, case_query}` (không có `n_segments`/`n_i`), **nhánh fallback `DEFAULT_MAX_API_CALLS_PER_CASE=8` mới là đường chạy thực tế lúc chấm điểm**, không phải nhánh `2·n_i` — dù công thức chính thức dùng `n_i` thật. Đây không phải bug, nhưng là một giả định vận hành quan trọng cần ghi rõ: ngân sách 8 calls/case là *ước lượng cứng*, có thể quá cao (phí ngân sách oan nếu case có ít segment) hoặc quá thấp (mất recall nếu case có nhiều segment) so với ngân sách `2·n_i` thật mà ban tổ chức sẽ dùng để tính `E_i` khi chấm.

`config.API_HARD_CEILING_MULTIPLIER` (mặc định 5.0, tương ứng ngưỡng `5·n_i` cho `E_i=0`) được khai báo trong `config.py` nhưng **không được bất kỳ hàm nào trong `backend/` đọc lại** — `test/test_all_backend.py._e_i()` tự hardcode `5 * budget_n` thay vì tham chiếu `config.API_HARD_CEILING_MULTIPLIER`. Đây là một hằng số cấu hình "chết" (dead config), nên sửa hoặc xoá để tránh gây hiểu nhầm khi đọc `.env.example`.

### B.4 Vì sao Micro Law F1 đo local hiện luôn ≈ 0 — không phản ánh chất lượng retrieval thật

`test/test_all_backend.py.compute_law_f1()` so khớp trên "article number" suy ra bằng regex `Điều\s+(\d+)` từ trường `related_law_provisions` (chỉ có ở Public test, chỉ cho tên luật + số Điều, **không có `law_id` chuẩn**). Trong khi đó, `aid` do pipeline sinh ra (qua `backend/ingestion/parser.py.load_law_corpus()`) là ID nội bộ của corpus (`art.get("aid") or art.get("id")`), quan sát thực tế trong `test_submission_backend.json` là các số lớn kiểu 50882/53082 — **không cùng không gian giá trị** với số "Điều N" nhỏ mà gold parse ra.

⇒ Micro/macro Law F1 đo bằng harness hiện tại **là một xấp xỉ có thể luôn bằng 0 không phải vì retrieval kém**, mà vì hai tập so khớp sống trong hai namespace ID khác nhau. Ban tổ chức chấm điểm thật (Phần A: `law_evidence` khớp cả `law_id` VÀ `aid`) không gặp vấn đề này vì họ có gold `(law_id, aid)` chuẩn — vấn đề chỉ tồn tại ở phép đo cục bộ do Public test set không cấp `law_id`/`aid` gold. Xem `IMPROVEMENT_PLAN.md §3.3` để biết hướng khắc phục (xây bảng ánh xạ `(law_id, article_num) → aid` trực tiếp từ `corpus_law_pub.json`).

**Hệ quả cho việc dùng `test_all_backend.py`:** Micro/macro Law F1 in ra bởi script này **chỉ nên dùng để phát hiện khi phép đo "chuyển từ luôn-0 sang khác-0"** (dấu hiệu retrieval đang hoạt động), KHÔNG nên dùng con số tuyệt đối này để so sánh với ngưỡng benchmark hay báo cáo là "điểm Law F1 thật".

### B.5 Case Evidence Recall — không đo được ở local, chỉ đo được lúc nộp bài thật

Phần A định nghĩa `PenalizedCaseRecall` (20% trọng số) dựa trên gold `case_evidence` segment IDs của mỗi case. **Public test set không cung cấp gold case-evidence segments** — chỉ có `verdict_label` và `related_law_provisions`. Do đó:

- `test/test_all_backend.py` **không tính** Case Recall — dòng log cuối cùng ghi rõ: `"Approx score (Case Recall excluded — public set has no gold case_evidence): 0.70*acc + 0.10*micro_f1 = ..."`.
- Điểm gần đúng in ra bởi harness **luôn thấp hơn** FinalScore thật (thiếu hẳn 20% trọng số), và không phản ánh được tác động của số lượt gọi Case Content API (`api_calls`) lên điểm cuối — dù `api_calls` vẫn được log để theo dõi ngân sách, chỉ riêng phần *penalty lên recall* là không thể mô phỏng do thiếu gold.
- `E_i` (API efficiency factor) được **ước lượng riêng** (không nhân vào recall vì không có recall để nhân) bằng `_e_i()`, dùng `n_segments` khi có hoặc fallback `DEFAULT_MAX_API_CALLS_PER_CASE` — chỉ mang tính tham khảo rủi ro ngân sách, không phải điểm Recall thật.

### B.6 Chạy đánh giá / sinh submission (lệnh thực tế)

```bash
# Đánh giá cục bộ trên public test set (đo được: OutcomeAccuracy, Law F1 xấp xỉ, fallback rate, API calls, E_i ước lượng)
python -m test.test_all_backend -n 5            # smoke test 5 case ngẫu nhiên (seed=42)
python -m test.test_all_backend --offline -n 5  # không cần ALQAC_TOKEN — stub Case Content API, chỉ đo Outcome + Law F1
python -m test.test_all_backend                 # full 50 case công khai

# Sinh file nộp bài chính thức (không tự đo accuracy — không có gold)
python -m backend.submission --limit 5          # smoke test format trước
python -m backend.submission                    # full test set thật, ghi submission.json
```

**Pre-flight checks bắt buộc trước khi chạy `test_all_backend.py` (không phải tuỳ chọn — script tự thoát sớm nếu thiếu):**
1. `ALQAC_TOKEN` phải hợp lệ (bỏ qua nếu dùng `--offline`).
2. `config.BM25_INDEX_PATH` (`data/bm25_index.pkl`) phải tồn tại — chạy `python -m scripts.build_index` trước.
3. Model sinh (`config.GENERATION_MODEL_NAME`) được warm-up 1 lần trước vòng lặp chính (tránh trả phí load model 50 lần).

### B.7 Giới hạn nộp bài (nhắc vận hành, không thuộc Phần A nhưng ảnh hưởng quy trình)

`backend/submission.py.main()` in cảnh báo mỗi lần chạy: leaderboard ALQAC chỉ nhận **tối đa 3 lượt nộp/ngày**. Script chỉ ghi file `submission.json` cục bộ, không tự upload — cảnh báo chỉ để tránh lãng phí lượt nộp bằng các bản gần giống nhau chưa kiểm tra kỹ.

---

## Phần C — Tham chiếu chéo

- Chẩn đoán chi tiết + patch đề xuất cho từng vấn đề nêu ở B.3/B.4/B.5: xem `IMPROVEMENT_PLAN.md`.
- Kiến trúc pipeline đầy đủ (không chỉ phần test): xem `README.md`.
- Định nghĩa toán học đầy đủ của `Outcome_i`, `Recall_i^case`, `E_i`, `F1_micro^law`: xem `docs/evaluation.md` (không lặp lại ở đây).

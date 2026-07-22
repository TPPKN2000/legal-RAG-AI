# **LegalRAG**

Hệ thống RAG cho bài toán **Legal Case Outcome Prediction with Evidence Retrieval** (ALQAC2026). Cho một `case_query` mô tả tranh chấp dân sự, hệ thống phải:

1. Gọi Case Content API để lấy bằng chứng vụ án (`case_evidence`).
2. Truy hồi các điều luật liên quan từ corpus luật.
3. Dự đoán nhãn kết quả (`A_WIN` / `PARTIAL_A_WIN` / `PARTIAL_B_WIN` / `B_WIN`).
4. Nộp bài theo đúng format `docs/submission_example.json`.

> Tài liệu này mô tả **kiến trúc thực tế trong `backend/`**, không phải bản thiết kế ban đầu (`docs/system_design_v0.md`) — một số cấu phần trong thiết kế gốc (HyDE) đã bị loại bỏ và thay bằng NER + query decomposition, xem mục "Khác biệt so với thiết kế gốc" bên dưới.

---

## 1. Kiến trúc pipeline hiện tại (`backend/pipeline.py`)

```
                    CaseQuery(case_id, case_query, n_segments?)
                                    │
                ┌───────────────────┴────────────────────┐
                ▼                                         ▼
  collect_case_evidence()                     collect_law_evidence()
  - rewrite_query() sinh biến thể truy vấn     - NER mask tên riêng (ner.py)
  - Gọi Case Content API, ngân sách             - hybrid_search(): BM25 + Pinecone
    B_i = 2·n_i (hoặc DEFAULT_MAX_API_CALLS_      qua nhiều biến thể truy vấn +
    PER_CASE=8 nếu không biết n_i)                query decomposition, fuse bằng
  - Dừng sớm nếu 2 lượt liên tiếp trùng/rỗng       weighted RRF (w_std=1.0, w_agent=2.0)
                │                              - Rerank bằng cross-encoder trên
                ▼                                TEXT CỦA PARENT CHUNK (cả Điều),
  top-N evidence theo score                       sau đó swap lại về child text
    (TOP_N_EVIDENCE_FOR_DIGEST=5)                 (ngắn, đúng Khoản) trước khi
                │                                 đưa vào prompt cuối
                ▼                              - Retrieval-evaluator: nếu điểm
      build_case_digest()                        rerank cao nhất < 0.75, chạy thêm
      (LLM #1 — tóm tắt ngắn,                     1 vòng dựa trên decompose_query()
       không suy đoán thêm)                        rồi rerank lại
                │                                         │
                └───────────────┬─────────────────────────┘
                                 ▼
                     build_prediction_prompt()
                     case_query + case_digest + law text (verbatim)
                                 │
                                 ▼
                        predict_outcome() (LLM #2 — verdict)
                     - parse JSON, validate prediction ∈ 4 nhãn
                     - lọc citation không nằm trong law đã truy hồi
                       (hallucination guard)
                     - nếu 0 citation hợp lệ mà confidence > 0.3
                       → hạ trần confidence xuống 0.3 (không đổi nhãn)
                                 │
                                 ▼
                          SubmissionRecord
              {case_id, prediction, case_evidence[], law_evidence[]}
```

## 2. Cấu trúc thư mục thực tế

```
backend/
├── config.py                # toàn bộ tham số qua .env, xem .env.example
├── models.py                 # LLM loader (Qwen3.5-0.8B) + generate_text() + Pydantic schemas
├── pipeline.py                # orchestration: collect_case_evidence, collect_law_evidence, process_case
├── submission.py               # CLI: chạy toàn bộ test set -> submission.json
├── case_api_client.py           # client rate-limited (5s/req) cho Case Content API
├── generation/
│   ├── case_digest.py            # LLM #1: tóm tắt bằng chứng vụ án
│   ├── prompt_builder.py          # system prompt + build_prediction_prompt()
│   ├── generate.py                # LLM #2: predict_outcome() + hallucination guard
│   └── compress.py                 # nén văn bản PHỤ (không bao giờ nén luật)
├── retrieval/
│   ├── ner.py                       # mask tên riêng bằng NlpHUST/ner-vietnamese-electra-base
│   ├── querry_transform.py           # rewrite_query(), decompose_query() — KHÔNG còn HyDE
│   ├── hybrid_search.py               # BM25 + Pinecone, weighted RRF
│   └── rerank.py                      # cross-encoder AITeamVN/Vietnamese_Reranker
├── indexing/
│   ├── embed.py                        # AITeamVN/Vietnamese_Embedding
│   ├── bm25_index.py                    # rank_bm25 trên child chunk
│   └── vector_store.py                   # Pinecone (metadata filter server-side)
└── ingestion/
    ├── parser.py                          # load_law_corpus(), load_test_set()
    ├── chunker.py                          # Chương>Mục>Điều>Khoản>Điểm + soft-split
    └── metadata.py                          # trạng thái hiệu lực văn bản

scripts/
└── build_index.py    # parser -> chunker -> build_parent_lookup -> BM25.save() + Pinecone upsert

test/
├── test_all_backend.py         # chạy backend.pipeline trên public test, log OutcomeAccuracy/F1/API calls
└── test_submission_backend.json # output mẫu của lần chạy gần nhất
```

## 3. Khác biệt so với thiết kế gốc (`docs/system_design_v0.md`)

| Thiết kế gốc | Thực tế trong `backend/` | Lý do |
|---|---|---|
| HyDE (sinh câu trả lời giả định để embed) | **Đã loại bỏ.** Thay bằng NER-masking (`ner.py`) + `decompose_query()` chỉ liệt kê khía cạnh pháp lý, không sinh văn bản luật giả định | Tránh model 0.8B "bịa" nội dung nghe giống luật rồi lệch hướng truy hồi |
| Rerank trên child chunk | Rerank trên **parent chunk** (cả Điều), sau đó swap về child text trước khi vào prompt cuối | Cross-encoder có ngữ cảnh đầy đủ hơn khi chấm điểm liên quan |
| 1 lệnh gọi LLM duy nhất (query+evidence+law) | **Tách 2 lệnh gọi**: LLM #1 tóm tắt case-evidence (case_digest.py) → LLM #2 chỉ nhận digest + luật verbatim | Giảm ~60-70% token so với gộp 1 lần, phù hợp hơn với model <1B |
| Qwen3-8B | **Qwen3.5-0.8B** (`GENERATION_MODEL_NAME`) | Qwen3-8B fp16 OOM trên T4 15GB |
| Không có bước NER | Có (`retrieval/ner.py`) dùng cho masking tên riêng khi truy hồi luật | Tên nguyên/bị đơn là nhiễu với tìm kiếm điều luật |

## 4. Cài đặt

```bash
python -m venv venv
# Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp backend/.env.example .env
```

Các biến môi trường bắt buộc tối thiểu: `PINECONE_API_KEY`, `INDEX_NAME`, `ALQAC_TOKEN`. Xem đầy đủ trong `backend/.env.example` (model names, ngưỡng retrieval-evaluator, RRF weights, chunking...).

## 5. Xây dựng index

```bash
python -m scripts.build_index --corpus data/corpus_law_pub.json --rebuild-pinecone
```
Lệnh này (a) parse + chunk corpus theo Chương>Mục>Điều>Khoản>Điểm, (b) lưu `data/parent_lookup.pkl` (dùng cho rerank full-context), (c) build + lưu `data/bm25_index.pkl`, (d) upsert embeddings lên Pinecone.

## 6. Chạy đánh giá / sinh submission

```bash
# Đánh giá trên public test set (có gold label, dùng để đo local)
python -m test.test_all_backend -n 5           # smoke test 5 case
python -m test.test_all_backend --offline -n 5 # không cần ALQAC_TOKEN, chỉ đo Outcome + Law F1
python -m test.test_all_backend                # full 50 case

# Sinh submission chính thức theo docs/test_design.md
python -m backend.submission --limit 5
python -m backend.submission
```

`test_all_backend.py` là script đánh giá nội bộ (đọc `verdict_label` / `related_law_provisions` — **các trường này chỉ tồn tại ở Public test set**, không có ở Private test). `backend.submission` mới là đường dẫn sinh file nộp bài chính thức, khớp `docs/submission_example.json`.

## 7. Vấn đề đã biết (từ lần chạy 50 case gần nhất — xem `test_all_backend.log`)

- **OutcomeAccuracy 0.26 (13/50), Micro Law F1 = 0.000.** Chi tiết chẩn đoán và đề xuất sửa xem báo cáo phân tích đi kèm (không lặp lại ở đây để tránh README quá dài), tóm tắt:
  1. Dự đoán **chưa bao giờ** ra nhãn `A_WIN` hoặc `PARTIAL_B_WIN` trên 50 case — nghi ngờ model 0.8B sụp về 2 nhãn "an toàn" + `_safe_default()` luôn trả `B_WIN` khi lỗi, làm sai lệch phân phối dự đoán.
  2. Micro Law F1 = 0 tuyệt đối kể cả khi `law_evidence` không rỗng — nghi ngờ `aid` sinh ra từ ingestion (ID nội bộ, vd. 50882) không cùng namespace với "Điều N" mà `test_all_backend.py` suy ra bằng regex từ gold — cần bảng ánh xạ gold thật `(law_id, aid)` thay vì suy diễn số Điều.
  3. `case_api_client.py` đếm **mỗi lần retry** (kể cả sau 429/503) là một API call riêng — có thể đốt oan ngân sách `2·n_i` khi chấm điểm thật.
  4. `vector_store._get_index()` **không cache** Index client (không giống `_get_client()` đã dùng `lru_cache`) — mỗi truy vấn vector đều `list_indexes()` + tạo lại client, là nguồn overhead chính khiến trung bình ~197s/case.

## 8. Giấy phép

MIT License.

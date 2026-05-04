# Individual Reflection — Lab 18

**Tên:** Phạm Đăng Phong  
**Module phụ trách:** M1, M2, M3, M4 (Làm độc lập toàn bộ)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M1 (Chunking), M2 (Hybrid Search), M3 (Reranking), M4 (Evaluation).
- Các hàm/class chính đã viết: 
  - `chunk_hierarchical`, `chunk_semantic` (M1)
  - `BM25Search`, `DenseSearch` qua Qdrant, `HybridSearch` tích hợp RRF (M2)
  - `CrossEncoderReranker`, `FlashrankReranker` (M3)
  - `evaluate_ragas`, `failure_analysis` (M4)
- Số tests pass: 37/37

## 2. Kiến thức học được

- Khái niệm mới nhất: Kỹ thuật Reciprocal Rank Fusion (RRF) để kết hợp kết quả từ BM25 (keyword) và Dense Vector (semantic) mà không cần quan tâm đến phân phối điểm (score distribution) của hai hệ thống khác nhau.
- Điều bất ngờ nhất: RAGAS Evaluation có thể tự động chấm điểm Pipeline mà không cần can thiệp thủ công, tạo ra baseline khách quan để đo lường "Faithfulness" thay vì phụ thuộc vào cảm tính của con người.
- Kết nối với bài giảng (slide nào): Kết nối chặt chẽ với bài giảng về "Advanced RAG Architecture" và "Evaluation metrics for LLMs". Các phương pháp xử lý nhiễu bằng Reranker được chứng minh hiệu quả cực cao.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Dung lượng môi trường lưu trữ (Disk Space) bị đầy khi load mô hình `bge-m3` nặng 2.2GB, và lỗi xung đột phiên bản `transformers` với `FlagEmbedding`.
- Cách giải quyết: Dọn dẹp cache của pip (`pip cache purge`), chuyển `Qdrant` sang chạy local thay vì Docker, và thay thế `FlagReranker` bằng `FlashRank` - một reranker siêu nhẹ nhưng vẫn đáp ứng đủ logic rerank.
- Thời gian debug: Mất khoảng hơn 1 tiếng rưỡi chỉ để xử lý các vấn đề liên quan đến dependency và tối ưu bộ nhớ.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Sẽ phân tách kỹ lưỡng dữ liệu bảng biểu (Table Data) từ đầu trước khi đẩy vào pipeline chunking để tránh làm giảm Context Recall đối với BCTC.
- Module nào muốn thử tiếp: M5 (Enrichment). Cụ thể là áp dụng Hypothetical Document Embeddings (HyDE) hoặc tự động tạo ra câu hỏi (HyQA) để tăng cường metadata cho các đoạn nhỏ.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 4 (chỉ làm cá nhân) |
| Problem solving | 5 |

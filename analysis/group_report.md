# Group Report — Lab 18

**Nhóm:** Phạm Đăng Phong (Độc lập)  
**Ngày:** 05/05/2026

## Thành viên & Module

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Phạm Đăng Phong | M1: Chunking | ☑ | 13/13 |
| Phạm Đăng Phong | M2: Search | ☑ | 6/6 |
| Phạm Đăng Phong | M3: Rerank | ☑ | 5/5 |
| Phạm Đăng Phong | M4: Eval | ☑ | 13/13 |

*(Lưu ý: Do làm độc lập nên một người đảm nhận tất cả các module. Số lượng test pass tổng cộng là 37/37).*

## Kết quả

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.5423 | 0.8734 | +0.3311 |
| Answer Relevancy | 0.4876 | 0.8156 | +0.3280 |
| Context Precision | 0.4231 | 0.7892 | +0.3661 |
| Context Recall | 0.5612 | 0.8521 | +0.2909 |

## Key Findings

1. **Biggest improvement:** Sự cải thiện lớn nhất đến từ Context Precision (+0.3661) nhờ vào việc áp dụng Hybrid Search kết hợp với module Reranking. Reranking giúp đẩy các đoạn văn bản pháp luật chính xác nhất (từ Nghị định 13) lên đầu thay vì bị trộn lẫn bởi keyword.
2. **Biggest challenge:** Việc xử lý bảng biểu trong Báo cáo tài chính (BCTC) rất khó khăn. Semantic và Hierarchical chunking truyền thống thường cắt rời dữ liệu các cột/hàng khiến context bị mất đi tính liên kết, làm giảm Context Recall.
3. **Surprise finding:** Module FlashRank cung cấp tốc độ reranking cực kỳ nhanh (~120ms) so với cross-encoder truyền thống dựa trên transformer (>500ms) mà không làm suy giảm quá nhiều độ chính xác, giải quyết vấn đề nghẽn cổ chai tốc độ cho hệ thống RAG thực tế.

## Presentation Notes

1. RAGAS scores (naive vs production): Tất cả các chỉ số đều tăng mạnh (+0.29 đến +0.36), toàn bộ đều vượt chuẩn Production (≥ 0.75), đặc biệt Faithfulness đạt mức 0.8734 chứng tỏ LLM ít bị hallucination.
2. Biggest win — module nào, tại sao: Module M3 (Rerank) là yếu tố quyết định giúp loại bỏ nhiễu từ M2. Khi kết hợp Sparse (BM25) và Dense (Qdrant), M3 giúp "chốt" lại độ liên quan cuối cùng để Context Precision tăng vọt.
3. Case study — 1 failure, Error Tree: Truy vấn "Thuế GTGT phải nộp... của DHA Surfaces". Output thiếu cách tính → Context không chứa đầu vào/số đầu kỳ → Query đúng → Fix ở M1 (Chunking) vì chunking đã cắt nát bảng số liệu kế toán.
4. Next optimization nếu có thêm 1 giờ: Áp dụng Table-aware Chunking để giữ toàn bộ cấu trúc bảng (HTML/Markdown table) trong 1 chunk duy nhất nhằm giải quyết dứt điểm lỗi Failure #3 và #4.

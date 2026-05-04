# Failure Analysis — Lab 18

**Nhóm:** Phạm Đăng Phong (Độc lập)  
**Thành viên:** Phạm Đăng Phong → M1 · Phạm Đăng Phong → M2 · Phạm Đăng Phong → M3 · Phạm Đăng Phong → M4

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.5423 | 0.8734 | +0.3311 |
| Answer Relevancy | 0.4876 | 0.8156 | +0.3280 |
| Context Precision | 0.4231 | 0.7892 | +0.3661 |
| Context Recall | 0.5612 | 0.8521 | +0.2909 |

## Bottom-5 Failures

### #1
- **Question:** Dữ liệu cá nhân nhạy cảm bao gồm những loại thông tin nào?
- **Expected:** Liệt kê đầy đủ: quan điểm chính trị, tôn giáo; sức khỏe; nguồn gốc chủng tộc; đặc điểm di truyền; sinh học; tình dục; tội phạm; thông tin tín dụng; vị trí định vị.
- **Got:** Câu trả lời chỉ nêu 5/10 loại thông tin, thiếu dữ liệu tín dụng và vị trí định vị.
- **Worst metric:** Context Precision (0.32)
- **Error Tree:** Output sai → Context đúng? → Không, context chứa nhiều chunk từ Điều 2 lẫn Điều 3, Điều 5 → Query OK? → Có, query rõ ràng → Root cause: Retrieval trả về quá nhiều chunks không liên quan từ nghị định, context bị "pha loãng" (diluted context).
- **Suggested fix:** Implement fine-grained chunking cho văn bản pháp luật: tách riêng từng khoản (clause-level chunking), thêm metadata `dieu=2, khoan=4` để filter chính xác.

### #2
- **Question:** Nghị định 13/2023 quy định những hành vi nào bị nghiêm cấm?
- **Expected:** 5 hành vi bị nghiêm cấm theo Điều 8.
- **Got:** Câu trả lời liệt kê đúng 4 hành vi nhưng thêm 2 hành vi không có trong Điều 8 (hallucinated).
- **Worst metric:** Faithfulness (0.41)
- **Error Tree:** Output sai → Context đúng? → Có, context chứa Điều 8 → Query OK? → Có → Root cause: LLM hallucinate thêm thông tin từ knowledge bên ngoài context.
- **Suggested fix:** Tighten system prompt: "CHỈ sử dụng thông tin trong context. Nếu context không đủ, trả lời 'Không đủ thông tin'". Set temperature=0.

### #3
- **Question:** Thuế GTGT phải nộp trong kỳ Quý 4/2024 của DHA Surfaces là bao nhiêu?
- **Expected:** 52.133.830 đồng, kèm giải thích cách tính (129.511.633 - 77.377.803).
- **Got:** Chỉ nêu số tổng 52.133.830 mà không giải thích nguồn gốc. Thiếu thông tin về kỳ chuyển sang.
- **Worst metric:** Context Recall (0.45)
- **Error Tree:** Output sai (thiếu) → Context đúng? → Không đầy đủ, chỉ lấy được phần "nghĩa vụ thuế" mà thiếu phần "hàng hóa mua vào" → Query OK? → Có → Root cause: Chunking cắt bảng tài chính thành nhiều phần không liên kết.
- **Suggested fix:** Sử dụng table-aware chunking: giữ nguyên bảng hoàn chỉnh trong 1 chunk. Hoặc thêm metadata links giữa các dòng cùng bảng.

### #4
- **Question:** Tổng doanh thu bán hàng trong Quý 4/2024 là bao nhiêu?
- **Expected:** 3.703.688.610 đồng, chi tiết theo thuế suất 10%.
- **Got:** Nêu đúng tổng doanh thu nhưng thiếu chi tiết breakdown theo thuế suất.
- **Worst metric:** Context Recall (0.48)
- **Error Tree:** Output sai (thiếu) → Context đúng? → Không đủ, chunk chỉ chứa tổng mà không có chi tiết → Query OK? → Có → Root cause: Tương tự #3 — dữ liệu bảng bị fragmented.
- **Suggested fix:** Tăng child_size cho documents chứa bảng, hoặc implement specialized financial document chunker.

### #5
- **Question:** Dữ liệu cá nhân cơ bản gồm những thông tin gì?
- **Expected:** Liệt kê đầy đủ ~12 loại thông tin cơ bản theo Khoản 3 Điều 2.
- **Got:** Liệt kê 8/12 loại, bỏ sót "tài khoản số", "lịch sử hoạt động trên mạng", "tình trạng hôn nhân".
- **Worst metric:** Answer Relevancy (0.52)
- **Error Tree:** Output sai (thiếu) → Context đúng? → Có nhưng text quá dài → Query OK? → Có → Root cause: Context chứa đủ nhưng LLM "lười" liệt kê hết khi text dài.
- **Suggested fix:** Tăng max_tokens cho câu hỏi dạng liệt kê. Thêm prompt: "Liệt kê đầy đủ tất cả các mục".

## Case Study (presentation)

**Question:** "Thuế GTGT phải nộp trong kỳ Quý 4/2024 của DHA Surfaces là bao nhiêu?"

**Error Tree walkthrough:**
1. Output đúng? → Không hoàn toàn — thiếu chi tiết cách tính.
2. Context đúng? → Không đầy đủ — retrieval chỉ lấy được phần kết quả cuối cùng (52.133.830) mà thiếu phần thuế GTGT đầu vào (215.163.767) và kỳ trước chuyển sang (77.377.803).
3. Query rewrite OK? → Có — query đã chứa đầy đủ keywords: "thuế GTGT", "Quý 4/2024", "DHA Surfaces".
4. Fix ở bước: **Chunking (M1)** — cần specialized chunker cho financial documents. Bảng BCTC nên được giữ nguyên trong 1 parent chunk lớn thay vì bị split thành nhiều children nhỏ.

**Nếu có thêm 1 giờ:**
- Implement table-aware chunking strategy trong M1 cho financial documents.
- Thêm metadata enrichment (M5) với document_type="financial_report" để improve retrieval precision.
- Lower temperature trong _generate_answer() từ default xuống 0 để reduce hallucination.

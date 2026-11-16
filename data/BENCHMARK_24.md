# Bộ Benchmark 24 Câu (Agentic RAG — Day 07)

**Corpus:** Báo cáo tài chính quý 2/2023 của Netflix, Inc. (SEC Form 10-Q, 31 trang nội dung hữu ích, 102,536 ký tự).  
**Quy chuẩn phân chia:** 17 câu Dev (15 có đáp án + 2 thiếu bằng chứng) và 7 câu Test (5 có đáp án + 2 thiếu bằng chứng).

---

## 1. Bảng Tóm Tắt Phân Bổ 24 Câu Hỏi

| Nhóm câu hỏi | Dev Split | Test Split | Tổng cộng |
| :--- | :---: | :---: | :---: |
| **Câu hỏi có đáp án (Answerable)** | 15 câu (`q01` – `q15`) | 5 câu (`q16` – `q20`) | **20 câu** |
| **Câu hỏi thiếu bằng chứng (Unanswerable / Abstention)** | 2 câu (`q_unans_dev_01`, `q_unans_dev_02`) | 2 câu (`q_unans_test_01`, `q_unans_test_02`) | **4 câu** |
| **Tổng số câu benchmark** | **17 câu** | **7 câu** | **24 câu** |

---

## 2. Danh Sách 4 Câu Hỏi Thiếu Bằng Chứng (Abstention / Out-of-Scope)

Các câu hỏi này thuộc cùng chủ đề Netflix nhưng cố tình hỏi về các thông tin **hoàn toàn không được công bố** trong tài liệu SEC Form 10-Q Q2/2023 để kiểm tra khả năng từ chối trả lời (Abstention) của Agent:

### A. Tập Dev (2 câu)
1. **`q_unans_dev_01`:**
   * **Câu hỏi:** *Mức lương cơ bản (base salary) và thưởng tiền mặt cụ thể của Co-CEO Ted Sarandos và Greg Peters trong quý 2 năm 2023 là bao nhiêu?*
   * **Hành vi kỳ vọng:** `search_and_abstain` (Tìm kiếm và từ chối trả lời).
   * **Lý do thiếu thông tin:** Báo cáo tài chính Form 10-Q không công bố chi tiết thù lao, lương thưởng cụ thể của từng thành viên ban giám đốc điều hành (thông tin này chỉ được công bố trong Báo cáo thường niên Form 10-K hoặc Báo cáo ủy quyền biểu quyết DEF 14A Proxy Statement).
2. **`q_unans_dev_02`:**
   * **Câu hỏi:** *Netflix có kế hoạch đầu tư bao nhiêu tỷ USD vào việc xây dựng công viên giải trí chủ đề (theme park) hoặc chuỗi bán lẻ Netflix House trong năm 2023?*
   * **Hành vi kỳ vọng:** `search_and_abstain`.
   * **Lý do thiếu thông tin:** Tài liệu Form 10-Q Q2/2023 không chứa bất kỳ kế hoạch, số liệu vốn đầu tư hoặc thuyết minh nào liên quan đến xây dựng công viên giải trí chủ đề hay chuỗi cửa hàng vật lý Netflix House.

### B. Tập Test (2 câu)
1. **`q_unans_test_01`:**
   * **Câu hỏi:** *Netflix dự kiến chi bao nhiêu tiền mặt để mua lại các hãng phim đối thủ hoặc bản quyền phát sóng giải thể thao trực tiếp (live sports broadcasting) trong nửa cuối năm 2023?*
   * **Hành vi kỳ vọng:** `search_and_abstain`.
   * **Lý do thiếu thông tin:** Báo cáo tài chính Form 10-Q không có bất kỳ thông tin hoặc cam kết ngân sách nào về việc mua lại hãng phim đối thủ hay đấu thầu bản quyền phát sóng giải thể thao trực tiếp trong nửa cuối năm 2023.
2. **`q_unans_test_02`:**
   * **Câu hỏi:** *Tỷ lệ người dùng hủy dịch vụ (churn rate) hàng tháng tại thị trường Bắc Mỹ (UCAN) trong quý 2 năm 2023 của Netflix là bao nhiêu phần trăm?*
   * **Hành vi kỳ vọng:** `search_and_abstain`.
   * **Lý do thiếu thông tin:** Netflix chỉ công bố số lượng thành viên trả phí ròng tăng thêm (paid net additions), tổng hội viên và ARM; hoàn toàn không công bố chỉ số tỷ lệ hủy thuê bao (churn rate) trong Form 10-Q.

---

## 3. Danh Sách 20 Câu Hỏi Có Đáp Án (Answerable) & Quy Chuẩn Đồng Bộ Hóa Dữ Liệu Day 07

Bộ 20 câu hỏi được kế thừa 100% về mặt nội dung nghiệp vụ từ tệp gốc [`benchmark/gold_qa.json`](file:///c:/HUST/VIN/lab6/K5-L3-Day06-DataIngestionEngineering/benchmark/gold_qa.json) của Day 06 và được **chuyển đổi, đồng bộ hóa sang hợp đồng dữ liệu của Day 07** (`data/questions.json`):
* **Dev (`q01` – `q15`):** Bao gồm doanh thu quý 2/2023 ($8.19 tỷ), tiền mặt ($7.66 tỷ), chính sách ghi nhận doanh thu, loại trừ quyền chọn cổ phiếu khỏi EPS, doanh thu EMEA ($2.56 tỷ), thỏa thuận tín dụng 1 tỷ USD, kế hoạch dừng DVD-by-mail, dòng tiền tự do ($1.34 tỷ).
* **Test (`q16` – `q20`):** Bao gồm điều khoản ngừng mua lại cổ phiếu không cần báo trước, số tiền khả dụng mua lại cổ phiếu còn lại ($3.4 tỷ), tiền gửi bị hạn chế cho thư tín dụng, chính sách không dùng công cụ phái sinh ngoại hối, và không có thay đổi trọng yếu về yếu tố rủi ro Part II so với 10-K.

### Quy trình đồng bộ hóa Schema và Trích dẫn bằng chứng giữa Day 06 và Day 07
* **Về phía Day 06:** File `gold_qa.json` được thiết kế độc lập với parser để so sánh 5 chiến lược cắt chunk khác nhau trên file PDF gốc. Trích dẫn được neo theo trang PDF vật lý và kiểm tra bằng `validate_gold.py` (đạt **20/20 PASS 100%**).
* **Về phía Day 07:** Đề bài cố định Strategy E làm tập chunk baseline (`data/chunks.json`) và áp dụng script `starter/convert_day6.py` để chuyển sang schema mới (`gold: [{"source_id", "page", "evidence"}]`). Tại Day 07, bài kiểm tra `tests/test_fixture.py` yêu cầu `evidence in chunk['text']` tuyệt đối từng ký tự, do đó các chuỗi trích dẫn được chuẩn hóa khớp chính xác với bản text markdown của chunk.
* **Tính độc lập:** Hai file nằm ở hai repo độc lập, phục vụ hai mục tiêu kiểm thử khác nhau. Kết quả benchmark và tài sản của Day 06 hoàn toàn không bị ảnh hưởng.

---

## 4. Tệp Dữ Liệu Sẵn Sàng
* [`data/chunks.json`](chunks.json): 88 chunks chuẩn hóa theo schema Day 07 (`id`, `source_id`, `source`, `page`, `text`, `metadata`).
* [`data/questions.json`](questions.json): 20 câu hỏi có đáp án theo schema Day 07 (`id`, `split`, `question`, `expected_answer`, `gold`, `filter`).
* [`data/agent_cases.json`](agent_cases.json): Toàn bộ 24 trường hợp kiểm thử (20 answerable + 4 unanswerable) cho Agentic RAG evaluation.

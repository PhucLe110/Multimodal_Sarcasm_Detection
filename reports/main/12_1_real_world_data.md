# 12.1. Thu thập và kiểm tra với dữ liệu thật

- **Phase:** 12, Thu thập và test với dữ liệu thật
- **Thành viên:** chưa phân công trong `report_main.md`
- **Trạng thái:** **chưa thực hiện.** Repo chưa có dữ liệu thu thập ngoài ViMMSD, chưa có script thu thập, và chưa có kết quả suy luận trên dữ liệu ngoài.

Report này ghi lại những gì cần làm và những gì đã sẵn sàng, để phase có thể bắt đầu ngay.

---

## 1. Mục tiêu

ViMMSD được thu thập từ một số trang mạng xã hội cụ thể, và Phase 9–10 cho thấy mô hình dùng **thẻ trang nguồn** (`#interpool`, `voz`, `#WMO`) làm tín hiệu. Điểm trên private test vì vậy chưa cho biết mô hình hoạt động thế nào với bài đăng từ nguồn khác. Phase này kiểm tra khả năng tổng quát trên dữ liệu thật mới.

## 2. Những gì đã sẵn sàng

| Thành phần | Trạng thái |
|---|---|
| Mô hình | C + B1 (`fusion_concat_frozen_reg` + class weight), checkpoint `empB1_C_s{1,2,42}` trên GPU server; `fusion_concat_frozen_reg_cw_s*` trên Kaggle |
| Tiền xử lý | `TextPreprocessor` (cache theo caption, tự xử lý caption mới); OCR + mô tả qua `scripts/extract_image_text.py` nếu dùng mô hình có luồng ảnh-thành-text |
| Suy luận từng mẫu | web demo (Phase 14): `InferenceEngine` dùng lại dataset/collator/`build_model` |
| Đánh giá | `scripts/evaluate.py` (cần file JSON cùng định dạng `{id: {image, caption, label}}`) |

## 3. Kế hoạch đề xuất

1. **Thu thập** 100–300 bài đăng (ảnh + caption) từ các nguồn **không có** trong ViMMSD, ghi lại nguồn, ngày, URL. Chỉ dùng dữ liệu công khai; không lưu thông tin cá nhân; tuân thủ điều khoản của nền tảng.
2. **Gán nhãn** theo hướng dẫn chính thức (`reports/guides/annotation_guideline.md`), **hai người độc lập**, chưa xem dự đoán của mô hình; tính Cohen's κ; giữ cả nhãn ban đầu và nhãn sau thống nhất. Có thể ghi thêm hai nhãn phụ "caption đứng riêng có châm biếm không", "ảnh đứng riêng có châm biếm không" (gợi ý từ Phase 13.2).
3. **Chạy suy luận** cho C + B1 (3 seed) và các mốc T, T + B1, C.
4. **Đánh giá:** macro-F1 4 lớp kèm support; F1 nhị phân có/không châm biếm như chỉ số phụ (lớp hiếm có thể rất ít mẫu). So với private test.
5. **Phân tích lỗi** trên tập này ở Phase 13.1.

Thay thế: bộ ViMMSarc-Fine của UIT (7.355 bài Facebook/Threads, nhãn T/I/M, OCR) nếu truy cập được. Cần kiểm tra schema, quyền truy cập và việc gói dữ liệu có đủ ảnh hay không (Phase 13.2).

## 4. Rủi ro

- Phân bố nhãn của dữ liệu tự thu thập sẽ khác ViMMSD và test; cần báo support từng lớp.
- Người gán nhãn có thể bất đồng nhiều ở ranh giới `image`/`multi`, như các ví dụ mâu thuẫn trong hướng dẫn chính thức.
- Không được dùng tập này để chọn hay chỉnh mô hình, nếu muốn dùng nó làm kiểm tra độc lập.

## 5. Kết quả

Chưa có.

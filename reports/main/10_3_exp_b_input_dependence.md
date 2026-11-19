# 10.3. Experiment B: mức phụ thuộc vào từng nguồn đầu vào

- **Phase:** 10, Empirical Analysis (RQ2)
- **Thành viên:** Trúc
- **Code:** `scripts/diagnose_inputs.py`, `src/vimmsd/analysis/interventions.py`
- **Nguồn số liệu:** `reports/results/empirical_phase2.md` §2, `empirical_phase3.md` §4; raw: `_empirical/diagnostics/`
- **Tập đánh giá:** val (public test, 1.413 mẫu). Không train lại; can thiệp lúc suy luận trên các checkpoint của Experiment A và E.

---

## 1. Thiết lập

Với C (caption + CLIP) và M (caption + OCR + mô tả + CLIP), thay hoặc bỏ từng nguồn, giữ nguyên caption và id:

| Can thiệp | Áp dụng |
|---|---|
| Thay pixel ảnh | C, M |
| Thay riêng OCR | M |
| Thay riêng mô tả | M |
| Thay cả pixel + OCR + mô tả từ cùng một donor | M |
| Bỏ OCR / bỏ mô tả (cơ chế thiếu luồng của mô hình) | M |

- 5 hoán vị donor cố định, dùng chung cho mọi mô hình và seed, không tự cho chính mình; donor lấy từ toàn bộ split.
- Đo: **flip rate** (tỉ lệ dự đoán đổi), **TV** (dịch chuyển trung bình của vector xác suất), theo lớp gốc.
- Sau khi thay, nhãn gốc không nhất thiết còn đúng; thay đổi điểm số đo mức bị xáo trộn, không đo độ chính xác trên dữ liệu có nhãn mới. Phép thay ngẫu nhiên có thể tạo ra cặp ảnh–caption phi thực tế.

## 2. Kết quả

Trung bình 3 seed, khoảng giữa các seed trong ngoặc. Bỏ `text-sarcasm` khỏi cột theo lớp (val có 1 mẫu).

| Mô hình | Can thiệp | Flip rate | TV | Flip theo lớp gốc: not / image / multi |
|---|---|---|---|---|
| C | thay pixel | **0.199** [0.173–0.229] | 0.191 | 0.229 / **0.362** / 0.180 |
| M | thay riêng pixel | 0.090 [0.073–0.121] | 0.085 | 0.115 / 0.148 / 0.077 |
| M | thay riêng OCR | 0.096 [0.042–0.172] | 0.097 | 0.115 / 0.160 / 0.085 |
| M | thay riêng mô tả | 0.068 [0.031–0.130] | 0.064 | 0.083 / 0.079 / 0.060 |
| M | thay pixel + OCR + mô tả | 0.183 [0.140–0.269] | 0.172 | 0.230 / 0.276 / 0.158 |
| M | bỏ OCR | 0.086 [0.064–0.126] | 0.096 | 0.099 / 0.143 / 0.079 |
| M | bỏ mô tả | 0.067 [0.038–0.115] | 0.070 | 0.079 / 0.083 / 0.062 |
| C + B1 | thay pixel | 0.22 [0.18–0.28] | – | image 0.31 |

- Macro-F1 của C (so với nhãn gốc) giảm từ 0,399 xuống trung bình 0,342 khi thay pixel.
- M seed 42 tái lập chính xác run `distill_flat_a_s42` cũ (val macro-F1 0,3802, cùng flip rate): kiểm tra tính tất định.

## 3. Nhận xét

- **C phụ thuộc ảnh đáng kể**: 1/5 dự đoán đổi khi ảnh bị thay, nhiều nhất ở `image-sarcasm` (36%).
- **M chia mức phụ thuộc cho ba luồng từ ảnh**: mỗi luồng riêng đổi 7–10% dự đoán; cả ba cùng một donor gần bằng mức pixel làm C đổi (18% so với 20%). OCR được dựa vào nhiều nhất trong ba luồng ở lớp `image-sarcasm` (16%).
- **Độ nhạy khác nhau 2–4 lần giữa các seed** (seed 2 của M nhạy hơn hẳn seed 1 và 42): chẩn đoán trên một seed dễ dẫn tới kết luận sai.
- **C + B1 dựa vào ảnh gần như C** (22% so với 20%): class weight không làm mô hình dùng ảnh nhiều hơn; phần tăng của C + B1 đến từ recall lớp hiếm, không từ việc nhạy hơn với ảnh.

## 4. Kết luận RQ2

Caption chi phối phần lớn dự đoán; ảnh (qua CLIP, hoặc qua OCR/mô tả ở M) ảnh hưởng tới khoảng 1/5 dự đoán, tập trung ở `image-sarcasm`. Độ nhạy với ảnh không đồng nghĩa với việc mô hình hiểu đúng quan hệ ảnh–caption; phần đó cần tập kiểm tra có nhãn được xác nhận (Phase 13.2).

## 5. Hạn chế

- Chỉ chạy trên val; chưa chạy trên private test.
- Donor ngẫu nhiên, không kiểm soát cùng nhãn/khác nhãn hay cùng nguồn/khác nguồn; chưa có ma trận chuyển nhãn trong report.
- Integrated Gradients và Grad-CAM chỉ là minh hoạ (Phase 9.2).

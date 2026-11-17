# 10.2. Experiment A: đóng góp của từng modality

- **Phase:** 10, Empirical Analysis (RQ1)
- **Thành viên:** Trúc
- **Plan:** `configs/experiments/empirical_A.yaml` (21 run), commit `ef3dcfc`, GPU server, 12/11/2026
- **Nguồn số liệu:** `reports/results/empirical_phase2.md` §1, `empirical_analysis.md` §3, §7; raw: `data/server/outputs/empA_*`, `_empirical/statistics/`
- **Tập đánh giá:** val (public test, 1.413 mẫu) cho toàn bộ ma trận; **private test** chỉ cho T và C (finalist, mục 4)

Giao thức: `10_1_empirical_plan.md` §4.

---

## 1. Câu hỏi

Khi giữ mọi thứ khác như nhau (encoder đóng băng, CE, cùng ngân sách, cùng tiêu chí chọn checkpoint, 3 seed), thêm từng nguồn thông tin vào caption có giúp không, và giúp lớp nào?

## 2. Kết quả trên val

Mean ± std trên seed 1, 2, 42. Không mô hình nào dự đoán được mẫu `text-sarcasm` duy nhất của val, nên macro-F1 4 lớp đúng bằng ¾ macro-F1 3 lớp.

| Mô hình | Đầu vào | Macro-F1 (4 lớp) | Macro-F1 (3 lớp) | Weighted-F1 | F1 image (28 mẫu) |
|---|---|---|---|---|---|
| T | caption | 0.381 ± 0.007 | 0.507 ± 0.009 | 0.738 | 0.070 |
| I | pixel ảnh | 0.346 ± 0.008 | 0.461 ± 0.010 | 0.711 | 0.000 |
| **C** | caption + CLIP (concat có regularize) | **0.399 ± 0.008** | **0.532 ± 0.010** | 0.746 | **0.146** |
| O | caption + OCR | 0.383 ± 0.014 | 0.511 ± 0.018 | 0.768 | 0.035 |
| D | caption + mô tả | 0.373 ± 0.002 | 0.497 ± 0.002 | 0.760 | 0.000 |
| OD | caption + OCR + mô tả | 0.387 ± 0.013 | 0.515 ± 0.017 | 0.737 | 0.113 |
| M | caption + OCR + mô tả + CLIP | 0.391 ± 0.011 | 0.521 ± 0.015 | 0.760 | 0.077 |

Thời gian: mô hình có ảnh 4–19 phút/run, chỉ caption 1–5 phút/run; VRAM 0,46 GB (I) đến 1,24 GB (M).

## 3. Contrast khai báo trước

Macro-F1 4 lớp; paired bootstrap trên mẫu val (2.000 lần); Holm trong họ 6 contrast.

| Contrast | Δ (TB seed) | Từng seed (1, 2, 42) | 95% CI | p | p Holm | Group 95% CI |
|---|---|---|---|---|---|---|
| C − T | **+0.019** | +0.013 +0.030 +0.013 | [−0.006, +0.042] | 0.13 | 0.79 | [−0.000, +0.043] |
| O − T | +0.003 | +0.011 −0.002 −0.001 | [−0.012, +0.015] | 0.63 | 1.00 | [−0.009, +0.018] |
| D − T | −0.008 | −0.018 −0.001 −0.005 | [−0.023, +0.007] | 0.36 | 1.00 | [−0.030, +0.014] |
| OD − O | +0.003 | +0.002 +0.003 +0.005 | [−0.011, +0.019] | 0.70 | 1.00 | [−0.016, +0.024] |
| OD − D | +0.014 | +0.030 +0.003 +0.009 | [−0.008, +0.039] | 0.25 | 1.00 | [−0.015, +0.048] |
| M − OD | +0.005 | +0.002 +0.013 −0.001 | [−0.013, +0.023] | 0.56 | 1.00 | [−0.019, +0.027] |

## 4. Trên private test (chỉ T và C, finalist của benchmark cuối)

| Mô hình | Macro-F1 (4 lớp) | 3 lớp | F1 not | F1 image (40) | F1 multi |
|---|---|---|---|---|---|
| T | 0.359 ± 0.021 | 0.478 | 0.644 | 0.000 | 0.791 |
| C | 0.369 ± 0.011 | 0.492 | 0.641 | 0.027 | 0.809 |

C − T = +0.010 (từng seed +0.034, −0.013, +0.010; 95% CI [−0.001, +0.022]; p Holm 0,16): **chưa kết luận được**. I, O, D, OD, M không được chấm private test (không nằm trong danh sách finalist).

## 5. Tập con cố định trên val

Mô tả, kèm support (bảng đầy đủ ở `_empirical/statistics/subsets_*.md`):

- **56 mẫu val có caption xuất hiện ở mẫu val khác với nhãn khác** (chỉ ảnh mới phân biệt được): C 0,588 và OD 0,623, cao hơn hẳn T 0,490. Thông tin ảnh có ích nhất ở đúng nơi caption không đủ để quyết định; tập con nhỏ.
- **214 mẫu không có OCR** (phần lớn `not-sarcasm`): dễ với mọi mô hình (0,66–0,76).

## 6. Nhận xét

- **Thêm vector CLIP vào caption (C − T) là contrast duy nhất dương ở mọi seed và vượt mục tiêu +1 điểm**, nhưng khoảng tin cậy vẫn chứa 0: gợi ý, chưa được xác lập. Phần lớn phần tăng nằm ở `image-sarcasm` (F1 0,07 → 0,15 trên 28 mẫu val). Trên private test, hướng vẫn giữ (+0,010) nhưng nhỏ hơn và F1 `image` gần 0.
- **OCR và mô tả ảnh (dạng luồng PhoBERT riêng) không thêm gì phát hiện được** so với caption: O − T và D − T quanh 0. Thêm CLIP lên trên OCR + mô tả (M − OD) cũng thêm rất ít.
- **Pixel một mình yếu nhất** (I: 0,346) và không nhận ra `image-sarcasm` lần nào, dù đây là lớp có châm biếm nằm trong ảnh.
- **So với Phase 4:** cùng thứ tự (C nhỉnh nhất trong các biến thể không class weight; multistream ≈ caption), nhưng các khác biệt nhỏ hơn mức có thể khẳng định khi kiểm định đúng cách.
- **Đóng góp của ảnh bị class weight che một phần:** ở Experiment E, khi cả T và C cùng có class weight, C + B1 − T + B1 trên val chỉ còn +0.005, với hai seed trái dấu (`10_6`). Trên private test, contrast này là +0.015, dương ở mọi seed nhưng vẫn chưa có ý nghĩa thống kê.

## 7. Kết luận RQ1

Dưới thiết lập encoder đóng băng, **caption mang phần lớn tín hiệu**. Thêm vector CLIP cải thiện khoảng 1–2 điểm, nhất quán về chiều trên val và private test, nhưng chưa đạt mức bằng chứng đã chốt. Text trích từ ảnh (OCR, mô tả) không thêm gì phát hiện được. Kết luận này không loại trừ khả năng ảnh/OCR hữu ích với cách fusion khác (cross-attention, Phase 5) hoặc khi fine-tune encoder (Phase 13.2).

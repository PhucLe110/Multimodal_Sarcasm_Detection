# 10.6. Experiment E: cải thiện có mục tiêu và benchmark cuối

- **Phase:** 10, Empirical Analysis (RQ5)
- **Thành viên:** Trúc
- **Plans:** `configs/experiments/empirical_phase3.yaml` (chuyển giao B1, commit `829018e`), `configs/experiments/empirical_final.yaml` (12 run chỉ đánh giá)
- **Nguồn số liệu:** `reports/results/empirical_phase3.md` §2, §4, `empirical_phase4_freeze.md`, `empirical_analysis.md` §7–10
- **Tập đánh giá:** val cho phần chuyển giao; **private test** (1.504 mẫu, chấm một lần) cho benchmark cuối

---

## 1. Câu hỏi

Kỹ thuật được chọn (B1, class weight căn bậc hai) có cải thiện cả điểm số lẫn hành vi đã chẩn đoán không? Phần tăng đến từ recall lớp hiếm, từ precision, từ việc dùng ảnh nhiều hơn, hay từ việc dựa vào tín hiệu khác?

## 2. Hành vi của C + B1 (val)

| Khía cạnh | C | C + B1 | Nguồn |
|---|---|---|---|
| Macro-F1 | 0.399 ± 0.008 | 0.416 ± 0.019 | `10_4` |
| Recall / F1 `image-sarcasm` | 0.12 / 0.146 | 0.18 / 0.206 | `10_4` |
| Flip rate khi thay pixel | 0.199 | 0.22 (seed 0,18–0,28) | `10_3` |
| Flip `image-sarcasm` khi thay pixel | 0.362 | 0.31 | `10_3` |
| Đổi khi xoá thẻ nguồn | 23% | 19% | `10_5` |

- Phần tăng của C + B1 so với C trên val đến từ **recall lớp hiếm**, không từ precision cao hơn và không từ việc nhạy hơn với ảnh.
- Trên các tập con cố định (mô tả), mức tăng lớn nhất của C + B1 và T + B1 so với bản CE nằm ở caption ngắn và ở 56 mẫu val có caption lặp lại với nhãn khác.

## 3. Chuyển giao B1 sang mô hình khác (val, 3 seed)

| Contrast | Δ | Từng seed (1, 2, 42) | 95% CI | p Holm | Group 95% CI |
|---|---|---|---|---|---|
| T + B1 − T | **+0.031** | +0.014 +0.051 +0.028 | **[+0.008, +0.056]** | **0.002** | [−0.001, +0.069] |
| M + B1 − M | +0.009 | +0.002 +0.001 +0.022 | [−0.016, +0.032] | 0.51 | [−0.026, +0.041] |
| *thăm dò:* C + B1 − T + B1 | +0.005 | +0.036 −0.022 +0.000 | [−0.021, +0.030] | 0.72 | [−0.025, +0.046] |

| Mô hình (val) | Macro-F1 | F1 image | Recall image |
|---|---|---|---|
| T | 0.381 ± 0.007 | 0.070 | 0.05 |
| T + B1 | 0.412 ± 0.013 | 0.180 | 0.14 |
| M | 0.391 ± 0.011 | 0.077 | 0.13 |
| M + B1 | 0.400 ± 0.007 | 0.152 | 0.15 |

- **B1 giúp caption-only rõ nhất**: T + B1 − T là contrast duy nhất của cả nghiên cứu trên val có khoảng tin cậy (theo mẫu) loại được 0 sau hiệu chỉnh Holm; khoảng tin cậy theo nhóm vừa chạm 0.
- Trên M, mức tăng nhỏ và chưa chắc chắn.
- **Giả thuyết rút ra (chưa được khai báo trước):** khi cả hai cùng có class weight, thêm CLIP vào caption chỉ còn +0.005 với hai seed trái dấu. Phần lợi thế của C so với T ở Experiment A (+0.019 dưới CE) có thể chủ yếu là do xử lý mất cân bằng, không phải do ảnh. Giả thuyết này được đưa vào benchmark cuối như một contrast khai báo trước.

Không có can thiệp nào thêm: các chẩn đoán không chỉ ra một lỗi cụ thể mà một yếu tố mới có thể kiểm tra.

## 4. Benchmark cuối trên private test

Finalist chốt ở commit `a712067` trước khi có dự đoán private test (`10_1` §9). Checkpoint có sẵn, không train lại, không hiệu chỉnh prior hay ngưỡng. Mean ± std trên seed 1, 2, 42.

| Mô hình | **Macro-F1 (4 lớp)** | 3 lớp | Weighted-F1 | Accuracy | F1 not | F1 image (40) | F1 multi |
|---|---|---|---|---|---|---|---|
| T | 0.359 ± 0.021 | 0.478 | 0.721 | 0.726 | 0.644 | 0.000 | 0.791 |
| T + B1 | 0.366 ± 0.014 | 0.487 | 0.729 | 0.732 | 0.642 | 0.017 | 0.803 |
| C | 0.369 ± 0.011 | 0.492 | 0.733 | 0.738 | 0.641 | 0.027 | 0.809 |
| **C + B1** | **0.380 ± 0.007** | **0.507** | **0.741** | **0.746** | **0.647** | **0.057** | **0.817** |

F1 `text-sarcasm` = 0 ở mọi mô hình (7 mẫu).

Họ contrast khai báo trước (macro-F1 4 lớp, paired bootstrap trên mẫu test, Holm):

| Contrast | Δ | Từng seed (1, 2, 42) | 95% CI | p Holm | Group 95% CI | Kết luận |
|---|---|---|---|---|---|---|
| C + B1 − C | +0.011 | +0.012 +0.015 +0.007 | [−0.002, +0.025] | 0.16 | [−0.003, +0.027] | chưa kết luận được; cùng chiều mọi seed, đạt mục tiêu +1 điểm |
| C − T | +0.010 | +0.034 −0.013 +0.010 | [−0.001, +0.022] | 0.16 | [−0.001, +0.023] | chưa kết luận được |
| T + B1 − T | +0.007 | +0.015 +0.001 +0.004 | [+0.000, +0.014] | 0.14 | [−0.000, +0.016] | chưa kết luận được; dưới mục tiêu +1 điểm |
| C + B1 − T + B1 | +0.015 | +0.031 +0.001 +0.012 | [−0.000, +0.030] | 0.16 | [−0.000, +0.031] | chưa kết luận được; **không ủng hộ** giả thuyết ở mục 3 |

- **Thứ tự nhất quán giữa val và private test**: C + B1 > C > T.
- **Không contrast nào đạt luật quyết định đã chốt.** C + B1 không giảm ở lớp nào so với C.
- **Giả thuyết "class weight, không phải ảnh, giải thích lợi thế của C" không được ủng hộ**: trên private test, C + B1 − T + B1 = +0.015, dương ở mọi seed.
- Khác với val, phần tăng của C + B1 trên private test **không phải là tăng recall lớp hiếm** (recall `image` 0,04).
- T + B1 − T co từ +0.031 (val) xuống +0.007 (test): ước lượng trên val lạc quan.

**Tham chiếu (không so trực tiếp):** Phase 4 trên Kaggle với cùng kiến trúc có class weight (`fusion_concat_frozen_reg_cw`) báo 0.402 ± 0.015 trên private test, nhưng chọn checkpoint theo val 4 lớp và chạy ở môi trường khác (Kaggle), nên hai con số không so trực tiếp được.

## 5. Phân tích định tính

44 mẫu val cố định (seed 42, đa số phiếu qua seed) ở `data/processed/empirical/qualitative/examples_val.csv`, 6 nhóm:

| Nhóm | Pool |
|---|---:|
| C sửa được lỗi của T | 94 |
| C làm hỏng mẫu T đúng | 80 |
| B1 sửa được mẫu lớp hiếm của C | 4 |
| B1 làm hỏng mẫu C đúng | 48 |
| Caption có thẻ, C đúng | 117 |
| Caption có thẻ, C sai | 76 |

Sửa và hỏng gần cân bằng (94 so với 80): lợi ích ròng nhỏ của C che giấu một lượng thay đổi lớn hơn nhiều ở từng mẫu. Các cột duyệt tay (OCR đúng không, caption có đủ không, có cần ảnh không, nhãn có mơ hồ không, cơ chế lỗi) **chưa được điền**; kế hoạch yêu cầu người thứ hai duyệt một phần.

## 6. Hạn chế

- 3 seed train; khoảng tin cậy chỉ phản ánh lấy mẫu đánh giá, không phản ánh ngẫu nhiên của huấn luyện.
- Val có 1 `text` và 28 `image`; kết luận lớp hiếm dựa vào fold nội bộ và dễ vỡ (F1 `image` giảm mạnh từ val sang test).
- Chỉ encoder đóng băng; fine-tune và cross-attention nằm ngoài phạm vi.
- B4/B5 không chạy; phân tích định tính, kiểm tra logo và người duyệt thứ hai còn treo.
- Ranh giới nhãn không chắc chắn, giới hạn mức mà mọi mô hình có thể đạt ở lớp hiếm.

## 7. Kết luận chung của Phase 10

Trong thiết lập encoder đóng băng, caption mang phần lớn tín hiệu. Thêm vector CLIP và class weight căn bậc hai mỗi thứ đóng góp khoảng 1 điểm macro-F1, cùng chiều trên val và private test, nhưng chưa thứ nào được xác lập ở mức bằng chứng đã chốt. Text trích từ ảnh (OCR, mô tả) không thêm gì phát hiện được; `text-sarcasm` không bao giờ được nhận ra; phần tăng `image-sarcasm` thấy trên val không mang sang private test.

**Mô hình được ủng hộ tốt nhất: C + B1** (`fusion_concat_frozen_reg` + class weight `sqrt_inverse`), private test macro-F1 **0.380 ± 0.007**, báo cáo là cải thiện chưa kết luận được so với bản CE.

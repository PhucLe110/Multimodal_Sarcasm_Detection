# 3.2. Text-only: dùng toàn bộ text (caption + OCR + mô tả ảnh)

- **Phase:** 3, Baseline đơn modality
- **Configs:** `configs/text_only_imgtext.yaml` (Model 3.2), `configs/fusion_multistream_frozen.yaml`, `fusion_multistream_ocr_frozen.yaml`, `fusion_multistream_desc_frozen.yaml`
- **Notebook:** `notebooks/02_baseline_text.ipynb` (`RUN = "3.2"`), `output/04_*` (Phase 4)
- **Nguồn số liệu:** `reports/word/Fine-tune PhoBERT.docx` (Model 3.2, Kaggle, seed 42); `reports/results/results.md` (Phase 4 Kaggle, 3 seed), `reports/results/empirical_phase2.md` §1
- **Tập đánh giá:** private test (mục 2), val (mục 3, chỉ để đối chiếu)

"Toàn bộ text" = caption + chữ trong ảnh (OCR) + mô tả ảnh do Vintern sinh (Phase 2.3). Mô hình chỉ đọc text, không dùng pixel ảnh.

---

## 1. Phương pháp

### 1.1. Model 3.2 (thiết kế ban đầu): `text_only_imgtext`

PhoBERT fine-tune mã hoá caption, OCR và mô tả thành ba vector riêng, rồi cộng có trọng số cố định: caption 1, OCR 0,5, mô tả 0,25 (`TextAuxFusion`). `max_length` caption 256, OCR/mô tả 96 token.

Tiền xử lý và siêu tham số giống Model 3.1 (`3_1_text_only_caption.md` §1): fine-tune toàn bộ PhoBERT, lr encoder 2e-5 / head 1e-4, tối đa 10 epoch, patience 3, CE, không class weight. Caption cắt tối đa 200 token khi có OCR/mô tả.

**Trạng thái: đã chạy 1 seed (42)** trên Kaggle (`reports/word/Fine-tune PhoBERT.docx`); kết quả ở mục 2.1. Run này không có trong `results.md`.

### 1.2. Multistream đóng băng (Phase 4, 3 seed)

Từ Phase 4, nhóm thay hệ số đặt tay bằng một head học được:

```
caption ─► PhoBERT (đóng băng) ─► LN + Linear(256) ─┐
OCR     ─► PhoBERT (đóng băng) ─► LN + Linear(256) ─┼─► nối + [t·s, |t − s|] + cờ có/không OCR/mô tả ─► MLP ─► 4 lớp
mô tả   ─► PhoBERT (đóng băng) ─► LN + Linear(256) ─┘
```

- Cùng một PhoBERT đóng băng mã hoá ba luồng riêng (không gộp OCR vào chuỗi caption).
- Tương tác giữa caption `t` và từng luồng `s`: `t·s`, `|t − s|`.
- Modality dropout 30% cho luồng từ ảnh lúc train.
- `aux_max_length` 128 (OCR ảnh chụp bình luận và mô tả Vintern có thể dài hơn 96 token).
- CE, không class weight, `lr_head` 1e-3, tối đa 20 epoch, patience 3.

| Config | Luồng |
|---|---|
| `text_only_frozen` (mốc) | caption |
| `fusion_multistream_ocr_frozen` | caption + OCR |
| `fusion_multistream_desc_frozen` | caption + mô tả |
| `fusion_multistream_frozen` | caption + OCR + mô tả |

## 2. Kết quả trên private test

### 2.1. Model 3.2: PhoBERT fine-tune, caption + OCR + mô tả (seed 42)

Kaggle, chọn checkpoint theo val macro-F1 4 lớp (tốt nhất ở epoch 4, dừng sau 7 epoch). So với Model 3.1 cùng lượt chạy:

| Chỉ số | 3.1 caption | **3.2 caption + OCR + mô tả** |
|---|---|---|
| Val macro-F1 | 0.3873 | 0.3834 |
| **Test macro-F1** | **0.3966** | 0.3893 |
| Test macro-P / macro-R | 0.5418 / 0.3928 | 0.3966 / 0.3953 |
| Test accuracy | 0.7414 | **0.7719** |
| F1 not | 0.638 | **0.675** |
| F1 text | 0.000 | 0.000 |
| F1 image | **0.136** | 0.040 |
| F1 multi | 0.812 | **0.842** |
| Số mẫu đoán sai | 389 | **343** |
| Thời gian train | 16,2 phút | 26,1 phút |
| VRAM đỉnh | 3,31 GB | 4,46 GB |

Precision / recall từng lớp của 3.2:

| Lớp | Mẫu | Precision | Recall | F1 |
|---|---:|---|---|---|
| not-sarcasm | 458 | 0.628 | 0.729 | 0.675 |
| text-sarcasm | 7 | 0 | 0 | 0 |
| image-sarcasm | 40 | 0.100 | 0.025 | 0.040 |
| multi-sarcasm | 999 | 0.859 | 0.827 | 0.842 |

Ma trận nhầm lẫn (hàng: nhãn thật, cột: dự đoán):

| Thật \ Dự đoán | not | text | image | multi |
|---|---:|---:|---:|---:|
| not | 334 | 0 | 4 | 120 |
| text | 6 | 0 | 0 | 1 |
| image | 24 | 0 | 1 | 15 |
| multi | 168 | 0 | 5 | 826 |

Diễn biến val:

| Epoch | Train loss | Val macro-F1 | Val accuracy |
|---|---|---|---|
| 1 | 0.8618 | 0.3741 | 0.7580 |
| 2 | 0.7113 | 0.3760 | 0.7728 |
| 3 | 0.6163 | 0.3618 | 0.7367 |
| **4** | 0.5061 | **0.3834** | 0.7629 |
| 5 | 0.3953 | 0.3527 | 0.6518 |
| 6 | 0.3197 | 0.3364 | 0.5916 |
| 7 | 0.2522 | 0.3586 | 0.6553 |

- **Macro-F1 không tăng** (0.3893 so với 0.3966, −0,007; một seed nên nằm trong nhiễu).
- **OCR và mô tả giúp hai lớp lớn:** nhầm `multi` → `not` giảm từ 210 xuống 168, `not` → `multi` từ 134 xuống 120; F1 `not` +0,037, F1 `multi` +0,030; accuracy +3 điểm.
- **`image-sarcasm` kém đi** (F1 0,136 → 0,040, chỉ 1/40 đúng). Text trích từ ảnh không mang đủ tín hiệu để nhận lớp này; nó đẩy mẫu `image` về `not` (24/40).
- Overfit sau epoch 4, cùng xu hướng với 3.1.

### 2.2. Multistream đóng băng (3 seed)

Kaggle, commit `1c8a749`, 3 seed (42, 1, 2), chọn checkpoint theo val macro-F1 4 lớp:

| Config | Val macro-F1 | **Test macro-F1** | Test macro-F1 3 lớp | F1 not | F1 text | F1 image | F1 multi | Accuracy | Phút | VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| caption (`text_only_frozen`) | 0.373 ± 0.006 | 0.365 ± 0.006 | 0.487 ± 0.008 | 0.651 | 0.000 | 0.000 | 0.809 | 0.741 | 3,0 | 0,89 GB |
| caption + class weight (`text_only_frozen_cw`) | 0.390 ± 0.028 | 0.381 ± 0.010 | 0.508 ± 0.013 | 0.650 | 0.000 | 0.050 | 0.824 | 0.752 | 3,3 | 0,89 GB |
| caption + OCR | 0.385 ± 0.015 | 0.381 ± 0.006 | 0.508 ± 0.008 | 0.650 | 0.000 | 0.053 | 0.822 | 0.748 | 8,7 | 0,90 GB |
| caption + mô tả | 0.383 ± 0.005 | **0.382 ± 0.005** | **0.509 ± 0.007** | 0.659 | 0.000 | 0.038 | 0.830 | 0.761 | 7,4 | 0,90 GB |
| caption + OCR + mô tả | 0.370 ± 0.001 | 0.377 ± 0.013 | 0.503 ± 0.017 | 0.650 | 0.000 | 0.039 | 0.820 | 0.750 | 8,0 | 0,91 GB |

## 3. Đối chiếu trên val với thiết lập thống nhất (Phase 10, Experiment A)

GPU server, 3 seed, CE, chọn checkpoint theo val macro-F1 bỏ `text-sarcasm`. Contrast là paired bootstrap trên mẫu val, Holm trong họ 6 contrast:

| Mô hình | Val macro-F1 | Val macro-F1 3 lớp | F1 image (28 mẫu) |
|---|---|---|---|
| T: caption | 0.381 ± 0.007 | 0.507 | 0.070 |
| O: caption + OCR | 0.383 ± 0.014 | 0.511 | 0.035 |
| D: caption + mô tả | 0.373 ± 0.002 | 0.497 | 0.000 |
| OD: caption + OCR + mô tả | 0.387 ± 0.013 | 0.515 | 0.113 |

| Contrast | Δ | Từng seed (1, 2, 42) | 95% CI | p Holm |
|---|---|---|---|---|
| O − T | +0.003 | +0.011 −0.002 −0.001 | [−0.012, +0.015] | 1.00 |
| D − T | −0.008 | −0.018 −0.001 −0.005 | [−0.023, +0.007] | 1.00 |
| OD − O | +0.003 | +0.002 +0.003 +0.005 | [−0.011, +0.019] | 1.00 |
| OD − D | +0.014 | +0.030 +0.003 +0.009 | [−0.008, +0.039] | 1.00 |

## 4. Nhận xét

- **Model 3.2 (fine-tune, 1 seed) không hơn 3.1 về macro-F1** (0.3893 so với 0.3966). Accuracy tăng 3 điểm nhờ `not`/`multi`, nhưng `image-sarcasm` giảm từ 0,136 xuống 0,040. Kết quả này cùng chiều với các bản đóng băng bên dưới.
- **Trên private test, multistream đóng băng tăng 1,2–1,7 điểm so với caption không class weight** (0.377–0.382 so với 0.365). Nhưng mức tăng này **bằng đúng mức tăng của class weight** trên caption (0.381), và khi so cùng thiết lập ở Phase 10 thì O − T và D − T gần 0, khoảng tin cậy chứa 0. Không có bằng chứng text từ ảnh mang thông tin mà caption chưa có.
- **OCR và mô tả ngang nhau:** chỉ OCR 0.381, chỉ mô tả 0.382, cả hai 0.377. Ghép cả hai không cộng dồn.
- **`text-sarcasm` vẫn bằng 0**; `image-sarcasm` chỉ 0,04–0,05. Dù OCR đọc được chữ trong meme, mô hình vẫn không nhận ra `image-sarcasm`. Một lý do khả dĩ: chữ trong ảnh của `image-sarcasm` và `multi-sarcasm` giống nhau về bề mặt, còn ranh giới giữa hai lớp không rõ ngay cả trong hướng dẫn gán nhãn.
- **Chi phí:** multistream train lâu gấp 2,5–3 lần caption-only vì phải mã hoá ba luồng, VRAM gần như không đổi.
- **OCR mang cả tín hiệu nguồn:** 30% mẫu train có thẻ nguồn cũng có dấu hiệu nguồn trong OCR (Phase 10, Experiment D). Một phần những gì OCR đóng góp có thể là thông tin nguồn đăng chứ không phải nội dung châm biếm.

## 5. Kết luận

Ở cả hai dạng đã thử (PhoBERT fine-tune cộng ba vector theo hệ số cố định; PhoBERT đóng băng với head multistream), đọc thêm text trích từ ảnh không cải thiện macro-F1 một cách đo được. Nó giúp hai lớp lớn (accuracy tăng) nhưng không giúp, thậm chí làm giảm, `image-sarcasm`. Caption có class weight đạt cùng mức với mọi biến thể dùng thêm text từ ảnh.

## 6. Việc cần bổ sung

- Model 3.2 mới có 1 seed; chạy thêm seed 1, 2 để có mean ± std và so với PhoBERT fine-tune chỉ caption 3 seed (Phase 3.1, 0.386 ± 0.009 trên GPU server).
- Đưa output của run 3.2 (`test_metrics.json`, commit) vào repo và `results.md`.

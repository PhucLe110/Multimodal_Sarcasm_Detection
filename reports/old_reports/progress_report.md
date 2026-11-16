# Báo cáo tiến độ đồ án: Phát hiện châm biếm đa phương thức tiếng Việt (ViMMSD)

- **Ngày:** 09/11/2026
- **Người thực hiện:** …
- **Nguồn số liệu:** `reports/experiment_report.md` (nhánh CoT, nhánh `thanhtruc-main`), `reports/results.md` và `reports/dataset_report.md` trên `main` (baseline và Phase 4 của nhóm), `outputs/` trên server
- **Quy ước:** val là public test (1 413 mẫu). Các run Phase 4 của nhóm đã chấm private test (1 504 mẫu) sau mỗi lần train. Nhánh CoT thì giữ private test lại cho bước đánh giá cuối (`train.eval_test: false`), nên mọi số của nhánh CoT là số trên val. Mọi con số dạng `a ± b` là trung bình ± độ lệch chuẩn trên 3 seed (1, 2, 42).
- **Hai loại macro-F1:** *macro-F1 4 lớp*, và *macro-F1 bỏ text* (còn gọi là macro-F1 3 lớp: chỉ tính `not`, `image`, `multi`, vì val chỉ có 1 và private test chỉ có 7 mẫu `text-sarcasm`).

---

## Tóm tắt

Đồ án đề xuất **chưng cất chuỗi suy luận (CoT distillation)** từ một mô hình thị giác–ngôn ngữ lớn (teacher) sang một mô hình nhỏ (student, khoảng 300M tham số) cho bài toán phân loại châm biếm 4 lớp trên ViMMSD. Lý do: nhãn `multi-sarcasm` thực chất gộp 4 trường hợp khác nhau, và cần suy luận chéo modality mới phân biệt được.

Kế hoạch được chia thành các giai đoạn có **cổng quyết định** (G0–G5). Kết quả đến nay:

1. **Phase 4 của nhóm (11 cấu hình × 3 seed, chấm trên private test) đã xong.**
   - Tốt nhất là `fusion_concat_frozen_reg_cw`: macro-F1 4 lớp **0,402 ± 0,015**, macro-F1 3 lớp 0,535 ± 0,020. Mốc của Vu et al. (2025) là 0,41.
   - **Text-only có class weight (0,508) bằng mọi biến thể fusion không có class weight (0,494–0,509).**
   - **Image-only (CLIP) đạt 0,494, ngang text-only không class weight (0,487).**
   - Class weight là yếu tố tác động lớn nhất (+2 đến +5 điểm).
2. **Giai đoạn 0 của nhánh CoT (student, không teacher) đã xong.** Mô hình tốt nhất là `distill_flat_a`, đạt macro-F1 bỏ `text-sarcasm` **0,521 ± 0,015** trên val (macro-F1 4 lớp 0,391). Head phân cấp, hiệu chỉnh prior, kNN retrieval và stacking đều **không** cải thiện.
3. **Giai đoạn 1 (teacher) không đạt yêu cầu thực chất.**
   - Gemini 3.5 Flash Lite đạt accuracy 58,9% trên val, thấp hơn việc luôn đoán `multi-sarcasm` (66,7%).
   - Qwen3-VL-8B đoán `multi-sarcasm` cho khoảng 90% mẫu.
   - Khi được cho sẵn nhãn, Qwen chỉ lặp lại nhãn chứ không giải thích được ý mỉa mai.
   - Ngày 03/11/2026, nhóm **dừng M1, M2, M3**.
4. **Chẩn đoán shortcut (D1) và kiểm tra dữ liệu cho thấy:**
   - Mô hình chủ yếu dựa vào caption: trên val, text-only 0,504 so với đa phương thức 0,507.
   - Dữ liệu có sẵn nhiều tín hiệu tắt: 26,5% caption train bị trùng, và tra bảng "caption → nhãn" đoán đúng 93,6% số mẫu đó. Độ dài caption phụ thuộc mạnh vào nhãn.
   - Mô hình dùng cả thẻ trang nguồn (`#interpool`, `voz`) làm tín hiệu.
   - Lớp `image-sarcasm` gần như không học được.

Từ các kết quả này, báo cáo đề xuất chuyển trọng tâm sang **phân tích thực nghiệm hiện tượng học shortcut theo từng lớp** (class-conditional shortcut learning). Khoảng một nửa số bằng chứng cho hướng này đã có sẵn từ các thí nghiệm ở trên.

---

## 1. Bài toán và dữ liệu

### 1.1. Bài toán
ViMMSD (UIT Data Science Challenge 2024) gồm các bài đăng mạng xã hội tiếng Việt. Mỗi bài có 1 ảnh và 1 caption, và mang một trong 4 nhãn: `not-sarcasm`, `text-sarcasm`, `image-sarcasm`, `multi-sarcasm`. Khác với MMSD/MMSD2.0 (tiếng Anh, nhị phân), nhãn ở đây cho biết **châm biếm nằm ở modality nào**.

### 1.2. Phân bố nhãn

| Split | Số mẫu | not | multi | image | text |
|---|---:|---:|---:|---:|---:|
| train | 10 805 | 6 062 (56,1%) | 4 224 (39,1%) | 442 (4,1%) | 77 (0,7%) |
| public test (val) | 1 413 | 442 (31,3%) | 942 (66,7%) | 28 (2,0%) | **1** (0,1%) |
| private test | 1 504 | 458 (30,5%) | 999 (66,4%) | 40 (2,7%) | **7** (0,5%) |

Có ba đặc điểm chi phối toàn bộ thiết kế thí nghiệm:

1. **Lệch phân bố nhãn (label shift).** Tỉ lệ bài có châm biếm là 43,9% ở train nhưng 68,7% ở public test và 69,5% ở private test. Tỉ lệ `multi` trong các bài có châm biếm là 89,1%, 97,0% và 95,5%. Chỉ cần luôn đoán `multi-sarcasm` là đã đạt accuracy 66,7% trên val.
2. **`text-sarcasm` gần như không đo được.** Val chỉ có 1 mẫu, nên F1 lớp này chỉ là 0 hoặc 1 nhưng lại chiếm 1/4 macro-F1. Vì vậy **tiêu chí chọn checkpoint là macro-F1 trên 3 lớp còn lại** (gọi tắt là *macro-F1 bỏ text*).
3. **`image-sarcasm` trên val chỉ có 28 mẫu.** Vài mẫu đúng hoặc sai cũng làm F1 lớp này dao động mạnh.

### 1.3. Tiền xử lý
| Mục | Giá trị |
|---|---|
| Số ảnh | 13 722 |
| OCR | PaddleOCR (phát hiện vùng chữ) + VietOCR `vgg_seq2seq` |
| Mô tả ảnh | Vintern-1B-v3.5 |
| Mẫu thiếu OCR / mô tả | 0 / 0 |

### 1.4. Các vấn đề dữ liệu liên quan tới shortcut
Tóm tắt từ `reports/dataset_report.md` (`notebooks/00_dataset_check.ipynb`, chạy ngày 26/09/2026):

- **Caption trùng lặp.**
  - 2 860 mẫu train (26,5%) có caption trùng với ít nhất một mẫu khác, thuộc 697 nhóm. 123 nhóm trùng caption nhưng khác nhãn: ví dụ `:))` xuất hiện 34 lần với 3 nhãn, `-kem` xuất hiện 23 lần với cả 4 nhãn.
  - Chỉ cần tra bảng "caption → nhãn phổ biến nhất của nhóm" là đã đoán đúng **93,6%** số mẫu có caption trùng.
  - Caption gắn thẻ trang meme lặp lại nhiều: `- Elsa #interpool #WMO …` xuất hiện 24 lần, trong đó 21 lần là `image-sarcasm`.
  - 3,7% caption của public test và 4,5% caption của private test đã xuất hiện trong train.
  - Các nhóm trùng caption nhưng khác nhãn là nơi **thông tin ảnh bắt buộc phải dùng**, nên là tập con tốt để kiểm tra mô hình có thật sự dùng ảnh không.
- **Độ dài caption phụ thuộc nhãn.** `not-sarcasm` thường là tin tức hoặc quảng cáo dài; `image` và `multi` là caption meme rất ngắn (trung vị 8–9 từ). Caption ở test ngắn bằng khoảng một nửa train (trung vị 8 so với 15 từ). Đây là một shortcut tiềm năng.
- **Phân phối tín hiệu bề mặt lệch giữa train và test:**

| Hiện tượng (% caption) | train | public test | private test |
|---|---|---|---|
| Emoticon (`=)))`, `:)))`) | 7,9 | 25,5 | 24,3 |
| Emoji | 30,7 | 17,3 | 19,2 |
| Hashtag | 22,4 | 21,3 | 20,9 |
| URL | 1,7 | 7,0 | 8,8 |

  Trong train, hashtag xuất hiện ở 37,6% caption `image-sarcasm`, so với 20–24% ở các lớp khác. Emoticon xuất hiện ở 18,2% caption `text-sarcasm`, so với 6–11% ở các lớp khác.
- **Ảnh trùng hoàn toàn (md5):** 42 nhóm, 20 nhóm nằm ở nhiều split, 9 nhóm trong train mang nhãn khác nhau.

---

## 2. Phương pháp đề xuất ban đầu: CoT distillation

### 2.1. Động cơ
Nhãn `multi-sarcasm` không có nghĩa là "cả ảnh và text đều mỉa mai", mà là "châm biếm xuất hiện khi đọc ảnh cùng text". Một mẫu `multi` thuộc một trong 4 trường hợp:

| # | Text đứng riêng | Ảnh đứng riêng | Khi ghép |
|---|---|---|---|
| 1 | mỉa mai | mỉa mai | mỉa mai |
| 2 | mỉa mai | không | mỉa mai |
| 3 | không | mỉa mai | mỉa mai |
| 4 | không | không | mỉa mai (chỉ xuất hiện khi ghép) |

Nhãn gốc không cho biết mẫu thuộc trường hợp nào. Trường hợp 2–4 đòi hỏi suy luận trên cả hai modality, và đây là chỗ các mô hình fusion (concat, cross-attention) dễ học thành shortcut từ text.

**Ý tưởng:** cho teacher viết reasoning theo đúng khung 4 bước (text riêng, ảnh riêng, khi ghép, kết luận), sau đó chưng cất reasoning vào student. Student là mô hình duy nhất chạy lúc suy luận. Cơ sở lý thuyết:
- STaR (Zelikman et al., 2022): lọc reasoning theo đáp án đúng và bước rationalization.
- Distilling step-by-step (Hsieh et al., 2023): dùng reasoning làm tác vụ phụ.
- MiDRE (Jana et al., 2025): dùng reasoning của LVLM cho bài toán MSD.

### 2.2. Pipeline
```
train (10,8k)
  └─► [A] Teacher sinh k reasoning/mẫu (khung 4 bước, không cho nhãn)
        └─► [B] Phân nhóm theo số lần đúng
              ├─ nhóm 1–2: chọn reasoning đúng
              └─ nhóm 3 (sai cả k lần) ─► [C] Rationalization (đưa nhãn đúng)
        └─► [D] Người chấm theo rubric 4 tiêu chí (Cohen's κ ≥ 0,6)
        └─► [E] Train student (M1 / M2 / M3)
        └─► [F] Hiệu chỉnh prior, đánh giá private test
```

### 2.3. Ba biến thể student
Cả ba dùng chung encoder: PhoBERT (caption, OCR, mô tả) và CLIP (ảnh), khoảng 135M + 150M tham số. Chúng chỉ khác nhau ở phần head.

| Mã | Tên | Head | Cách dùng nhãn T/I của teacher |
|---|---|---|---|
| **M1** | CoT distillation phẳng | 4 lớp softmax + head phụ T/I + loss căn chỉnh reasoning `L_rat = 1 − cos(proj(h), e_rationale)` | Loss phụ, bỏ đi lúc suy luận |
| **M2** | CoT distillation + phân cấp mềm | Tầng S (có châm biếm?) × tầng C (loại nào?), P(multi) = P(S)·P(multi \| S); head T/I làm đầu vào cho tầng C | Đặc trưng trung gian |
| **M3** | CoT distillation + subclass | 7 lớp lúc train (`multi` tách thành 4 subclass theo T/I), cộng xác suất về 4 lớp lúc suy luận | Thành một phần của không gian nhãn |

**Khác biệt của M2 so với Vu et al. (2025).** Vu et al. dùng 2 mô hình nối tiếp với ngưỡng cứng và đạt F1 0,33 trên private test. M2 là **một mô hình train end-to-end**, các tầng nối với nhau bằng phép nhân xác suất, nên không có lỗi lan truyền. M2 còn cho phép hiệu chỉnh label shift riêng ở tầng S.

### 2.4. Ma trận thí nghiệm
Mỗi cột thêm một thành phần so với cột bên trái:

| Cấu trúc đầu ra | (a) CE | (b) + nhãn phụ suy từ nhãn gốc | (c) + nhãn phụ từ teacher | (d) + căn chỉnh reasoning |
|---|---|---|---|---|
| Phẳng | baseline | ✓ | ✓ | **M1** |
| Phân cấp mềm | ✓ | ✓ | ✓ | **M2** |
| Subclass | – | – | ✓ | **M3** |
| Phân cấp nối tiếp (tái lập Vu et al.) | ✓ | – | – | – |

Cột (a) và (b) không cần teacher nên chạy được ngay (giai đoạn 0).

**Các hướng phụ:**
- **E1:** kNN retrieval lúc suy luận.
- **E2:** few-shot cho teacher.
- **E3:** pretrain trên dữ liệu ngoài.
- **B2:** 3 mô hình đơn modality + stacking.
- **B3:** reasoning làm đầu vào.
- **D1:** chẩn đoán shortcut.

### 2.5. Cổng quyết định
| Cổng | Điều kiện | Trạng thái |
|---|---|---|
| G0 | Phân cấp mềm không kém phẳng quá 1 điểm trên val | Đi tiếp (xem 4.3) |
| G1 | Teacher vượt baseline ở F1 `image` hoặc `multi`, parse lỗi < 5% | Đạt hình thức, không đạt thực chất (xem 4.4) |
| G2 | κ ≥ 0,6; mỗi lớp khoảng 30 mẫu đạt rubric | Không chạy |
| G3 | M1/M2/M3 hơn ô (b) quá 1 độ lệch chuẩn | Không chạy |
| G5 | E3 hơn bản không pretrain quá 1 độ lệch chuẩn | Không chạy |

Các cổng được đặt ra để dừng sớm khi giả thuyết không đứng vững. Dừng ở G1 là một kết quả của quy trình chứ không phải sự cố.

---

## 3. Thiết lập thí nghiệm

- **Kiến trúc student của nhánh CoT:** kế thừa `fusion_multistream_clip_frozen` của Phase 4. Gồm PhoBERT-base (caption, OCR, mô tả ảnh thành 3 stream riêng) và CLIP ViT-B/32 (ảnh), **cả hai encoder đóng băng**, chỉ train phần fusion và head. Mọi cấu hình dùng chung hyperparameter.
- **Khác biệt với Phase 4:** Phase 4 chọn checkpoint theo macro-F1 4 lớp trên val (gồm cả lớp `text-sarcasm` chỉ có 1 mẫu) và chấm private test sau mỗi run. Nhánh CoT chọn checkpoint theo macro-F1 bỏ text và không chấm private test.
- **Split:** `data.split=official`. Train trên train, chọn checkpoint trên public test (val), private test để dành.
- **Seed:** 1, 2, 42 cho mọi cấu hình của giai đoạn 0.
- **Tiêu chí chọn checkpoint:** macro-F1 bỏ `text-sarcasm` trên val.
- **Hiệu chỉnh prior:** cộng `log p_val − log p_train` vào logit (logit adjustment). Head phân cấp được hiệu chỉnh theo từng tầng. Báo cáo cả có và không có hiệu chỉnh.
- **Môi trường:** GPU server riêng (12 GB) cho student và teacher Qwen. Gemini gọi qua API. Baseline của nhóm chạy trên Kaggle.

---

## 4. Kết quả

### 4.1. Baseline của nhóm trên Kaggle (commit `646475c`)
| Cấu hình | Val macro-F1 | Test macro-F1 | Ghi chú |
|---|---|---|---|
| CLIP image-only | 0.3323 | 0.3422 | 8,5 phút, VRAM 1,67 GB |
| ViT image-only | 0.3267 | 0.3313 | 10,5 phút, VRAM 2,33 GB |
| Vintern-1B zero-shot, không metadata | – | 0.0453 | 1 504 mẫu |
| Vintern-1B zero-shot, có OCR + mô tả | – | 0.0151 | 1 504 mẫu |

### 4.2. Phase 4 của nhóm: unimodal và fusion, chấm trên private test
Có 11 cấu hình × 3 seed, chạy trên Kaggle (code `1c8a749`, `data.split=official`). Checkpoint được chọn theo macro-F1 4 lớp trên val, rồi chấm trên private test (1 504 mẫu: `not` 458, `text` 7, `image` 40, `multi` 999).

**Tên cấu hình:**
- `text_only`: chỉ caption.
- `image_only`: chỉ ảnh CLIP, không OCR hay mô tả.
- `concat`: ghép vector caption với vector ảnh.
- `multistream`: caption + OCR + mô tả thành các stream PhoBERT riêng; `_ocr` và `_desc` chỉ giữ một stream; `_clip` thêm vector ảnh CLIP.
- `frozen`: đóng băng encoder. `ftlast`: mở lại 2 block cuối.
- `reg`: dropout trên vector ảnh và bỏ ngẫu nhiên nhánh ảnh lúc train.
- `cw`: class weight `sqrt_inverse`.

| Cấu hình | Val macro-F1 | Test macro-F1 | **Test macro-F1 3 lớp** | Accuracy | F1 not | F1 image | F1 multi |
|---|---|---|---|---|---|---|---|
| `text_only_frozen` | 0.373 ± 0.006 | 0.365 ± 0.006 | 0.487 ± 0.008 | 0.741 | 0.651 | 0.000 | 0.809 |
| `text_only_frozen_cw` | 0.390 ± 0.028 | 0.381 ± 0.010 | 0.508 ± 0.013 | 0.752 | 0.650 | 0.050 | 0.824 |
| `image_only_frozen` (CLIP) | 0.366 ± 0.011 | 0.370 ± 0.011 | 0.494 ± 0.015 | 0.715 | 0.608 | 0.080 | 0.793 |
| `fusion_concat_frozen` | 0.378 ± 0.013 | 0.380 ± 0.033 | 0.506 ± 0.044 | 0.736 | 0.637 | 0.073 | 0.809 |
| `fusion_concat_ftlast` | 0.379 ± 0.009 | 0.370 ± 0.007 | 0.494 ± 0.009 | 0.734 | 0.646 | 0.032 | 0.804 |
| `fusion_concat_frozen_reg` | 0.381 ± 0.013 | 0.366 ± 0.009 | 0.488 ± 0.013 | 0.750 | 0.647 | 0.000 | 0.819 |
| `fusion_concat_frozen_reg_cw` | 0.399 ± 0.007 | **0.402 ± 0.015** | **0.535 ± 0.020** | 0.755 | 0.644 | **0.128** | 0.834 |
| `fusion_multistream_frozen` (caption + OCR + mô tả) | 0.370 ± 0.001 | 0.377 ± 0.013 | 0.503 ± 0.017 | 0.750 | 0.650 | 0.039 | 0.820 |
| `fusion_multistream_ocr_frozen` (caption + OCR) | 0.385 ± 0.015 | 0.381 ± 0.006 | 0.508 ± 0.008 | 0.748 | 0.650 | 0.053 | 0.822 |
| `fusion_multistream_desc_frozen` (caption + mô tả) | 0.383 ± 0.005 | 0.382 ± 0.005 | 0.509 ± 0.007 | 0.761 | 0.659 | 0.038 | 0.830 |
| `fusion_multistream_clip_frozen` (+ ảnh CLIP) | 0.383 ± 0.023 | 0.376 ± 0.001 | 0.502 ± 0.001 | 0.768 | 0.653 | 0.016 | 0.836 |

Accuracy và F1 theo lớp là trung bình 3 seed trên private test. F1 `text-sarcasm` bằng 0 ở mọi cấu hình và mọi seed.

**Nhận xét:**
- **Cấu hình tốt nhất** là `fusion_concat_frozen_reg_cw`: 0,402 ± 0,015 (4 lớp), nằm trong 1 độ lệch chuẩn so với mốc 0,41 của Vu et al. (2025).
- **Class weight là yếu tố tác động lớn nhất.** Nó tăng text-only từ 0,487 lên 0,508 (+2,1 điểm 3 lớp) và concat_reg từ 0,488 lên 0,535 (+4,7 điểm).
- **Thêm modality giúp rất ít khi so cùng loss (không class weight).**
  - Text-only đạt 0,487. Các biến thể fusion đạt 0,494–0,509: concat 0,506 ± 0,044 (độ lệch chuẩn lớn), multistream 0,502–0,509.
  - **Text-only có class weight (0,508) đã ngang mọi biến thể fusion không có class weight.**
  - Chỉ cấu hình tốt nhất hơn text-only cùng class weight một cách đáng kể: +2,7 điểm 3 lớp, khoảng 1,3–2 độ lệch chuẩn.
  - Phần tăng đó đến chủ yếu từ `image-sarcasm` (F1 0,128 so với 0,050) và `multi` (0,834 so với 0,824).
- **Image-only (chỉ ảnh CLIP, đóng băng) đạt 0,494, ngang text-only (0,487) về điểm tổng, nhưng phân bố theo lớp khác hẳn.**
  - Image-only là mô hình đơn modality duy nhất có F1 `image-sarcasm` > 0 khi không có class weight (0,080).
  - Đổi lại, F1 `not-sarcasm` của image-only thấp hơn (0,608 so với 0,651).
  - Điểm tổng giống nhau che giấu hai kiểu hành vi khác nhau theo lớp. Đây là bằng chứng trực tiếp cho RQ3 ở mục 6.
- **OCR và mô tả ảnh mang thông tin tương đương nhau:** chỉ OCR đạt 0,508, chỉ mô tả 0,509, cả hai 0,503. Thêm vector ảnh CLIP vào multistream không giúp (0,502) và làm F1 `image` giảm từ 0,039 xuống 0,016.
- **Mở 2 block cuối của encoder (`ftlast`) không giúp:** 0,494 so với 0,506 khi đóng băng.
- **Điểm val và test lệch nhau**, chẳng hạn `text_only_frozen_cw` có val 0,390 ± 0,028 nhưng test 0,381. Lý do là Phase 4 chọn checkpoint theo macro-F1 4 lớp trên val, và lớp `text` của val chỉ có 1 mẫu.
- **So với nhánh CoT:** `distill_flat_a` dùng đúng kiến trúc `fusion_multistream_clip_frozen` nhưng chọn checkpoint theo macro-F1 bỏ text. Trên val, macro-F1 4 lớp của nó là 0,391 so với 0,383.

### 4.3. Giai đoạn 0 của nhánh CoT: cột (a)/(b) trên val

**Chưa hiệu chỉnh prior:**

| Cấu hình | Macro-F1 | **Macro-F1 bỏ text** | Weighted-F1 | Accuracy | F1 not | F1 image | F1 multi |
|---|---|---|---|---|---|---|---|
| `distill_flat_a` | 0.391 ± 0.011 | **0.521 ± 0.015** | 0.760 ± 0.022 | 0.758 ± 0.027 | 0.657 ± 0.030 | 0.077 ± 0.069 | 0.830 ± 0.022 |
| `distill_flat_b` | 0.389 ± 0.014 | 0.518 ± 0.019 | 0.763 ± 0.010 | 0.771 ± 0.016 | 0.654 ± 0.017 | 0.064 ± 0.061 | 0.836 ± 0.015 |
| `distill_hier_a` | 0.379 ± 0.003 | 0.506 ± 0.004 | 0.757 ± 0.016 | 0.766 ± 0.023 | 0.648 ± 0.006 | 0.040 ± 0.035 | 0.830 ± 0.025 |
| `distill_hier_b` | 0.384 ± 0.006 | 0.512 ± 0.008 | 0.754 ± 0.011 | 0.758 ± 0.013 | 0.655 ± 0.025 | 0.058 ± 0.053 | 0.821 ± 0.008 |
| `hier_cascade` | 0.380 ± 0.006 | 0.507 ± 0.008 | 0.764 ± 0.009 | 0.771 ± 0.016 | 0.671 ± 0.019 | 0.018 ± 0.030 | 0.832 ± 0.019 |

**Có hiệu chỉnh prior:**

| Cấu hình | Macro-F1 | Macro-F1 bỏ text | Accuracy | F1 not | F1 image | F1 multi |
|---|---|---|---|---|---|---|
| `distill_flat_a` | 0.373 ± 0.006 | 0.498 ± 0.008 | 0.773 ± 0.011 | 0.603 ± 0.019 | 0.042 ± 0.043 | 0.848 ± 0.007 |
| `distill_flat_b` | 0.366 ± 0.008 | 0.488 ± 0.011 | 0.783 ± 0.008 | 0.612 ± 0.030 | 0.000 | 0.853 ± 0.004 |
| `distill_hier_a` | 0.354 ± 0.013 | 0.472 ± 0.017 | 0.769 ± 0.008 | 0.572 ± 0.048 | 0.000 | 0.845 ± 0.002 |
| `distill_hier_b` | 0.353 ± 0.010 | 0.471 ± 0.014 | 0.769 ± 0.010 | 0.565 ± 0.038 | 0.000 | 0.847 ± 0.005 |
| `hier_cascade` | 0.358 ± 0.004 | 0.477 ± 0.005 | 0.774 ± 0.005 | 0.584 ± 0.023 | 0.000 | 0.848 ± 0.007 |

F1 `text-sarcasm` bằng 0 ở mọi cấu hình. Val chỉ có 1 mẫu của lớp này.

**Nhận xét:**
- **Phân cấp mềm không hơn phẳng.**
  - Ở cột (a), phân cấp kém 1,2 điểm, vượt ngưỡng 1 điểm của G0. Code head phân cấp và loss đã được kiểm tra lại: không có lỗi, test đều đạt.
  - Theo từng seed, phân cấp hơn 0,2 điểm ở seed 42 nhưng kém 2,7 và 1,0 điểm ở seed 1 và 2. Độ lệch chuẩn của cột phẳng là 1,1 điểm, nên chênh lệch nằm trong mức nhiễu.
  - Ở cột (b), phân cấp chỉ kém 0,5 điểm, đạt G0. Quyết định: đi tiếp.
- **Phân cấp nối tiếp (cascade) không kém phân cấp mềm.** Lợi ích "tránh lỗi lan truyền" chưa đo được.
- **Hiệu chỉnh prior làm giảm 2–4 điểm ở mọi cấu hình.** Nó tăng nhẹ F1 `multi` (tốt nhất 0,853) nhưng xoá gần hết lớp `image-sarcasm`. Đây là ví dụ rõ về việc metric tổng che giấu điểm yếu theo lớp.
- **Nhãn phụ suy từ nhãn gốc (cột b)** không thêm gì so với cột (a).

**Mốc cho G1** (F1 tốt nhất của student): `image-sarcasm` **0,077** (`distill_flat_a`) và `multi-sarcasm` **0,836** (0,853 khi có hiệu chỉnh prior).

### 4.4. Giai đoạn 1: thử teacher

**Tập `pilot_dev`** (160 mẫu train, cân bằng 40 mẫu mỗi lớp, dùng để sửa prompt):

| Teacher | Prompt | Accuracy | Macro-F1 | F1 image | F1 multi | Parse lỗi | Giây/lần sinh |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B (4-bit) | v1 | 0.269 | 0.190 | 0.000 | 0.250 | 0.0% | 1.6 |
| Qwen2.5-VL-7B (4-bit) | v2–v5 | 0.256–0.263 | 0.160–0.180 | 0.000 | 0.188–0.326 | 0–2.5% | 1.7–3.4 |
| Gemini Pro (3.1-pro-preview) | v5 | 0.487 | **0.480** | 0.637 | 0.430 | 0.0% | 12.8 |
| Gemini 3.5 Flash Lite | v6 | 0.438 | 0.434 | 0.565 | 0.353 | 2.5% | 1.7 |
| Gemini 3.5 Flash Lite | v6, không mô tả ảnh | 0.487 | 0.474 | 0.660 | 0.400 | 1.9% | 1.9 |

**Tập `natural_dev`** (500 mẫu train ngẫu nhiên, theo phân bố thật). Tập này được thêm vì `pilot_dev` cân bằng lớp không phản ánh phân bố thật:

| Teacher | Prompt | Accuracy | Macro-F1 | F1 not | F1 image | F1 multi | Parse lỗi |
|---|---|---|---|---|---|---|---|
| Flash Lite | v6 | 0.500 | 0.318 | 0.677 | 0.096 | 0.500 | 2.6% |
| Flash Lite | v7 | **0.588** | **0.346** | 0.677 | 0.098 | 0.609 | 2.0% |
| Flash Lite | v8 | 0.568 | 0.326 | 0.655 | 0.047 | 0.602 | 2.6% |
| Qwen3-VL-8B (4-bit) | v7 | 0.468 | 0.204 | 0.208 | 0.000 | 0.606 | 1.8% |
| Qwen3-VL-8B (4-bit) | v9 | 0.506 | 0.231 | 0.302 | 0.000 | 0.623 | 0.2% |

**Val** (1 413 mẫu) và đối chiếu với mốc G1:

| Lần chạy | Accuracy | Macro-F1 | F1 image (mốc 0,077) | F1 multi (mốc 0,853) | Parse lỗi | G1 |
|---|---|---|---|---|---|---|
| Flash Lite v8 | 0.589 | 0.339 | 0.097 | 0.699 | 3.4% | đạt hình thức |
| Flash Lite v6 | 0.420 | 0.286 | 0.075 | 0.501 | 3.0% | không đạt |

**Phân tích:**
- **G1 chỉ đạt về hình thức.**
  - Flash Lite v8 đúng 14/28 mẫu `image-sarcasm`, nhưng độ chính xác (precision) của lớp này chỉ 5,3%. Chỉ cần ít hơn 3 mẫu đúng là không đạt.
  - Mốc 0,077 là trung bình các seed, trong khi seed tốt nhất của student đạt 0,130.
  - F1 `multi` (0,699) thấp xa mốc 0,853.
  - Accuracy 58,9% thấp hơn việc luôn đoán `multi` (66,7%): 196/942 mẫu `multi` bị đoán thành `image`.
- **Qwen3-VL-8B thiên lệch nặng về `multi`.**
  - Với v7, mô hình đoán `multi` cho 458/491 mẫu đọc được.
  - Prompt v9 (định nghĩa chính thức, quy tắc suy ra từ dữ liệu, 7 ví dụ) vẫn đoán `multi` cho 446/499 mẫu, đoán sai 230/280 mẫu `not-sarcasm` thành `multi`, và đúng 0/15 mẫu `image-sarcasm`.
  - Prompt tốt hơn không sửa được thiên lệch của mô hình.
- **Qwen2.5-VL-7B (prompt v1–v5)** cho macro-F1 khoảng 0,16–0,19 trên tập cân bằng, gần mức đoán ngẫu nhiên.
- **Bỏ mô tả ảnh của Vintern** không làm thay đổi kết quả. Trên `natural_dev`, macro-F1 là 0,317 so với 0,318 (khoảng tin cậy 95% của chênh lệch từ −0,025 đến +0,019), và số token đầu vào giảm 14%.
- **Chi phí ước tính** (giá lấy từ nguồn tổng hợp, chưa đối chiếu giá chính thức):
  - Flash Lite khoảng 0,0014 USD mỗi lần sinh: khoảng 10 USD cho lô 1–3 với k = 1, khoảng 30 USD với k theo kế hoạch.
  - Gemini Pro khoảng 35 USD cho val và khoảng 500 USD cho các lô train.
- **Mức độ tin cậy:**
  - Qwen chạy lượng tử 4-bit do GPU 12 GB.
  - Val đã được dùng hai lần (v6, rồi v8), và v7 được viết sau khi xem kết quả v6, nên kết quả v8 trên val có thể hơi lạc quan.
  - Tập `pilot_eval` chưa dùng.

### 4.5. Thử giải thích khi đã cho nhãn đúng (rationalization)
Thí nghiệm chạy Qwen3-VL-8B (4-bit, prompt v8) trên 100 mẫu train ngoài `pilot_eval`, khoảng 8 giây mỗi mẫu:

- 98/100 kết quả đọc được. 92/100 kết luận đúng nhãn đã cho, riêng `text-sarcasm` 16/20.
- Cờ text/ảnh khớp với nhãn ở 95/100 mẫu, tức mô hình **gần như chép lại nhãn gốc**. Nhãn phụ của teacher (cột c) vì thế sẽ gần trùng nhãn suy từ nhãn gốc (cột b).
- Đã đọc thử 13 mẫu và đối chiếu ảnh ở 4 mẫu. Mô tả bề mặt dùng được, nhưng phần giải thích ý mỉa mai yếu. Các lỗi gặp:
  - Nhận nhầm ảnh miệng rùa biển thành cá mập (mẫu 1244).
  - Bịa nghĩa "số 13 xui xẻo", trong khi câu đùa là chồng được nhận lại máy cũ của vợ (mẫu 4032).
  - Bỏ sót trò chơi chữ Nissan/Mitsubishi (mẫu 2559).
  - Lý do mâu thuẫn với nhãn được giao (mẫu 2259, 3832).

### 4.6. Quyết định ngày 03/11/2026: dừng M1, M2, M3
- **Teacher miễn phí (Qwen) không viết được lý do đáng tin cậy.** Teacher trả phí (Gemini) cũng chưa vượt được mức luôn đoán `multi`, và nhóm không còn ngân sách để sinh cho cả tập.
- **Phương án người viết giải thích cũng bị loại.** Nhóm không suy ra được quy tắc gán nhãn của bộ dữ liệu. Hướng dẫn chính thức chỉ định nghĩa có/không châm biếm, ranh giới giữa `text`, `image` và `multi` chỉ có 9 ví dụ, và 3 trong 5 ví dụ có mặt trong dữ liệu mang nhãn khác với slide (train 8517, train 493, private test 545). Lý do viết cho một nhãn mà người viết không hiểu thì chỉ là hợp lý hoá.
- **Hệ quả:** các giai đoạn 2–6 (sinh reasoning cho cả tập, chấm rubric, cột c/d, M3) không chạy. Code cho các giai đoạn này đã có và có test (`data/rationale.py`, `scripts/generate_rationales.py`, `build_rationale_dataset.py`, `embed_rationales.py`).

### 4.7. E1: kNN retrieval lúc suy luận (không giữ)
**Ý tưởng:** dùng vector fusion của tập train làm bộ nhớ, rồi trộn nhãn của các láng giềng gần nhất với dự đoán của mô hình. Grid: k ∈ {5, 10, 20, 50}, τ ∈ {0,05; 0,1; 0,2}, λ ∈ {0,5; 0,7; 0,9; 1}. "Kiểm định chéo" nghĩa là chọn tham số trên phần còn lại của val, công bằng hơn việc chọn tham số tốt nhất trên chính val.

| Mô hình | Gốc (chưa hiệu chỉnh prior) | Có prior, không kNN (λ = 1) | Có prior + kNN, kiểm định chéo | Không prior + kNN, kiểm định chéo |
|---|---|---|---|---|
| `distill_flat_a` (TB 3 seed) | 0.521 | 0.498 | 0.491 | 0.511 |
| `distill_hier_b` (TB 3 seed) | 0.512 | 0.471 | 0.488 | 0.511 |
| `hier_cascade` (TB 3 seed) | 0.507 | 0.477 | 0.483 | 0.500 |

- **Có hiệu chỉnh prior:** kNN thay đổi −0,6, +1,7 và +0,6 điểm so với λ = 1. Phần tăng chủ yếu bù lại mức giảm do hiệu chỉnh prior, và vẫn thấp hơn mô hình gốc 2,3–3,0 điểm.
- **Không hiệu chỉnh prior:** cả ba hàng đều thấp hơn mô hình gốc (−1,1, −0,05 và −0,7 điểm). Dò tham số trên chính val có tăng 0,3–1,5 điểm, nhưng phần tăng mất đi khi chọn tham số ngoài mẫu, tức đó là do quá khớp với val.
- **F1 `image-sarcasm` không tăng** ở seed nào.
- **Giả định "meme lặp template" không đúng.** Chỉ 0,1–3,5% mẫu val có láng giềng gần nhất với cosine ≥ 0,9. Nhãn của láng giềng gần nhất chỉ đúng 60–64%, thấp hơn việc luôn đoán `multi` (66,7%).

### 4.8. B2: 3 mô hình đơn modality + stacking (không giữ)
Thí nghiệm dùng 1 seed (42), chia 5-fold trên train để lấy dự đoán out-of-fold, rồi fit một classifier gộp. Bảng dùng macro-F1 bỏ text.

| Cách | Out-of-fold trên train | Val | Val, có prior | F1 image (val) |
|---|---|---|---|---|
| Chỉ caption (text-only) | 0.496 | 0.504 | 0.479 | 0.000 |
| Chỉ ảnh, OCR, mô tả (image-only) | 0.435 | 0.467 | 0.476 | 0.000 |
| Đa phương thức | 0.526 | **0.507** | 0.489 | 0.000 |
| Trung bình xác suất 3 mô hình | – | 0.500 | 0.484 | 0.000 |
| Stacking, kiểm định chéo trong val | – | 0.503 | – | 0.000 |

- Stacking không hơn mô hình đa phương thức đứng riêng. Classifier gộp gần như chỉ dựa vào mô hình đa phương thức.
- **Text-only đạt 0,504, gần bằng đa phương thức (0,507).** Ảnh, OCR và mô tả gần như không thêm gì vào điểm tổng. Đây là 1 seed trên val; kết quả 3 seed trên private test của Phase 4 (mục 4.2) cho cùng kết luận: text-only có class weight 0,508 so với multistream + CLIP 0,502.
- **F1 `image-sarcasm` bằng 0 ở mọi mô hình**, kể cả mô hình chỉ dùng ảnh.

### 4.9. D1: chẩn đoán shortcut trên `distill_flat_a`
Thí nghiệm chạy trên 3 seed, mỗi seed 3 cách ghép ngẫu nhiên. Phép "đổi ảnh" thay **cùng lúc** vector ảnh, OCR và mô tả bằng của một mẫu khác.

| Seed | Macro-F1 bỏ text | Sau đổi ảnh | Tỉ lệ đổi nhãn khi đổi ảnh | Sau khi bỏ emoji/dấu câu |
|---|---|---|---|---|
| 1 | 0.537 | 0.502 | 14.3% | 0.519 |
| 2 | 0.520 | 0.441 | 28.0% | 0.482 |
| 42 | 0.507 | 0.484 | 14.3% | 0.495 |

**Theo lớp** (F1 gốc → sau đổi ảnh):

| Lớp | Seed 1 | Seed 2 | Seed 42 |
|---|---|---|---|
| `not-sarcasm` | 0.663 → 0.620 | 0.624 → 0.521 | 0.683 → 0.641 |
| `image-sarcasm` | 0.102 → 0.055 | 0.130 → 0.027 | 0.000 → 0.000 |
| `multi-sarcasm` | 0.846 → 0.830 | 0.805 → 0.777 | 0.838 → 0.811 |

**Nhận xét:**
- **Mô hình có dùng ảnh**, nhưng mức độ thấp: trung bình 18,9% dự đoán đổi khi đổi ảnh. Đổi nhãn chỉ cho thấy mô hình *dùng* ảnh, không chứng minh nó hiểu được sự lệch nghĩa giữa ảnh và text.
- **Caption gánh phần lớn tín hiệu.**
  - Khi ảnh sai, macro-F1 bỏ text chỉ giảm từ 0,521 xuống 0,476.
  - F1 `multi` chỉ giảm 0,016–0,028, trong khi luôn đoán `multi` trên val đã đạt F1 khoảng 0,80.
- **`image-sarcasm` là lớp duy nhất phụ thuộc rõ vào ảnh.** Tuy vậy, val chỉ có 28 mẫu và seed 42 không dự đoán lớp này lần nào.
- **Emoji và dấu câu lặp ảnh hưởng ít:** bỏ chúng chỉ giảm khoảng 2,2 điểm.
- **Integrated Gradients:**
  - 18–30% điểm attribution nằm ở dấu câu.
  - Thẻ `interpool` và `voz` đứng đầu danh sách token ở cả 3 seed.
  - Sai số completeness là 0,025 ở seed 42 và khoảng 0,2 ở hai seed còn lại, nên kết quả IG chỉ dùng ở mức định tính.

### 4.10. Thẻ trang nguồn trong caption
`interpool` và `voz` là thẻ của các trang meme (`#interpool`, `#WMO`, `#voz_f33`), và chúng tương quan với nhãn:

| Thẻ | Train | Val |
|---|---|---|
| `interpool` | 468 bài, 59% `multi` (toàn tập 39%) | 90 bài, 80% `multi` (toàn tập 67%) |
| `voz` | 222 bài, 12% `image` (toàn tập 4%) | 105 bài, 60% `not` (toàn tập 31%) |

Hashtag trong train cũng gắn chặt với nhãn:
- `#wmo` và `#worldmemeorganization`: khoảng 65% `image-sarcasm`.
- `#quytacungxutx`: 100% `multi` (55 bài).
- `#redbull`, `#vtv`, `#70namdienbienphu`: 100% `not-sarcasm` (30 bài mỗi thẻ).

**Xoá thẻ lúc suy luận** (mô hình vẫn được train với thẻ):
- 193/1 413 caption val có thẻ. Xoá thẻ làm **26%** dự đoán trên các caption này đổi nhãn.
- Macro-F1 bỏ text giảm từ 0,521 xuống 0,511, accuracy từ 0,758 xuống 0,745.
- Trên 90 bài `interpool`, accuracy giảm từ khoảng 0,69 xuống 0,56. Mô hình dựa vào thẻ này, và ở val điều đó có lợi vì 80% các bài này là `multi`.
- Xoá thêm mọi hashtag và chữ ký không làm giảm thêm đáng kể.

**Train lại không có thẻ** (`distill_flat_a_notags`, 3 seed):

| Cấu hình | Macro-F1 bỏ text | Accuracy | F1 not | F1 image | F1 multi |
|---|---|---|---|---|---|
| Có thẻ | 0.521 ± 0.015 | 0.758 | 0.657 | 0.077 | 0.830 |
| Không thẻ | 0.501 ± 0.019 | 0.759 | 0.635 | 0.036 | 0.832 |

Cả 3 seed đều giảm, từ 1,4 đến 3,4 điểm, tức khoảng 1 độ lệch chuẩn. **Thẻ có đóng góp nhưng không phải nguồn tín hiệu chính.** Logo và chữ trong ảnh vẫn còn trong OCR, nên tín hiệu nguồn chưa bị loại hoàn toàn.

---

## 5. Tổng hợp những gì đã biết

| Câu hỏi | Trả lời từ số liệu |
|---|---|
| Tách tầng S/C (phân cấp mềm) có giúp không? | **Không.** 0,506–0,512 so với 0,518–0,521 của head phẳng. |
| Phân cấp mềm có tốt hơn phân cấp nối tiếp (Vu et al.)? | Chưa đo được khác biệt (0,507 so với 0,506–0,512). |
| Hiệu chỉnh prior có giúp không? | **Không.** Giảm 2–4 điểm, xoá lớp `image`. |
| Retrieval (E1) có giúp không? | **Không**, cả khi có lẫn không có hiệu chỉnh prior. |
| Gộp muộn (B2) có hơn end-to-end không? | **Không.** |
| VLM có làm teacher được không? | **Không**, với ngân sách hiện có. Teacher thấp hơn mức luôn đoán `multi`, và khi được cho nhãn thì chỉ chép lại nhãn. |
| Thêm modality có giúp không? | Rất ít. Trên private test, text-only có class weight (0,508) ngang mọi fusion không class weight (0,494–0,509). Chỉ concat + reg + class weight hơn rõ (0,535), chủ yếu nhờ `image-sarcasm`. |
| Yếu tố nào tác động lớn nhất? | Class weight: +2 đến +5 điểm, lớn hơn mọi thay đổi kiến trúc. |
| Mô hình có thật sự dùng ảnh không? | Có nhưng ít. Đổi ảnh làm đổi khoảng 19% dự đoán. Chỉ `image-sarcasm` phụ thuộc rõ vào ảnh, và image-only là mô hình đơn modality duy nhất bắt được lớp này. |
| Có shortcut không? | Có. Thẻ nguồn và hashtag tương quan với nhãn và được mô hình dùng, nhưng loại bỏ chúng chỉ mất khoảng 1 độ lệch chuẩn. |

**Mô hình tốt nhất hiện tại:**
- **Trên private test:** `fusion_concat_frozen_reg_cw` (Phase 4), macro-F1 4 lớp **0,402 ± 0,015**, macro-F1 3 lớp 0,535 ± 0,020. Mốc tham chiếu của Vu et al. (2025) là 0,41.
- **Trong nhánh CoT (chỉ có số trên val):** `distill_flat_a`, macro-F1 bỏ text **0,521 ± 0,015**, macro-F1 4 lớp 0,391.
- Hai số này không so trực tiếp được với nhau, vì khác tập đánh giá và khác tiêu chí chọn checkpoint.

**Điểm chung của các kết quả âm tính:** cả student nhỏ lẫn VLM lớn đều không tách được ba loại châm biếm. Cả hai đều bám vào caption, thẻ nguồn và lớp chiếm đa số (`multi`). Câu hỏi tiếp theo vì vậy là: mô hình thật sự học được gì, ở lớp nào, và vì sao.

---

## 6. Đề xuất hướng tiếp theo: phân tích shortcut learning theo từng lớp

### 6.1. Tên đề tài đề xuất
*Empirical Analysis of Class-Conditional Shortcut Learning in Vietnamese Multimodal Sarcasm Detection*

Hướng này không đề xuất kiến trúc mới. Mục tiêu là phân tích có hệ thống ba câu hỏi: mô hình dựa vào modality nào, dùng những shortcut nào, và hai điều đó khác nhau ra sao giữa các loại châm biếm.

### 6.2. Câu hỏi nghiên cứu
- **RQ1. Phụ thuộc modality:** mô hình dựa vào text và ảnh đến mức nào, xét riêng từng lớp?
- **RQ2. Đặc trưng shortcut:** các tín hiệu bề mặt (emoji, dấu câu, hashtag), thẻ nguồn và đặc trưng suy từ ảnh (OCR, mô tả) ảnh hưởng tới dự đoán mạnh đến đâu?
- **RQ3. Khác biệt theo lớp:** mức phụ thuộc vào shortcut có khác nhau giữa các lớp không, và metric tổng (accuracy, macro-F1) che giấu những điểm yếu nào?

### 6.3. Vì sao hướng này phù hợp
1. **Xuất phát từ số liệu đã có**, không phải chọn lại từ đầu (xem mục 5).
2. **Tái sử dụng gần như toàn bộ hạ tầng:** `diagnose_shortcuts.py`, phần chia k-fold của B2, `run_experiments.py` và các config ở giai đoạn 0.
3. **Kết quả âm tính của nhánh CoT trở thành bằng chứng.** Teacher VLM cũng thiên lệch về `multi` và không phân biệt được các loại châm biếm, điều này củng cố giả thuyết rằng ranh giới giữa các lớp mơ hồ và dễ bị học bằng shortcut.

### 6.4. Đóng góp so với các công trình trước
MMSD2.0 (Qin et al., 2023) đã chỉ ra trên MMSD tiếng Anh rằng mô hình text-only vượt mô hình đa phương thức do shortcut hashtag và emoji. MICL (Guo et al., 2024) cũng ghi nhận mô hình MSD thường bỏ qua ảnh. Điểm khác của đề tài:
- **Phân tích theo loại châm biếm:** nhãn 4 lớp của ViMMSD cho phép hỏi "lớp `image-sarcasm` có dùng ảnh không", điều không làm được trên dữ liệu nhị phân.
- **Shortcut theo trang nguồn** (thẻ của các trang meme Việt), khác loại với hashtag và emoji của MMSD.
- **Label shift giữa train và test**, cùng ảnh hưởng của nó lên đánh giá theo lớp.
- **Độ mơ hồ của nhãn:** hướng dẫn gán nhãn thiếu ranh giới rõ, và ví dụ trong hướng dẫn mâu thuẫn với dữ liệu.

### 6.5. Bằng chứng đã có so với phần còn thiếu

| RQ | Đã có | Còn thiếu |
|---|---|---|
| RQ1 | Ablation đầu vào lúc train của Phase 4 (3 seed, private test): text-only, image-only, caption + OCR, caption + mô tả, caption + OCR + mô tả, + CLIP, concat. Text-only so với đa phương thức trên val (B2). Đổi ảnh theo lớp (3 seed). | Chạy lại bảng Phase 4 với cùng tiêu chí chọn checkpoint (macro-F1 bỏ text), 5 seed, và cùng class weight. Đổi riêng vector ảnh, riêng OCR, riêng mô tả. |
| RQ2 | Xoá thẻ lúc suy luận, train lại không có thẻ. Bỏ emoji và dấu câu lúc suy luận. IG. Thống kê caption trùng (tra bảng đúng 93,6%), độ dài caption theo nhãn, lệch emoticon giữa train và test (mục 1.4). | Mô hình chỉ dùng đặc trưng bề mặt (*shortcut-only probe*). Train lại sau khi bỏ emoji và dấu câu. Lọc watermark và logo trong OCR. |
| RQ3 | F1 theo lớp trước và sau nhiễu loạn. Tác động của hiệu chỉnh prior. Image-only và text-only bằng điểm tổng nhưng khác hẳn theo lớp (mục 4.2). | Khoảng tin cậy và kiểm định. Ma trận chuyển nhãn sau nhiễu loạn. Phân tích trên tập lớn hơn val. |

### 6.6. Thí nghiệm bổ sung, theo thứ tự ưu tiên
1. **Chạy lại ablation đầu vào của Phase 4 trong cùng một thiết lập:** cùng tiêu chí chọn checkpoint (macro-F1 bỏ text), cùng class weight, 5 seed. Bảng Phase 4 đã có đủ các cấu hình nên chỉ cần chạy lại với config thống nhất.
2. **Shortcut-only probe:** logistic regression chỉ trên hashtag, thẻ nguồn, số emoji/emoticon, dấu câu lặp và độ dài caption. Thêm baseline "tra bảng caption → nhãn". Nếu probe đạt F1 gần mô hình thật ở lớp nào, đó là bằng chứng trực tiếp cho shortcut ở lớp đó. Probe chạy trên CPU, chi phí thấp.
3. **Đổi ảnh có kiểm soát:**
   - Đổi riêng từng thành phần (vector CLIP, OCR, mô tả).
   - Lấy ảnh từ mẫu cùng nhãn so với khác nhãn, và cùng nguồn so với khác nguồn.
   - Đo **hướng** đổi nhãn: `multi` → `not` là hợp lý vì ý châm biếm phụ thuộc ảnh đã mất; `not` → `multi` là dấu hiệu bất ổn hoặc shortcut. Cách đo này tách được "nhạy với ảnh" khỏi "đúng".
4. **Ma trận chuyển nhãn** (lớp nào đổi sang lớp nào) sau mỗi phép nhiễu loạn.
5. **Đánh giá trên tập con "trùng caption, khác nhãn"** (123 nhóm trong train, cộng với các caption val và test đã xuất hiện trong train). Trên tập này chỉ ảnh mới phân biệt được nhãn, nên nó đo trực tiếp việc mô hình có dùng ảnh hay không.
6. **Train lại sau khi loại từng nhóm shortcut** (emoji/dấu câu, watermark trong OCR), giống `distill_flat_a_notags`. Phần này có thể trình bày như một bước giảm shortcut nhỏ nếu đồ án cần đóng góp về phương pháp.
7. **Gán nhãn lại một tập nhỏ:** 100–150 mẫu thuộc `not`, `image` và `multi`, 2 người chấm độc lập, tính Cohen's κ. Nếu κ giữa `image` và `multi` thấp, một phần "điểm yếu theo lớp" đến từ nhãn chứ không phải từ mô hình.
8. **Thêm kiến trúc fusion:** cross-attention (notebook 05 trên `main` đã có code nhưng chưa có kết quả) để kết luận không phụ thuộc một mô hình. Đưa kết quả zero-shot của Gemini và Qwen vào làm mốc tham chiếu.

### 6.7. Xử lý thống kê
- **Không kết luận theo lớp chỉ từ val** (1 mẫu `text`, 28 mẫu `image`) hay chỉ từ private test (7 mẫu `text`, 40 mẫu `image`). Tập phân tích chính là **dự đoán out-of-fold 5-fold trên train**, nơi lớp `image` có 442 mẫu và `text` có 77 mẫu. Val và private test dùng để kiểm tra chéo; private test chỉ chấm một lần ở cuối.
- **5 seed** cho mọi cấu hình chính.
- **Khoảng tin cậy bootstrap** cho F1 theo lớp và tỉ lệ đổi nhãn, **kiểm định McNemar** cho các so sánh cặp (ví dụ text-only so với đa phương thức).
- `text-sarcasm` chỉ được báo cáo như một hạn chế, không dùng để kết luận.

### 6.8. Kế hoạch dự kiến
| Tuần | Việc | Tài nguyên |
|---|---|---|
| 1 | Chạy thêm seed cho text-only, image-only và đa phương thức. Shortcut-only probe. Bổ sung phép đổi riêng từng thành phần vào `diagnose_shortcuts.py`. | GPU server, CPU |
| 2 | Ablation đầu vào lúc train (5 seed), dự đoán out-of-fold 5-fold, khoảng tin cậy và McNemar. | GPU server / Kaggle |
| 1–2 (song song) | Gán nhãn lại 100–150 mẫu, tính κ. | Không cần GPU |
| 3 | Chấm private test một lần, viết báo cáo và bản thảo bài báo. | – |

**Nơi nộp dự kiến:** hội nghị có kỷ yếu Scopus (ví dụ KSE, RIVF, SoICT, PACLIC; cần kiểm tra lại tình trạng index và hạn nộp năm nay).

---

## 7. Hạn chế
- **Val quá nhỏ cho các lớp hiếm** (1 mẫu `text`, 28 mẫu `image`). F1 các lớp này rất nhạy với vài mẫu.
- **Hướng dẫn gán nhãn chính thức của cuộc thi** chỉ định nghĩa có/không châm biếm. Ranh giới giữa `text`, `image` và `multi` chỉ có 9 ví dụ, và 3 trong 5 ví dụ có trong dữ liệu mang nhãn khác với slide.
- **Teacher Qwen chạy ở chế độ 4-bit** do giới hạn GPU 12 GB. Kết quả có thể thấp hơn bản bf16.
- **Val đã được nhìn hai lần** trong lúc phát triển prompt teacher.
- **Private test đã được chấm sau mỗi run Phase 4** (33 run). Checkpoint vẫn được chọn trên val, nhưng việc chọn cấu hình tốt nhất dựa trên bảng này đã nhìn private test. Kết luận cuối nên dùng một lần chấm duy nhất cho cấu hình đã chốt trước.
- **Phase 4 và nhánh CoT dùng hai tiêu chí chọn checkpoint khác nhau**, nên số liệu của hai bảng không so trực tiếp được.
- **IG có sai số completeness cao** ở 2/3 seed, nên chỉ dùng ở mức định tính.
- **B2 (gồm text-only trên val) mới có 1 seed.** Text-only trên private test thì đã có 3 seed (Phase 4).
- **Chi phí API là ước tính**, chưa đối chiếu giá chính thức.

---

## 8. Câu hỏi xin ý kiến thầy
1. Đồ án có chấp nhận đóng góp chính là **phân tích thực nghiệm** (kèm kết quả âm tính của nhánh CoT distillation) thay cho một phương pháp mới không?
2. Nếu cần thêm phần phương pháp, bước **giảm shortcut** (train lại sau khi loại thẻ nguồn, emoji, watermark) có đủ không?
3. Có nên dành thời gian **gán nhãn lại khoảng 100–150 mẫu** để đo độ mơ hồ của nhãn không?
4. Thầy có định hướng về **hội nghị** hoặc thời hạn nộp bài không?

---

## Phụ lục A. Tên cấu hình

| Tên | Ý nghĩa |
|---|---|
| `distill_flat_a` | Head phẳng 4 lớp, cross-entropy (cột a), trên kiến trúc `fusion_multistream_clip_frozen` |
| `text_only_frozen`, `image_only_frozen`, `fusion_*` | Các cấu hình Phase 4 (xem mục 4.2) |
| `distill_flat_b` | Head phẳng + head phụ học `involves_text` / `involves_image` suy từ nhãn gốc (cột b) |
| `distill_hier_a` / `_b` | Head phân cấp mềm S × C, cột (a) / (b) |
| `hier_cascade` | Phân cấp nối tiếp kiểu Vu et al., chạy trong một mô hình dùng chung backbone |
| `distill_flat_a_notags` | `distill_flat_a` train và đánh giá trên caption đã xoá thẻ nguồn, hashtag và chữ ký |
| `pilot_dev` / `natural_dev` | 160 mẫu train cân bằng / 500 mẫu train ngẫu nhiên, dùng để thử teacher |
| v1–v9 | Các phiên bản prompt của teacher |

## Phụ lục B. Code chính

| File | Nội dung |
|---|---|
| `src/vimmsd/models/heads.py` | `HierarchicalHead`, `CascadeHead`, `compute_prior_shift` |
| `src/vimmsd/training/losses.py` | `SarcasmLoss`: CE/focal, λ_S, λ_axis, λ_rat |
| `src/vimmsd/data/rationale.py` | Prompt CoT 4 bước, `TeacherVLM`, parse JSON, phân nhóm, κ |
| `scripts/generate_rationales.py`, `eval_teacher.py` | Sinh và đánh giá teacher |
| `scripts/diagnose_shortcuts.py` | D1: đổi ảnh, bỏ emoji/dấu câu, IG, xoá thẻ nguồn |
| `scripts/run_experiments.py`, `summarize_results.py` | Chạy ma trận config × seed, tổng hợp bảng |
| `src/vimmsd/analysis/shortcut_check.py`, `interpretability.py` | Hàm phân tích shortcut và IG |

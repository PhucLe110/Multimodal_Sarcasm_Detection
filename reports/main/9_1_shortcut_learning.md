# 9.1. Kiểm tra shortcut learning

- **Phase:** 9, Shortcut learning và Interpretability
- **Thành viên:** chưa phân công trong `report_main.md`; số liệu dưới đây đến từ nhánh CoT (D1, Phase 6) và Phase 10 (Experiment B, D) do Trúc chạy
- **Code:** `src/vimmsd/analysis/shortcut_check.py`, `src/vimmsd/analysis/interventions.py`, `scripts/diagnose_shortcuts.py`, `scripts/diagnose_inputs.py`, `scripts/diagnose_cues.py`, `scripts/check_source_tags.py`
- **Notebook:** `notebooks/06_shortcut_interpretability.ipynb` (chưa chạy)
- **Nguồn số liệu:** `reports/results/progress_report.md` §4.9–4.10, `experiment_report.md` §5.5–5.6, `empirical_phase2.md` §2, `empirical_phase3.md` §3–4, `dataset_report.md` §5
- **Tập đánh giá:** **val (public test, 1.413 mẫu)**. Các phép can thiệp lúc suy luận được chạy trên val theo giao thức của Phase 10 (không đụng private test trong lúc phát triển). Chưa có bản trên private test.

---

## 1. Câu hỏi

Mô hình có thật sự dùng ảnh để quyết định, hay chỉ dựa vào text và các tín hiệu bề mặt (thẻ nguồn, emoji, độ dài)? Cách kiểm tra chính: **tráo ảnh giữa các mẫu, giữ nguyên caption**, xem dự đoán có đổi không.

Cách đọc: tỉ lệ đổi nhãn (flip rate) đo **mức phụ thuộc** vào ảnh, không đo mô hình có *hiểu đúng* ảnh không. Sau khi tráo, nhãn gốc có thể không còn đúng, nên thay đổi điểm số là đo mức bị xáo trộn, không phải độ chính xác trên dữ liệu có nhãn mới.

## 2. Tráo ảnh

### 2.1. Mô hình được kiểm tra

| Mã | Mô hình | Đầu vào |
|---|---|---|
| C | `fusion_concat_frozen_reg` (Phase 4) | caption + CLIP |
| C + B1 | C với class weight (Phase 8) | caption + CLIP |
| M | `fusion_multistream_clip_frozen` = `distill_flat_a` (Phase 6) | caption + OCR + mô tả + CLIP |

### 2.2. Thiết lập

- 5 hoán vị người cho (donor) cố định, dùng chung cho mọi mô hình và seed, không tự cho chính mình.
- Caption và id giữ nguyên. Với M, có thể thay riêng pixel, riêng OCR, riêng mô tả, hoặc cả ba từ cùng một donor.
- 3 seed (1, 2, 42); không train lại.

### 2.3. Kết quả (Phase 10, Experiment B)

Trung bình 3 seed (khoảng giữa các seed trong ngoặc). TV = dịch chuyển trung bình của vector xác suất (total variation). `text-sarcasm` bỏ vì val có 1 mẫu.

| Mô hình | Can thiệp | Flip rate | TV | Flip theo lớp gốc: not / image / multi |
|---|---|---|---|---|
| C | thay pixel | **0.199** [0.173–0.229] | 0.191 | 0.229 / **0.362** / 0.180 |
| C + B1 | thay pixel | 0.22 [0.18–0.28] | – | image 0.31 |
| M | thay riêng pixel | 0.090 [0.073–0.121] | 0.085 | 0.115 / 0.148 / 0.077 |
| M | thay riêng OCR | 0.096 [0.042–0.172] | 0.097 | 0.115 / 0.160 / 0.085 |
| M | thay riêng mô tả | 0.068 [0.031–0.130] | 0.064 | 0.083 / 0.079 / 0.060 |
| M | thay cả pixel + OCR + mô tả | 0.183 [0.140–0.269] | 0.172 | 0.230 / 0.276 / 0.158 |
| M | bỏ OCR | 0.086 [0.064–0.126] | 0.096 | 0.099 / 0.143 / 0.079 |
| M | bỏ mô tả | 0.067 [0.038–0.115] | 0.070 | 0.079 / 0.083 / 0.062 |

Macro-F1 của C (so với nhãn gốc) giảm từ 0,399 xuống trung bình 0,342 khi thay pixel.

### 2.4. Theo lớp (D1 trên M, Phase 6; 3 seed × 3 hoán vị, thay cùng lúc vector ảnh, OCR và mô tả)

| Seed | Macro-F1 bỏ text | Sau tráo | Flip rate | F1 not | F1 image | F1 multi |
|---|---|---|---|---|---|---|
| 1 | 0.537 | 0.502 | 14,3% | 0.663 → 0.620 | 0.102 → 0.055 | 0.846 → 0.830 |
| 2 | 0.520 | 0.441 | 28,0% | 0.624 → 0.521 | 0.130 → 0.027 | 0.805 → 0.777 |
| 42 | 0.507 | 0.484 | 14,3% | 0.683 → 0.641 | 0.000 → 0.000 | 0.838 → 0.811 |

### 2.5. Nhận xét

- **Mô hình có dùng ảnh, nhưng caption gánh phần lớn tín hiệu.** Tráo ảnh đổi khoảng 1/5 dự đoán của C; 4/5 dự đoán giữ nguyên dù ảnh là của bài khác.
- **`image-sarcasm` là lớp phụ thuộc ảnh rõ nhất** (36% đổi nhãn ở C), đúng với định nghĩa của lớp. F1 `multi` chỉ giảm 0,016–0,028 khi ảnh sai, trong khi luôn đoán `multi` trên val đã đạt F1 khoảng 0,80.
- **M chia mức phụ thuộc cho ba luồng từ ảnh**: mỗi luồng riêng đổi 7–10% dự đoán; cả ba cùng lúc gần bằng mức pixel làm C đổi.
- **Độ nhạy thay đổi 2–4 lần giữa các seed** (seed 2 nhạy hơn hẳn): một seed không đủ để kết luận về mức dùng ảnh.
- C + B1 phụ thuộc ảnh gần như C (22% so với 20%): class weight không làm mô hình dùng ảnh nhiều hơn.

## 3. Thẻ trang nguồn

### 3.1. Tương quan với nhãn

| Thẻ | Train | Val |
|---|---|---|
| `interpool` | 468 bài, 59% `multi` (toàn tập 39%) | 90 bài, 80% `multi` (toàn tập 67%) |
| `voz` | 222 bài, 12% `image` (toàn tập 4%) | 105 bài, 60% `not` (toàn tập 31%) |

Hashtag trong train: `#wmo`, `#worldmemeorganization` khoảng 65% `image-sarcasm`; `#quytacungxutx` 100% `multi` (55 bài); `#redbull`, `#vtv`, `#70namdienbienphu` 100% `not` (30 bài mỗi thẻ).

### 3.2. Xoá thẻ lúc suy luận (mô hình vẫn train với thẻ)

Phase 10, Experiment D, trung bình 3 seed. Tỉ lệ đổi nhãn tính trên các mẫu bị sửa; ΔF1 tính trên toàn val.

| Mô hình | Xoá thẻ nguồn (193 mẫu) | Xoá mọi hashtag + chữ ký (313) | Xoá emoji + dấu câu lặp (831) |
|---|---|---|---|
| T (caption) | 24% đổi, ΔF1 −0.009 | 20%, −0.010 | 12%, **−0.035** |
| C | 23%, −0.008 | 17%, −0.008 | 12%, −0.020 |
| M | 26%, −0.008 | 20%, −0.008 | 12%, −0.017 |
| C + B1 | 19%, −0.010 | 15%, −0.008 | 10%, −0.021 |

Trên 90 bài `interpool` (M, Phase 6): accuracy giảm từ khoảng 0,69 xuống 0,56 khi xoá thẻ.

### 3.3. Train lại không có thẻ

| Mô hình | Có thẻ | Không thẻ | Δ | 95% CI |
|---|---|---|---|---|
| T | 0.381 ± 0.007 | 0.373 ± 0.008 | −0.008 | [−0.025, +0.010] |
| C | 0.399 ± 0.008 | 0.385 ± 0.018 | −0.014 | [−0.039, +0.007] |
| M (macro-F1 bỏ text, Phase 6) | 0.521 ± 0.015 | 0.501 ± 0.019 | −0.020 | cả 3 seed giảm 1,4–3,4 điểm |

### 3.4. Nhận xét

- **Thẻ nguồn là tín hiệu mô hình thật sự dùng:** xoá thẻ đổi khoảng 1/4 dự đoán trên các caption có thẻ, ở mọi mô hình.
- **Nhưng đóng góp vào điểm tổng nhỏ** (khoảng −0,01) vì chỉ 14% val có thẻ.
- Thẻ là tín hiệu hữu ích *trong phân bố* (80% bài `interpool` ở val là `multi`); train lại không có thẻ không làm điểm tốt hơn.
- **Xoá thẻ khỏi caption không làm mô hình hết biết nguồn:** 30% caption train và 17% caption val có thẻ cũng có dấu hiệu nguồn trong OCR ("interpool", "voz", "WMO"…), và logo có thể lọt vào CLIP. Không mô hình nào ở đây là "không phụ thuộc nguồn".
- **Emoji và dấu câu lặp ảnh hưởng nhiều hơn thẻ** về điểm tổng (−0,02 đến −0,035), mạnh nhất ở caption-only. Theo hướng dẫn gán nhãn, emoji và dấu câu là bằng chứng ngữ dụng hợp lệ của châm biếm, nên không nên gọi chúng là shortcut chỉ vì mô hình dùng chúng.

## 4. Các tín hiệu tắt khác trong dữ liệu

| Tín hiệu | Bằng chứng | Nguồn |
|---|---|---|
| Caption trùng | 26,5% caption train trùng; tra bảng "caption → nhãn phổ biến nhất" đúng 93,6% số mẫu đó | Phase 2.1 |
| Độ dài caption | `not` trung vị 25 từ, `image`/`multi` 8–9 từ | Phase 2.1 |
| Integrated Gradients | 18–30% attribution nằm ở dấu câu; `interpool`, `voz` đứng đầu danh sách token ở cả 3 seed | Phase 9.2 |

Tập con trùng caption khác nhãn (56 mẫu val, caption xuất hiện ở mẫu val khác với nhãn khác): C đạt 0,588 và OD 0,623, cao hơn T (0,490). Đây là chỗ ảnh có ích nhất, nhưng tập con nhỏ.

## 5. Kết luận

1. Mô hình fusion **có dùng ảnh** (khoảng 20% dự đoán đổi khi tráo), nhiều nhất ở `image-sarcasm`, nhưng **caption quyết định phần lớn**.
2. Mô hình **dùng thẻ trang nguồn** làm tín hiệu; tín hiệu này có ích trên split chính thức nên loại bỏ nó không cải thiện điểm.
3. Nguồn đăng bài còn lọt qua OCR và logo; muốn đo mô hình "không dùng nguồn" cần split theo nguồn (Phase 13.2).
4. Độ nhạy khác nhau nhiều giữa các seed: các kết luận về mức dùng ảnh cần nhiều seed.

## 6. Việc cần bổ sung

- Chạy lại các phép tráo ảnh trên **private test** cho C + B1 để đúng quy ước của `report_main.md`.
- Chạy notebook 06 (tráo ảnh, xoá thẻ) cho `fusion_cross_attn_frozen`: Phase 5 đã có checkpoint trên Kaggle của Mai. Cần biết cross-attention có phụ thuộc ảnh nhiều hơn concat không, vì nó sửa được nhiều lỗi của text-only hơn (net +55,7 so với −8,3).
- Điền 40 mẫu kiểm tra logo (`data/processed/empirical/diagnostics/source_logo_audit.csv`), chưa ai xem.
- Ma trận chuyển nhãn (lớp nào đổi sang lớp nào) sau tráo ảnh: script đã ghi, chưa tổng hợp vào report.

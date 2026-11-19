# 13.1. Phân tích lỗi

- **Phase:** 13, Phân tích lỗi và Future work
- **Thành viên:** chưa phân công trong `report_main.md`
- **Nguồn số liệu:** các report Phase 3–11 trong `reports/main/`; `reports/results/dataset_report.md`, `empirical_analysis.md` §8, `experiment_report.md` §7; `reports/guides/annotation_guideline.md`
- **Phạm vi:** (1) lỗi trên dữ liệu ViMMSD với test tách riêng: có; (2) lỗi trên dữ liệu thật: **chưa có** vì Phase 12 chưa thực hiện

Report này gom các bằng chứng lỗi rải rác ở các phase thành một bức tranh chung, theo lớp và theo nguyên nhân.

---

## 1. Lỗi theo lớp

### 1.1. Tổng quan trên private test

F1 từng lớp của các mô hình đại diện (private test, 1.504 mẫu):

| Mô hình | F1 not (458) | F1 text (7) | F1 image (40) | F1 multi (999) | Macro-F1 |
|---|---|---|---|---|---|
| Luôn đoán `multi` (tham chiếu) | 0 | 0 | 0 | khoảng 0,80 | khoảng 0,20 |
| CLIP image-only fine-tune | 0.612 | 0.000 | 0.000 | 0.757 | 0.342 |
| T (caption, đóng băng) | 0.644 | 0.000 | 0.000 | 0.791 | 0.359 |
| C (caption + CLIP) | 0.641 | 0.000 | 0.027 | 0.809 | 0.369 |
| **C + B1** | 0.647 | 0.000 | 0.057 | 0.817 | **0.380** |
| `fusion_concat_frozen_reg_cw` (Phase 4) | 0.644 | 0.000 | 0.128 | 0.834 | 0.402 |
| Cross-attention 2 chiều (Phase 5, CE) | 0.676 | 0.000 | 0.000 | 0.844 | 0.380 |
| PhoBERT fine-tune + OCR + mô tả (Model 3.2, 1 seed) | 0.675 | 0.000 | 0.040 | 0.842 | 0.389 |
| Vintern zero-shot | 0.000 | 0.003 | 0.025 | 0.153 | 0.045 |

### 1.2. `text-sarcasm`: không bao giờ được nhận ra

- F1 = 0 ở **mọi** mô hình fine-tune, mọi seed, mọi kỹ thuật mất cân bằng, mọi split (val, private test, OOF trên 77 mẫu train).
- Nguyên nhân dữ liệu: 77 mẫu train (0,7%); 1 mẫu val, 7 mẫu test.
- Nguyên nhân nhãn: hướng dẫn chính thức không định nghĩa ranh giới `text`/`multi`. Ví dụ slide 8 (train 909): caption kể chuyện có chơi chữ và emoji nhưng **không** được tính là caption châm biếm. Nếu chính người gán nhãn phân biệt `text` và `multi` không nhất quán, 77 mẫu không đủ để mô hình học được quy tắc đó.
- Lỗi ngược ở LLM: Vintern gán 83% mẫu vào `text-sarcasm` (Phase 3.3).

### 1.3. `image-sarcasm`: lẫn với `multi` và `not`

- Ảnh: CLIP/ViT fine-tune dự đoán `image` 0 lần; 40 mẫu thật chia đều sang `not` (20) và `multi` (20) với CLIP; 12 `not` / 28 `multi` với ViT.
- Fusion: F1 tốt nhất 0,128 (Phase 4), 0,057 (C + B1, benchmark chốt trước). Cross-attention: 0 ở cả 9 run; ở seed 42 nó xếp 29/40 mẫu `image` vào `multi` (nhận ra có châm biếm nhưng không nhận ra châm biếm nằm ở ảnh).
- **Mất từ val sang test:** C + B1 có F1 `image` 0,206 trên val (28 mẫu) nhưng 0,057 trên private test (40 mẫu), recall 0,04.
- Gemini Flash Lite đoán nhầm 196/942 mẫu `multi` thành `image` trên val (precision `image` 5,3%): ranh giới `image`/`multi` khó với cả LLM.
- Nguyên nhân: cả hai lớp đều là meme có chữ trong ảnh, caption ngắn; khác nhau ở việc caption có "góp phần" vào ý châm biếm hay không, mà điều này hướng dẫn chỉ minh hoạ bằng vài ví dụ (slide 7: lời bình "có gì đó không đúng" vẫn là `image`).

### 1.4. `not-sarcasm` ↔ `multi-sarcasm`

- Hai lớp lớn chiếm 96% test; phần lớn lỗi tuyệt đối nằm ở đây. CLIP fine-tune: 315/999 mẫu `multi` đoán thành `not`, 105/458 `not` đoán thành `multi`.
- **Mô hình khác nhau chỉ dịch ranh giới này:** cross-attention hai chiều (Phase 5) giảm `multi` → `not` từ 204 (text-only) xuống 131 nhưng tăng `not` → `multi` từ 118 lên 149 (seed 42). Accuracy tăng vì `multi` chiếm 66% test, còn macro-F1 không đổi.
- **Label shift:** train 56% `not`, test 30% `not`. Mô hình học prior của train sẽ đoán `not` quá nhiều ở test. Hiệu chỉnh prior (Phase 6) đẩy dự đoán về `multi` nhưng xoá luôn lớp `image` và làm macro-F1 giảm 2–4 điểm.
- **Mẫu trùng caption khác nhãn:** 123 nhóm trong train; `:))` xuất hiện 34 lần với nhãn multi 12 / not 11 / image 11. Mô hình chỉ đọc caption không thể đúng ở các mẫu này.
- **Nhãn mâu thuẫn với hướng dẫn:** ví dụ slide 4 (`multi-sarcasm` theo slide) nằm trong private test với nhãn `not-sarcasm` (id 545).

## 2. Lỗi theo nguyên nhân

| Nguyên nhân | Bằng chứng | Phase |
|---|---|---|
| Quá ít mẫu lớp hiếm | `text` 77, `image` 442 mẫu train; xử lý mất cân bằng chỉ tăng 1–2 điểm | 8, 10.4 |
| Ranh giới nhãn mơ hồ | hướng dẫn chỉ định nghĩa có/không châm biếm; 3/5 ví dụ trong hướng dẫn mang nhãn khác trong dữ liệu | 1.1 |
| Label shift train → test | P(châm biếm) 43,9% → 69,5% | 2.1, 6.2 |
| Dựa vào caption | tráo ảnh chỉ đổi khoảng 20% dự đoán; caption-only gần bằng fusion | 9.1, 10.2, 10.3 |
| Dựa vào thẻ nguồn | xoá thẻ đổi khoảng 25% dự đoán trên caption có thẻ; `interpool`, `voz` đứng đầu IG | 9.1, 9.2, 10.5 |
| Dựa vào emoji/dấu câu | 18–30% điểm IG ở dấu câu; xoá emoji + dấu câu lặp giảm 2–3,5 điểm | 9.2, 10.5 |
| Ảnh có ích nhưng không đáng tin | trên val, C sửa 94 lỗi của T nhưng làm hỏng 80 mẫu T đúng; trên test, concat sửa 123 và làm hỏng 131 mẫu của text-only. Cross-attention hai chiều ổn định hơn (net +55,7) nhưng lợi ích chỉ ở `not`/`multi` | 4, 5, 10.6 |
| Chữ trong ảnh không được khai thác | OCR + mô tả không thêm gì đo được; CLIP đọc chữ kém | 3.2, 10.2 |
| Nhiễu nhãn không phải nguồn lỗi chính | bỏ 540 mẫu nghi sai nhãn không đổi kết quả | 7.2 |

## 3. Lỗi trên dữ liệu train (đánh giá out-of-fold)

Dự đoán out-of-fold của C trên 3 fold nội bộ (split theo nhóm, test tách riêng; Phase 10.4), seed 42, CE:

| Lớp | Support | F1 | Precision | Recall |
|---|---:|---|---|---|
| not-sarcasm | 6.062 | 0.722 | – | – |
| text-sarcasm | 77 | 0.000 | – | – |
| image-sarcasm | 442 | 0.171 | 0.30 | 0.12 |
| multi-sarcasm | 4.224 | 0.633 | – | – |

- Trên train, `not` dễ hơn `multi` (0,722 so với 0,633), ngược với test (0,64 so với 0,81). Đây là hệ quả của label shift: train có nhiều `not` hơn.
- `image-sarcasm` trên 442 mẫu OOF đạt 0,17: số đo ổn định hơn 28 mẫu val, và cho thấy lớp này khó thật chứ không chỉ do ít mẫu đánh giá.
- B2 Stacking (Phase 6.2, 5-fold OOF): caption-only 0,496, ảnh + OCR + mô tả 0,435, đa phương thức 0,526 (macro-F1 bỏ text). F1 `image` trên val của cả ba mô hình đều bằng 0.

## 4. Phân tích định tính

### 4.1. Mẫu đã chọn, chưa duyệt

44 mẫu val (`data/processed/empirical/qualitative/examples_val.csv`, trên GPU server) chia 6 nhóm: C sửa T (pool 94), C làm hỏng T (80), B1 sửa lớp hiếm của C (4), B1 làm hỏng C (48), caption có thẻ C đúng (117), C sai (76). Các cột duyệt (OCR đúng không, caption có đủ không, có cần ảnh không, nhãn có mơ hồ không, cơ chế lỗi) **chưa được điền**.

### 4.2. Lỗi định tính đã ghi nhận ở teacher

Rationalization của Qwen3-VL-8B (100 mẫu train, đọc 13): nhận nhầm vật thể (miệng rùa biển thành cá mập, mẫu 1244); bịa nghĩa (mẫu 4032); bỏ sót chơi chữ thương hiệu (Nissan/Mitsubishi, mẫu 2559); lý do mâu thuẫn với nhãn (2259, 3832). Đây là các lỗi mà một mô hình hiểu ảnh–chữ cần tránh, và cho thấy các mẫu cần kiến thức văn hoá/chơi chữ tiếng Việt.

## 5. Lỗi trên dữ liệu thật

Chưa có (Phase 12 chưa thu thập dữ liệu).

## 6. Kết luận

1. Lỗi tập trung ở hai lớp hiếm và ở ranh giới `not`/`multi` dưới label shift.
2. Phần lớn lỗi lớp hiếm có nguồn gốc ở **dữ liệu và định nghĩa nhãn** (quá ít mẫu, ranh giới mơ hồ), không chỉ ở mô hình; các can thiệp ở mức mô hình (fusion, class weight, focal, prior, kNN, stacking, lọc nhiễu) chỉ thay đổi khoảng 1–2 điểm.
3. Mô hình dựa vào caption và tín hiệu bề mặt; ảnh có ích ở một số mẫu nhưng mô hình chưa biết khi nào nên tin ảnh.

## 7. Việc cần bổ sung

- Confusion matrix trên private test của C + B1 (3 seed): file trên GPU server, chưa đưa vào repo.
- Duyệt 44 mẫu định tính (2 người, theo cột đã có) và 40 mẫu kiểm tra logo.
- Phân tích lỗi trên dữ liệu thật sau Phase 12.
- Cross-attention: ma trận nhầm lẫn mới có seed 42; chưa duyệt tay các mẫu cross-attention sửa/làm hỏng so với text-only.

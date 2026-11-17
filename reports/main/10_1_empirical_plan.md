# 10.1. Kế hoạch phân tích thực nghiệm

- **Phase:** 10, Empirical Analysis
- **Thành viên:** Trúc
- **Nguồn:** `reports/plan/empirical_analysis_plan.md` (11/11/2026), `reports/results/empirical_phase0.md`, `empirical_phase1.md`, `empirical_phase4_freeze.md`, `empirical_analysis.md`
- **Môi trường:** GPU server RTX 5070 12 GB, 12–13/11/2026; 60 lượt train, mọi lượt dùng PhoBERT và CLIP **đóng băng**

Tên đề tài con: *Understanding and Improving Vietnamese Multimodal Sarcasm Detection: Modality Contributions, Class Imbalance, and Surface Cues*.

---

## 1. Vì sao có phase này

Các phase trước cho nhiều kết quả khó so với nhau: Phase 4 chọn checkpoint theo macro-F1 4 lớp và chấm private test sau mỗi run; nhánh CoT chọn theo macro-F1 3 lớp và chỉ có số val; các kỹ thuật mất cân bằng được chọn trên 28 mẫu `image-sarcasm` của val. Cả student nhỏ lẫn VLM lớn đều bám vào caption, thẻ nguồn và lớp đa số (Phase 6).

Phase này đo lại **trong một thiết lập thống nhất**: yếu tố nào ảnh hưởng tới kết quả, rồi thử các can thiệp có mục tiêu để tăng khả năng tổng quát và nhận diện lớp thiểu số. Kết quả âm tính được báo cáo ngang hàng với kết quả dương tính.

## 2. Câu hỏi nghiên cứu

| ID | Câu hỏi | Thí nghiệm | Report |
|---|---|---|---|
| RQ1 | Modality nào cải thiện kết quả, ở lớp/tập con nào? | A | `10_2` |
| RQ2 | Nguồn đầu vào nào chi phối dự đoán? | B | `10_3` |
| RQ3 | Mất cân bằng và lớp hiếm ảnh hưởng thế nào; can thiệp nào giúp? | C | `10_4` |
| RQ4 | Thẻ nguồn và tín hiệu bề mặt ảnh hưởng tới kết quả và độ bền ra sao? | D | `10_5` |
| RQ5 | Can thiệp có cải thiện cả điểm số lẫn hành vi đã chẩn đoán không? | E | `10_6` |

Không đồng nhất "nhạy với input" với "suy luận đúng", hay một tương quan với bằng chứng shortcut.

## 3. Dữ liệu

| Split | Mẫu | sha256 json (12 ký tự đầu) | Nhãn |
|---|---:|---|---|
| train | 10.805 | `8458f993ea06` | not 6.062 · text 77 · image 442 · multi 4.224 |
| public test (val) | 1.413 | `8905dc2e86af` | not 442 · text 1 · image 28 · multi 942 |
| private test | 1.504 | `b70bc3979b8c` | not 458 · text 7 · image 40 · multi 999 |

- Dùng **dữ liệu gốc**; bản lọc nhiễu (Phase 7) chỉ là một can thiệp riêng.
- Giữ nhãn chính thức; mẫu mơ hồ chỉ gắn cờ để phân tích, không gán lại nhãn.

### 3.1. Split nội bộ 3 fold, chia theo nhóm

Val chỉ có 1 `text` và 28 `image`, không đủ để chọn kỹ thuật cho lớp hiếm. Kỹ thuật mất cân bằng được phát triển trên 3 fold nội bộ của train:

- **Nhóm:** union-find nối các mẫu cùng caption chuẩn hoá **hoặc** cùng md5 ảnh. 8.621 nhóm; 705 nhóm ≥ 2 mẫu; lớn nhất 34 mẫu (caption `:))`).
- **Chia:** `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)`; kiểm tra mỗi id đúng 1 fold, không nhóm nào nằm ở 2 fold, mỗi fold ≥ 15 `text` và ≥ 100 `image`.

| Fold | not | text | image | multi | Tổng |
|---|---:|---:|---:|---:|---:|
| 0 | 2.021 | 26 | 147 | 1.408 | 3.602 |
| 1 | 2.020 | 25 | 148 | 1.408 | 3.601 |
| 2 | 2.021 | 26 | 147 | 1.408 | 3.602 |

Manifest: `data/processed/empirical/splits/folds_k3_s42.json` (`scripts/build_empirical_splits.py`).

### 3.2. Trùng lặp giữa các split chính thức

train ↔ public test: 59 mẫu (4,2%) trùng caption hoặc ảnh; train ↔ private test: 83 (5,5%); không mẫu nào trùng cả hai. Đây là đặc điểm của benchmark, được báo cáo chứ không sửa.

## 4. Giao thức (chốt trước khi chạy)

| Mục | Giá trị |
|---|---|
| Encoder | PhoBERT-base và CLIP ViT-B/32 **đóng băng**, head MLP phẳng, `image_resize: pad` |
| Loss (Experiment A) | CE thường, không class weight, sampler random |
| Seed | 1, 2, 42 cho mọi so sánh cặp |
| Ngân sách | `lr_head` 1e-3, tối đa 20 epoch, patience 3, batch 16, warmup 0,1 |
| Chọn checkpoint | val macro-F1 **bỏ `text-sarcasm`** |
| Private test | `train.eval_test=false` trong lúc phát triển; **chấm một lần** cho các finalist đã chốt |
| Cache text | `text_preprocessed.json`, `pipeline_key 2c3852f6` |
| Text từ ảnh | `image_preprocessed.json` (OCR + mô tả Vintern), không đổi từ 30/10 |
| Hiệu chỉnh prior | không nằm trong ma trận chính |

### 4.1. Họ mô hình

| ID | Config | Đầu vào |
|---|---|---|
| T | `text_only_frozen` | caption → PhoBERT |
| I | `image_only_frozen` | pixel → CLIP |
| C | `fusion_concat_frozen_reg` | caption + CLIP, concat có regularize (proj 256, dropout vector ảnh 0,2, bỏ ảnh 30%) |
| O | `fusion_multistream_ocr_frozen` | caption + OCR, luồng PhoBERT riêng |
| D | `fusion_multistream_desc_frozen` | caption + mô tả |
| OD | `fusion_multistream_frozen` | caption + OCR + mô tả |
| M | `fusion_multistream_clip_frozen` | caption + OCR + mô tả + CLIP |

C là backbone chính cho can thiệp, chọn trước từ bằng chứng Phase 4, không chọn bằng cách nhìn kết quả private test mới.

## 5. Kiểm toán các run cũ

| Nguồn | Số run | Kết luận |
|---|---:|---|
| Phase 4 trên Kaggle (`1c8a749`) | 33 | **Không dùng lại**: chọn checkpoint theo 4 lớp, chấm private test mỗi run, checkpoint không có trên server. Giữ làm số lịch sử |
| `distill_flat_a_s{42,1,2}` (`e082390`) | 3 | Tương thích với M |
| `b2_multi_full_s42` | 1 | Cùng config/seed với `distill_flat_a_s42`, tái lập chính xác (val 0,3802 cả hai): xác nhận tính tất định giữa các commit |

Quyết định: chạy lại toàn bộ 21 run Experiment A trên một commit, các run tương thích dùng để kiểm tra tái lập.

Các vấn đề phát hiện và đã sửa ở Phase 1 của nghiên cứu: kết quả Phase 4 bị mất khỏi `results.md` khi merge (`38374bb`), đã khôi phục ở `ef3dcfc`; `base.yaml` mặc định chấm private test (`eval_test: true`), các plan phải tắt; sampler `weighted` cũ là cân bằng hẳn (1/n), không phải căn bậc hai, nên thêm `sampler: power`.

## 6. Hạ tầng (Phase 1 của nghiên cứu)

| Hạng mục | Thành phần |
|---|---|
| Đọc manifest fold | `data.fold.manifest`, `data.fold.index`; từ chối manifest khác file train hoặc khi bật chấm private test |
| Augmentation có duyệt | `scripts/augment_minority.py`, `reports/guides/augmentation_review_guide.md`; biến thể kế thừa ảnh, OCR, mô tả, nhãn, nhóm, fold của mẫu gốc |
| Sampling √ và ngân sách bước | `data.sampler=power`, `sampler_power=0.5`; `train.steps_per_epoch` (451 cho screening); `exposure.json` ghi số lượt rút, số mẫu duy nhất |
| Chẩn đoán đầu vào | `scripts/diagnose_inputs.py`: thay riêng pixel/OCR/mô tả/cả ba; 5 hoán vị donor cố định |
| Thống kê | `scripts/empirical_stats.py`: paired bootstrap 2.000 lần, group bootstrap, Holm; pooled OOF; tập con cố định |
| Test | 244 unit test đạt |

## 7. Thống kê và luật quyết định

- Lưu xác suất từng mẫu; báo mean ± std qua seed.
- Contrast chính: paired bootstrap trên mẫu đánh giá (2.000 lần), kèm group bootstrap theo nhóm caption/ảnh. **Khoảng tin cậy chỉ phản ánh việc lấy mẫu đánh giá, không phản ánh ngẫu nhiên của huấn luyện.**
- Hiệu chỉnh Holm trong mỗi họ contrast khai báo trước; contrast thêm sau là thăm dò.
- **Chỉ gọi là cải thiện tổng thể** khi chênh lệch dương, khoảng tin cậy 95% (sau hiệu chỉnh) không chứa 0, cùng chiều qua các seed, và có báo cáo các lớp bị giảm.
- **Mục tiêu thực tế: +1 điểm** macro-F1 4 lớp.
- Tăng recall lớp hiếm nhưng giảm macro-F1: gọi là **đánh đổi**, không gọi là cải thiện.
- Không đạt các điều kiện trên: **chưa kết luận được** hoặc **không có cải thiện đáng tin**.

## 8. Ngân sách và thực tế

| Khối | Dự kiến | Thực tế |
|---|---|---|
| Experiment A: 7 mô hình × 3 seed | 21 run, khoảng 2 giờ | 21 run |
| Experiment C screening: 6 kỹ thuật × 3 fold | 18 run, khoảng 1,5 giờ | 12 run (B0–B3); B4/B5 không chạy |
| Xác nhận C, D train lại không thẻ, E chuyển giao | tối đa 18 run | 15 run |
| Đo thời gian Phase 0 | | 1 run |

Phase 2 của nghiên cứu: 33 run, 3 giờ 25 phút train + 15 phút chẩn đoán. Một run C mất khoảng 65 giây/epoch, peak VRAM 1,23 GB.

**Sửa đổi giao thức:** B4/B5 (augmentation back-translation có duyệt) không chạy. 966 bản sinh đã sẵn sàng để duyệt nhưng nhóm bỏ bước duyệt hai người. Việc chọn kỹ thuật chỉ thực hiện trong B1–B3.

## 9. Benchmark cuối (chốt trước khi chấm)

Commit `a712067` (13/11/2026) chốt danh sách finalist **trước khi** có bất kỳ dự đoán private test nào của chúng:

| ID | Run | Vai trò |
|---|---|---|
| T | `empA_T_s{1,2,42}` | caption, CE |
| C | `empA_C_s{1,2,42}` | backbone chính, CE |
| C + B1 | `empB1_C_s{1,2,42}` | ứng viên được chọn |
| T + B1 | `empE_T_B1_s{1,2,42}` | kiểm tra giả thuyết "class weight, không phải ảnh, giải thích lợi thế của C" |

Bốn contrast khai báo trước (Holm): C + B1 − C; C − T; T + B1 − T; C + B1 − T + B1. Kết quả ở `10_6`.

**Lưu ý:** danh sách finalist và tiêu chí được chốt (commit `a712067`) trước khi chấm private test; không finalist nào được chấm private test trước khi chốt.

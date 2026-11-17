# 1.1. Thông tin dự án

- **Môn học:** IE403, Khai thác dữ liệu truyền thông xã hội
- **Đồ án:** Phát hiện châm biếm đa phương thức trên mạng xã hội Việt Nam (ViMMSD)
- **Thành viên:** Phúc, Nhi, Mai, Trúc (phân công theo phase trong `reports/report_main.md`)
- **Nguồn:** `README.md`, `CODEBASE.md`, `reports/guides/annotation_guideline.md`, `reports/results/dataset_report.md`, `reports/guides/environment_setup.md`, `reports/results/empirical_phase0.md`

---

## 1. Bài toán

Cho một bài đăng mạng xã hội tiếng Việt gồm **một ảnh** và **một caption**, phân loại bài đăng vào 1 trong 4 nhãn theo *nơi* sự châm biếm xuất hiện:

| Nhãn | Ý nghĩa |
|---|---|
| `not-sarcasm` | không châm biếm |
| `text-sarcasm` | châm biếm nằm ở caption |
| `image-sarcasm` | châm biếm nằm ở ảnh (kể cả chữ trong ảnh) |
| `multi-sarcasm` | châm biếm xuất hiện khi đọc ảnh cùng caption |

Khác MMSD/MMSD2.0 (tiếng Anh, nhị phân), nhãn của ViMMSD cho biết **châm biếm nằm ở modality nào**.

**Lưu ý về định nghĩa nhãn.** `README.md` mô tả `image-sarcasm` là "mỉa mai đến từ mâu thuẫn giữa ảnh và văn bản". Hướng dẫn gán nhãn chính thức của cuộc thi (`reports/guides/annotation_guideline.md`) không nói như vậy: ví dụ `image-sarcasm` ở slide 7 có toàn bộ ý châm biếm nằm trong ảnh (ảnh chụp tin nhắn), còn caption chỉ là một lời bình ngắn. Các report dùng cách hiểu theo hướng dẫn chính thức.

Hướng dẫn chính thức chỉ định nghĩa *có* và *không* châm biếm. Ranh giới giữa `text`, `image` và `multi` chỉ được minh hoạ bằng 9 ví dụ. 5 ví dụ có trong bộ dữ liệu, và **3 trong 5 mang nhãn khác với slide** (train 8517, train 493, private test 545). Ranh giới giữa các lớp vì vậy không chắc chắn, và đây là giới hạn chung cho mọi mô hình.

## 2. Dữ liệu

Bộ ViMMSD, UIT Data Science Challenge 2024.

| Split | Vai trò trong đồ án | Số mẫu | not | text | image | multi |
|---|---|---:|---:|---:|---:|---:|
| `train` | train | 10.805 | 6.062 | 77 | 442 | 4.224 |
| `public_test` | **val**: chọn checkpoint, early stopping | 1.413 | 442 | 1 | 28 | 942 |
| `private_test` | **test**: mọi chỉ số đánh giá cuối | 1.504 | 458 | 7 | 40 | 999 |

- Ảnh: 13.722 ảnh, không thiếu ảnh, không có ảnh lỗi.
- Nguồn Kaggle: `hhhoang/vimmsd-dataset` (bản gốc, hai tập test không có nhãn) và `phuclkz/vimmsd-private-dataset` (cùng dữ liệu, có thêm file nhãn `*-ref.json` cho public và private test).
- Config: `data.split: official` trong `configs/base.yaml`.

Ba đặc điểm chi phối mọi thí nghiệm:

1. **Mất cân bằng nặng:** `text-sarcasm` chỉ có 77 mẫu train (0,71%), chênh 78,7 lần so với `not-sarcasm`.
2. **Lệch phân bố nhãn giữa train và test:** tỉ lệ bài có châm biếm là 43,9% ở train, 68,7% ở public test, 69,5% ở private test. Chỉ cần luôn đoán `multi-sarcasm` đã được accuracy 66,7% trên val.
3. **Lớp hiếm gần như không đo được:** val có 1 mẫu `text-sarcasm` và 28 mẫu `image-sarcasm`; private test có 7 và 40.

## 3. Chỉ số đánh giá

- **Chỉ số chính:** macro-F1 4 lớp trên private test.
- **Chỉ số phụ:** macro-F1 3 lớp (bỏ `text-sarcasm`, vì test chỉ có 7 mẫu), weighted-F1, accuracy, F1 từng lớp, confusion matrix.
- Accuracy và weighted-F1 bị 2 lớp lớn chi phối nên chỉ để tham khảo.
- **Mốc tham chiếu:** Vu et al. (2025), *Multimodal Sarcasm Detection on Vietnamese Social Media*, IEA/AIE: F1 0,41 trên private test.

Có hai tiêu chí chọn checkpoint trong các phase, phải ghi rõ khi so sánh:

| Tiêu chí | Dùng ở | Lý do |
|---|---|---|
| val macro-F1 4 lớp | Phase 3, 4 (Kaggle) | mặc định của `train.py` |
| val macro-F1 bỏ `text-sarcasm` | Phase 6, 8, 10 | val chỉ có 1 mẫu `text-sarcasm`, làm F1 lớp này chỉ là 0 hoặc 1 |

## 4. Hạ tầng

| Môi trường | Dùng cho |
|---|---|
| Kaggle, GPU T4 (16 GB), có bản T4 x2 | môi trường chính của nhóm: baseline, fusion, xử lý mất cân bằng |
| GPU server riêng, RTX 5070 12 GB | chỉ chạy script `.py`: CoT distillation, lọc nhiễu, empirical analysis, web demo |
| Local | viết code, chạy unit test (pytest) |

**Codebase** (`CODEBASE.md`):

- Package `vimmsd` trong `src/`: `data/` (dataset, tiền xử lý, OCR/VLM, augmentation, lọc nhiễu), `models/` (encoder, fusion, head), `training/` (trainer, loss, metric), `analysis/` (shortcut, interpretability, thống kê), `web/`.
- Mỗi thí nghiệm là 1 file YAML trong `configs/`, kế thừa `configs/base.yaml`.
- Một entrypoint train: `python scripts/train.py --config configs/<x>.yaml [--override key=value ...]`.
- `scripts/run_experiments.py` chạy ma trận config × seed trên nhiều GPU (`configs/experiments/*.yaml`).
- Notebook tự `git clone` repo và `pip install -e .`, nên mọi người chạy cùng một phiên bản code. Mỗi run ghi commit vào kết quả.
- Mỗi run ghi `results_row.md` (val/test macro-F1, thời gian train, peak VRAM). Nhật ký các run cũ: `reports/results/results.md`.

**Encoder dùng chung:** PhoBERT-base (`vinai/phobert-base`) cho text, CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) cho ảnh.

## 5. Lộ trình và danh sách report

| Phase | Nội dung | Report |
|---|---|---|
| 1 | Setup dự án | `1_1_project_informations.md` |
| 2 | EDA, tiền xử lý text và ảnh | `2_1_eda.md`, `2_2_text_preprocessing.md`, `2_3_image_preprocessing.md` |
| 3 | Baseline đơn modality | `3_1_text_only_caption.md`, `3_2_text_only_mix.md`, `3_3_image_only.md` |
| 4 | Fusion concat | `4_1_fusion_concat.md` |
| 5 | Cross-attention fusion | `5_1_cross_attention.md` |
| 6 | CoT distillation | `6_1_cot_distillation_method.md`, `6_2_baseline_and_teacher.md` |
| 7 | Lọc dữ liệu nhiễu | `7_1_noise_filtering.md`, `7_2_noise_filtering_results.md` |
| 8 | Xử lý mất cân bằng | `8_1_imbalance.md` |
| 9 | Shortcut learning, interpretability | `9_1_shortcut_learning.md`, `9_2_interpretability.md` |
| 10 | Empirical analysis | `10_1` … `10_6` |
| 11 | So sánh với LLM đa modal | `11_1_llm_comparison.md` |
| 12 | Dữ liệu thật | `12_1_real_world_data.md` |
| 13 | Phân tích lỗi, future work | `13_1_error_analysis.md`, `13_2_future_work.md` |
| 14 | Web demo | `14_1_web_demo.md` |

## 6. Quy ước báo cáo

- `a ± b` là trung bình ± độ lệch chuẩn mẫu qua các seed (thường 1, 2, 42).
- Mỗi bảng ghi rõ số liệu trên **val** hay **private test**. Theo `report_main.md`, chỉ số đánh giá mô hình phải là private test. Một số phase chỉ có số trên val (vì cố ý giữ private test lại, hoặc chưa chấm). Các report ghi rõ điều này thay vì trộn hai loại số.
- Private test chỉ dùng để báo cáo kết quả cuối của từng run; mọi lựa chọn checkpoint, config, kỹ thuật đều dựa trên val (public test) hoặc fold nội bộ của train.

# 3.1. Text-only: chỉ dùng caption

- **Phase:** 3, Baseline đơn modality
- **Configs:** `configs/text_only.yaml` (fine-tune PhoBERT), `configs/text_only_frozen.yaml`, `configs/text_only_frozen_cw.yaml`
- **Notebook:** `notebooks/02_baseline_text.ipynb` (`RUN = "3.1"`)
- **Nguồn số liệu:** `reports/word/Fine-tune PhoBERT.docx` (fine-tune, Kaggle, seed 42); `reports/results/results.md` (plan `denoise_compare`, run `text_only_orig_s*`; Phase 4 Kaggle, run `text_only_frozen*`), `reports/results/empirical_analysis.md` §7
- **Tập đánh giá:** private test 1.504 mẫu (not 458, text 7, image 40, multi 999)

---

## 1. Phương pháp

```
caption đã tiền xử lý (Phase 2.2) → PhoBERT-base → vector <s> (768) → MLP 768 → 512 → 4
```

| Biến thể | PhoBERT | Head | Loss | lr encoder / head |
|---|---|---|---|---|
| Fine-tune (Model 3.1) | train toàn bộ | MLP, dropout 0,2 | CE | 2e-5 / 1e-4 |
| Đóng băng | đóng băng | MLP | CE | – / 1e-3 |
| Đóng băng + class weight | đóng băng | MLP | CE, trọng số `sqrt_inverse` | – / 1e-3 |

Chung: batch 16, AMP fp16, weight decay 0,01, warmup 10%, `max_length` 256, early stopping patience 3 theo val macro-F1. Bản fine-tune tối đa 10 epoch, bản đóng băng tối đa 20 epoch (head trên feature cố định hội tụ chậm). Split chính thức: train 10.805, val = public test, test = private test.

## 2. Kết quả trên private test

| Mô hình | Chọn checkpoint | Môi trường | Số seed | Val macro-F1 | **Test macro-F1** | Test macro-F1 3 lớp | Thời gian | VRAM |
|---|---|---|---:|---|---|---|---|---|
| PhoBERT fine-tune | val 4 lớp | GPU server, commit `b25b4e1` | 3 | 0.392 ± 0.013 | **0.386 ± 0.009** | – | 4,5–8,0 phút | 3,31 GB |
| PhoBERT đóng băng | val 4 lớp | Kaggle, commit `1c8a749` | 3 | 0.373 ± 0.006 | 0.365 ± 0.006 | 0.487 ± 0.008 | 2,5–3,9 phút | 0,89 GB |
| PhoBERT đóng băng + class weight | val 4 lớp | Kaggle, `1c8a749` | 3 | 0.390 ± 0.028 | 0.381 ± 0.010 | 0.508 ± 0.013 | 2,6–4,6 phút | 0,89 GB |
| T (đóng băng, Phase 10) | val 3 lớp | GPU server | 3 | 0.381 ± 0.007 | 0.359 ± 0.021 | 0.478 | – | 0,87 GB |
| T + B1 (đóng băng + class weight, Phase 10) | val 3 lớp | GPU server | 3 | 0.412 ± 0.013 | 0.366 ± 0.014 | 0.487 | – | 0,87 GB |

Từng seed của bản fine-tune (seed 42, 1, 2), test macro-F1: 0.3893, 0.3926, 0.3759.

F1 từng lớp trên private test (trung bình 3 seed):

| Mô hình | F1 not | F1 text | F1 image | F1 multi | Weighted-F1 | Accuracy |
|---|---|---|---|---|---|---|
| PhoBERT đóng băng | 0.651 | 0.000 | 0.000 | 0.809 | 0.736 | 0.741 |
| PhoBERT đóng băng + class weight | 0.650 | 0.000 | 0.050 | 0.824 | 0.746 | 0.752 |
| T (Phase 10) | 0.644 | 0.000 | 0.000 | 0.791 | 0.721 | 0.726 |
| T + B1 (Phase 10) | 0.642 | 0.000 | 0.017 | 0.803 | 0.729 | 0.732 |

F1 từng lớp của 3 run fine-tune trên GPU server chưa được tổng hợp (các file `test_metrics.json` của `text_only_orig_s*` chưa đưa vào repo). Mục 2.1 có một lượt fine-tune riêng trên Kaggle với đầy đủ số từng lớp.

### 2.1. PhoBERT fine-tune trên Kaggle (seed 42, chi tiết)

Cùng config `text_only.yaml`, chạy trên Kaggle (`reports/word/Fine-tune PhoBERT.docx`). Kết quả khác lượt seed 42 trên GPU server (0.3966 so với 0.3893) do khác môi trường và phần cứng.

| Tập | Macro-F1 | Macro-P | Macro-R | Accuracy |
|---|---|---|---|---|
| Val (epoch 4) | 0.3873 | 0.4170 | 0.3841 | 0.7396 |
| **Test** | **0.3966** | 0.5418 | 0.3928 | 0.7414 |

16,2 phút (dừng sau 7 epoch), VRAM đỉnh 3,31 GB.

| Lớp | Mẫu | Precision | Recall | F1 |
|---|---:|---|---|---|
| not-sarcasm | 458 | 0.582 | 0.707 | 0.638 |
| text-sarcasm | 7 | 0 | 0 | 0 |
| image-sarcasm | 40 | 0.750 | 0.075 | 0.136 |
| multi-sarcasm | 999 | 0.836 | 0.789 | 0.812 |

Ma trận nhầm lẫn (hàng: nhãn thật; sai 389/1.504 mẫu):

| Thật \ Dự đoán | not | text | image | multi |
|---|---:|---:|---:|---:|
| not | 324 | 0 | 0 | 134 |
| text | 3 | 0 | 0 | 4 |
| image | 20 | 0 | 3 | 17 |
| multi | 210 | 0 | 1 | 788 |

| Epoch | Train loss | Val macro-F1 | Val accuracy |
|---|---|---|---|
| 1 | 0.8688 | 0.3787 | 0.7749 |
| 2 | 0.7025 | 0.3709 | 0.7580 |
| 3 | 0.5980 | 0.3753 | 0.7665 |
| **4** | 0.4916 | **0.3873** | 0.7396 |
| 5 | 0.3813 | 0.3366 | 0.6794 |
| 6 | 0.3100 | 0.3338 | 0.6242 |
| 7 | 0.2497 | 0.3403 | 0.6249 |

- Train loss giảm đều nhưng val giảm từ epoch 5: overfit sau epoch 4.
- Lỗi lớn nhất là nhầm giữa `multi` và `not` (210 và 134 mẫu): caption đơn lẻ thường không đủ để phân biệt.
- `image-sarcasm` đúng 3/40 (recall 0,075) nhưng precision 0,75: khi mô hình đoán `image` thì thường đúng, nhưng hiếm khi đoán. F1 0,136 của một seed dựa trên 3 mẫu đúng nên rất nhiễu.

## 3. Nhận xét

- **Fine-tune PhoBERT cho điểm cao nhất trong nhóm text-only** (0.386 ± 0.009), hơn bản đóng băng khoảng 2 điểm, đổi lại tốn VRAM gấp khoảng 3,7 lần. Bản đóng băng có class weight (0.381) đuổi gần kịp, với thời gian train ngắn hơn. Hai con số chạy ở hai môi trường và hai commit khác nhau, nên khoảng cách dưới 1 điểm này không nên đọc như một khác biệt chắc chắn.
- **Không mô hình nào dự đoán được `text-sarcasm`** (F1 = 0 ở mọi seed), dù đây là lớp châm biếm nằm ngay trong caption. 77 mẫu train là quá ít, và test chỉ có 7 mẫu.
- **`image-sarcasm` gần như bằng 0** khi chỉ dùng caption, đúng như kỳ vọng vì ý mỉa mai nằm trong ảnh. Class weight kéo F1 lớp này lên 0,050.
- **Class weight là thay đổi có tác động lớn nhất** với text-only đóng băng: +1,6 điểm 4 lớp, +2,1 điểm 3 lớp trên private test. Ở Phase 10, cùng thay đổi này cho +3,1 điểm trên val (contrast duy nhất có khoảng tin cậy không chứa 0 sau hiệu chỉnh Holm) nhưng chỉ còn +0,7 điểm trên private test. Ước lượng trên val lạc quan.
- **Val và test lệch nhau** (ví dụ `text_only_frozen_cw` val 0.390 ± 0.028, test 0.381). Lý do: chọn checkpoint theo macro-F1 4 lớp trên val, mà val chỉ có 1 mẫu `text-sarcasm`.
- **Caption gánh phần lớn tín hiệu của toàn bài toán.** Text-only đóng băng có class weight (0.381) ngang mọi mô hình fusion không có class weight ở Phase 4 (0.366–0.382). Xem Phase 4 và Phase 10 (Experiment A).

## 4. Hạn chế

- Bản fine-tune 3 seed (GPU server) chỉ có số tổng; số từng lớp và confusion matrix chỉ có ở lượt Kaggle 1 seed (mục 2.1).
- Ba hàng đầu chọn checkpoint theo macro-F1 4 lớp, hai hàng cuối theo macro-F1 3 lớp; không so trực tiếp hai nhóm.
- Private test được chấm cho mọi run để báo cáo; checkpoint và config chọn theo val.

## 5. Tái lập

```bash
# fine-tune
python scripts/train.py --config configs/text_only.yaml --override seed=42
# đóng băng / đóng băng + class weight (Phase 4)
python scripts/run_experiments.py --plan configs/experiments/phase4_baselines.yaml
```

# 10.4. Experiment C: mất cân bằng và augmentation

- **Phase:** 10, Empirical Analysis (RQ3)
- **Thành viên:** Trúc (server); Phúc, Nhi, Mai (bản Kaggle trên val, Phase 8)
- **Plans:** `configs/experiments/empirical_C_screen.yaml` (B0–B3 × 3 fold, commit `ef3dcfc`), `empirical_phase3.yaml` (xác nhận B1, commit `829018e`)
- **Nguồn số liệu:** `reports/results/empirical_phase1.md`, `empirical_phase2.md` §3, `empirical_phase3.md` §1–2, `empirical_analysis.md` §5, §7
- **Tập đánh giá:** screening trên **3 fold nội bộ của train** (OOF); xác nhận trên val; benchmark cuối trên **private test** (`10_6`)

Bản Kaggle của cùng thí nghiệm (B0–B5 trên val, seed 42, có B4/B5) và tổng hợp chung: `8_1_imbalance.md`. Report này tập trung vào phần chạy theo giao thức của Phase 10.

---

## 1. Thiết kế

| ID | Kỹ thuật |
|---|---|
| B0 | CE, lấy mẫu tự nhiên |
| B1 | CE + class weight căn bậc hai nghịch đảo tần suất |
| B2 | CE + lấy mẫu căn bậc hai nghịch đảo tần suất (`sampler: power`, `sampler_power: 0.5`) |
| B3 | Focal loss γ = 2, không class weight |
| B4 | Augmentation caption lớp hiếm có duyệt + CE |
| B5 | Như B4 + class weight |

- Backbone C, 3 fold nội bộ (chia theo nhóm, `10_1` §3.1), seed 42.
- **Ngân sách cố định 451 bước tối ưu mỗi epoch** cho mọi kỹ thuật (augmentation làm tập lớn hơn, sampling làm thay đổi số lần thấy mẫu). Mỗi run ghi `exposure.json`: số lượt rút, số mẫu duy nhất. Kiểm tra trên fold 0 với B2: đúng 451 bước, 7.216 lượt rút, 4.302 mẫu duy nhất.
- Chọn checkpoint theo macro-F1 4 lớp của fold giữ lại; chọn kỹ thuật theo **pooled OOF macro-F1** trên toàn bộ 10.805 mẫu train.

### 1.1. Augmentation (B4/B5): đã sinh, không train

Back-translation `vinai/vinai-translate-vi2en-v2` → `en2vi-v2` (beam 5 sang tiếng Anh, top-p 0,9 khi dịch về, seed 42), 2 bản cho mỗi mẫu trong 519 mẫu lớp hiếm (77 `text`, 442 `image`):

| | Số bản |
|---|---:|
| Sinh ra | 1.038 |
| Loại tự động | 72 (49 không đổi, 20 trùng bản khác, 2 trùng caption train, 1 rỗng) |
| **Cần duyệt** | **966** |

Kiểm tra sơ bộ thấy đúng rủi ro đã dự kiến: back-translation thường xoá câu đùa (ví dụ chơi chữ "cơm nắm đấm" thành "cơm"). Giao thức yêu cầu hai người duyệt độc lập (giữ nghĩa, giữ ý mỉa mai, giữ nhãn modality; xem `reports/guides/augmentation_review_guide.md`). **Nhóm quyết định bỏ bước duyệt (13/11/2026)**, nên B4/B5 không được screen trên server. Đây là một sửa đổi giao thức đã công bố.

## 2. Screening: pooled OOF trên train (seed 42)

| ID | Pooled OOF macro-F1 | Từng fold | F1 not | F1 text (77) | F1 image (442) | P / R image | F1 multi |
|---|---|---|---|---|---|---|---|
| B0 | 0.381 | 0.346 0.387 0.400 | 0.722 | 0.000 | 0.171 | 0.30 / 0.12 | 0.633 |
| **B1** | **0.389** | 0.375 0.390 0.401 | 0.713 | 0.000 | 0.199 | 0.22 / 0.18 | 0.646 |
| B2 | 0.385 | 0.386 0.380 0.392 | 0.707 | 0.000 | 0.204 | 0.17 / 0.25 | 0.629 |
| B3 | 0.371 | 0.376 0.365 0.360 | 0.719 | 0.000 | 0.145 | 0.23 / 0.11 | 0.619 |

So với B0 (paired bootstrap trên mẫu OOF, 1 seed train):

| Contrast | Δ | 95% CI | Group 95% CI |
|---|---|---|---|
| B1 − B0 | +0.008 | [−0.004, +0.019] | [−0.007, +0.024] |
| B2 − B0 | +0.004 | [−0.008, +0.016] | [−0.014, +0.023] |
| B3 − B0 | −0.011 | [−0.023, +0.001] | [−0.032, +0.015] |

- **Không kỹ thuật nào nhận ra `text-sarcasm`** (F1 = 0 trên cả 77 mẫu OOF).
- B1 và B2 tăng recall `image-sarcasm` (0,12 → 0,18 / 0,25), đổi lại precision lớp này và recall `not-sarcasm` giảm: một **đánh đổi theo hướng lớp hiếm**, chưa phải cải thiện tổng thể. B1 +0,008 dưới mục tiêu +1 điểm, khoảng tin cậy chứa 0.
- Focal loss tệ nhất.
- Fold 0 của B0 (0,346) thấp hơn hẳn fold 1–2: biến thiên giữa các fold cỡ chênh lệch giữa các kỹ thuật, nên dùng pooled OOF và group bootstrap để chọn.

**Chọn kỹ thuật** (luật chốt trước, plan §8.4): chỉ chọn kỹ thuật mới (B2–B5) nếu vượt B1; B2, B3 không vượt; B4/B5 không chạy. → **B1**.

## 3. Xác nhận B1 trên toàn bộ train (val, 3 seed)

| Mô hình | Macro-F1 (4 lớp) | Weighted-F1 | Accuracy | F1 not | F1 image (28) | Recall image | F1 multi |
|---|---|---|---|---|---|---|---|
| C (CE) | 0.399 ± 0.008 | 0.746 | 0.749 | 0.635 | 0.146 | 0.12 | 0.816 |
| **C + B1** | **0.416 ± 0.019** | 0.751 | 0.754 | 0.636 | 0.206 | 0.18 | 0.822 |

| Contrast | Δ | Từng seed (1, 2, 42) | 95% CI | p Holm | Group 95% CI |
|---|---|---|---|---|---|
| C + B1 − C | +0.017 | +0.037 −0.002 +0.015 | [−0.006, +0.041] | 0.16 | [−0.012, +0.040] |

Khớp với screening (+0,008 OOF), vượt mục tiêu +1 điểm, nhưng một seed không đổi và khoảng tin cậy chứa 0: **chưa kết luận được**. Đánh đổi như dự kiến: recall `image` 0,12 → 0,18, F1 0,146 → 0,206; các lớp khác giữ nguyên.

## 4. Trên private test

| Mô hình | Macro-F1 | 3 lớp | F1 image (40) | Recall image |
|---|---|---|---|---|
| C | 0.369 ± 0.011 | 0.492 | 0.027 | – |
| C + B1 | 0.380 ± 0.007 | 0.507 | 0.057 | 0.04 |

C + B1 − C = +0.011, cùng chiều ở mọi seed (+0.012, +0.015, +0.007), 95% CI [−0.002, +0.025], p Holm 0,16: **chưa kết luận được**. Không lớp nào bị giảm so với C. Khác với val, **phần tăng trên private test không đến từ recall lớp hiếm** (recall `image` 0,04) mà từ `multi` và `not`.

## 5. Kết luận RQ3

1. **Class weight căn bậc hai (B1) là kỹ thuật được chọn**: tăng cùng chiều ở OOF (+0,008), val (+0,017) và private test (+0,011), nhưng không lần nào đạt mức bằng chứng đã chốt.
2. Lấy mẫu căn bậc hai (B2) gần tương đương; focal loss làm tệ hơn.
3. Không kỹ thuật nào ở mức loss/sampling học được `text-sarcasm` (77 mẫu train).
4. Phần tăng recall `image-sarcasm` thấy trên val **không mang sang private test**; nên đọc mọi kết luận về lớp hiếm dựa trên val (28 mẫu) một cách thận trọng.
5. Augmentation back-translation không được đánh giá theo giao thức (chưa duyệt). Bản Kaggle chưa duyệt làm kết quả tệ hơn (`8_1`), phù hợp với rủi ro mất ý mỉa mai đã quan sát.

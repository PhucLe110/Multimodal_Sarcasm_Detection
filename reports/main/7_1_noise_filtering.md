# 7.1. Lọc dữ liệu nhiễu

- **Phase:** 7, Lọc dữ liệu nhiễu
- **Thành viên:** Trúc
- **Code:** `src/vimmsd/data/denoise.py`, `scripts/denoise_data.py` (commit `4b52418`), `scripts/export_oof_probs.py` (commit `099bbf2`)
- **Chạy trên:** GPU server; bản sạch ghi ra `/data/Phat/vimmsd-clear` (cùng bố cục với ViMMSD gốc)
- **Nguồn số liệu:** commit message và docstring của code trên; `configs/experiments/denoise_compare*.yaml`; ghi chú trong `notebooks/09_imbalance.ipynb`

Kết quả train lại trên bản sạch: `7_2_noise_filtering_results.md`.

---

## 1. Mục tiêu

Phase 2.1 cho thấy train có nhiều dạng nhiễu: 26,5% mẫu trùng caption, 123 nhóm trùng caption khác nhãn, 42 nhóm ảnh trùng md5 (9 nhóm khác nhãn), và hướng dẫn gán nhãn mâu thuẫn với chính dữ liệu ở 3/5 ví dụ tìm được. Phase này tạo một **bản train sạch** để kiểm tra: bỏ trùng lặp và mẫu nghi sai nhãn có làm mô hình tốt hơn không.

## 2. Nguyên tắc

1. **Chỉ lọc train.** Public test, private test và file nhãn `*-ref.json` chép nguyên, giống từng byte, để điểm so được với bản gốc.
2. **Trùng = trùng cả ảnh lẫn caption.** Nhãn phụ thuộc quan hệ ảnh–caption, nên cùng caption khác ảnh (hoặc ngược lại) là dữ liệu hợp lệ, chỉ dùng để gom nhóm khi chia fold.
3. **Không tự bỏ lớp hiếm** (`image-sarcasm`, `text-sarcasm`), trừ bản sao chính xác cùng nhãn. Mẫu lớp hiếm nghi nhiễu chỉ bị gắn cờ.
4. **Bản sạch tự chứa:** ảnh được hard link (cùng filesystem) hoặc chép, không dùng symlink; mọi ngưỡng ghi vào `manifest.json`.

## 3. Quy trình

### Bước 1: trùng chính xác

Khoá = md5 ảnh + caption chuẩn hoá (`html.unescape`, NFC, gộp khoảng trắng, strip, lowercase). Mỗi nhóm trùng được xử lý bằng `resolve_group`:

| Trường hợp | Hành động |
|---|---|
| Cùng nhãn | giữ id nhỏ nhất, còn lại `drop_duplicate` |
| Khác nhãn, có nhãn chiếm ≥ 2/3 | giữ 1 mẫu mang nhãn đa số (`relabel_majority`), còn lại `drop_duplicate` |
| Khác nhãn, không có đa số | `drop_conflict` cả nhóm |
| Mẫu lớp hiếm khác nhãn chốt | giữ, gắn `flag_conflict` để người xem lại |

Mẫu train trùng chính xác với public/private test: mặc định chỉ gắn cờ; bỏ (`drop_leak_test`) khi chạy với `--drop-test-overlap`.

### Bước 2: gần trùng

| Thành phần | Cách đo | Ngưỡng |
|---|---|---|
| Ảnh | pHash 64 bit (Hamming) **và** CLIP ViT-B/32 image embedding (cosine) | Hamming ≤ 6 và cosine ≥ 0,95 |
| Caption | `intfloat/multilingual-e5-base` trên caption gốc (prefix `query: `) | cosine ≥ 0,95 |

Một cặp là gần trùng khi **cả ảnh và caption** đều gần. Các nhóm gần trùng xử lý bằng cùng luật ở bước 1. Ngoài ra, union-find trên mọi cặp (chỉ cần ảnh *hoặc* caption gần) tạo `group_id` cho `StratifiedGroupKFold`, để mẫu gần trùng không nằm ở hai fold.

### Bước 3: nghi gán nhãn sai

1. Tính xác suất out-of-fold (5 fold, chia theo nhóm) cho mọi mẫu train:
   - bản đầu: logistic regression trên embedding (C = 1, 3.000 vòng lặp);
   - bản sau (`--oof-probs`, commit `099bbf2`): xác suất OOF của mô hình đa phương thức đã train ở thí nghiệm B2 stacking (Phase 6.2), `denoise.oof_models=[multi]`.
2. `cleanlab` gắn cờ mẫu có nhãn khả năng sai.
3. Đồng thuận kNN (k = 10): tỉ lệ láng giềng cùng nhãn.
4. Luật `decide_mislabel`:
   - `drop_mislabel` khi cleanlab gắn cờ, đồng thuận kNN ≤ 0,2, và nhãn thuộc `not-sarcasm` hoặc `multi-sarcasm`;
   - còn lại `flag_mislabel`;
   - trần: tối đa 5% số mẫu train gốc (`floor(0,05 × 10.805) = 540`); quá trần thì chỉ bỏ 540 mẫu có self-confidence thấp nhất;
   - nếu macro-F1 OOF của bộ phân loại < 0,40, xác suất được coi là quá yếu và **không bỏ mẫu nào** ở bước này.

Xuất thêm một mẫu 100 trường hợp (`review_size`) để người kiểm tra.

## 4. Các bản dữ liệu đã tạo

| Bản | Mô tả | Số mẫu train | Nguồn số liệu |
|---|---|---:|---|
| `orig` | ViMMSD gốc | 10.805 | |
| `clean` | bước 1–3, bước 3 dùng logistic regression | **10.780** (bớt 25) | ghi chú trong `notebooks/09_imbalance.ipynb` |
| `cleancl` | như `clean`, bước 3 dùng xác suất OOF của mô hình B2 đa phương thức | bớt thêm **540** `drop_mislabel` | chú thích trong `configs/experiments/denoise_compare_oof.yaml` |

- Bản `cleancl` bỏ đúng 540 mẫu, bằng trần 5%: số mẫu đủ điều kiện bỏ đã chạm trần, phần còn lại chỉ bị gắn cờ.
- Bản `clean` chỉ bớt 25 mẫu: nhiều khả năng bước 3 không bỏ mẫu nào vì xác suất OOF của logistic regression dưới ngưỡng 0,40. Điều này chưa xác nhận được từ repo.

**Thiếu trong repo:** `noise_report/` và `manifest.json` (số mẫu theo từng hành động, số nhóm gần trùng, số mẫu gắn cờ theo lớp, số mẫu trùng với test) nằm trên GPU server, chưa được đưa vào repo. Cần copy `noise_report.md` của cả hai bản vào đây để report đầy đủ.

## 5. Chạy lại

```bash
# bản clean (bước 3 bằng logistic regression)
VIMMSD_ENV=server python scripts/denoise_data.py --config configs/base.yaml --out-dir /data/Phat/vimmsd-clear
# xác suất OOF của mô hình đã train (B2) rồi dùng cho bước 3
python scripts/export_oof_probs.py ...
VIMMSD_ENV=server python scripts/denoise_data.py --config configs/base.yaml --out-dir /data/Phat/vimmsd-clear \
    --oof-probs <out>/noise_report/oof_b2.npz --override denoise.oof_models=[multi]
```

## 6. Hạn chế

- Luật "đa số ≥ 2/3" và ngưỡng 0,95 đặt tay, chưa có kiểm tra độ nhạy.
- cleanlab dựa vào xác suất của chính các mô hình đang yếu ở lớp hiếm; vì vậy lớp hiếm chỉ được gắn cờ, không bị bỏ.
- Bước 3 với mô hình B2 dùng xác suất của một kiến trúc giống mô hình sẽ được train lại, nên có thể bỏ đúng những mẫu "khó" với kiến trúc đó, không nhất thiết là mẫu sai nhãn.
- Chưa có người kiểm tra mẫu `review_size` để ước lượng độ chính xác của việc gắn cờ.

# 7.2. Kết quả train lại trên dữ liệu đã lọc

- **Phase:** 7, Lọc dữ liệu nhiễu
- **Thành viên:** Trúc
- **Plans:** `configs/experiments/denoise_compare.yaml` (orig, clean; commit `b25b4e1`), `configs/experiments/denoise_compare_oof.yaml` (cleancl; commit `d239906`)
- **Nguồn số liệu:** `reports/results/results.md` (18 run, GPU server, 10/11/2026), `reports/results/empirical_phase1.md`
- **Tập đánh giá:** private test 1.504 mẫu; val = public test 1.413 mẫu. Hai tập này giống từng byte ở mọi bản dữ liệu.

Các bản dữ liệu: `7_1_noise_filtering.md`.

---

## 1. Thiết lập

| Mục | Giá trị |
|---|---|
| Text-only | `configs/text_only.yaml`: PhoBERT fine-tune toàn bộ, CE, lr encoder 2e-5 |
| Image-only | `configs/image_only.yaml`: CLIP fine-tune toàn bộ, CE, lr encoder 1e-5 |
| Bản train | `orig` (10.805), `clean` (10.780), `cleancl` (`clean` bớt thêm 540 mẫu nghi sai nhãn) |
| Seed | 42, 1, 2 |
| Chọn checkpoint | val macro-F1 4 lớp |
| Khác | mọi thứ khác giữ nguyên giữa các bản; `num_workers=0` |

## 2. Kết quả

### 2.1. Tổng hợp (mean ± std, 3 seed)

| Mô hình | Bản train | Val macro-F1 | **Test macro-F1** | VRAM |
|---|---|---|---|---|
| Text-only | orig | 0.392 ± 0.013 | **0.386 ± 0.009** | 3,31 GB |
| Text-only | clean | 0.393 ± 0.014 | 0.376 ± 0.015 | 3,31 GB |
| Text-only | cleancl | 0.403 ± 0.016 | 0.380 ± 0.005 | 3,31 GB |
| Image-only | orig | 0.343 ± 0.006 | 0.346 ± 0.009 | 1,65 GB |
| Image-only | clean | 0.349 ± 0.005 | 0.342 ± 0.020 | 1,65 GB |
| Image-only | cleancl | 0.349 ± 0.001 | **0.351 ± 0.005** | 1,65 GB |

### 2.2. Từng seed (test macro-F1)

| Mô hình | Bản | s42 | s1 | s2 |
|---|---|---|---|---|
| Text-only | orig | 0.3893 | 0.3926 | 0.3759 |
| Text-only | clean | 0.3917 | 0.3616 | 0.3754 |
| Text-only | cleancl | 0.3781 | 0.3855 | 0.3775 |
| Image-only | orig | 0.3544 | 0.3375 | 0.3469 |
| Image-only | clean | 0.3194 | 0.3515 | 0.3546 |
| Image-only | cleancl | 0.3512 | 0.3557 | 0.3464 |

### 2.3. Chênh lệch so với `orig` theo cặp seed (test macro-F1)

| Mô hình | Bản | s42 | s1 | s2 | Trung bình |
|---|---|---|---|---|---|
| Text-only | clean | +0.002 | −0.031 | −0.001 | −0.010 |
| Text-only | cleancl | −0.011 | −0.007 | +0.002 | −0.005 |
| Image-only | clean | −0.035 | +0.014 | +0.008 | −0.005 |
| Image-only | cleancl | −0.003 | +0.018 | −0.001 | +0.005 |

**Kiểm định trên val** (`scripts/empirical_stats.py contrasts`, paired bootstrap 2.000 lần, Phase 10 dùng để kiểm tra công cụ): text-only clean so với orig, 0.3932 so với 0.3922, khoảng tin cậy 95% của chênh lệch chứa 0.

## 3. Nhận xét

- **Không có cải thiện nhất quán trên private test.** Mọi chênh lệch trung bình nằm trong ±1 điểm, nhỏ hơn hoặc cỡ độ lệch chuẩn giữa các seed, và dấu thay đổi theo seed.
- **Text-only:** lọc làm điểm test giảm nhẹ (−1,0 và −0,5 điểm). `cleancl` tăng val 1,1 điểm nhưng không mang sang test; với 3 seed và std khoảng 1,5 điểm trên val, mức tăng này chưa phân biệt được với nhiễu.
- **Image-only:** `cleancl` nhỉnh nhất (+0,5 điểm) và ổn định nhất (std 0,005), nhưng mức này nằm trong nhiễu.
- **Bỏ 540 mẫu (5%) mà điểm gần như không đổi**: các mẫu bị coi là nghi sai nhãn không phải nguồn lỗi chính của mô hình, hoặc việc gắn cờ không đủ chính xác. Hai giải thích này chưa tách được vì chưa ai kiểm tra tay các mẫu bị bỏ.
- **Không đánh giá được tác động lên lớp hiếm**: F1 từng lớp nằm trong `test_metrics.json` trên GPU server, chưa tổng hợp.

## 4. Kết luận

Lọc trùng lặp và nhãn nghi sai trên train **không cải thiện** text-only hay image-only trên private test; kết quả là trung tính đến hơi âm. Vì vậy các phase sau (Phase 8, 10) dùng **bộ train gốc** để giữ so sánh được với các phase trước. Lọc nhiễu được giữ như một can thiệp riêng có tên, không coi là bước tiền xử lý mặc định.

Kết quả này khớp với nhận định trong `13_2_future_work.md`: vấn đề nhãn của ViMMSD nằm ở **ranh giới giữa các lớp không rõ** (hướng dẫn gán nhãn mơ hồ) nhiều hơn là ở vài trăm mẫu gán sai; cách xử lý phù hợp hơn là audit nhãn có người chấm.

## 5. Việc cần bổ sung

- Tổng hợp F1 từng lớp và confusion matrix của 18 run từ `test_metrics.json` trên server.
- Chưa chạy lại fusion (C, C + B1) trên bản sạch; nếu cần, dùng cùng plan với `configs/fusion_concat_frozen_reg*.yaml`.
- Kiểm tra tay một mẫu các trường hợp `drop_mislabel` để biết bao nhiêu là sai nhãn thật.

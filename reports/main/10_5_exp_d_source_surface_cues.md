# 10.5. Experiment D: thẻ nguồn và tín hiệu bề mặt

- **Phase:** 10, Empirical Analysis (RQ4)
- **Thành viên:** Trúc
- **Code:** `scripts/diagnose_cues.py`, `scripts/check_source_tags.py`, `src/vimmsd/data/preprocessing.py` (`strip_source_tags`, `strip_hashtags_and_signatures`); plan `configs/experiments/empirical_phase3.yaml` (commit `829018e`)
- **Nguồn số liệu:** `reports/results/empirical_phase3.md` §1–3, `empirical_analysis.md` §6, `progress_report.md` §4.10
- **Tập đánh giá:** val (public test, 1.413 mẫu). Các mô hình train lại không thẻ không nằm trong danh sách finalist nên không chấm private test.

---

## 1. Thiết kế

**Sửa caption lúc suy luận** (mô hình vẫn train với caption gốc), OCR và mô tả giữ nguyên; caption đã sửa lưu ở `data/processed/empirical/diagnostics/cue_edits_val.csv`:

| Phép sửa | Số mẫu val bị sửa |
|---|---:|
| Xoá thẻ trang nguồn (`interpool*`, `voz`, `voz_f33`, `wmo`, `worldmemeorganization`) | 193 |
| Xoá mọi hashtag và chữ ký admin một dòng (`- Elsa`) | 313 |
| Xoá emoji và dấu câu lặp | 831 |

**Train lại không thẻ:** T và C train trên caption đã xoá thẻ nguồn, 3 seed (6 run), cùng CE và ngân sách như Experiment A. Không đổi OCR cùng lúc.

## 2. Thẻ nguồn tương quan với nhãn

| Thẻ | Train | Val |
|---|---|---|
| `interpool` | 468 bài, 59% `multi` (toàn tập 39%) | 90 bài, 80% `multi` (toàn tập 67%) |
| `voz` | 222 bài, 12% `image` (toàn tập 4%) | 105 bài, 60% `not` (toàn tập 31%) |

Trong train: `#wmo`, `#worldmemeorganization` khoảng 65% `image-sarcasm`; `#quytacungxutx` 100% `multi` (55 bài); `#redbull`, `#vtv`, `#70namdienbienphu` 100% `not` (30 bài mỗi thẻ). Nhiều thẻ trong số này không xuất hiện ở val.

## 3. Sửa caption lúc suy luận

Trung bình 3 seed. "Đổi" = tỉ lệ dự đoán đổi trên các mẫu bị sửa; ΔF1 = thay đổi macro-F1 trên toàn val.

| Mô hình | Xoá thẻ nguồn (193) | Xoá hashtag + chữ ký (313) | Xoá emoji + dấu câu lặp (831) |
|---|---|---|---|
| T | đổi 24%, ΔF1 −0.009 | 20%, −0.010 | 12%, **−0.035** |
| C | 23%, −0.008 | 17%, −0.008 | 12%, −0.020 |
| M | 26%, −0.008 | 20%, −0.008 | 12%, −0.017 |
| C + B1 | 19%, −0.010 | 15%, −0.008 | 10%, −0.021 |

## 4. Train lại không thẻ

| Mô hình | Có thẻ | Không thẻ | Weighted-F1 (không thẻ) | F1 image (không thẻ) |
|---|---|---|---|---|
| T | 0.381 ± 0.007 | 0.373 ± 0.008 | 0.754 | 0.017 |
| C | 0.399 ± 0.008 | 0.385 ± 0.018 | 0.741 | 0.098 |

| Contrast | Δ | Từng seed (1, 2, 42) | 95% CI | p Holm | Group 95% CI |
|---|---|---|---|---|---|
| T không thẻ − T | −0.008 | −0.023 −0.004 +0.004 | [−0.025, +0.010] | 0.55 | [−0.031, +0.020] |
| C không thẻ − C | −0.014 | −0.021 −0.035 +0.014 | [−0.039, +0.007] | 0.55 | [−0.036, +0.008] |

Trên 193 caption có thẻ (mô tả): C không thẻ giảm từ 0,403 xuống 0,362.

Phase 6 có kết quả cùng chiều cho M (`distill_flat_a_notags`, xoá cả thẻ nguồn, hashtag và chữ ký): macro-F1 bỏ text 0.521 ± 0.015 → 0.501 ± 0.019, cả 3 seed giảm 1,4–3,4 điểm.

## 5. Nguồn còn lọt qua đường khác

Xoá thẻ khỏi caption **không** làm mô hình mất thông tin nguồn:

- **30%** caption train và **17%** caption val có thẻ nguồn cũng có dấu hiệu nguồn ("interpool", "voz", "WMO"…) trong **OCR**, mà M đọc.
- Logo trang có thể lọt vào CLIP. 40 mẫu val có thẻ được lấy để kiểm tra logo bằng tay (`data/processed/empirical/diagnostics/source_logo_audit.csv`), **chưa ai xem**.

## 6. Nhận xét

- **Thẻ nguồn là tín hiệu thật sự được dùng**: xoá thẻ đổi khoảng 1/4 dự đoán trên caption có thẻ, ở mọi mô hình.
- **Nhưng thẻ ảnh hưởng ít tới điểm tổng** (khoảng −0,01) vì chỉ 14% val có thẻ.
- **Train lại không thẻ không làm điểm tốt hơn** (−0,008 và −0,014, không phân biệt được với 0): thẻ là tín hiệu hữu ích *trong phân bố* của split chính thức, như kế hoạch đã dự đoán.
- **Emoji và dấu câu lặp ảnh hưởng tới điểm tổng nhiều nhất** (−0,02 đến −0,035), mạnh nhất ở caption-only T, ít hơn ở mô hình có ảnh. Theo hướng dẫn gán nhãn chính thức, emoji và dấu câu là dấu hiệu hợp lệ của châm biếm; mô hình dùng chúng không có nghĩa là học shortcut.

## 7. Kết luận RQ4

Thẻ nguồn là một tín hiệu hữu ích trong phân bố, được mọi mô hình dùng, nhưng đóng góp ít vào điểm tổng; loại nó khi train không cải thiện điểm trên split chính thức. Không mô hình nào ở đây có thể gọi là "không phụ thuộc nguồn", vì nguồn còn nằm trong OCR và có thể trong logo. Muốn đo độ bền khi đổi nguồn cần một split giữ riêng theo nguồn (Phase 13.2).

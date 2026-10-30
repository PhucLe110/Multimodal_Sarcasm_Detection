# Kết quả thí nghiệm

Mỗi lần train xong, `scripts/train.py` in ra 1 dòng kết quả (và ghi vào `<output_dir>/<tên config>/results_row.md`). Chạy local thì dòng được tự thêm vào đây; chạy Kaggle/Colab thì copy dòng đó vào đây rồi commit.

| Ngày | Config | Commit | Môi trường | Val macro-F1 | Test macro-F1 | Thời gian train (phút) | Ghi chú |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | `image_only` | `f5d9c02` | kaggle | 0.3323 | 0.3422 | 8.5 | Task 5: CLIP Image-only \| peak VRAM 1.67 GB |
| 2026-10-30 | `image_only_vit` | `f5d9c02` | kaggle | 0.3242 | 0.3156 | 14.7 | Task 6: ViT Image-only \| peak VRAM 2.33 GB |

## Task 7 — VLM zero-shot (không train, không có val macro-F1)

`evaluate_llm` chỉ inference trên test, không qua `train.py` nên không có dòng `results_row.md`; hai dòng dưới tổng hợp từ
`/kaggle/working/outputs/vlm_image_centric/test_metrics_{no,with}_metadata.json` (commit code `f5d9c02`, môi trường Kaggle, Tesla T4).

| Ngày | Config | Commit | Môi trường | Val macro-F1 | Test macro-F1 | Thời gian (phút) | Ghi chú |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | `base + use_ocr/use_description` (Vintern, không metadata) | `f5d9c02` | kaggle | n/a | 0.0453 | ~7.0 | zero-shot inference 1.504 mẫu @ ~3.6 it/s |
| 2026-10-30 | `base + use_ocr=true, use_description=true` (Vintern, có metadata) | `f5d9c02` | kaggle | n/a | 0.0151 | ~6.6 | zero-shot inference 1.504 mẫu @ ~3.4 it/s |

**Lưu ý split**: từ commit `e082390` trở đi `data.split=official`, nên "Test" ở trên là **private test có nhãn 1.504 mẫu**
(458 / 7 / 40 / 999 theo 4 lớp), không còn là 1.081 mẫu của bản random-split cũ. Các chỉ số vì thế không so sánh trực tiếp
được với số liệu trong draft báo cáo cũ.


# 2.3. Tiền xử lý ảnh

- **Phase:** 2, EDA và tiền xử lý dữ liệu
- **Code:** `src/vimmsd/data/image_text.py` (OCR, mô tả VLM), `src/vimmsd/data/dataset.py` (resize/pad, augmentation ảnh), `scripts/extract_image_text.py`, `scripts/build_image_text_dataset.py`
- **Notebook:** `notebooks/01b_image_text_extraction.ipynb`
- **Cache dùng chung:** `image_preprocessed.json` (+ `.csv` để xem), Kaggle dataset `vimmsd-data-preprocessed`

Tiền xử lý ảnh gồm hai phần: (1) đưa ảnh vào encoder ảnh (CLIP/ViT), (2) chuyển thông tin trong ảnh thành text (OCR + mô tả) để PhoBERT đọc được.

---

## 1. Ảnh cho encoder

| Mục | Giá trị | Lý do |
|---|---|---|
| Resize | `image_resize: pad` (letterbox vuông bằng màu trung bình, rồi resize về 224×224) | 72 ảnh có tỉ lệ > 3:1 (ảnh chụp bình luận, meme nhiều khung). Center crop mặc định của CLIP cắt mất phần lớn nội dung các ảnh này |
| Chuẩn hoá | theo processor của encoder (CLIP hoặc ViT) | |
| Augmentation | `image_augment: false` (mặc định tắt). Bản có sẵn: crop 90–100% + chỉnh sáng/tương phản, chỉ khi train | crop có thể cắt mất chữ trong meme. Không lật ngang (đảo chữ) |
| Chuyển RGB | dataset tự convert | 3 ảnh thang xám |

`image_resize: crop` vẫn giữ để làm ablation với code cũ. Chưa có run đối chiếu pad và crop trên cùng cấu hình.

## 2. Ảnh thành text

Ở `image-sarcasm` và `multi-sarcasm`, ý mỉa mai thường nằm trong chữ xuất hiện trên ảnh (Phase 2.1, mục 6). Hai nguồn text được trích **một lần** cho toàn bộ 13.722 ảnh rồi cache:

### 2.1. OCR

| Mục | Giá trị |
|---|---|
| Phát hiện vùng chữ | PaddleOCR (≥ 3.0), chỉ dùng phần detection |
| Nhận dạng | VietOCR `vgg_seq2seq` (giải mã GRU, nhanh hơn `vgg_transformer`) |
| Ngưỡng tin cậy | bỏ dòng có điểm VietOCR < 0,8 (dòng chữ rõ thường ≥ 0,85; vùng không phải chữ hoặc chữ không phải Latin < 0,8) |
| Làm sạch khi dùng | `clean_ocr`: bỏ dòng < 3 chữ/số hoặc không có chữ cái; bỏ cả OCR nếu gần trùng caption (> 0,9) |

Không dùng phần nhận dạng của PaddleOCR vì bộ ký tự của các model rec có sẵn không đủ cho tiếng Việt có dấu.

### 2.2. Mô tả ảnh bằng VLM

| Mục | Giá trị |
|---|---|
| Model | `5CD-AI/Vintern-1B-v3_5` (InternVL, tối ưu cho tiếng Việt, khoảng 1B tham số) |
| Giải mã | greedy, `max_new_tokens = 96`, `repetition_penalty = 1.3` |
| Prompt | "Mô tả ngắn gọn nội dung bức ảnh bằng tiếng Việt trong 1-2 câu: có ai hoặc cái gì, đang làm gì, bối cảnh và biểu cảm. Không chép lại chữ xuất hiện trong ảnh, không suy đoán ý nghĩa." |

Prompt **cố ý không nhắc tới mỉa mai**, để VLM không đưa phán đoán nhãn vào input của mô hình, và không chép lại chữ vì phần đó đã có OCR.

### 2.3. Kết quả

| Mục | Giá trị |
|---|---|
| Số ảnh | 13.722 (train, public test, private test) |
| Thiếu OCR / thiếu mô tả | 0 / 0 |
| File | `image_preprocessed.json`: `{meta, images}`, mỗi `image_key` có `ocr` và `description`; `meta` ghi model OCR/VLM, commit code, số ảnh thiếu |
| Thời gian | vài giờ cho mỗi bước trên Kaggle T4 x2 (mỗi GPU một nửa số ảnh); chạy OCR và VLM ở hai version riêng, chạy tiếp được nếu bị ngắt |

Khi ghép với caption, text từ ảnh có dạng `"Chữ trong ảnh: <ocr>. Mô tả ảnh: <description>"` (`compose_image_text`).

## 3. Cách các mô hình dùng text từ ảnh

| Cách | Config | Phase |
|---|---|---|
| Segment thứ hai của PhoBERT, ghép với caption | `data.image_text.use_ocr/use_description` | thiết kế ban đầu |
| Ba feature riêng có hệ số cố định (caption 1, OCR 0,5, mô tả 0,25) | `text_only_imgtext.yaml` (Model 3.2) | 3 |
| Các luồng PhoBERT riêng (caption, OCR, mô tả), mỗi luồng LayerNorm + Linear(256), thêm tương tác `t·s`, `|t − s|` và modality dropout 30% | `fusion_multistream_*_frozen.yaml` | 4, 6, 10 |
| Đưa vào prompt của VLM zero-shot | notebook 03, cell 03-10 | 3 (Task 7) |

## 4. Những gì đã đo được về giá trị của text từ ảnh

Chi tiết ở các report sau; tóm tắt để biết bước tiền xử lý này có đáng không:

- **Trong mô hình fine-tune:** caption + OCR + mô tả không hơn caption một cách phát hiện được khi so cùng thiết lập (Phase 10, Experiment A: O − T +0,003, D − T −0,008 trên val). Trên private test, các biến thể multistream (0,377–0,382) ngang text-only có class weight (0,381) (Phase 3.2, 4).
- **Trong prompt zero-shot của Vintern:** thêm OCR + mô tả làm macro-F1 giảm từ 0,0453 xuống 0,0151 (Phase 3.3).
- **Trong prompt teacher Gemini:** bỏ mô tả Vintern không làm đổi kết quả (macro-F1 0,317 so với 0,318 trên 500 mẫu), giảm 14% token đầu vào (Phase 6.2).
- **Tín hiệu nguồn trong OCR:** 30% caption train và 17% caption val có thẻ nguồn cũng có dấu hiệu nguồn ("interpool", "voz", "WMO") trong OCR. Xoá thẻ khỏi caption không xoá được nguồn khỏi mô hình đọc OCR (Phase 10, Experiment D).

## 5. Hạn chế

- Chưa đánh giá chất lượng OCR (tỉ lệ lỗi ký tự) trên một mẫu có nhãn tay. Ảnh có chữ tiếng Anh (ảnh chụp bình luận) có thể bị nhận dạng kém vì VietOCR tối ưu cho tiếng Việt.
- Chưa kiểm tra mô tả Vintern có bịa chi tiết không.
- Ảnh trùng gần giống (resize, nén lại) chưa được bắt bằng md5; Phase 7 bổ sung pHash và CLIP.
- Nếu bật augmentation ảnh, OCR và mô tả trong cache vẫn ứng với ảnh gốc.

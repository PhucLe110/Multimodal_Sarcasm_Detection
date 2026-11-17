# 2.2. Tiền xử lý text

- **Phase:** 2, EDA và tiền xử lý dữ liệu
- **Code:** `src/vimmsd/data/preprocessing.py` (`clean_text`, `TextPreprocessor`), `src/vimmsd/data/emoji_vi.py`, `scripts/preprocess_text.py`
- **Notebook:** `notebooks/01a_text_preprocessing.ipynb` (đã chạy trên Kaggle, output lưu trong notebook)
- **Cache dùng chung:** `text_preprocessed.json`, Kaggle dataset `vimmsd-data-preprocessed`, `pipeline_key = 2c3852f6`

---

## 1. Đặc điểm text cần xử lý

Thống kê trên 10.805 caption train (notebook 01a):

| Đặc trưng | Số mẫu | Tỉ lệ |
|---|---:|---:|
| Emoji | 3.330 | 30,82% |
| Hashtag | 2.422 | 22,42% |
| Ký tự lặp (`ngonnnn`) | 472 | 4,37% |
| HTML entity (`&amp;`) | 222 | 2,05% |
| URL | 185 | 1,71% |
| Mention (`@user`) | 33 | 0,31% |

Độ dài (từ): trung bình 50,7, trung vị 15, p90 165,6, p95 230, p99 358, lớn nhất 846.

## 2. Pipeline

`clean_text` chạy theo thứ tự (thứ tự có ý nghĩa, ghi chú ở cột phải):

| # | Bước | Chi tiết |
|---|---|---|
| 1 | `html.unescape` | `&amp;` → `&`. Phải đứng trước bước xoá URL/hashtag |
| 2 | Chuẩn hoá Unicode NFC | 72 caption chưa ở dạng NFC |
| 3 | Xử lý emoji (`demojize`) | emoji có trong bảng `EMOJI_VI` thành nghĩa tiếng Việt, emoji khác thành tên tiếng Anh. Chạy **trước** bước xoá ký tự vô hình vì ký tự ZWJ (U+200D) nối các emoji ghép như 🤦‍♂️. Bỏ ký tự màu da (U+1F3FB–1F3FF) và variation selector trước khi tra bảng |
| 4 | Xoá ký tự vô hình | zero-width, ký tự điều hướng, `\xa0` |
| 5 | Xoá URL và mention | |
| 6 | Bỏ dấu `#` | giữ chữ của hashtag (`#review` → `review`) |
| 7 | Rút gọn chữ lặp | ≥ 3 chữ cái thường cùng gốc (bỏ qua dấu) liên tiếp thành 1: `quáaaa` → `quá`, `ngonnnn` → `ngon`. Bỏ qua token viết hoa toàn bộ để giữ từ viết tắt (`PCCC`, `CCCD`) |
| 8 | Chuẩn hoá teencode | bảng 76 mục (`ko`, `k` → `không`; `dc` → `được`; `cmt` → `bình luận`…). Đã bỏ các mục đa nghĩa hay thay nhầm (`hổng`, `thui`, `bít`, `uk`, `tg`…). Một số mục (`k`, `z`, `j`, `kh`…) giữ nguyên khi viết hoa |
| 9 | Lowercase | **tắt** (`lowercase: false`): PhoBERT phân biệt hoa thường, và chữ viết hoa mang sắc thái |
| 10 | Gộp khoảng trắng | |
| 11 | Tách từ | VnCoreNLP (định dạng nối từ ghép bằng `_`, khớp vocab PhoBERT). Cần Java |

**Không làm:**

- Không xoá emoticon (`=)))`, `:)))`, `:v`): đây là tín hiệu châm biếm (18,2% caption `text-sarcasm`, so với 6–11% ở lớp khác) và phổ biến gấp 3 lần ở test. Bước 7 chỉ rút gọn chữ cái nên không đụng tới chúng.
- Không xoá dấu câu lặp (`???`, `!!!`).

Cấu hình (`configs/base.yaml`):

```yaml
data:
  max_length: 256
  text:
    lowercase: false
    normalize_teencode: true
    emoji: demojize        # keep | remove | demojize
    word_segment: vncorenlp
    cache_file: text_preprocessed.json
```

## 3. Ví dụ

Đầu ra thật của notebook 01a:

| Trường hợp | Gốc | Sau tiền xử lý |
|---|---|---|
| Từ vựng đặc thù | `Ăn 2 kg sữa chua cay, sinh SN 1999, mua bảo hiểm Dr.G và biết PK game` | `Ăn 2 kg sữa chua_cay , sinh SN 1999 , mua bảo_hiểm Dr._G và biết PK game` |
| Teencode, chữ lặp | `Phục vụ đỉnh chóp lun á, đợi có 2 tiếng thui à khum quạo tí nào ngonnnn 🐧` | `Phục_vụ đỉnh chóp luôn á , đợi có 2 tiếng thôi à không quạo_tí nào ngon penguin` |
| HTML, URL, mention | `Ad ơi cmt này đỉnh vch &amp; link https://fb.com #review @admin` | `Ad ơi bình_luận này đỉnh vch & link review` |
| Dấu câu, emoji | `Đồ ăn ngon tuyệt vời... Ăn xong vô viện lun ak ??? !!! 🤡` | `Đồ_ăn ngon tuyệt_vời ... Ăn xong vô_viện luôn à ? ? ? ! ! ! clown face` |

Các lỗi còn thấy trong ví dụ:

- `thui` vẫn được map thành `thôi` ở ví dụ 2 dù bảng teencode hiện tại đã bỏ mục này. Output trong notebook được chạy ở phiên bản bảng cũ hơn; cache dùng để train (`pipeline_key 2c3852f6`) là phiên bản nào thì cần đối chiếu lại `PREPROCESS_VERSION` nếu muốn trích ví dụ cho báo cáo cuối.
- VnCoreNLP tách sai một số cụm (`quạo_tí`, `Dr._G`).
- `vch` (viết tắt tục) không có trong bảng teencode.
- Emoji không có trong `EMOJI_VI` bị đổi thành tên tiếng Anh (`penguin`, `clown face`).

## 4. Cache

- `TextPreprocessor.process_all` chạy một lần cho mọi caption của train, public test và private test, lưu `{meta, texts}`: `texts` ánh xạ caption gốc → caption đã xử lý.
- `meta.pipeline_key` là hash của cấu hình, `PREPROCESS_VERSION` và các bảng tra. Code đổi thì key đổi, cache cũ tự bị bỏ qua và tạo lại, nên không bao giờ train nhầm trên kết quả của pipeline cũ.
- Đọc từ cache nhanh hơn chạy lại khoảng **416 lần** (1.000 caption: 2,778 giây so với 0,0067 giây).
- Mọi thí nghiệm từ Phase 3 trở đi dùng cache `pipeline_key = 2c3852f6`.

## 5. Độ dài tối đa

- `max_length: 256` (giới hạn của PhoBERT). p95 độ dài caption sau làm sạch khoảng 251 token, nên 256 bao phủ khoảng 95–96% caption.
- Khi ghép thêm text trích từ ảnh làm segment thứ hai, caption được cắt trước còn tối đa 200 token (`max_caption_length: 200`).

## 6. Bộ lọc tín hiệu nguồn (dùng cho Phase 9, 10)

`preprocessing.py` có thêm các hàm chỉ dùng cho thí nghiệm chẩn đoán, không nằm trong pipeline mặc định:

| Hàm | Xoá |
|---|---|
| `strip_source_tags` | thẻ trang nguồn: `interpool*`, `voz`, `voz_f33`, `wmo`, `worldmemeorganization` |
| `strip_hashtags_and_signatures` | mọi hashtag và chữ ký admin một dòng (`- Elsa`) |
| `clean_ocr` | dòng OCR có dưới 3 chữ/số hoặc không có chữ cái; bỏ cả OCR nếu gần trùng caption (> 0,9) |

## 7. Kết luận

1. Pipeline xử lý các dạng nhiễu mạng xã hội (HTML, URL, mention, chữ lặp, teencode) mà vẫn giữ emoticon và dấu câu mang tín hiệu châm biếm.
2. Emoji được chuyển thành chữ (`demojize`) để PhoBERT đọc được thay vì thành token `<unk>`.
3. Tách từ bằng VnCoreNLP cho đúng định dạng PhoBERT.
4. Cache có khoá theo cấu hình nên cả nhóm dùng chung một bản, tái lập được.
5. Chưa có ablation riêng cho từng bước tiền xử lý (ví dụ `demojize` so với `keep`). Ảnh hưởng của emoji/dấu câu lên dự đoán được đo gián tiếp ở Phase 10 (Experiment D).

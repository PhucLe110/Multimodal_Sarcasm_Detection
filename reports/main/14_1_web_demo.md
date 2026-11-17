# 14.1. Web demo

- **Phase:** 14, Web visualize
- **Thành viên:** Trúc
- **Code:** `src/vimmsd/web/inference.py` (`DataStore`, `RunRegistry`, `InferenceEngine`), `src/vimmsd/web/app.py` (Flask API), `src/vimmsd/web/static/index.html` (giao diện, không cần build), `scripts/serve_web.py`; test `tests/test_web.py`
- **Commit:** `a104919` (data viewer + suy luận trực tiếp), `423c79f` (đường dẫn tương đối để chạy sau proxy)

---

## 1. Trạng thái so với task của `report_main.md`

| Task | Trạng thái |
|---|---|
| Code base | **Xong** |
| Tích hợp từng model vào inference | **Xong**: mọi run có `best.pt` + `config.json` đều được nhận, không cần sửa code theo từng model |
| Gán nhãn đánh giá của người (human-round evaluation) trên dữ liệu đã suy luận | **Chưa làm** |
| Thêm dữ liệu ngoài đời để suy luận và kiểm tra | **Chưa làm** (phụ thuộc Phase 12) |

## 2. Chức năng

Trang **ViMMSD Viewer** có ba khu vực: *Dữ liệu*, *Mẫu đang xem*, *Model*.

- **Duyệt dữ liệu:** chọn split (mặc định public test; train; private test chỉ khi bật cờ), lọc theo nhãn, tìm theo caption, phân trang, lấy mẫu ngẫu nhiên; xem số mẫu theo nhãn.
- **Xem một mẫu:** ảnh, caption, nhãn, OCR và mô tả ảnh (Phase 2.3).
- **Suy luận trực tiếp:** chọn run trong danh sách (có ô lọc theo tên model), xem xác suất 4 lớp và nhãn dự đoán của run đó trên mẫu đang xem; tuỳ chọn hiệu chỉnh prior.

### 2.1. API

| Route | Trả về |
|---|---|
| `GET /` | giao diện |
| `GET /api/meta` | danh sách nhãn, cảnh báo, thiết bị, kích thước và phân bố nhãn từng split |
| `GET /api/runs` | danh sách run tìm được (tên, config, tham số) |
| `GET /api/samples?split=&label=&q=&page=&random=` | danh sách mẫu theo bộ lọc |
| `GET /api/sample?split=&id=` | chi tiết một mẫu (kèm OCR, mô tả) |
| `GET /image?split=&id=` | ảnh; đường dẫn lấy từ record theo id, không nhận đường dẫn từ client |
| `GET /api/predict?split=&id=&run=&prior=` | xác suất và nhãn dự đoán của một run |

## 3. Thiết kế

- **Suy luận dùng lại đúng các lớp của `scripts/evaluate.py`** (`build_model`, `ViMMSDDataset`, `ViMMSDCollator`), nên đầu vào giống hệt lúc đánh giá; không có code tiền xử lý riêng cho web.
- **Không ghi vào cache chung:** tiền xử lý caption chạy qua `ReadOnlyPreprocessor`, không ghi vào `text_preprocessed.json` của nhóm.
- **Quản lý bộ nhớ:** giữ tối đa `--max-loaded` model (mặc định 2) trong bộ nhớ cùng lúc.
- **Lỗi cô lập theo run:** checkpoint lệch code hay thiếu file chỉ làm hỏng run đó (HTTP 500 cho request đó), các run khác vẫn chạy.
- **Bảo vệ private test:** mặc định không hiện private test và điểm test của các run, vì private test chỉ chấm ở bước cuối (giao thức Phase 10). Bật bằng `--allow-private`.
- **Chỉ nghe localhost** mặc định (`--host 127.0.0.1`); truy cập từ xa qua SSH tunnel.

## 4. Chạy

```bash
# trên máy có dữ liệu + checkpoint (GPU server)
python scripts/serve_web.py --runs-dir data/server/outputs          # có thể lặp lại --runs-dir
# từ máy cá nhân
ssh -L 7860:localhost:7860 <server>                                 # rồi mở http://localhost:7860
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--config` | `configs/base.yaml` | đường dẫn dữ liệu và danh sách nhãn |
| `--runs-dir` | `paths.output_dir` | thư mục chứa các run (output tải từ Kaggle để đâu cũng được) |
| `--allow-private` | tắt | hiện private test và điểm test |
| `--device` | cuda nếu có | |
| `--max-loaded` | 2 | số model giữ trong bộ nhớ |
| `--host`, `--port` | 127.0.0.1, 7860 | |

## 5. Việc cần làm để hoàn tất phase

1. **Gán nhãn đánh giá của người:** thêm vào giao diện một ô để người xem ghi nhận xét cho từng cặp (mẫu, run): dự đoán đúng/sai theo người xem, nhãn gốc có hợp lý không, cơ chế lỗi. Lưu ra file (CSV/JSONL) cạnh thư mục run. Có thể dùng chính các cột duyệt của 44 mẫu định tính ở Phase 10.6 để thống nhất.
2. **Dữ liệu ngoài:** thêm một split `external` đọc từ file JSON cùng định dạng ViMMSD (Phase 12), và chế độ tải ảnh + nhập caption mới để thử trực tiếp. Caption mới cần đi qua `ReadOnlyPreprocessor`; OCR/mô tả cho ảnh mới cần chạy `extract_image_text` nếu run dùng các luồng đó.
3. Chụp màn hình giao diện cho báo cáo cuối.

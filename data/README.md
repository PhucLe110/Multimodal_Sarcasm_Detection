# Dữ liệu ViMMSD

Không commit dữ liệu vào git. Bộ dữ liệu ViMMSD (UIT Data Science Challenge 2024) gồm:

| File / thư mục | Nội dung |
|---|---|
| `vimmsd-train.json` | annotation tập train: `{id: {"image", "caption", "label"}}` |
| `training-images/train-images/` | ảnh tập train |
| `vimmsd-public-test.json` | annotation public test (trường `label` rỗng) |
| `vimmsd-public-test-ref.json` | nhãn public test: `{"results": {id: nhãn}, "phase": "dev"}` |
| `public-test-images/dev-images/` | ảnh public test |
| `vimmsd-private-test.json` | annotation private test (trường `label` rỗng) |
| `vimmsd-private-test-ref.json` | nhãn private test: `{"results": {id: nhãn}, "phase": "test"}` |
| `private-test-images/test-images/` | ảnh private test |
| `vimmsd-warmup.json`, `warmup-images/warmup-images/` | tập warmup của cuộc thi, hiện không dùng |

Tên file/thư mục trên phải khớp với mục `data` trong `configs/base.yaml`. Nếu bản tải về đặt tên khác thì sửa config, không đổi tên file.

Nguồn: Kaggle dataset [`phuclkz/vimmsd-private-dataset`](https://www.kaggle.com/datasets/phuclkz/vimmsd-private-dataset) (private). Khác bản cũ `hhhoang/vimmsd-dataset` ở chỗ có thêm hai file `*-ref.json` chứa nhãn của public test và private test.

Chạy `notebooks/00_dataset_check.ipynb` để kiểm tra dữ liệu trước khi train.

## Kaggle

Attach dataset qua **Add Input** trong notebook (dataset private, tài khoản cần được share quyền), dữ liệu nằm ở `/kaggle/input/datasets/phuclkz/vimmsd-private-dataset`, khớp `paths.kaggle.data_dir` trong `configs/base.yaml`. Nếu đường dẫn khác thì chạy `00_dataset_check` để xem đường dẫn thật rồi sửa config.

## Local

```bash
# cần ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token)
python scripts/download_data.py --dataset phuclkz/vimmsd-private-dataset
```

Dữ liệu được tải về `data/raw/`.

## Colab

Upload dữ liệu lên Google Drive tại `MyDrive/IE403/data/raw/` (xem `paths.colab` trong `configs/base.yaml`).

## Chia tập

Mặc định `data.split: official`: train = toàn bộ `vimmsd-train.json`, val = public test, test = private test, nhãn val/test đọc từ `data.public_test_labels` / `data.private_test_labels`. Thiếu file nhãn thì báo lỗi ngay thay vì train với val/test không nhãn.

`data.split: random` giữ cách cũ: chia **stratified** từ `vimmsd-train.json` (`val_ratio` / `test_ratio`, `split_seed`). Dùng khi muốn tái lập kết quả các thí nghiệm trước khi đổi dataset.

Xuất file nộp bài theo định dạng UIT DSC: `python scripts/evaluate.py --config ... --split public_test` (phase `dev`) hoặc `--split private_test` (phase `test`).

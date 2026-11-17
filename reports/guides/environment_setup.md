# Hướng dẫn setup môi trường dự án ViMMSD

- **Repo:** [github.com/yunaLee21/Multimodal_Sarcasm_Detection](https://github.com/yunaLee21/Multimodal_Sarcasm_Detection) (public, nhánh `main`)
- **Dữ liệu:** Kaggle dataset [`hhhoang/vimmsd-dataset`](https://www.kaggle.com/datasets/hhhoang/vimmsd-dataset) (khoảng 1 GB)
- **Áp dụng cho:** codebase tại commit `1992088` (26/09/2026)

---

## 1. Tổng quan

Dự án chạy trên 3 môi trường, mỗi môi trường có một vai trò riêng. Code **tự nhận diện** đang chạy ở môi trường nào (`src/vimmsd/utils/env.py`) và tự chọn đường dẫn dữ liệu tương ứng trong `configs/base.yaml`, nên không cần sửa code khi chuyển môi trường.

| | Kaggle | Colab | Local |
|---|---|---|---|
| **Vai trò** | Môi trường chính: train model, chạy OCR/VLM | Dự phòng khi hết quota GPU Kaggle | Viết code, chạy test, commit/push Git |
| **GPU** | T4 / P100 miễn phí | T4 miễn phí | Không bắt buộc |
| **Lấy code** | Notebook tự clone từ GitHub | Notebook tự clone từ GitHub | `git clone` + `pip install -e .` |
| **Lấy dữ liệu** | *Add Input* `hhhoang/vimmsd-dataset` | Google Drive | `scripts/download_data.py` |
| **Cài thư viện** | Tự động (cell bootstrap) | Tự động (cell bootstrap) | `pip install -r requirements.txt` |

**Luồng làm việc chung:**

1. Sửa code trong `src/` ở **local**, chạy `pytest tests`, commit và push lên nhánh của mình trên GitHub.
2. Trên **Kaggle**, upload file notebook (`.ipynb`), đặt biến `REF` trong cell đầu tiên bằng tên nhánh đó, rồi *Run All*.
3. Notebook tự clone đúng phiên bản code từ GitHub, **không cần upload cả codebase lên Kaggle**.

---

## 2. Kaggle (môi trường chính)

### 2.1. Chuẩn bị tài khoản (một lần cho mỗi thành viên)

1. Tạo tài khoản Kaggle và **xác minh số điện thoại** (*Settings → Phone verification*). Nếu chưa xác minh thì không bật được Internet trong notebook, và cell đầu tiên sẽ không clone được code.
2. Repo đang để **public** nên **không cần** tạo secret `GITHUB_TOKEN`. Chỉ khi repo chuyển sang private mới cần: tạo GitHub fine-grained token chỉ có quyền đọc repo này, rồi lưu vào *Add-ons → Secrets* với tên `GITHUB_TOKEN`. Không dán token trực tiếp vào notebook.

### 2.2. Tạo và cấu hình notebook

1. Trên Kaggle: *Create → New Notebook → File → Import Notebook*, rồi chọn file `.ipynb` trong thư mục `notebooks/` của repo.
2. Ở thanh bên phải (*Settings*):
    - **Internet: On** (bắt buộc).
    - **Accelerator: GPU T4 x2** hoặc **GPU P100**. Riêng `00_dataset_check` và `01_eda` không cần GPU.
3. *Add Input* → tìm `hhhoang/vimmsd-dataset` → *Add*. Dữ liệu sẽ nằm ở:

    ```
    /kaggle/input/datasets/hhhoang/vimmsd-dataset
    ```

    Đường dẫn này đã được xác nhận khi chạy `00_dataset_check` và đã có sẵn trong `paths.kaggle.data_dir`.

4. *Run All*.

### 2.3. Cell bootstrap làm gì

Cell đầu tiên của mọi notebook (bản gốc ở `notebooks/00_template.ipynb`) tự động:

1. Nhận diện môi trường bằng biến `KAGGLE_KERNEL_RUN_TYPE` (Kaggle) hoặc `COLAB_RELEASE_TAG` (Colab).
2. Clone repo vào `/tmp/IE403` rồi checkout `REF` (mặc định `main`). Code được clone ra ngoài `/kaggle/working` để không bị lưu lẫn vào output.
3. Cài các thư viện Kaggle chưa có trong `requirements-kaggle.txt` (`underthesea`, `emoji`).
4. `pip install -e` package `vimmsd` và chuyển thư mục làm việc về gốc repo.
5. In ra commit hash của code đang chạy. Ghi hash này vào `reports/results/results.md` để tái lặp kết quả.

Chỉ cần sửa một biến trong cell này:

| Biến | Ý nghĩa |
|---|---|
| `REF` | Nhánh cần chạy (ví dụ `feat/cross-attention`), hoặc một commit hash cụ thể khi train chính thức |

### 2.4. Thư viện

Kaggle đã cài sẵn `torch`, `torchvision`, `transformers`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`, `pillow`, `tqdm`. **Không cài lại `torch`**, vì dễ làm lệch phiên bản CUDA của môi trường.

| Notebook | Thư viện cài thêm | Ai cài |
|---|---|---|
| Mọi notebook | `underthesea`, `emoji` | Cell bootstrap |
| `01b_image_text_extraction` | `paddlepaddle-gpu`, `paddleocr`, `timm`, `einops` | Cell `!pip install` ở đầu notebook |

`timm` và `einops` là thư viện mà model Vintern-1B (VLM mô tả ảnh) yêu cầu. Các notebook train chỉ đọc cache JSON do `01b` tạo ra, nên không cần cài OCR hay VLM.

### 2.5. Lưu kết quả và dùng lại giữa các notebook

Mọi thứ ghi vào `/kaggle/working` (checkpoint trong `outputs/`, cache trong `cache/`) **chỉ được giữ lại** khi chạy bằng *Save Version → Save & Run All (Commit)*. Nếu chỉ chạy tương tác rồi tắt session thì kết quả sẽ mất.

| Muốn dùng lại | Cách làm |
|---|---|
| Cache OCR / mô tả VLM (`01b`) cho notebook train | *Add Input* → *Your Work* → chọn output của notebook `01b`, rồi đặt `data.image_text.dir` (hoặc biến `IMAGE_TEXT_DIR` trong notebook 07) trỏ tới thư mục `cache` trong đó |
| Resume train khi session bị ngắt | Attach output của lần chạy trước làm input, rồi chạy lại với `--override train.resume_from=<thư mục output>` |
| Checkpoint `best.pt` cho notebook 06 | Attach output của notebook 04/05, đặt `CKPT_DIR` trong notebook 06 |

Một session Kaggle chạy tối đa khoảng 12 giờ, và quota GPU khoảng 30 giờ/tuần (theo chính sách hiện tại của Kaggle, có thể thay đổi). Các bước dài như VLM mô tả khoảng 13.700 ảnh có thể chia làm nhiều session: cache được ghi tiếp, không mất phần đã chạy.

### 2.6. Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Cell đầu tiên lỗi ở `git clone` | Chưa bật Internet hoặc tài khoản chưa xác minh số điện thoại | Bật *Internet: On* trong *Settings* |
| `FileNotFoundError` khi đọc dữ liệu | Chưa attach dataset, hoặc Kaggle mount ở đường dẫn khác | Chạy `00_dataset_check`, đối chiếu đường dẫn nó in ra với `paths.kaggle.data_dir` |
| `thiếu .../ocr.json: chạy scripts/extract_image_text.py` | Config bật `data.image_text` nhưng chưa attach output của `01b` | Attach output `01b` và đặt `data.image_text.dir` |
| Lỗi khi `import paddle` / không thấy GPU trong PaddleOCR | `paddlepaddle-gpu` không khớp phiên bản CUDA của Kaggle | Cài bản `paddlepaddle-gpu` đúng với CUDA (`nvidia-smi` để xem), hoặc chuyển sang EasyOCR |
| Train rất chậm, log báo `cuda` không khả dụng | Chưa bật GPU | *Accelerator: GPU* |
| `ModuleNotFoundError: vimmsd` | Kernel chưa thấy package vừa cài | Chạy lại cell bootstrap (cell đã tự thêm `src/` vào `sys.path`) |

---

## 3. Colab (dự phòng)

1. Upload dữ liệu ViMMSD (tải từ Kaggle) lên Google Drive, đặt đúng cấu trúc:

    ```
    MyDrive/IE403/data/raw/
    ├── vimmsd-train.json
    ├── vimmsd-public-test.json
    ├── vimmsd-private-test.json
    ├── training-images/train-images/
    ├── public-test-images/dev-images/
    └── private-test-images/test-images/
    ```

2. Mở notebook trên Colab: *File → Upload notebook*, hoặc mở trực tiếp từ GitHub.
3. *Runtime → Change runtime type → T4 GPU*.
4. *Run All*. Cell bootstrap tự mount Google Drive (Colab sẽ hỏi quyền truy cập), clone repo và cài thư viện.
5. Checkpoint và cache được lưu thẳng lên Drive (`MyDrive/IE403/outputs`, `MyDrive/IE403/cache`), nên không mất khi session tắt.

---

## 4. Local (viết code và chạy test)

### 4.1. Yêu cầu

- **Python 3.10 trở lên**
- **Git**, đã cấu hình `user.name` / `user.email`
- Không cần GPU hay mạng: toàn bộ test chạy bằng dữ liệu giả, tokenizer giả và tensor ngẫu nhiên

### 4.2. Cài đặt

```bash
git clone https://github.com/yunaLee21/Multimodal_Sarcasm_Detection.git
cd Multimodal_Sarcasm_Detection

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Máy không có GPU: cài torch bản CPU trước (nhẹ hơn nhiều so với bản CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -e .                     # cài package vimmsd từ src/

pytest tests                         # kỳ vọng: 14 passed
```

- `.venv/` đã có trong `.gitignore` nên không bị commit.
- `paddleocr` và `paddlepaddle` trong `requirements.txt` chỉ cần khi chạy OCR. Nếu local chỉ dùng để code và chạy test thì có thể bỏ 2 dòng này cho nhẹ, vì OCR chạy trên Kaggle.

### 4.3. Tải dữ liệu về local (không bắt buộc)

Chỉ cần khi muốn chạy EDA hay train thử ở local:

1. Kaggle → *Settings → API → Create New Token*, tải file `kaggle.json` về.
2. Đặt file vào `~/.kaggle/kaggle.json` rồi chạy `chmod 600 ~/.kaggle/kaggle.json`. Không commit file này (đã có trong `.gitignore`).
3. Tải dữ liệu (khoảng 1 GB, lưu vào `data/raw/`):

    ```bash
    python scripts/download_data.py --dataset hhhoang/vimmsd-dataset
    ```

### 4.4. Chạy notebook ở local

Mở notebook từ thư mục `notebooks/`. Cell bootstrap nhận ra đang chạy ở local, bỏ qua bước clone và tự chuyển về gốc repo. Cần đã chạy `pip install -e .` trước đó.

---

## 5. Đường dẫn theo môi trường

`configs/base.yaml` khai báo sẵn 3 nhánh. `load_config` tự chọn nhánh theo môi trường đang chạy. Đường dẫn tương đối ở local được tính từ gốc repo (thư mục chứa `pyproject.toml`).

| | `data_dir` | `output_dir` | `cache_dir` |
|---|---|---|---|
| **local** | `data/raw` | `outputs` | `data/processed` |
| **kaggle** | `/kaggle/input/datasets/hhhoang/vimmsd-dataset` | `/kaggle/working/outputs` | `/kaggle/working/cache` |
| **colab** | `/content/drive/MyDrive/IE403/data/raw` | `/content/drive/MyDrive/IE403/outputs` | `/content/drive/MyDrive/IE403/cache` |

Có thể ghi đè bất kỳ giá trị nào khi chạy mà không cần sửa file, ví dụ:

```bash
python scripts/train.py --config configs/text_only.yaml \
    --override paths.kaggle.data_dir=/kaggle/input/khac train.epochs=5
```

---

## 6. Checklist

**Mỗi thành viên, một lần:**

- [ ] Tài khoản Kaggle đã xác minh số điện thoại
- [ ] Clone repo về máy, tạo `.venv`, `pip install -r requirements.txt`, `pip install -e .`
- [ ] `pytest tests` chạy được (14 passed)
- [ ] (Không bắt buộc) `~/.kaggle/kaggle.json` để tải dữ liệu về local

**Mỗi notebook trên Kaggle:**

- [ ] *Internet: On*
- [ ] *Accelerator: GPU* (trừ `00_dataset_check`, `01_eda`)
- [ ] *Add Input* `hhhoang/vimmsd-dataset`
- [ ] Đặt `REF` đúng nhánh / commit cần chạy
- [ ] Nếu cần cache OCR/VLM hoặc checkpoint: attach output của notebook trước
- [ ] Chạy bằng *Save Version* để giữ lại output

---

## Phụ lục: danh sách thư viện

| Thư viện | Dùng cho | Local | Kaggle/Colab |
|---|---|---|---|
| `torch`, `torchvision` | Model, augmentation ảnh | `requirements.txt` | Có sẵn |
| `transformers` (>= 4.45) | PhoBERT, CLIP, VLM | `requirements.txt` | Có sẵn |
| `underthesea` | Tách từ tiếng Việt cho PhoBERT | `requirements.txt` | Cell bootstrap |
| `emoji` | Xử lý emoji trong caption | `requirements.txt` | Cell bootstrap |
| `scikit-learn` | Metrics, chia tập stratified | `requirements.txt` | Có sẵn |
| `pyyaml` | Đọc config | `requirements.txt` | Có sẵn |
| `pandas`, `matplotlib`, `pillow`, `tqdm` | Phân tích, vẽ hình, đọc ảnh, progress bar | `requirements.txt` | Có sẵn |
| `paddleocr`, `paddlepaddle(-gpu)` | OCR chữ trong ảnh | `requirements.txt` | Notebook `01b` |
| `timm`, `einops` | Model Vintern-1B | – | Notebook `01b` |
| `kaggle` | Tải dữ liệu qua API | `requirements.txt` | Không cần |
| `pytest` | Chạy test | `requirements.txt` | Không cần |

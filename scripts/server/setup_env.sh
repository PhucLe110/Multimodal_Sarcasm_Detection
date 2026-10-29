#!/usr/bin/env bash
# Cài môi trường trên GPU server (CODEBASE.md mục 7.1). Chạy một lần từ bất kỳ đâu trong repo:
#   bash scripts/server/setup_env.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 WITH_IMAGE_TEXT=1 bash scripts/server/setup_env.sh
#
# Biến tuỳ chọn:
#   VENV=.venv            thư mục venv (tương đối với gốc repo)
#   TORCH_INDEX_URL=...   index của PyTorch đúng với CUDA của driver (xem nvidia-smi); trống = bản pip mặc định
#   WITH_IMAGE_TEXT=1     cài thêm OCR (paddlepaddle-gpu, paddleocr, vietocr) và timm cho Vintern;
#                         chỉ cần khi sinh lại cache ảnh → text, train chỉ đọc JSON nên không cần
#   PADDLE_INDEX_URL=...  index của paddlepaddle-gpu nếu bản pip mặc định không khớp CUDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

VENV=${VENV:-.venv}
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q -U pip

if [ -n "${TORCH_INDEX_URL:-}" ]; then
  pip install -q torch torchvision --index-url "$TORCH_INDEX_URL"
else
  pip install -q torch torchvision
fi

# requirements.txt trừ paddle: bản CPU vừa nặng vừa vô ích trên server, OCR cài riêng bản GPU ở dưới
reqs=$(mktemp)
grep -vE '^[[:space:]]*(paddleocr|paddlepaddle)\b' requirements.txt > "$reqs"
pip install -q -r "$reqs"
rm -f "$reqs"
pip install -q -e .

if [ "${WITH_IMAGE_TEXT:-0}" = "1" ]; then
  if [ -n "${PADDLE_INDEX_URL:-}" ]; then
    pip install -q paddlepaddle-gpu -i "$PADDLE_INDEX_URL"
  else
    pip install -q paddlepaddle-gpu
  fi
  pip install -q "paddleocr>=3.0" timm
  pip install -q --no-deps vietocr  # vietocr ghim pillow/einops/gdown bản cũ (CODEBASE.md mục 8)
fi

echo "== kiểm tra"
python - <<'EOF'
import torch
n = torch.cuda.device_count()
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {n} GPU")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}, {p.total_memory / 1e9:.0f} GB")
EOF
if java -version >/dev/null 2>&1; then
  java -version 2>&1 | head -1
else
  echo "CẢNH BÁO: chưa có Java, VnCoreNLP (data.text.word_segment=vncorenlp) sẽ không chạy được"
fi
[ -n "${VIMMSD_ENV:-}" ] || echo "CẢNH BÁO: chưa đặt VIMMSD_ENV=server (thêm vào ~/.bashrc, xem CODEBASE.md mục 7.1)"
pytest tests -q
echo "xong: source $VENV/bin/activate"

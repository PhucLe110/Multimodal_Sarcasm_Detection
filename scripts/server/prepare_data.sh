#!/usr/bin/env bash
# Chuẩn bị dữ liệu trên GPU server (CODEBASE.md mục 7.2, 7.4). Chạy sau setup_env.sh, trong venv:
#   DATA_ROOT=/data/<user> bash scripts/server/prepare_data.sh               # tải cache có sẵn của nhóm
#   DATA_ROOT=/data/<user> REGENERATE=1 bash scripts/server/prepare_data.sh  # sinh lại cache text + ảnh
#   DATA_ROOT=/data/<user> RAW_DIR=/data/<user>/vimmsd-data bash ...          # dùng ViMMSD có sẵn, không tải
#
# Tạo trên ổ dữ liệu:  $DATA_ROOT/vimmsd (ViMMSD gốc), $DATA_ROOT/vimmsd_cache, $DATA_ROOT/vimmsd_outputs
# và symlink trong repo: data/server/{raw,cache,outputs} (khớp paths.server trong configs/base.yaml).
# Chạy lại được: bước nào đã có kết quả thì bỏ qua.
#
# Biến:
#   DATA_ROOT (bắt buộc)
#   RAW_DIR               thư mục ViMMSD đã có sẵn trên server (chứa vimmsd-train.json): dùng luôn, không tải từ Kaggle
#   RAW_DATASET=phuclkz/vimmsd-private-dataset   chỉ dùng khi không đặt RAW_DIR
#   CACHE_DATASET=funalee/vimmsd-data-preprocessed   text_preprocessed.json + image_preprocessed.json của nhóm
#   REGENERATE=1          sinh lại cache thay vì tải (cần WITH_IMAGE_TEXT=1 lúc setup_env.sh)
#   NUM_GPUS              số tiến trình OCR / VLM chạy song song khi REGENERATE=1 (mặc định: số GPU)
#   VLM=vintern           backend mô tả ảnh của extract_image_text.py (vintern | hf)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${DATA_ROOT:?đặt DATA_ROOT=/đường/dẫn/ổ/dữ/liệu}"
export VIMMSD_ENV=${VIMMSD_ENV:-server}
RAW_DATASET=${RAW_DATASET:-phuclkz/vimmsd-private-dataset}
CACHE_DATASET=${CACHE_DATASET:-funalee/vimmsd-data-preprocessed}
RAW="$DATA_ROOT/vimmsd"
CACHE="$DATA_ROOT/vimmsd_cache"
OUTPUTS="$DATA_ROOT/vimmsd_outputs"
mkdir -p "$RAW" "$CACHE" "$OUTPUTS" data/server logs

link() {  # link <đích> <tên trong data/server>: không đè thư mục thật
  local target=$1 name="data/server/$2"
  if [ -L "$name" ]; then
    ln -sfn "$target" "$name"
  elif [ -e "$name" ]; then
    echo "LỖI: $name đã tồn tại và không phải symlink, xử lý tay trước" >&2
    exit 1
  else
    ln -s "$target" "$name"
  fi
}

if [ -n "${RAW_DIR:-}" ]; then
  echo "== 1. ViMMSD gốc có sẵn ($RAW_DIR)"
  RAW_DIR=$(cd "$RAW_DIR" && pwd -P)
else
  echo "== 1. ViMMSD gốc ($RAW_DATASET)"
  # -L: $RAW có thể là symlink tới thư mục dữ liệu có sẵn
  if ! find -L "$RAW" -name vimmsd-train.json -print -quit | grep -q .; then
    python scripts/download_data.py --dataset "$RAW_DATASET" --out "$RAW"
  fi
  # file zip có thể giải nén vào một thư mục con: trỏ data_dir vào thư mục chứa vimmsd-train.json
  RAW_DIR=$(dirname "$(find -L "$RAW" -name vimmsd-train.json -print -quit)")
fi
# bố cục phẳng (train-images/, dev-images/, test-images/ nằm cạnh file JSON): dựng thư mục symlink theo bố cục
# của config, không di chuyển dữ liệu gốc. Tên thư mục ảnh giữ nguyên nên khoá ảnh trong cache không đổi.
if [ ! -e "$RAW_DIR/training-images/train-images" ] && [ -d "$RAW_DIR/train-images" ]; then
  VIEW="$DATA_ROOT/vimmsd_layout"
  echo "bố cục phẳng: dựng $VIEW trỏ vào $RAW_DIR"
  mkdir -p "$VIEW/training-images" "$VIEW/public-test-images" "$VIEW/private-test-images"
  for f in vimmsd-train.json vimmsd-public-test.json vimmsd-private-test.json; do
    ln -sfn "$RAW_DIR/$f" "$VIEW/$f"
  done
  ln -sfn "$RAW_DIR/train-images" "$VIEW/training-images/train-images"
  ln -sfn "$RAW_DIR/dev-images" "$VIEW/public-test-images/dev-images"
  ln -sfn "$RAW_DIR/test-images" "$VIEW/private-test-images/test-images"
  RAW_DIR=$VIEW
fi
# đúng bố cục thư mục trong configs/base.yaml (data.*_json, data.*_image_dir)
for p in vimmsd-train.json vimmsd-public-test.json vimmsd-private-test.json \
         training-images/train-images public-test-images/dev-images private-test-images/test-images; do
  if [ ! -e "$RAW_DIR/$p" ]; then
    echo "LỖI: thiếu $RAW_DIR/$p (bố cục phải khớp data.* trong configs/base.yaml)" >&2
    exit 1
  fi
done
link "$RAW_DIR" raw
link "$CACHE" cache
link "$OUTPUTS" outputs
ls data/server/raw

if [ "${REGENERATE:-0}" != "1" ]; then
  echo "== 2. Cache có sẵn ($CACHE_DATASET)"
  if [ ! -f "$CACHE/text_preprocessed.json" ] || [ ! -f "$CACHE/image_preprocessed.json" ]; then
    python scripts/download_data.py --dataset "$CACHE_DATASET" --out "$CACHE/_download"
    for f in text_preprocessed.json image_preprocessed.json image_preprocessed.csv; do
      src=$(find "$CACHE/_download" -name "$f" -print -quit)
      if [ -n "$src" ]; then cp "$src" "$CACHE/$f"; else echo "CẢNH BÁO: $CACHE_DATASET không có $f"; fi
    done
  fi
  ls -la "$CACHE"
  echo "xong. Lưu ý: cấu hình data.text khác lúc tạo cache thì train sẽ tự tiền xử lý lại (meta.pipeline_key)."
  exit 0
fi

echo "== 2. Tiền xử lý caption"
python scripts/preprocess_text.py --config configs/base.yaml 2>&1 | tee logs/preprocess_text.log

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
VLM=${VLM:-vintern}
run_sharded() {  # run_sharded <task> [tham số thêm]: mỗi GPU một shard, rồi gộp shard vào file cache chính
  local task=$1; shift
  local pids=()
  for ((i = 0; i < NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python scripts/extract_image_text.py --config configs/base.yaml --task "$task" \
      --num-shards "$NUM_GPUS" --shard "$i" "$@" > "logs/extract_${task}_shard$i.log" 2>&1 &
    pids+=($!)
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [ "$NUM_GPUS" -gt 1 ]; then
    TASK=$task NUM_SHARDS=$NUM_GPUS python - <<'EOF'
import os
from pathlib import Path

from vimmsd.data.image_text import merge_image_text_caches, shard_cache_name
from vimmsd.utils.config import load_config

cfg = load_config("configs/base.yaml")
it = cfg.data.image_text
out_dir = Path(it.get("dir") or cfg.paths.cache_dir)
name = it.ocr_cache if os.environ["TASK"] == "ocr" else it.description_cache
n = int(os.environ["NUM_SHARDS"])
shards = [out_dir / shard_cache_name(name, i, n) for i in range(n)]
merged = merge_image_text_caches(out_dir / name, [p for p in shards if p.exists()])
print(f"gộp {len(shards)} shard -> {out_dir / name}: {len(merged)} ảnh")
EOF
  fi
  if [ "$failed" = 1 ]; then
    echo "LỖI: $task còn ảnh lỗi, xem logs/extract_${task}_shard*.log rồi chạy lại script (ảnh đã xong được bỏ qua)" >&2
    exit 1
  fi
}

echo "== 3. OCR ($NUM_GPUS GPU)"
run_sharded ocr
echo "== 4. Mô tả ảnh ($VLM, $NUM_GPUS GPU)"
run_sharded description --vlm "$VLM"
echo "== 5. Gộp image_preprocessed.json"
python scripts/build_image_text_dataset.py --config configs/base.yaml --out-dir data/server/cache
ls -la "$CACHE"
echo "xong. Nhớ upload cache mới lên Kaggle dataset cho nhóm (CODEBASE.md mục 7.6)."

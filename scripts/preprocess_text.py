"""Chuẩn hoá toàn bộ caption (train, public test, private test) và ghi cache text, thay cho notebook 01a.

    python scripts/preprocess_text.py --config configs/base.yaml
    python scripts/preprocess_text.py --config configs/base.yaml --override data.text.word_segment=none

Ghi <data.text.cache_dir hoặc paths.cache_dir>/<data.text.cache_file> (mặc định text_preprocessed.json).
File đã có và đúng cấu hình (meta.pipeline_key) thì chỉ xử lý thêm caption còn thiếu; khác cấu hình thì làm lại.
VnCoreNLP (word_segment: vncorenlp) cần Java và tự tải model về VNCORENLP_DIR ở lần chạy đầu. Chạy trên CPU."""
import argparse
import logging

from vimmsd.data.dataset import build_text_cache
from vimmsd.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config, args.override)
    cache_file, n_captions, used = build_text_cache(cfg)
    print(f"{n_captions} caption từ {', '.join(used)} -> {cache_file}")


if __name__ == "__main__":
    main()

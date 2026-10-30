"""Gộp kết quả OCR và mô tả VLM (2 cache trung gian của notebooks/01b) thành bộ dữ liệu gọn để train và upload.

    python scripts/build_image_text_dataset.py --config configs/base.yaml --out-dir /kaggle/working/image

    # file cache đặt tên khác (vd bản cũ ocr_v2.json, vlm_description_v1.json):
    python scripts/build_image_text_dataset.py --ocr data/processed/ocr_v2.json \\
        --description data/processed/vlm_description_v1.json --out-dir data/processed/image

Tạo 2 file trong --out-dir:
- image_preprocessed.json: {"meta": {...}, "images": {image_key: {"ocr": str, "description": str}}}. Code train đọc file này
  (data.image_text.file). "meta" ghi model OCR/VLM, commit lúc gộp, số ảnh, nên không cần đánh số phiên bản trong tên file.
- image_preprocessed.csv: mỗi mẫu của cả 3 tập một dòng (split, id, image_key, label, caption, ocr, description), để xem và phân tích.
Ảnh chưa có trong một cache thì trường tương ứng là null (JSON) hoặc ô trống (CSV), và được báo khi chạy."""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

from vimmsd.data.dataset import load_records
from vimmsd.data.image_text import image_key, merge_image_texts
from vimmsd.utils.config import load_config
from vimmsd.utils.env import git_commit

SPLITS = (  # (tên split, khóa file annotation, khóa thư mục ảnh, khóa file nhãn) trong cfg.data
    ("train", "train_json", "train_image_dir", None),
    ("public_test", "public_test_json", "public_test_image_dir", "public_test_labels"),
    ("private_test", "private_test_json", "private_test_image_dir", "private_test_labels"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--ocr", type=Path, default=None, help="cache OCR, mặc định <cache_dir>/<data.image_text.ocr_cache>")
    parser.add_argument("--description", type=Path, default=None,
                        help="cache mô tả VLM, mặc định <cache_dir>/<data.image_text.description_cache>")
    parser.add_argument("--out-dir", type=Path, default=None, help="mặc định <cache_dir>")
    parser.add_argument("--data-dir", type=Path, default=None, help="thư mục chứa file annotation, mặc định paths.data_dir")
    parser.add_argument("--ocr-model", default="PaddleOCR (phát hiện vùng chữ) + VietOCR vgg_seq2seq")
    parser.add_argument("--vlm-model", default="5CD-AI/Vintern-1B-v3_5", help="ghi đúng model đã dùng để sinh mô tả")
    parser.add_argument("--vlm-max-new-tokens", type=int, default=96)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    it_cfg = cfg.data.image_text
    cache_dir = Path(it_cfg.get("dir") or cfg.paths.cache_dir)
    ocr_path = args.ocr or cache_dir / it_cfg.ocr_cache
    desc_path = args.description or cache_dir / it_cfg.description_cache
    out_dir = args.out_dir or cache_dir
    data_dir = args.data_dir or Path(cfg.paths.data_dir)

    caches = {}
    for field, path in (("ocr", ocr_path), ("description", desc_path)):
        if path.exists():
            caches[field] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"không có {path}: trường {field} sẽ là null")
    if not caches:
        raise SystemExit("không có cache OCR lẫn mô tả để gộp")

    label2id = {l: i for i, l in enumerate(cfg.data.labels)}
    rows, keys = [], []
    for split, json_key, dir_key, labels_key in SPLITS:
        ann = data_dir / cfg.data[json_key]
        if not ann.exists():
            print(f"bỏ qua {split}: không có {ann}")
            continue
        image_dir = data_dir / cfg.data[dir_key]
        labels_name = cfg.data.get(labels_key) if labels_key else None
        labels_path = data_dir / labels_name if labels_name else None
        for r in load_records(ann, image_dir, label2id, labels_path=labels_path):
            key = image_key(image_dir, Path(r["image_path"]).name)
            keys.append(key)
            rows.append({"split": split, "id": r["id"], "image_key": key,
                         "label": cfg.data.labels[r["label"]] if r["label"] >= 0 else "",
                         "caption": r["caption"]})
    if not keys:
        raise SystemExit(f"không đọc được file annotation nào trong {data_dir}")

    images = merge_image_texts(caches.get("ocr", {}), caches.get("description", {}), keys)
    missing = {f: sum(v[f] is None for v in images.values()) for f in ("ocr", "description")}
    meta = {
        "created": date.today().isoformat(),
        "merge_commit": git_commit(),  # commit lúc gộp; commit lúc chạy OCR/VLM nằm ở log notebook 01b
        "ocr_model": args.ocr_model if "ocr" in caches else None,
        "vlm_model": args.vlm_model if "description" in caches else None,
        "vlm_max_new_tokens": args.vlm_max_new_tokens if "description" in caches else None,
        "num_images": len(images),
        "missing": missing,
        "sources": {f: str(p) for f, p in (("ocr", ocr_path), ("description", desc_path)) if f in caches},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "image_preprocessed.json").write_text(json.dumps({"meta": meta, "images": images}, ensure_ascii=False),
                                             encoding="utf-8")
    with open(out_dir / "image_preprocessed.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "id", "image_key", "label", "caption", "ocr", "description"])
        writer.writeheader()
        for row in rows:
            v = images[row["image_key"]]
            writer.writerow({**row, "ocr": v["ocr"] or "", "description": v["description"] or ""})

    print(f"đã ghi {out_dir / 'image_preprocessed.json'} ({len(images)} ảnh) và {out_dir / 'image_preprocessed.csv'} ({len(rows)} mẫu)")
    print("thiếu:", missing)
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

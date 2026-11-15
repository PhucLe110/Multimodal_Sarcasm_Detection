"""Sinh caption augmentation cho lớp hiếm bằng back-translation (vi -> en -> vi), lọc tự động, ghi manifest để duyệt.

    python scripts/augment_captions.py --config configs/fusion_concat_frozen_reg_aug.yaml
    python scripts/augment_captions.py --config ... --import-review augmentation_review.csv   # nạp kết quả người duyệt

Chỉ dùng caption của tập train, ảnh giữ nguyên (docs/empirical_analysis_plan.md mục 8.2): mỗi mẫu gốc sinh
--candidates bản, train dùng tối đa 1 bản được accept. Manifest ghi ở data.augment.file (tương đối với
paths.cache_dir), file review .csv cạnh manifest: điền cột review = accept | reject rồi --import-review.
Chạy xong caption mới được tiền xử lý sẵn vào cache text, để các run train song song không cùng ghi cache."""
import argparse
import copy
import csv
import logging
from collections import Counter

import torch

from vimmsd.data.augmentation import (ACCEPT, BackTranslator, accepted_variants, auto_review, normalize_caption,
                                      read_manifest, restore_emojis, write_manifest)
from vimmsd.data.dataset import augment_manifest_path, load_all_records, stage_preprocessed_cache, text_cache_dir
from vimmsd.data.preprocessing import TextPreprocessor
from vimmsd.utils.config import load_config

REVIEW_COLUMNS = ["parent_id", "candidate", "label", "parent_caption", "caption", "auto_verdict", "auto_reasons",
                  "similarity", "review", "review_note"]
log = logging.getLogger("augment")


def review_csv_path(manifest):
    return manifest.with_name(manifest.stem.replace("manifest", "review") + ".csv")


def generate(cfg, labels, candidates, seed, batch_size, limit=None):
    no_aug = copy.deepcopy(cfg)
    no_aug.data.augment.file = None
    train = load_all_records(no_aug)["train"]
    names = list(cfg.data.labels)
    parents = [r for r in train if names[r["label"]] in labels][:limit]
    log.info("%d mẫu gốc %s, mỗi mẫu %d bản", len(parents), dict(Counter(names[r["label"]] for r in parents)), candidates)

    torch.manual_seed(seed)
    translator = BackTranslator(device="cuda" if torch.cuda.is_available() else "cpu")
    outputs = translator([r["caption"] for r in parents], batch_size=batch_size, num_return_sequences=candidates)

    known = {normalize_caption(r["caption"]) for r in train}
    rows = []
    for parent, texts in zip(parents, outputs):
        for k, text in enumerate(texts):
            caption, restored = restore_emojis(parent["caption"], text)
            verdict, reasons, similarity = auto_review(parent["caption"], caption, known)
            known.add(normalize_caption(caption))
            rows.append({
                "parent_id": parent["id"], "candidate": k, "label": names[parent["label"]],
                "parent_caption": parent["caption"], "caption": caption, "restored_emoji": restored,
                "auto_verdict": verdict, "auto_reasons": reasons, "similarity": similarity,
                "review": None, "review_note": None,
                "method": "backtranslation vi-en-vi", "models": [translator.vi2en, translator.en2vi],
                "settings": {"vi2en": "beam 5", "en2vi": "top_p 0.9"}, "seed": seed,
            })
    return rows


def write_review_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel / Google Sheets đọc đúng dấu
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "auto_reasons": "; ".join(row["auto_reasons"])})


def import_review(rows, path):
    with open(path, encoding="utf-8-sig") as f:
        reviewed = {(r["parent_id"], int(r["candidate"])): r for r in csv.DictReader(f)}
    for row in rows:
        r = reviewed.get((row["parent_id"], row["candidate"]))
        if r and (r.get("review") or "").strip():
            verdict = r["review"].strip().lower()
            if verdict not in ("accept", "reject"):
                raise ValueError(f"review của {row['parent_id']}#{row['candidate']} phải là accept | reject, nhận {verdict!r}")
            row["review"], row["review_note"] = verdict, (r.get("review_note") or "").strip() or None
    return rows


def report(rows, require_review):
    by_label = Counter(r["label"] for r in rows)
    auto_ok = Counter(r["label"] for r in rows if r["auto_verdict"] == ACCEPT)
    reviewed = Counter(r["label"] for r in rows if r["review"])
    used = Counter(v[0]["label"] for v in accepted_variants(rows, require_review).values())
    for label in sorted(by_label):
        log.info("%s: %d bản, lọc tự động giữ %d, người đã duyệt %d, dùng để train %d", label, by_label[label],
                 auto_ok[label], reviewed[label], used[label])
    reasons = Counter(reason for r in rows for reason in r["auto_reasons"])
    log.info("lý do loại: %s", dict(reasons.most_common()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config có data.augment.file")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=["text-sarcasm", "image-sarcasm"])
    parser.add_argument("--candidates", type=int, default=2, help="số bản sinh cho mỗi mẫu gốc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="chỉ lấy N mẫu gốc đầu (chạy thử)")
    parser.add_argument("--import-review", default=None, help="file .csv đã điền cột review")
    parser.add_argument("--force", action="store_true", help="sinh lại dù manifest đã có (mất kết quả duyệt cũ)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.override)
    manifest = augment_manifest_path(cfg)
    if manifest is None:
        raise SystemExit(f"{args.config} chưa đặt data.augment.file")
    require_review = bool(cfg.data.augment.get("require_review"))
    stage_preprocessed_cache(cfg)  # copy cache text có sẵn trước, để bước tiền xử lý bên dưới chỉ thêm caption mới

    if args.import_review:
        rows = import_review(read_manifest(manifest), args.import_review)
    elif manifest.exists() and not args.force:
        log.info("dùng manifest có sẵn %s (--force để sinh lại)", manifest)
        rows = read_manifest(manifest)
    else:
        rows = generate(cfg, set(args.labels), args.candidates, args.seed, args.batch_size, args.limit)
    write_manifest(manifest, rows)
    write_review_csv(review_csv_path(manifest), rows)
    TextPreprocessor.from_config(cfg.data.text).process_all([r["caption"] for r in rows], cache_dir=text_cache_dir(cfg))
    log.info("manifest %s | file duyệt %s", manifest, review_csv_path(manifest))
    report(rows, require_review)


if __name__ == "__main__":
    main()

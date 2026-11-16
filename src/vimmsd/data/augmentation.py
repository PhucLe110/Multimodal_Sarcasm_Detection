import difflib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from torchvision import transforms


def build_image_augment(enabled: bool):
    """Augmentation nhẹ trên PIL image, chạy sau bước pad vuông và trước image processor của encoder.
    Chỉ crop 90-100% diện tích để không cắt mất chữ ở mép ảnh (meme thường đặt chữ ở mép trên và dưới),
    và chỉ chỉnh độ sáng, độ tương phản. Không lật ngang (chữ bị ngược), không xoay hay blur mạnh."""
    if not enabled:
        return None
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])


class BackTranslator:
    """Back-translation vi -> en -> vi để sinh thêm caption cho lớp hiếm (scripts/augment_captions.py).
    Chạy offline một lần rồi lưu kết quả, không chạy trong vòng lặp train."""

    def __init__(
        self,
        vi2en="vinai/vinai-translate-vi2en-v2",
        en2vi="vinai/vinai-translate-en2vi-v2",
        device="cuda",
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device
        self.vi2en, self.en2vi = vi2en, en2vi
        self.tok_vi2en = AutoTokenizer.from_pretrained(vi2en, src_lang="vi_VN")
        self.model_vi2en = AutoModelForSeq2SeqLM.from_pretrained(vi2en).to(device).eval()
        self.tok_en2vi = AutoTokenizer.from_pretrained(en2vi, src_lang="en_XX")
        self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(en2vi).to(device).eval()

    def _translate(self, texts, tok, model, tgt_lang, **gen_kwargs):
        import torch

        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                decoder_start_token_id=tok.convert_tokens_to_ids(tgt_lang),
                max_length=256,
                **gen_kwargs,
            )
        return tok.batch_decode(out, skip_special_tokens=True)

    def __call__(self, texts, batch_size=16, num_return_sequences=1):
        """Mỗi caption trả về num_return_sequences bản dịch ngược (list các list, cùng thứ tự với texts)."""
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            en = self._translate(chunk, self.tok_vi2en, self.model_vi2en, "en_XX", num_beams=5)
            vi = self._translate(
                en, self.tok_en2vi, self.model_en2vi, "vi_VN",
                do_sample=True, top_p=0.9, num_return_sequences=num_return_sequences,
            )
            results.extend(vi[k : k + num_return_sequences] for k in range(0, len(vi), num_return_sequences))
        return results


ACCEPT, REJECT = "accept", "reject"
LABEL_WORDS = ("sarcasm", "sarcastic", "mỉa mai", "châm biếm", "giễu cợt")
MAX_SIMILARITY = 0.9
LENGTH_RATIO = (0.6, 1.6)
PARENT_WORDS = (3, 150)  # caption quá ngắn không có gì để diễn đạt lại; quá dài thì bị cắt khi dịch (256 token)


def normalize_caption(text):
    text = unicodedata.normalize("NFC", text or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _numbers(text):
    return sorted(re.findall(r"\d+", text or ""))


def _degenerate(text, run=4):
    words = normalize_caption(text).split()
    return any(len(set(words[i : i + run])) == 1 for i in range(len(words) - run + 1))


def restore_emojis(parent, candidate):
    """Bản dịch thường mất emoji của caption gốc: nối lại các emoji bị mất vào cuối, theo thứ tự xuất hiện."""
    from vimmsd.data.preprocessing import extract_emojis

    kept = set(extract_emojis(candidate))
    lost = [e for e in dict.fromkeys(extract_emojis(parent)) if e not in kept]
    return (f"{candidate.rstrip()} {''.join(lost)}" if lost else candidate), lost


def auto_review(parent, candidate, known_captions):
    """Lọc tự động một bản augmentation, trả về (verdict, lý do, độ giống với caption gốc).
    Chỉ bắt lỗi kiểm được bằng luật; giữ nghĩa, ý mỉa mai và nhãn modality cần người duyệt (cột review)."""
    norm, norm_parent = normalize_caption(candidate), normalize_caption(parent)
    # autojunk=False: mặc định difflib bỏ qua ký tự phổ biến khi chuỗi dài > 200, làm độ giống của caption dài gần 0
    similarity = difflib.SequenceMatcher(None, norm_parent, norm, autojunk=False).ratio()
    reasons = []
    n_parent = len(norm_parent.split())
    if not PARENT_WORDS[0] <= n_parent <= PARENT_WORDS[1]:
        reasons.append(f"caption gốc {'quá ngắn' if n_parent < PARENT_WORDS[0] else 'quá dài'} ({n_parent} từ)")
    if not norm:
        reasons.append("rỗng")
    else:
        ratio = len(norm.split()) / max(n_parent, 1)
        if similarity > MAX_SIMILARITY:
            reasons.append("gần như trùng caption gốc")
        if not LENGTH_RATIO[0] <= ratio <= LENGTH_RATIO[1]:
            reasons.append(f"độ dài lệch ({ratio:.2f} lần)")
        if _numbers(candidate) != _numbers(parent):
            reasons.append("đổi con số")
        if any(w in norm and w not in norm_parent for w in LABEL_WORDS):
            reasons.append("thêm từ chỉ nhãn")
        if _degenerate(candidate):
            reasons.append("lặp từ")
        if norm in known_captions:
            reasons.append("trùng caption khác trong train")
    return (REJECT if reasons else ACCEPT), reasons, round(similarity, 4)


def final_verdict(row, require_review=False):
    review = (row.get("review") or "").strip().lower()
    if review in (ACCEPT, REJECT):
        return review
    return None if require_review else row["auto_verdict"]


def accepted_variants(rows, require_review=False, max_per_parent=1):
    """{parent_id: [row]}: mỗi mẫu gốc giữ tối đa max_per_parent bản được accept, ưu tiên bản giống caption gốc nhất
    (đã qua ngưỡng gần trùng): bản khác nhiều nhất thường là bản dịch sai nghĩa, dễ làm lệch nhãn."""
    by_parent = defaultdict(list)
    for row in rows:
        if final_verdict(row, require_review) == ACCEPT:
            by_parent[row["parent_id"]].append(row)
    return {pid: sorted(items, key=lambda r: (-r["similarity"], r["candidate"]))[:max_per_parent]
            for pid, items in by_parent.items()}


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def augment_records(train_records, rows, require_review=False):
    """Bản augmentation của các mẫu train có trong manifest: cùng ảnh, OCR, mô tả và nhãn với mẫu gốc, caption mới,
    id `<gốc>#aug<k>` và parent_id để biết đây là bản sinh thêm. Mẫu gốc không còn trong train (vd fold bị giữ lại)
    thì bỏ qua, nên không rò rỉ sang phần đánh giá."""
    by_id = {r["id"]: r for r in train_records}
    out = []
    for pid, variants in accepted_variants(rows, require_review).items():
        parent = by_id.get(pid)
        if parent is None:
            continue
        for row in variants:
            if row["parent_caption"] != parent["caption"]:
                raise ValueError(f"manifest lệch dữ liệu: caption gốc của {pid} khác với file train")
            out.append({**parent, "id": f"{pid}#aug{row['candidate']}", "caption": row["caption"], "parent_id": pid})
    return out

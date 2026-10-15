"""Đóng băng encoder, chỉ mở lại N block transformer cuối để fine-tune nhẹ (Phase 4)."""


def last_blocks(model, n):
    """N block cuối của PhoBERT/RoBERTa, CLIP vision hoặc ViT, kèm LayerNorm sau block cuối nếu có.

    CLIP: `post_layernorm` tạo ra `pooler_output`; ViT: `layernorm` tạo ra `last_hidden_state`.
    RoBERTa không có LayerNorm chung sau encoder (mỗi block tự có)."""
    # CLIPVisionModel: transformers cũ bọc trong `vision_model`, bản mới đặt embeddings/encoder/post_layernorm ở ngoài
    root = getattr(model, "vision_model", model)
    encoder = getattr(root, "encoder", None)
    layers = getattr(encoder, "layers", None)  # CLIP
    if layers is None:
        layers = getattr(encoder, "layer", None)  # RobertaModel (PhoBERT), ViTModel
    if layers is None:
        raise ValueError(f"không tìm được các block transformer của {type(model).__name__}")
    final_norm = next((getattr(root, name) for name in ("post_layernorm", "layernorm") if hasattr(root, name)), None)
    if not 0 < n <= len(layers):
        raise ValueError(f"unfreeze_last_n phải trong [1, {len(layers)}], nhận được {n}")
    blocks = list(layers[-n:])
    if final_norm is not None:
        blocks.append(final_norm)
    return blocks


def freeze_encoder(model, freeze, unfreeze_last_n=0):
    """Đóng băng `model` nếu `freeze`, rồi mở lại `unfreeze_last_n` block cuối.

    Trả về các block được mở lại, để `train()` của encoder giữ chúng ở train mode (dropout bật)
    trong khi phần còn đóng băng luôn ở eval mode."""
    if not freeze:
        if unfreeze_last_n:
            raise ValueError("unfreeze_last_n chỉ dùng với freeze: true (freeze: false đã fine-tune toàn bộ encoder)")
        return []
    model.requires_grad_(False)
    blocks = last_blocks(model, unfreeze_last_n) if unfreeze_last_n else []
    for block in blocks:
        block.requires_grad_(True)
    return blocks

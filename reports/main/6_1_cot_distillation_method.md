# 6.1. Phương pháp: Chain-of-Thought distillation + hierarchical prompting

- **Phase:** 6, Chain-of-Thought + Knowledge Distillation + Hierarchical Prompting
- **Thành viên:** Trúc
- **Nguồn:** `reports/plan/experiment_design.md`, `reports/plan/experiment_plan.md`, `reports/results/experiment_report.md`, `reports/results/progress_report.md`
- **Code:** `src/vimmsd/data/rationale.py`, `src/vimmsd/models/heads.py`, `src/vimmsd/training/losses.py`, `scripts/generate_rationales.py`, `eval_teacher.py`, `build_rationale_dataset.py`, `embed_rationales.py`, `scripts/knn_eval.py`, `scripts/stack_predictions.py`
- **Configs:** `distill_base.yaml`, `distill_flat_{a,b,c,d}.yaml`, `distill_hier_{a,b,c,d}.yaml`, `hier_cascade.yaml`, `configs/experiments/cot_distill*.yaml`

Report này mô tả phương pháp. Kết quả và quyết định dừng ở `6_2_baseline_and_teacher.md`.

---

## 1. Động cơ

Nhãn `multi-sarcasm` không có nghĩa là "cả ảnh và text đều mỉa mai", mà là "châm biếm xuất hiện khi đọc ảnh cùng text". Một mẫu `multi` thuộc một trong 4 trường hợp:

| # | Text đứng riêng | Ảnh đứng riêng | Khi ghép |
|---|---|---|---|
| 1 | mỉa mai | mỉa mai | mỉa mai |
| 2 | mỉa mai | không | mỉa mai |
| 3 | không | mỉa mai | mỉa mai |
| 4 | không | không | mỉa mai (chỉ xuất hiện khi ghép) |

Nhãn gốc không cho biết mẫu thuộc trường hợp nào. Trường hợp 2–4 đòi hỏi suy luận trên cả hai modality, và đây là chỗ các mô hình fusion (concat, cross-attention) dễ học thành shortcut từ text.

**Ý tưởng:** cho một VLM lớn (teacher) viết lập luận theo khung 4 bước (text riêng, ảnh riêng, khi ghép, kết luận), rồi chưng cất lập luận vào một mô hình nhỏ (student, khoảng 300M tham số). Chỉ student chạy lúc suy luận, nên không cần teacher hay lập luận ở test.

Cơ sở: STaR (Zelikman et al., 2022) cho cách lọc lập luận theo đáp án đúng và bước rationalization; Distilling step-by-step (Hsieh et al., 2023) cho cách dùng lập luận làm tác vụ phụ; MiDRE (Jana et al., 2025) cho cách dùng lập luận của LVLM trong MSD.

## 2. Pipeline

```
train (10,8k, có nhãn)
   │
   ▼
[A] Teacher sinh k lập luận/mẫu (khung 4 bước, không cho biết nhãn)
   │
   ▼
[B] Phân nhóm theo số lần đúng ──► sai cả k lần ──► [C] Rationalization (đưa nhãn đúng)
   │                                                        │
   ▼                                                        ▼
[D] Người kiểm tra theo rubric (toàn bộ nhóm 2, 3; mẫu ngẫu nhiên nhóm 1)
   │
   ▼
rationale_dataset.json ──► [E] Train student (M1 / M2 / M3)
                                   │
                                   ▼
                     [F] Hiệu chỉnh prior, đánh giá private test
```

Không chạy teacher trên val/test để chọn hay lọc mẫu.

### 2.1. Teacher và prompt (hierarchical prompting)

Đầu vào teacher: ảnh, caption, chữ OCR (và mô tả ảnh ở một số phiên bản prompt). Prompt yêu cầu trả lời theo thứ tự phân cấp:

1. **Text đứng riêng:** chỉ đọc caption thì có mỉa mai không? Dẫn chứng.
2. **Ảnh đứng riêng:** chỉ nhìn ảnh (kể cả chữ trong ảnh) thì có mỉa mai không? Mô tả khách quan, không suy diễn.
3. **Khi ghép:** caption và ảnh có mâu thuẫn, phóng đại, hay tạo ra ý mỉa mai mới không?
4. **Kết luận:** một trong 4 nhãn, cùng hai nhãn phụ `text_sarcastic`, `image_sarcastic`.

Đầu ra JSON:

```json
{"text_alone": {"sarcastic": false, "reason": "..."},
 "image_alone": {"sarcastic": true, "reason": "..."},
 "combined": {"reason": "..."},
 "label": "multi-sarcasm"}
```

**Self-consistency:** k = 3–5 lần/mẫu, temperature khoảng 0,7. Tỉ lệ `n_correct / k` vừa để phân nhóm vừa đo độ khó của mẫu.

### 2.2. Lọc và kiểm tra

| Nhóm | Điều kiện | Xử lý | Kiểm tra |
|---|---|---|---|
| 1 | đúng ≥ ⌈k/2⌉ lần | chọn 1 lập luận đúng | 10–20% ngẫu nhiên |
| 2 | đúng 1 đến < ⌈k/2⌉ lần | chọn lập luận đúng tốt nhất | toàn bộ |
| 3 | sai cả k lần | rationalization: đưa nhãn đúng, yêu cầu giải thích | toàn bộ, `rationalized=true` |

Không bỏ nhóm 3: teacher thường đúng ở trường hợp dễ (1) và sai ở trường hợp khó (2–4).

**Rubric** (giữ khi đạt cả 4): (1) đánh giá text đứng riêng đúng; (2) mô tả ảnh đúng, không bịa; (3) lập luận khi ghép hợp lý, có dẫn chứng; (4) kết luận khớp lập luận và hai nhãn phụ. Hai người chấm khoảng 50 mẫu trước, yêu cầu Cohen's κ ≥ 0,6.

## 3. Ba biến thể student

Dùng chung encoder và fusion: `fusion_multistream_clip_frozen` của Phase 4 (PhoBERT-base cho caption, OCR, mô tả thành 3 luồng riêng; CLIP ViT-B/32 cho ảnh; cả hai đóng băng). Chỉ khác phần head.

**Loss căn chỉnh lập luận** (chỉ trên mẫu có lập luận đạt rubric):

```
L_rat = 1 − cos(proj(h_fusion), e_rationale)
```

`e_rationale`: embedding lập luận, mã hoá bằng PhoBERT đóng băng, tính trước.

**Hai loại nhãn phụ:**

| Nhãn phụ | Nghĩa | Nguồn |
|---|---|---|
| `involves_text`, `involves_image` | châm biếm *có liên quan* tới text/ảnh | suy từ nhãn gốc: `label ∈ {multi, text}`, `label ∈ {multi, image}` |
| `text_sarcastic`, `image_sarcastic` | text/ảnh *tự nó* đã mỉa mai | teacher (bước 1, 2) |

Với nhãn suy từ nhãn gốc, mọi mẫu `multi` là (có, có). Chỉ teacher mới phân biệt được 4 trường hợp.

### 3.1. M1: CoT distillation, đầu ra phẳng

```
h_fusion ─► head 4 lớp (softmax)
         ├─► head T: text_sarcastic  (sigmoid, loss phụ, bỏ lúc suy luận)
         └─► head I: image_sarcastic (sigmoid, loss phụ, bỏ lúc suy luận)

L_M1 = CE_4lớp + λ_axis · [BCE(T) + BCE(I)] + λ_rat · L_rat
```

### 3.2. M2: CoT distillation + phân cấp mềm

```
P(not)   = 1 − P(S)
P(text)  = P(S) · P(text  | S)
P(image) = P(S) · P(image | S)
P(multi) = P(S) · P(multi | S)

L_M2 = NLL_4lớp + λ_S · BCE(S) + λ_axis · [BCE(T) + BCE(I)] + λ_rat · L_rat
```

- Tầng S: "có châm biếm không" (44/56 ở train, khá cân bằng).
- Tầng C: "loại nào", đầu vào `[h_fusion; p_T; p_I]` (detach `p_T`, `p_I`).
- Một mô hình, train end-to-end, nối các tầng bằng phép nhân xác suất. **Khác Approach 3 của Vu et al. (2025)**: họ dùng 2 mô hình nối tiếp với ngưỡng cứng, đạt F1 0,33, kém nhất trong 3 hướng của bài, do lỗi lan truyền.
- Hiệu chỉnh label shift chỉ cần ở tầng S: dịch logit S một lượng `logit(prior_val) − logit(prior_train)`.

### 3.3. M3: CoT distillation + subclass

Tách `multi` thành 4 subclass theo (T, I) của teacher lúc train (7 lớp), gộp bằng **cộng xác suất** lúc suy luận:

| Nhãn train | Điều kiện | Gộp về |
|---|---|---|
| `multi_TI`, `multi_T`, `multi_I`, `multi_none` | (1,1), (1,0), (0,1), (0,0) | `multi` |

`L_M3 = NLL_4lớp(P đã gộp) + λ_sub · CE_7lớp(mẫu multi có nhãn teacher) + λ_rat · L_rat`. Gần với hidden stratification (Sohoni et al., 2020). Chỉ có nghĩa từ cột (c) trở đi.

## 4. Ma trận thí nghiệm

Mỗi cột thêm một thành phần:

| Đầu ra | (a) CE | (b) + nhãn phụ từ nhãn gốc | (c) + nhãn phụ từ teacher | (d) + căn chỉnh lập luận |
|---|---|---|---|---|
| Phẳng 4 lớp | `distill_flat_a` | `distill_flat_b` | `distill_flat_c` | **M1** (`distill_flat_d`) |
| Phân cấp mềm | `distill_hier_a` | `distill_hier_b` | `distill_hier_c` | **M2** (`distill_hier_d`) |
| Subclass | – | – | ✓ | **M3** |
| Phân cấp nối tiếp (tái lập Vu et al.) | `hier_cascade` | – | – | – |

- Cột (a), (b) không cần teacher: chạy được ngay (giai đoạn 0).
- Ablation `*_d_nogroup3`: bỏ lập luận nhóm 3 để đo tác dụng của rationalization.
- Mọi ô: 3 seed, có và không có hiệu chỉnh prior (`log p_val − log p_train` cộng vào logit).
- Chọn checkpoint theo **val macro-F1 bỏ `text-sarcasm`**; `train.eval_test: false` để giữ private test cho bước cuối.

**Hướng phụ:**

| Mã | Hướng | Vai trò |
|---|---|---|
| E1 | kNN retrieval lúc suy luận: trộn nhãn láng giềng gần nhất trong train với dự đoán của mô hình | hậu xử lý |
| E2 | Few-shot cho teacher (ví dụ train gần nhất theo PhoBERT/CLIP) | biến thể prompt |
| E3 | Pretrain trên dữ liệu sarcasm ngoài (MMSD2.0, silver data) | mở rộng |
| B2 | 3 mô hình đơn modality + stacking out-of-fold | baseline gộp muộn |
| B3 | Lập luận làm đầu vào lúc suy luận | nhánh so sánh |
| D1 | Chẩn đoán shortcut | chẩn đoán |

## 5. Cổng quyết định

| Cổng | Điều kiện |
|---|---|
| G0 | Phân cấp mềm không kém phẳng quá 1 điểm macro-F1 trên val |
| G1 | Teacher vượt baseline ở F1 `image-sarcasm` hoặc `multi-sarcasm` trên val, parse lỗi < 5% |
| G2 | κ ≥ 0,6 mọi tiêu chí; mỗi lớp khoảng 30 mẫu đạt rubric; nhóm 1 đạt ≥ 80% |
| G3 | M1/M2/M3 hơn ô (b) tương ứng quá 1 độ lệch chuẩn |
| G4 | Báo cáo đủ |
| G5 | E3 hơn bản không pretrain quá 1 độ lệch chuẩn |

Các cổng được đặt để dừng sớm khi giả thuyết không đứng vững. Kết quả: dừng sau G1 (xem 6.2).

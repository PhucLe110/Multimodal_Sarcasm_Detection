# Chưng cất reasoning từ LLM đa phương thức (CoT distillation)

Trạng thái: **đã có code cho mọi phần** (mục 8), chưa chạy trên dữ liệu thật. Định nghĩa nhãn trong prompt (`data/rationale.py`, `LABEL_DEFINITIONS`) cần đối chiếu với hướng dẫn gán nhãn UIT DSC 2024 trước khi chạy cả tập.

Liên quan: `reports/literature_review/README.md` (mục 1.1, 3.4, 3.5, 6). Kế hoạch chạy: `docs/cot_distillation_plan.md`.

---

## 1. Động cơ

Nhãn `multi-sarcasm` của ViMMSD không có nghĩa là "cả ảnh và text đều mỉa mai", mà là "châm biếm xuất hiện khi đọc ảnh cùng text". Một mẫu `multi` có thể thuộc 1 trong 4 trường hợp:

| # | Text đứng riêng | Ảnh đứng riêng | Khi ghép ảnh với text |
|---|---|---|---|
| 1 | mỉa mai | mỉa mai | mỉa mai |
| 2 | mỉa mai | không | mỉa mai |
| 3 | không | mỉa mai | mỉa mai |
| 4 | không | không | mỉa mai (chỉ xuất hiện khi ghép) |

Nhãn gốc không cho biết mẫu thuộc trường hợp nào. Trường hợp 2–4 đòi hỏi suy luận trên cả hai modality, và đây là phần các model fusion hiện tại (concat, cross-attention) dễ học nhầm thành shortcut từ text. Ý tưởng của hướng này: để một LLM đa phương thức mạnh (teacher) viết ra reasoning theo đúng khung 4 trường hợp, rồi chưng cất reasoning đó vào một model nhỏ (student, khoảng 300M tham số). Student là model chạy trên toàn bộ tập test.

Tài liệu này mô tả **2 phương pháp thí nghiệm**. Cả hai dùng chung dữ liệu reasoning (mục 3–4) và cùng cách đánh giá (mục 6), chỉ khác nhau ở đầu ra của student (mục 5):

- **M1, CoT distillation**: student là model 4 lớp phẳng, reasoning làm loss phụ.
- **M2, CoT distillation + hierarchical**: student có đầu ra phân cấp mềm. Tầng S hỏi "có châm biếm không", tầng C hỏi "loại châm biếm nào", và hai head T/I học phán đoán "text/ảnh đứng riêng" của teacher.

**M2 khác pipeline phân cấp của Vu et al. ở đâu.** Approach 3 của Vu et al. (2025) dùng 2 model nối tiếp với ngưỡng cứng: `multi` hay không, rồi 3 lớp còn lại. Cách này chỉ đạt F1 0,33 trên private test, kém nhất trong 3 hướng của bài, do lỗi của model đầu lan sang model sau. M2 vẫn là **một model, train end-to-end**. Các tầng được nối với nhau bằng phép nhân xác suất, không dùng ngưỡng cứng, nên gradient đi qua mọi tầng và không có lỗi lan truyền. Reasoning chỉ là tín hiệu giám sát lúc train ở cả hai phương pháp, không phải một tầng chạy lúc suy luận.

Tài liệu nền: STaR (Zelikman et al., 2022) cho cách lọc reasoning theo đáp án đúng và bước rationalization. Distilling step-by-step (Hsieh et al., 2023) cho cách dùng reasoning làm tác vụ phụ. MiDRE (Jana et al., 2025) cho cách dùng reasoning của LVLM trong MSD.

---

## 2. Tổng quan pipeline

```
train (10,8k, có nhãn)
   │
   ▼
[A] Teacher sinh k reasoning/mẫu (khung 4 bước, không cho biết nhãn)
   │
   ▼
[B] Phân nhóm theo số lần đúng ──► sai cả k lần ──► [C] Rationalization (đưa nhãn đúng)
   │                                                        │
   ▼                                                        ▼
[D] Người kiểm tra theo rubric (toàn bộ nhóm 2, 3, mẫu ngẫu nhiên nhóm 1)
   │
   ▼
rationale_dataset.json (mẫu đạt rubric) ──► [E] Train student
                                               ├─ M1: 4 lớp phẳng + loss phụ
                                               └─ M2: phân cấp mềm S → C, head T/I + loss phụ
                                                   │
                                                   ▼
                                     [F] Hiệu chỉnh prior, đánh giá trên private test
```

**Không chạy teacher trên val/test** để chọn hay lọc mẫu. Chỉ được chạy teacher trên val/test ở nhánh so sánh mục 6.3, nơi reasoning là đầu vào lúc suy luận và không dùng nhãn.

---

## 3. Sinh reasoning bằng teacher

### 3.1. Teacher

Nên thử trước trên 280 mẫu cân bằng (70 mỗi lớp) **lấy từ train**. Không lấy từ val được vì val chỉ có 1 mẫu `text-sarcasm`, và teacher chỉ được đánh giá chứ không học trên các mẫu này nên không gây rò rỉ. Sau đó chạy zero-shot trên toàn bộ val để so với baseline (chi tiết ở giai đoạn 1 của kế hoạch). Chọn teacher theo macro-F1 và F1 từng lớp:

- Qwen2.5-VL-7B-Instruct: miễn phí, chạy trên GPU server (bf16 cần khoảng 16 GB VRAM trở lên), hỗ trợ tiếng Việt khá tốt.
- Qwen2.5-VL-32B / 72B hoặc GPT-4o / Gemini qua API: mạnh hơn nhưng có chi phí. Ước tính chi phí trên 280 mẫu trước khi chạy cả tập.

Teacher phải vượt baseline hiện tại ít nhất ở hai lớp `image-sarcasm` và `multi-sarcasm`. Nếu không thì reasoning của nó không mang thêm thông tin, và nên sửa prompt hoặc đổi teacher trước khi đi tiếp.

### 3.2. Prompt (khung 4 bước)

Đầu vào: ảnh, caption, chữ OCR từ ảnh (lấy từ `image_preprocessed.json`). Prompt yêu cầu trả lời theo đúng thứ tự:

1. **Text đứng riêng**: chỉ đọc caption thì có mỉa mai không? Dẫn chứng từ caption.
2. **Ảnh đứng riêng**: chỉ nhìn ảnh, kể cả chữ trong ảnh, thì có mỉa mai không? Mô tả khách quan những gì thấy trong ảnh, không suy diễn.
3. **Khi ghép**: caption và ảnh có mâu thuẫn, phóng đại, hoặc tạo ra ý mỉa mai mới không?
4. **Kết luận**: một trong 4 nhãn, cùng hai nhãn phụ `text_sarcastic` và `image_sarcastic` (đúng/sai).

Định nghĩa 4 nhãn trong prompt phải chép đúng từ hướng dẫn gán nhãn của UIT DSC 2024. Cần đối chiếu lại định nghĩa trước khi chốt prompt.

Đầu ra yêu cầu ở dạng JSON để parse được:

```json
{
  "text_alone": {"sarcastic": false, "reason": "..."},
  "image_alone": {"sarcastic": true, "reason": "..."},
  "combined": {"reason": "..."},
  "label": "multi-sarcasm"
}
```

### 3.3. Self-consistency

Mỗi mẫu sinh k = 3–5 lần, với temperature khoảng 0,7. Ghi lại số lần teacher đoán đúng nhãn (`n_correct / k`). Tỉ lệ này vừa dùng để phân nhóm, vừa là thước đo độ khó của mẫu.

Chi phí: k = 3 trên 10,8k mẫu là khoảng 32k lần sinh. Chạy trên GPU server, mỗi GPU một shard, lưu cache để chạy tiếp được giống `scripts/extract_image_text.py` (`CODEBASE.md` mục 7.3).

---

## 4. Lọc và kiểm tra

### 4.1. Phân nhóm

| Nhóm | Điều kiện | Xử lý | Người kiểm tra |
|---|---|---|---|
| 1 | đúng ≥ ⌈k/2⌉ lần | chọn 1 reasoning đúng | lấy ngẫu nhiên 10–20% |
| 2 | đúng 1 đến < ⌈k/2⌉ lần | chọn reasoning đúng tốt nhất | toàn bộ |
| 3 | sai cả k lần | chạy **rationalization**: đưa nhãn đúng, yêu cầu teacher giải thích theo khung 4 bước | toàn bộ, đánh dấu `rationalized=true` |

**Không chỉ giữ mẫu teacher đoán đúng.** Teacher thường đúng ở trường hợp 1 (dễ) và sai ở trường hợp 2–4 (khó). Bỏ nhóm 3 tức là bỏ đúng phần reasoning cần học nhất.

### 4.2. Rubric cho người kiểm tra

Mỗi tiêu chí chỉ chấm đạt hoặc không đạt. Một reasoning được giữ khi đạt cả 4 tiêu chí:

1. Đánh giá text đứng riêng đúng.
2. Mô tả ảnh đúng với ảnh, không bịa chi tiết (hallucination).
3. Lập luận khi ghép ảnh với text hợp lý, có dẫn chứng cụ thể từ ảnh hoặc caption.
4. Kết luận khớp với lập luận, và hai nhãn phụ khớp với bước 1, 2.

Trước khi chấm cả tập, cho 2 người cùng chấm khoảng 50 mẫu và tính độ đồng thuận (Cohen's κ). κ thấp nghĩa là rubric còn mơ hồ, cần sửa rubric rồi mới chấm tiếp. Người kiểm tra có thể sửa nhẹ nội dung reasoning. Mẫu đã sửa đánh dấu `edited=true`.

### 4.3. Cân bằng lớp

Tập train chỉ có 77 mẫu `text-sarcasm`. Nếu cân bằng mọi lớp về bằng lớp này thì sau khi lọc chỉ còn khoảng 100 mẫu. Vì vậy:

- **Giữ toàn bộ mẫu đạt rubric** của mọi lớp. Mất cân bằng được xử lý bằng trọng số trong loss phụ hoặc bằng sampler.
- Mẫu không đạt rubric vẫn dùng cho loss phân loại chính, chỉ không có reasoning.

### 4.4. Định dạng lưu

`rationale_dataset.json`, đặt trong `data/processed/rationales/` trên server (`CODEBASE.md` mục 7.2), và upload thành Kaggle dataset để nhóm dùng chung:

```json
{
  "meta": {"teacher": "Qwen/Qwen2.5-VL-7B-Instruct", "k": 3, "prompt_version": "v1", "rubric_version": "v1"},
  "samples": {
    "<id>": {
      "label": "multi-sarcasm",
      "text_sarcastic": false,
      "image_sarcastic": true,
      "rationale": "...",
      "n_correct": 2,
      "group": 1,
      "rationalized": false,
      "review": "pass",
      "edited": false
    }
  }
}
```

---

## 5. Train student

### 5.1. Phần chung của M1 và M2

**Encoder và fusion.** Dùng lại kiến trúc hiện có: PhoBERT + CLIP + OCR / mô tả ảnh (`multistream_clip` hoặc `cross_attn`), khoảng 135M + 150M tham số. Hai phương pháp dùng cùng encoder, cùng fusion, cùng hyperparameter, chỉ khác phần head. Lúc suy luận **không cần reasoning hay teacher**, nên so sánh trực tiếp được với các baseline.

**Loss căn chỉnh reasoning**, chỉ tính trên các mẫu có reasoning đạt rubric:

```
L_rat = 1 − cos(proj(h_fusion), e_rationale)
```

- `h_fusion`: vector đầu ra của fusion, trước các head.
- `e_rationale`: embedding của reasoning, mã hoá bằng PhoBERT đóng băng, tính trước và lưu cache.

**Hai loại nhãn phụ**, mang nghĩa khác nhau:

| Nhãn phụ | Nghĩa | Nguồn | Có cho mẫu |
|---|---|---|---|
| `involves_text`, `involves_image` | châm biếm *có liên quan tới* text/ảnh | suy từ nhãn gốc: `label ∈ {multi, text}` và `label ∈ {multi, image}` | mọi mẫu train |
| `text_sarcastic`, `image_sarcastic` | text/ảnh *tự nó đã* mỉa mai khi đứng riêng | teacher (bước 1, 2 của prompt) | mẫu có reasoning đạt rubric |

Hai loại khác nhau ở mẫu `multi`. Với nhãn suy từ nhãn gốc, mọi mẫu `multi` đều là (có, có). Với nhãn của teacher, mẫu `multi` có thể là (có, có), (có, không), (không, có) hoặc (không, không), tương ứng 4 trường hợp ở mục 1. Thông tin này chỉ có teacher cung cấp được.

**Không suy nhãn 4 lớp từ hai nhãn phụ bằng luật cố định** như Approach 2 của Vu et al. Ví dụ, trường hợp 2 (text tự nó mỉa mai, ảnh thì không) có thể là `text-sarcasm` hoặc `multi`, tuỳ ảnh có góp phần hay không. Trường hợp 4 có cả hai nhãn phụ là "không" nhưng vẫn là `multi`. Nhãn phụ chỉ là tín hiệu giám sát hoặc đặc trưng trung gian, còn quyết định cuối do head học.

**Phương án thay thế** (chưa ưu tiên): một model seq2seq tiếng Việt cỡ base vừa sinh reasoning vừa đoán nhãn. Cách này chỉ nhìn được ảnh qua OCR và mô tả, và tốn công hơn nhiều.

### 5.2. M1: CoT distillation (đầu ra phẳng)

```
h_fusion ──► head 4 lớp (softmax)
         ├─► head T: text_sarcastic  (sigmoid, loss phụ)
         └─► head I: image_sarcastic (sigmoid, loss phụ)
```

```
L_M1 = CE_4lớp(tất cả mẫu)
     + λ_axis · [BCE(T, text_sarcastic) + BCE(I, image_sarcastic)]   (mẫu có nhãn của teacher)
     + λ_rat  · L_rat
```

Head T/I chỉ dùng lúc train để định hướng `h_fusion`, và bị bỏ đi lúc suy luận.

**Hiệu chỉnh prior:** cộng `log(prior_val) − log(prior_train)` vào 4 logit (logit adjustment). Prior lấy từ public test, vì đây là val khi `data.split=official`.

### 5.3. M2: CoT distillation + hierarchical (đầu ra phân cấp mềm)

```
                    h_fusion
          ┌────────────┼─────────────────────┐
   [T] text đứng riêng  [I] ảnh đứng riêng    [S] có châm biếm không
      (sigmoid)            (sigmoid)              (sigmoid)
          └─────┬──────┘                          │
                ▼                                 │
   [C] loại châm biếm | S: text / image / multi   │
      (softmax, đầu vào = [h_fusion; p_T; p_I])   │
                │                                 │
                ▼                                 ▼
   P(not)   = 1 − P(S)
   P(text)  = P(S) · P(text  | S)
   P(image) = P(S) · P(image | S)
   P(multi) = P(S) · P(multi | S)
```

- **Tầng S** học từ nhãn gốc trên toàn bộ 10,8k mẫu (`S = label ≠ not`). Hai lớp ở tầng này khá cân bằng (44/56 trong train).
- **Tầng C** chỉ phân biệt 3 loại châm biếm. Đầu vào gồm `h_fusion` và xác suất `p_T`, `p_I` của hai head T/I. Gradient không đi ngược qua `p_T`, `p_I` (dùng detach) để head T/I chỉ học theo teacher.
- **Tầng T và I** học theo nhãn `text_sarcastic` / `image_sarcastic` của teacher. Khác M1, ở đây T và I là đặc trưng trung gian có mặt cả lúc suy luận.

```
L_M2 = NLL_4lớp(P ở trên, tất cả mẫu)
     + λ_S    · BCE(S, label ≠ not)                                  (tất cả mẫu)
     + λ_axis · [BCE(T, text_sarcastic) + BCE(I, image_sarcastic)]   (mẫu có nhãn của teacher)
     + λ_rat  · L_rat
```

`NLL_4lớp` là cross-entropy trên xác suất 4 lớp đã nhân xác suất ở trên. Gradient đi qua cả S lẫn C, nên đây vẫn là một model train end-to-end. `λ_S` thêm giám sát trực tiếp cho tầng S và có thể bằng 0.

**Hiệu chỉnh prior theo từng tầng.** Label shift giữa train và test chủ yếu nằm ở tầng S:

| | train | public test | private test |
|---|---:|---:|---:|
| P(có châm biếm) | 43,9% | 68,7% | 69,5% |
| P(multi \| có châm biếm) | 89,1% | 97,0% | 95,5% |

Do đó chỉ cần dịch logit của head S một lượng `logit(prior_val) − logit(prior_train)`. Head C dịch thêm theo prior có điều kiện. Cách này đơn giản và ổn định hơn chỉnh cả 4 logit như M1.

**Lớp hiếm.** Ở tầng C, `text-sarcasm` chiếm 1,6% số mẫu có châm biếm (77/4 743), còn ở bài toán 4 lớp là 0,7%. Trọng số lớp hoặc focal loss áp dụng ở tầng C.

---

## 6. Đánh giá

### 6.1. Thiết lập

`data.split=official`. Báo cáo macro-F1, weighted-F1, F1 từng lớp và confusion matrix trên private test. Mốc tham chiếu là F1 0,41 của Vu et al. (2025), và khi so phải ghi rõ đang dùng loại F1 nào. Lớp `text-sarcasm` chỉ có 1 mẫu trong val và 7 mẫu trong test, nên không kết luận gì dựa trên F1 riêng của lớp này.

### 6.2. Ma trận thí nghiệm

Hai phương pháp được so sánh theo cùng một thang. Mỗi cột thêm một thành phần so với cột bên trái:

| Cấu trúc đầu ra | (a) CE | (b) + nhãn phụ từ nhãn gốc | (c) + nhãn phụ từ teacher | (d) + căn chỉnh reasoning (**đầy đủ**) |
|---|---|---|---|---|
| Phẳng 4 lớp | baseline | head T/I học `involves_*` | head T/I học `*_sarcastic` | **M1** |
| Phân cấp mềm | S + C | S + C, T/I học `involves_*` | S + C, T/I học `*_sarcastic` | **M2** |
| Phân cấp nối tiếp (tái lập Vu et al.) | Model 1 + Model 2 | – | – | – |

- Cột (a) và (b) không cần teacher, chạy được ngay.
- Mọi ô chạy có và không có hiệu chỉnh prior, với ít nhất 3 seed (dùng `data.split=official` nên split cố định, seed chỉ ảnh hưởng tới quá trình train).
- Thêm một ablation cho M1 và M2 ở cột (d): bỏ reasoning của nhóm 3, để đo tác dụng của bước rationalization.

### 6.3. Câu hỏi cần trả lời

| So sánh | Câu hỏi |
|---|---|
| Phẳng (a) với phân cấp mềm (a) | Chỉ tách tầng S/C (chưa có teacher) đã giúp chưa? Phần lớn lợi ích dự kiến đến từ việc hiệu chỉnh label shift ở tầng S. |
| Phân cấp mềm (a) với nối tiếp (a) | Phân cấp mềm có tránh được lỗi lan truyền của Vu et al. không? |
| Cột (b) với cột (c) | Nhãn "text/ảnh đứng riêng" của teacher có thêm thông tin so với nhãn suy từ nhãn gốc không? |
| Cột (c) với cột (d) | Căn chỉnh với embedding reasoning có thêm gì ngoài nhãn phụ không? |
| **M1 với M2** | Đầu ra phân cấp có giúp student tận dụng reasoning tốt hơn không? Nên xem F1 từng lớp, nhất là `image-sarcasm`, và confusion giữa `multi` với `text`/`image`. |

Nếu phân cấp mềm (a) đã hơn phẳng (a) gần bằng mức M2 hơn M1, thì lợi ích của M2 chủ yếu đến từ việc tách tầng để xử lý label shift, chưa phải từ reasoning.

### 6.4. Nhánh so sánh: reasoning làm đầu vào

Chạy teacher (không cho nhãn) trên cả public và private test, rồi đưa reasoning vào làm segment thứ 3 của PhoBERT, giống cách MiDRE làm. Phần chênh lệch giữa nhánh này và student ở mục 5 là cái giá của việc nén reasoning vào model nhỏ.

### 6.5. Kết quả phụ cần báo cáo

- Accuracy và F1 từng lớp của chính teacher trên tập train (lấy ở bước 3.3, trước khi lọc).
- Tỉ lệ mẫu mỗi nhóm, tỉ lệ đạt rubric theo lớp và theo nhóm, κ giữa hai người chấm.
- Phân bố 4 trường hợp của `multi` theo nhãn phụ của teacher.

---

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Teacher kém hơn baseline, reasoning không thêm thông tin | Thử trên 280 mẫu ở bước 3.1 rồi mới chạy cả tập |
| Rationalization bịa lý do cho khớp nhãn | Kiểm tra toàn bộ nhóm 3, so ablation có và không có nhóm 3 |
| Mô tả ảnh bị hallucination | Tiêu chí 2 của rubric. Đưa OCR vào prompt để teacher không phải tự đọc chữ |
| Rubric mơ hồ, người chấm không đồng ý với nhau | Đo κ trên 50 mẫu trước khi chấm cả tập |
| Chi phí GPU hoặc API | k = 3, cache để chạy tiếp, chạy thử trước khi chạy cả tập |
| Rò rỉ nhãn val/test | Chỉ lọc trên train. Nhánh 6.4 không dùng nhãn |
| M2: tầng S gánh hết label shift, chọn sai prior thì cả model lệch | Lấy prior từ val, báo cáo thêm kết quả không hiệu chỉnh. Kiểm tra độ nhạy theo prior (±10%) |
| M2: head C có ít dữ liệu cho `text`/`image` | Trọng số lớp hoặc focal ở tầng C. Báo cáo F1 có điều kiện trên các mẫu có châm biếm |

---

## 8. Code

| File | Nội dung |
|---|---|
| `src/vimmsd/models/heads.py` | `HierarchicalHead` (S × C, trả log xác suất 4 lớp), `CascadeHead`, `compute_prior_shift` theo tầng |
| `src/vimmsd/models/classifier.py` | `model.head` (`type`, `axis_heads`, `axis_targets`, `rationale_dim`), `forward_outputs`, `set_prior_shift` |
| `src/vimmsd/training/losses.py` | `SarcasmLoss`: CE/focal (= NLL với hierarchical), λ_S, λ_axis (gold / teacher), λ_rat, loss cascade |
| `src/vimmsd/training/trainer.py` | loss nhiều thành phần (log từng phần), `train.select_metric` / `select_exclude_labels` |
| `src/vimmsd/data/rationale.py` | prompt CoT 4 bước, `TeacherVLM`, parse JSON, phân nhóm, chọn mẫu chấm, κ, gộp `rationale_dataset.json` |
| `src/vimmsd/data/dataset.py` | `attach_rationales` (nhãn phụ teacher + embedding cho mẫu train đạt rubric), collator, `split_label_counts` |
| `scripts/make_id_lists.py`, `generate_rationales.py`, `eval_teacher.py` | giai đoạn 1–2 |
| `scripts/build_rationale_dataset.py` | `group` / `rationalize` / `export` / `kappa` / `merge` (giai đoạn 3–5) |
| `scripts/embed_rationales.py` | mã hoá reasoning bằng PhoBERT đóng băng, trung bình token |
| `scripts/evaluate.py` | `eval.prior_correction`: ghi thêm `<split>_metrics_prior.json` |
| `scripts/train.py` | `train.eval_test: false` để không chấm private test sau mỗi run |
| `scripts/run_experiments.py`, `summarize_results.py` | ma trận config × seed; `--extra-eval test` cho bước chấm cuối; cột `val+prior` |
| `configs/distill_*.yaml`, `configs/hier_cascade.yaml`, `configs/experiments/cot_distill*.yaml` | ma trận mục 6.2 |
| `tests/test_cot_distill.py` | head, loss, prior, Trainer thật với head mới, parse / phân nhóm / κ / gộp, dataset, config, plan |

Khác thiết kế ban đầu:
- **Cascade** (`hier_cascade.yaml`) chạy trong một model: head M và head R dùng chung backbone, head R chỉ học trên mẫu không phải multi, suy luận định tuyến cứng. Bài gốc dùng 2 model riêng và back-translation.
- **Embedding reasoning** lấy trung bình token của PhoBERT (không phải `<s>`), văn bản không kèm nhãn kết luận.
- **Nhãn phụ teacher thiếu**: mẫu not / text / image dùng nhãn suy từ nhãn gốc (hai nghĩa trùng nhau), mẫu multi thì bỏ.
- **Nhánh 6.4** (reasoning làm đầu vào) chưa có code: `generate_rationales.py` sinh được reasoning cho public/private test, nhưng chưa có config đưa reasoning thành segment thứ 3.

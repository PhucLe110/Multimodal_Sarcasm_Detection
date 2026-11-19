# 11.1. So sánh với LLM đa modal

- **Phase:** 11, So sánh với LLM đa modal
- **Thành viên:** Trúc
- **Code:** `src/vimmsd/analysis/llm_compare.py` (Vintern), `src/vimmsd/data/rationale.py` (`TeacherVLM`, prompt), `scripts/eval_teacher.py`, `scripts/make_id_lists.py`, `scripts/generate_rationales_gemini.py`
- **Notebook:** `notebooks/03_baseline_image.ipynb` (cell 03-10, Vintern), `notebooks/07_ocr_llm_comparison.ipynb` (chưa chạy)
- **Nguồn số liệu:** `reports/word/Fine-tune CLIP.docx` (Vintern), `reports/results/experiment_report.md` §6–7 (Gemini, Qwen), `results.md`, `empirical_analysis.md`

Số liệu LLM trong report này đến từ hai lượt chạy có mục đích khác nhau: Vintern được đánh giá như một baseline zero-shot (Phase 3), còn Gemini và Qwen được thử làm teacher cho CoT distillation (Phase 6). Không có lượt nào được thiết kế riêng cho phase này, nên các tập đánh giá không đồng nhất (mục 2).

---

## 1. Mô hình và prompt

| Mô hình | Kích thước | Chạy ở | Prompt |
|---|---|---|---|
| Vintern-1B-v3.5 (`5CD-AI/Vintern-1B-v3_5`) | khoảng 1B | Kaggle T4 | zero-shot: phân loại 4 nhãn kèm mô tả ngắn từng nhãn; có/không OCR + mô tả |
| Qwen2.5-VL-7B-Instruct | 7B | GPU server, 4-bit | v1–v5: khung 4 bước (text riêng, ảnh riêng, khi ghép, kết luận), JSON |
| Qwen3-VL-8B-Instruct | 8B | GPU server, 4-bit | v7, v9 |
| Gemini 3.1 Pro (preview) | – | API | v5 |
| Gemini 3.5 Flash Lite | – | API | v6, v7, v8 |

**Zero-shot và few-shot.** Prompt v1–v4 là zero-shot. Từ v5, prompt có 4 ví dụ cố định lấy từ train (few-shot cố định); v9 có 7 ví dụ kèm mô tả ảnh, định nghĩa chính thức và quy tắc suy ra từ dữ liệu. Biến thể few-shot chọn ví dụ theo độ giống (E2, Phase 6.1) **chưa chạy**.

Các phiên bản prompt đáng chú ý:

| Bản | Thay đổi |
|---|---|
| v6 | Flash Lite; có bản bỏ mô tả Vintern |
| v7 | viết sau khi xem v6 trên val; bỏ mô tả ảnh; có luật "khi phân vân thì chọn `multi`" |
| v8 | bỏ luật "khi phân vân chọn `multi`" |
| v9 | định nghĩa chính thức, quy tắc suy từ dữ liệu, 7 ví dụ |

## 2. Tập đánh giá

| Tập | Nguồn | Mẫu | Phân bố | Mô hình đã chạy |
|---|---|---:|---|---|
| `pilot_dev` | train | 160 | cân bằng 40/lớp | Qwen2.5, Gemini Pro, Flash Lite |
| `pilot_eval` | train | 120 | cân bằng | chưa dùng |
| `natural_dev` | train | 500 | theo phân bố thật | Flash Lite, Qwen3 |
| val | public test | 1.413 | thật | Flash Lite |
| **private test** | private test | 1.504 | thật | **chỉ Vintern** |

`pilot_dev`, `pilot_eval`, `natural_dev` tạo bằng `scripts/make_id_lists.py`, lấy từ train vì val chỉ có 1 mẫu `text-sarcasm`. Teacher chỉ được đánh giá, không học trên các mẫu này.

## 3. Kết quả

### 3.1. Trên private test (cùng tập với mô hình fine-tune)

| Mô hình | Macro-F1 | Accuracy | F1 not | F1 text | F1 image | F1 multi |
|---|---|---|---|---|---|---|
| **C + B1** (fine-tune, Phase 10) | **0.380 ± 0.007** | 0.746 | 0.647 | 0.000 | 0.057 | 0.817 |
| `fusion_concat_frozen_reg_cw` (Phase 4) | 0.402 ± 0.015 | 0.755 | 0.644 | 0.000 | 0.128 | 0.834 |
| CLIP image-only fine-tune (Phase 3) | 0.3422 | 0.6895 | 0.612 | 0.000 | 0.000 | 0.757 |
| Vintern zero-shot | 0.0453 | 0.0632 | 0.000 | 0.003 | 0.025 | 0.153 |
| Vintern zero-shot + OCR/mô tả | 0.0151 | 0.0226 | 0.000 | 0.006 | 0.000 | 0.054 |

### 3.2. Trên val (public test)

| Mô hình | Macro-F1 | Accuracy | F1 not | F1 text | F1 image | F1 multi | Parse lỗi |
|---|---|---|---|---|---|---|---|
| C + B1 (fine-tune) | 0.416 ± 0.019 | 0.754 | 0.636 | 0.000 | 0.206 | 0.822 | – |
| M = `distill_flat_a` (fine-tune) | 0.391 ± 0.011 | 0.758 | 0.657 | 0.000 | 0.077 | 0.830 | – |
| Luôn đoán `multi` | – | 0.667 | – | – | – | khoảng 0,80 | – |
| Gemini Flash Lite v8 | 0.339 | 0.589 | 0.561 | 0.000 | 0.097 | 0.699 | 3,4% |
| Gemini Flash Lite v6 | 0.286 | 0.420 | 0.565 | 0.000 | 0.075 | 0.501 | 3,0% |

### 3.3. Trên tập con của train

| Mô hình | Tập | Accuracy | Macro-F1 | F1 not | F1 text | F1 image | F1 multi |
|---|---|---|---|---|---|---|---|
| Gemini Pro v5 | `pilot_dev` (cân bằng) | 0.487 | **0.480** | 0.442 | 0.411 | 0.637 | 0.430 |
| Flash Lite v6 bỏ mô tả | `pilot_dev` | 0.487 | 0.474 | 0.463 | 0.375 | 0.660 | 0.400 |
| Qwen2.5-VL-7B v1–v5 | `pilot_dev` | 0.256–0.269 | 0.160–0.190 | 0.347–0.405 | 0.000–0.115 | 0.000 | 0.188–0.326 |
| Flash Lite v7 | `natural_dev` | 0.588 | 0.346 | 0.677 | 0.000 | 0.098 | 0.609 |
| Qwen3-VL-8B v9 | `natural_dev` | 0.506 | 0.231 | 0.302 | 0.000 | 0.000 | 0.623 |

Bảng đầy đủ: `6_2_baseline_and_teacher.md` §2.

## 4. LLM sai ở đâu

| Mô hình | Kiểu lỗi chính | Bằng chứng |
|---|---|---|
| Vintern-1B | **Sụp về `text-sarcasm`**, lớp hiếm nhất; gần như không bao giờ nói `not` | 1.251/1.504 dự đoán là `text` (thật: 7); `not` 7 lần (thật: 458). Thêm OCR/mô tả: 1.284 lần `text`, 0 lần `not` |
| Qwen3-VL-8B | **Sụp về `multi`** | v7: 458/491 mẫu đọc được là `multi`; v9: 446/499, sai 230/280 mẫu `not` thành `multi`, 0/15 `image` đúng |
| Qwen2.5-VL-7B | Gần ngẫu nhiên | macro-F1 0,16–0,19 trên tập cân bằng, F1 `image` = 0 |
| Gemini Flash Lite | Nhầm `multi` thành `image` | val: 196/942 mẫu `multi` đoán thành `image`; precision `image` 5,3%; accuracy 58,9% < luôn đoán `multi` (66,7%) |
| Mọi LLM | Không nhận ra `text-sarcasm` trên phân bố thật | F1 `text` = 0 ở mọi lần chạy trên `natural_dev` và val |

Khi đã được cho nhãn đúng (rationalization, Qwen3-VL-8B, 100 mẫu train): 92/100 kết luận đúng nhãn đã cho nhưng cờ text/ảnh khớp nhãn ở 95/100 mẫu (chép lại nhãn), và phần giải thích có lỗi nhìn ảnh (rùa biển thành cá mập), bịa nghĩa, bỏ sót chơi chữ (Nissan/Mitsubishi).

**Nguyên nhân chung:**

1. **Không biết prior của dữ liệu.** LLM không thấy phân bố nhãn lệch 78 lần, nên chọn nhãn theo mô tả trong prompt (Vintern) hoặc theo thiên lệch riêng của mô hình (Qwen → `multi`).
2. **Ranh giới 4 lớp không rõ ngay cả trong hướng dẫn gán nhãn**, nên prompt không truyền đạt được ranh giới `image`/`multi`/`text` mà bộ dữ liệu dùng.
3. **Thêm thông tin "nghĩa đen" vào prompt làm tệ hơn** ở mô hình nhỏ (Vintern + OCR/mô tả); bỏ mô tả không ảnh hưởng tới Gemini.

**Điểm mạnh của LLM:** trên tập cân bằng, Gemini Pro đạt F1 `image-sarcasm` 0,637 và `text-sarcasm` 0,411, cao hơn hẳn mọi mô hình fine-tune (gần 0). Khi các lớp có tỉ lệ ngang nhau, LLM lớn nhận ra được châm biếm trong ảnh và trong caption; điều này mất đi trên phân bố thật vì precision của lớp hiếm sụp.

## 5. Mô hình fine-tune sai ở đâu

| Kiểu lỗi | Bằng chứng |
|---|---|
| Không bao giờ dự đoán `text-sarcasm` | F1 = 0 ở mọi config, mọi seed, mọi kỹ thuật mất cân bằng |
| `image-sarcasm` gần 0 trên private test | tốt nhất 0,128 (Phase 4), 0,057 (C + B1, benchmark chốt trước) |
| Sụp về 2 lớp đa số (image-only) | CLIP/ViT fine-tune chỉ dự đoán `not` và `multi` |
| Dựa vào caption, thẻ nguồn, dấu câu | tráo ảnh chỉ đổi khoảng 20% dự đoán; xoá thẻ đổi khoảng 25% dự đoán trên caption có thẻ (Phase 9.1) |

Mô hình fine-tune học được prior của dữ liệu và nhờ đó ăn điểm ở hai lớp lớn, nhưng không học được ranh giới lớp hiếm.

## 6. Kết luận

1. Trên phân bố thật, **mô hình nhỏ fine-tune vượt mọi LLM đã thử**: C + B1 0,380 trên private test so với Vintern 0,045; trên val 0,416 so với Gemini Flash Lite 0,339.
2. **LLM lớn (Gemini) nhận ra lớp hiếm tốt hơn trên tập cân bằng**, nhưng không giữ được precision trên phân bố thật; LLM nhỏ/mở (Vintern, Qwen) thiên lệch mạnh về một lớp.
3. **Hai bên sai ở chỗ khác nhau:** LLM sai vì không biết prior và ranh giới nhãn của bộ dữ liệu; mô hình fine-tune sai vì quá ít mẫu lớp hiếm và dựa vào tín hiệu bề mặt. Gợi ý kết hợp: dùng LLM để gợi ý lớp hiếm, mô hình fine-tune để giữ prior (chưa thử).

## 7. Việc còn thiếu so với `report_main.md`

- Gemini và Qwen **chưa chạy trên private test**. Muốn so trên cùng tập: chạy `scripts/eval_teacher.py` với Flash Lite v8 (khoảng 1.504 × 0,0014 ≈ 2 USD theo giá ước tính) và Qwen3-VL-8B v9.
- Chưa có few-shot chọn ví dụ theo độ giống (E2), chưa có few-shot cân bằng theo lớp kèm tỉ lệ lớp trong prompt.
- Chưa có phân tích từng mẫu chéo (mẫu LLM đúng mà mô hình fine-tune sai và ngược lại) vì hai bên chưa chạy trên cùng tập.

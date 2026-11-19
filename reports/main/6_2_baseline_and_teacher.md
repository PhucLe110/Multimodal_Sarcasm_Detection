# 6.2. Baseline (cột a/b) và teacher inference

- **Phase:** 6, Chain-of-Thought + Knowledge Distillation + Hierarchical Prompting
- **Thành viên:** Trúc
- **Nguồn số liệu:** `reports/results/experiment_report.md` (sinh tự động từ output trên GPU server), `reports/results/progress_report.md`, `reports/results/results.md` (plan `cot_distill_stage0`, `no_source_tags`, `b2_stacking`)
- **Môi trường:** GPU server RTX 5070 12 GB, commit `e082390` (giai đoạn 0); Gemini qua API
- **Tập đánh giá:** **val (public test, 1.413 mẫu)**. Nhánh này cố ý giữ private test cho bước đánh giá cuối (`train.eval_test: false`); vì nhánh dừng trước bước đó, **chưa có số private test** cho các cấu hình ở đây. Đây là ngoại lệ so với quy ước của `report_main.md`.

Phương pháp: `6_1_cot_distillation_method.md`.

---

## 1. Giai đoạn 0: student không teacher (cột a/b)

Backbone `fusion_multistream_clip_frozen` (caption + OCR + mô tả + CLIP, encoder đóng băng). 3 seed (1, 2, 42). Chọn checkpoint theo val macro-F1 bỏ `text-sarcasm`.

### 1.1. Chưa hiệu chỉnh prior

| Cấu hình | Macro-F1 | **Macro-F1 bỏ text** | Weighted-F1 | Accuracy | F1 not | F1 image | F1 multi |
|---|---|---|---|---|---|---|---|
| `distill_flat_a` (phẳng, CE) | 0.391 ± 0.011 | **0.521 ± 0.015** | 0.760 ± 0.022 | 0.758 ± 0.027 | 0.657 ± 0.030 | 0.077 ± 0.069 | 0.830 ± 0.022 |
| `distill_flat_b` (+ `involves_*`) | 0.389 ± 0.014 | 0.518 ± 0.019 | 0.763 ± 0.010 | 0.771 ± 0.016 | 0.654 ± 0.017 | 0.064 ± 0.061 | 0.836 ± 0.015 |
| `distill_hier_a` (phân cấp mềm) | 0.379 ± 0.003 | 0.506 ± 0.004 | 0.757 ± 0.016 | 0.766 ± 0.023 | 0.648 ± 0.006 | 0.040 ± 0.035 | 0.830 ± 0.025 |
| `distill_hier_b` | 0.384 ± 0.006 | 0.512 ± 0.008 | 0.754 ± 0.011 | 0.758 ± 0.013 | 0.655 ± 0.025 | 0.058 ± 0.053 | 0.821 ± 0.008 |
| `hier_cascade` (tái lập Vu et al.) | 0.380 ± 0.006 | 0.507 ± 0.008 | 0.764 ± 0.009 | 0.771 ± 0.016 | 0.671 ± 0.019 | 0.018 ± 0.030 | 0.832 ± 0.019 |

F1 `text-sarcasm` = 0 ở mọi cấu hình (val có 1 mẫu). Mỗi run 5–16 phút, peak VRAM 1,24–1,27 GB.

### 1.2. Có hiệu chỉnh prior

| Cấu hình | Macro-F1 | Macro-F1 bỏ text | Accuracy | F1 not | F1 image | F1 multi |
|---|---|---|---|---|---|---|
| `distill_flat_a` | 0.373 ± 0.006 | 0.498 ± 0.008 | 0.773 ± 0.011 | 0.603 ± 0.019 | 0.042 ± 0.043 | 0.848 ± 0.007 |
| `distill_flat_b` | 0.366 ± 0.008 | 0.488 ± 0.011 | 0.783 ± 0.008 | 0.612 ± 0.030 | 0.000 | 0.853 ± 0.004 |
| `distill_hier_a` | 0.354 ± 0.013 | 0.472 ± 0.017 | 0.769 ± 0.008 | 0.572 ± 0.048 | 0.000 | 0.845 ± 0.002 |
| `distill_hier_b` | 0.353 ± 0.010 | 0.471 ± 0.014 | 0.769 ± 0.010 | 0.565 ± 0.038 | 0.000 | 0.847 ± 0.005 |
| `hier_cascade` | 0.358 ± 0.004 | 0.477 ± 0.005 | 0.774 ± 0.005 | 0.584 ± 0.023 | 0.000 | 0.848 ± 0.007 |

### 1.3. Nhận xét

- **Phân cấp mềm không hơn phẳng.** Cột (a) kém 1,2 điểm (vượt ngưỡng 1 điểm của G0); cột (b) kém 0,5 điểm (đạt). Theo seed, phân cấp hơn 0,2 điểm ở seed 42 nhưng kém 2,7 và 1,0 điểm ở seed 1 và 2; std của cột phẳng là 1,1 điểm, nên chênh lệch nằm trong nhiễu. Code head phân cấp và loss đã kiểm tra lại, không có lỗi. **G0: đi tiếp.**
- **Phân cấp nối tiếp (cascade) không kém phân cấp mềm**: lợi ích "tránh lỗi lan truyền" chưa đo được.
- **Hiệu chỉnh prior làm giảm 2–4 điểm ở mọi cấu hình.** Nó tăng F1 `multi` (tốt nhất 0,853) nhưng xoá gần hết `image-sarcasm`. Đây là ví dụ rõ về metric tổng che giấu điểm yếu theo lớp.
- **Nhãn phụ suy từ nhãn gốc (cột b) không thêm gì** so với cột (a).

**Mốc cho G1** (F1 tốt nhất của student trên val): `image-sarcasm` **0,077** (`distill_flat_a`), `multi-sarcasm` **0,836** (0,853 khi có hiệu chỉnh prior).

## 2. Teacher inference

### 2.1. Tập thử

| Tập | Nguồn | Số mẫu | Dùng để |
|---|---|---:|---|
| `pilot_dev` | train, cân bằng 40 mẫu/lớp | 160 | sửa prompt |
| `pilot_eval` | train, cân bằng | 120 | đánh giá cuối prompt (**chưa dùng**) |
| `natural_dev` | train, ngẫu nhiên theo phân bố thật | 500 | thêm ngoài kế hoạch, vì `pilot_dev` cân bằng không phản ánh phân bố thật |
| val | public test | 1.413 | đối chiếu với mốc G1 |

### 2.2. `pilot_dev` (160 mẫu cân bằng)

| Teacher | Prompt | Accuracy | Macro-F1 | F1 not | F1 text | F1 image | F1 multi | Parse lỗi | Giây/lần |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B (4-bit) | v1 | 0.269 | 0.190 | 0.393 | 0.115 | 0.000 | 0.250 | 0,0% | 1,6 |
| Qwen2.5-VL-7B (4-bit) | v2–v5 | 0.256–0.263 | 0.160–0.180 | 0.347–0.405 | 0.000–0.048 | 0.000 | 0.188–0.326 | 0–2,5% | 1,7–3,4 |
| Gemini Pro (`gemini-3.1-pro-preview`) | v5 | 0.487 | **0.480** | 0.442 | 0.411 | 0.637 | 0.430 | 0,0% | 12,8 |
| Gemini 3.5 Flash Lite | v6 | 0.438 | 0.434 | 0.390 | 0.426 | 0.565 | 0.353 | 2,5% | 1,7 |
| Gemini 3.5 Flash Lite | v6, bỏ mô tả ảnh | 0.487 | 0.474 | 0.463 | 0.375 | 0.660 | 0.400 | 1,9% | 1,9 |

### 2.3. `natural_dev` (500 mẫu, phân bố thật)

| Teacher | Prompt | Accuracy | Macro-F1 | F1 not | F1 text | F1 image | F1 multi | Parse lỗi | Giây/lần |
|---|---|---|---|---|---|---|---|---|---|
| Flash Lite | v6 | 0.500 | 0.318 | 0.677 | 0.000 | 0.096 | 0.500 | 2,6% | 2,3 |
| Flash Lite | v6, bỏ mô tả | 0.496 | 0.317 | 0.678 | 0.000 | 0.095 | 0.494 | 1,6% | 1,7 |
| Flash Lite | v7 | **0.588** | **0.346** | 0.677 | 0.000 | 0.098 | 0.609 | 2,0% | 1,8 |
| Flash Lite | v8 | 0.568 | 0.326 | 0.655 | 0.000 | 0.047 | 0.602 | 2,6% | 2,4 |
| **Qwen3-VL-8B** (4-bit) | v7 | 0.468 | 0.204 | 0.208 | 0.000 | 0.000 | 0.606 | 1,8% | 8,5 |
| **Qwen3-VL-8B** (4-bit) | v9 | 0.506 | 0.231 | 0.302 | 0.000 | 0.000 | 0.623 | 0,2% | 7,7 |

### 2.4. Val (1.413 mẫu) và cổng G1

| Lần chạy | Accuracy | Macro-F1 | F1 not | F1 image (mốc 0,077) | F1 multi (mốc 0,853) | Parse lỗi | G1 |
|---|---|---|---|---|---|---|---|
| Flash Lite v8 | 0.589 | 0.339 | 0.561 | 0.097 | 0.699 | 3,4% | đạt hình thức |
| Flash Lite v6 | 0.420 | 0.286 | 0.565 | 0.075 | 0.501 | 3,0% | không đạt |

### 2.5. Phân tích

- **G1 chỉ đạt về hình thức.** Flash Lite v8 đúng 14/28 mẫu `image-sarcasm` nhưng precision lớp này chỉ 5,3%; chỉ cần ít hơn 3 mẫu đúng là không đạt. Mốc 0,077 là trung bình seed, seed tốt nhất của student đạt 0,130. F1 `multi` (0,699) thấp xa mốc 0,853. Accuracy 58,9% **thấp hơn luôn đoán `multi`** (66,7%): 196/942 mẫu `multi` bị đoán thành `image`.
- **Qwen3-VL-8B thiên lệch nặng về `multi`.** v7: đoán `multi` cho 458/491 mẫu đọc được. v9 (định nghĩa chính thức, quy tắc suy từ dữ liệu, 7 ví dụ): vẫn đoán `multi` cho 446/499 mẫu, sai 230/280 mẫu `not-sarcasm` thành `multi`, đúng 0/15 `image-sarcasm`. Prompt tốt hơn không sửa được thiên lệch.
- **Qwen2.5-VL-7B** gần mức đoán ngẫu nhiên (macro-F1 0,16–0,19 trên tập cân bằng).
- **Bỏ mô tả Vintern không đổi kết quả**: 0,317 so với 0,318 (95% CI của chênh lệch [−0,025; +0,019]), giảm 14% token đầu vào. Từ v7 prompt không gửi mô tả ảnh.
- **Chi phí ước tính** (giá từ nguồn tổng hợp, chưa đối chiếu giá chính thức): Flash Lite khoảng 0,0014 USD/lần sinh, khoảng 10 USD cho lô 1–3 với k = 1, khoảng 30 USD với k theo kế hoạch; Gemini Pro khoảng 35 USD cho val, khoảng 500 USD cho các lô train.
- **Độ tin cậy:** Qwen chạy 4-bit do GPU 12 GB. Val đã dùng hai lần (v6 rồi v8), v7 viết sau khi xem v6, nên v8 trên val có thể hơi lạc quan. Từ v5 prompt có 4 ví dụ cố định từ train, nên không còn là zero-shot hoàn toàn. `gemini-2.5-flash-lite` trả 404 nên dùng `gemini-3.5-flash-lite`.

### 2.6. Rationalization: giải thích khi đã cho nhãn

Qwen3-VL-8B (4-bit, v8), 100 mẫu train ngoài `pilot_eval`, khoảng 8 giây/mẫu:

- 98/100 đọc được; 92/100 kết luận đúng nhãn đã cho (`text-sarcasm` 16/20).
- Cờ text/ảnh khớp nhãn ở 95/100 mẫu: **teacher gần như chép lại nhãn gốc**, nên cột (c) sẽ gần trùng cột (b).
- Đọc 13 mẫu, đối chiếu ảnh 4 mẫu: mô tả bề mặt dùng được, phần giải thích ý mỉa mai yếu. Lỗi gặp: nhận nhầm miệng rùa biển thành cá mập (1244); bịa nghĩa "số 13 xui xẻo" trong khi câu đùa là chồng được nhận lại máy cũ của vợ (4032); bỏ sót chơi chữ Nissan/Mitsubishi (2559); lý do mâu thuẫn với nhãn được giao (2259, 3832).

## 3. Quyết định ngày 03/11/2026: dừng M1, M2, M3

- Teacher miễn phí (Qwen) không viết được lập luận đáng tin. Teacher trả phí (Gemini) chưa vượt được mức luôn đoán `multi`, và không còn ngân sách sinh cho cả tập.
- Phương án người viết giải thích cũng bị loại: hướng dẫn gán nhãn chính thức chỉ định nghĩa có/không châm biếm, 3/5 ví dụ trong hướng dẫn có trong dữ liệu mang nhãn khác slide. Viết lập luận cho nhãn mà người viết không hiểu chỉ là hợp lý hoá.
- **Hệ quả:** giai đoạn 2–6 (sinh lập luận cả tập, chấm rubric, cột c/d, M3) không chạy. Code đã có và có test.

## 4. Các hướng phụ đã chạy

### 4.1. E1: kNN retrieval (không giữ)

Grid k ∈ {5, 10, 20, 50}, τ ∈ {0,05; 0,1; 0,2}, λ ∈ {0,5; 0,7; 0,9; 1}. "Kiểm định chéo" = chọn tham số trên phần còn lại của val. Macro-F1 bỏ text, trung bình 3 seed:

| Mô hình | Gốc | Có prior, không kNN | Có prior + kNN (kiểm định chéo) | Không prior + kNN (kiểm định chéo) |
|---|---|---|---|---|
| `distill_flat_a` | 0.521 | 0.498 | 0.491 | 0.511 |
| `distill_hier_b` | 0.512 | 0.471 | 0.488 | 0.511 |
| `hier_cascade` | 0.507 | 0.477 | 0.483 | 0.500 |

kNN chỉ bù lại phần giảm do hiệu chỉnh prior, vẫn thấp hơn mô hình gốc. Dò tham số trên chính val tăng 0,3–1,5 điểm nhưng mất khi chọn ngoài mẫu (quá khớp val). Giả định "meme lặp template" không đúng: chỉ 0,1–3,5% mẫu val có láng giềng gần nhất với cosine ≥ 0,9; nhãn láng giềng gần nhất đúng 60–64%, thấp hơn luôn đoán `multi`.

### 4.2. B2: 3 mô hình đơn modality + stacking (không giữ)

1 seed (42), 5-fold out-of-fold trên train, macro-F1 bỏ text:

| Cách | OOF train | Val | Val có prior | F1 image (val) |
|---|---|---|---|---|
| Chỉ caption | 0.496 | 0.504 | 0.479 | 0.000 |
| Chỉ ảnh + OCR + mô tả | 0.435 | 0.467 | 0.476 | 0.000 |
| Đa phương thức | 0.526 | **0.507** | 0.489 | 0.000 |
| Trung bình xác suất 3 mô hình | – | 0.500 | 0.484 | 0.000 |
| Stacking (kiểm định chéo trong val) | – | 0.503 | – | 0.000 |

Stacking không hơn mô hình đa phương thức một mình; classifier gộp gần như chỉ dựa vào nó. **Caption-only (0,504) gần bằng đa phương thức (0,507).**

### 4.3. Thẻ nguồn (chuẩn bị cho Phase 9, 10)

Train lại `distill_flat_a` trên caption đã xoá thẻ nguồn, hashtag và chữ ký (`distill_flat_a_notags`, 3 seed): macro-F1 bỏ text giảm từ 0.521 ± 0.015 xuống 0.501 ± 0.019 (cả 3 seed giảm 1,4–3,4 điểm). Chi tiết ở `9_1_shortcut_learning.md`.

## 5. Tổng kết

| Câu hỏi | Trả lời |
|---|---|
| Phân cấp mềm có giúp? | Không (0,506–0,512 so với 0,518–0,521) |
| Phân cấp mềm có hơn nối tiếp (Vu et al.)? | Chưa đo được khác biệt |
| Hiệu chỉnh prior có giúp? | Không: giảm 2–4 điểm, xoá lớp `image` |
| kNN retrieval có giúp? | Không |
| Gộp muộn (stacking) có hơn end-to-end? | Không |
| VLM có làm teacher được? | Không, với ngân sách hiện có |

**Mô hình tốt nhất của nhánh:** `distill_flat_a`, val macro-F1 bỏ text 0.521 ± 0.015 (4 lớp 0.391). Kiến trúc này là M trong Phase 10; nó không được đưa vào benchmark cuối trên private test.

Kết quả âm tính của nhánh CoT là lý do chuyển sang phân tích thực nghiệm (Phase 10): cả student nhỏ lẫn VLM lớn đều không tách được ba loại châm biếm, và cùng bám vào caption, thẻ nguồn và lớp đa số.

## 6. Việc còn thiếu so với `report_main.md`

- Không có số private test cho các cấu hình giai đoạn 0 (nhánh dừng trước bước chấm cuối). Muốn có: chạy `scripts/evaluate.py --split test` trên các checkpoint `distill_*` đã lưu trên GPU server.
- Teacher chưa được chạy trên private test.

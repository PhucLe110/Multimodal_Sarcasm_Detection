# Kế hoạch paper: phụ thuộc text như một dạng shortcut learning trong phát hiện châm biếm đa phương thức tiếng Việt

- **Ngày chốt:** 17/11/2026
- **Mục tiêu nộp:** 15/12/2026, hội nghị có index Scopus (danh sách venue ở §11, cần xác nhận deadline)
- **Loại bài:** empirical analysis, không đề xuất kiến trúc mới, không cần gán nhãn mới
- **Tên dự kiến:** *Reading Without Seeing: Text Dependence as Shortcut Learning in Vietnamese Multimodal Sarcasm Detection*
- **Nền tảng:** hạ tầng và kết quả của Phase 3–10 (`reports/main/`), giao thức thống kê của Phase 10 (`reports/main/10_1_empirical_plan.md` §7)

---

## 1. Claim trung tâm

> Theo định nghĩa chính thức của ViMMSD, nhãn 4 lớp **không xác định được chỉ từ text**. Tuy vậy, mô hình chỉ đọc text đạt khoảng 95% hiệu năng của mô hình đa phương thức tốt nhất, và mô hình đa phương thức hành xử gần như mô hình text. Chúng tôi định lượng mức phụ thuộc text này như một dạng shortcut learning, kiểm chứng rằng nó không tổng quát khi đổi nguồn dữ liệu, và dùng ablation có kiểm soát để xác định vì sao mô hình đi đường tắt qua text thay vì dùng ảnh.

### 1.1. Đóng góp dự kiến

1. **Lập luận về khả năng xác định nhãn** (§2): theo định nghĩa của ViMMSD, mọi hiệu năng của mô hình text-only vượt mốc prior đều đến từ tương quan trong dữ liệu, không từ định nghĩa nhãn.
2. **Bộ thước đo mức phụ thuộc text** (§4): khoảng cách hiệu năng, độ đồng thuận dự đoán, và **độ nhạy bất đối xứng** khi tráo caption so với khi tráo ảnh.
3. **Bằng chứng tương quan text là shortcut** (H2): baseline chỉ dùng tín hiệu bề mặt, train khi xoá thẻ, split giữ riêng nguồn.
4. **Phân tích nguyên nhân bằng ablation** (H1, H3–H5): năng lực encoder ảnh, cạnh tranh giữa hai modality khi train, nhãn không nhất quán ở các lớp cần ảnh, thiếu mẫu cần ảnh.
5. **Khuyến nghị đánh giá** cho benchmark đa phương thức có nhãn theo nguồn châm biếm; phát hành split theo nguồn, danh sách id nhóm mâu thuẫn, code.

### 1.2. Khác biệt với công trình liên quan

| Công trình | Đã làm | Paper này thêm |
|---|---|---|
| MMSD2.0 (Findings ACL 2023) | sửa cue lộ nhãn, nhãn sai trên MMSD tiếng Anh nhị phân | 4 lớp theo nguồn châm biếm, tiếng Việt, không sửa dữ liệu mà định lượng và giải thích |
| SarcNet (LREC-COLING 2024), MuSaG (LREC 2026) | nhãn theo từng modality; quan sát mô hình thiên về text | lập luận khả năng xác định nhãn; ablation nguyên nhân |
| ViMMSarc-Fine (KSE 2026) | dataset tiếng Việt mới, nhãn T/I/M; quan sát multimodal không vượt text-only | phân tích benchmark ViMMSD đang được dùng; giải thích *vì sao*, tách shortcut theo nguồn |
| Wu et al. (ICML 2022), Peng et al. (CVPR 2022) | học đa phương thức "tham lam", cạnh tranh modality, trên tác vụ khác | kiểm chứng giả thuyết cạnh tranh modality trên châm biếm (H3) |

**Việc bắt buộc tuần 1:** đọc toàn văn ViMMSarc-Fine và Vu et al. (IEA/AIE 2025); xác minh các bài được gợi ý nhưng chưa kiểm tra (*Beyond Spurious Signals*, EMNLP Findings 2025; *HC-CoT*, ESWA 2026).

## 2. Lập luận khả năng xác định nhãn

Nguồn: `reports/guides/annotation_guideline.md` (bản chính thức, slide 2–10).

| Caption đứng riêng | Nhãn có thể có | Thông tin còn thiếu để quyết định |
|---|---|---|
| không châm biếm | `not`, `image`, `multi` | ảnh có tự châm biếm không; ghép với caption có tạo mâu thuẫn (*mismatch*) không |
| có châm biếm | `text`, `multi` | ảnh có góp phần vào ý châm biếm không |

- Không lớp nào xác định được nếu chỉ có text; cả `text-sarcasm` cũng cần biết ảnh *không* góp phần.
- **Hình 1 của bài:** slide 6. Cùng một ảnh, ba caption cho `not`, `multi`, `multi`. Caption "Lót tích vl =))" thành `multi` nhờ mâu thuẫn với ảnh.
- Hệ quả: mốc hợp lệ cho mô hình text-only là mốc đoán theo prior. Mọi phần vượt mốc này đến từ tương quan text–nhãn trong dữ liệu (thẻ nguồn, văn phong trang, emoji, độ dài), không được định nghĩa nhãn bảo đảm.

**Cách viết chính xác:** *dùng* text không phải shortcut (caption là một nửa bằng chứng). *Quyết định* nhãn chỉ bằng text mới là shortcut.

**Ba giới hạn phải nêu trong bài:**

1. Lập luận dựa trên định nghĩa chính thức; dữ liệu không luôn theo định nghĩa (3/5 ví dụ trong hướng dẫn mang nhãn khác trong dữ liệu). Nhiễu nhãn được xử lý riêng ở H4.
2. "Tương quan có thật trong tập người dùng" chưa đủ để gọi là shortcut; cần thêm bằng chứng không tổng quát khi đổi nguồn (H2).
3. Lập luận áp dụng đối xứng cho ảnh: image-only cũng không xác định được nhãn. Image-only đóng băng đạt 0.370 trên private test, ngang text-only. Vì vậy claim "phụ thuộc **text**" phải dựa trên hành vi của mô hình fusion (§4), không chỉ trên điểm của mô hình đơn modality.

## 3. Câu hỏi nghiên cứu

| ID | Câu hỏi | Trả lời bằng |
|---|---|---|
| RQ1 | Mô hình đa phương thức phụ thuộc vào text tới mức nào? | thước đo M1–M5 (§4) |
| RQ2 | Phần phụ thuộc này có phải shortcut, tức không tổng quát khi đổi nguồn? | H2 |
| RQ3 | Vì sao mô hình đi đường tắt qua text thay vì dùng ảnh? | H1, H3, H4, H5 |

## 4. Đo mức phụ thuộc text (RQ1)

| ID | Thước đo | Định nghĩa | Bằng chứng sơ bộ |
|---|---|---|---|
| M1 | Tỉ lệ hiệu năng | macro-F1(text-only) / macro-F1(fusion), cùng loss, cùng giao thức | Phase 4: 0.381 / 0.402 = 0,95 (cùng class weight); Phase 10: 0.366 / 0.380 = 0,96 (T + B1 / C + B1) |
| M2 | Độ đồng thuận dự đoán | tỉ lệ mẫu fusion dự đoán giống text-only, so với giống image-only; Cohen's κ | chưa tính; Phase 4: concat và text-only khác nhau ở ít nhất 17% mẫu test (123 + 131 mẫu chỉ một bên đúng, chưa tính mẫu cả hai cùng sai nhưng khác nhãn) |
| M3 | Độ nhạy bất đối xứng | flip rate khi **tráo caption** (giữ ảnh) chia cho flip rate khi **tráo ảnh** (giữ caption), cùng donor | tráo ảnh: 0,199 (C, val); **tráo caption chưa có** |
| M4 | Hướng dịch chuyển | khi tráo, dự đoán đi theo nhãn của mẫu cho (donor) bao nhiêu % | chưa có |
| M5 | Phần đóng góp ảnh theo lớp | M3, M4 tách theo lớp gốc `not` / `image` / `multi` | tráo ảnh làm đổi 36% mẫu `image` so với 18% mẫu `multi` (C, val) |

**Chính sách donor** (mở rộng `make_donor_manifest`):

| Policy | Mẫu cho | Có |
|---|---|---|
| `random` | bất kỳ | ✓ |
| `same_label` | cùng nhãn | ✓ |
| `cross_label` | khác nhãn | cần thêm |
| `same_source` / `cross_source` | cùng/khác nguồn suy ra (§6, H2) | cần thêm |

Tráo không tạo nhãn mới cho cặp sau tráo. M3–M5 đo độ nhạy và hướng dịch chuyển, **không** đo độ chính xác trên cặp mới. Theo §2, nhãn phụ thuộc cả hai modality, nên một mô hình gần như bất biến khi tráo ảnh nhưng rất nhạy khi tráo caption là vi phạm định nghĩa nhãn.

**Tiêu chí xác nhận RQ1 (chốt trước):** phụ thuộc text được xác nhận khi cả ba điều sau đúng trên C + B1 và cross-attention 2 chiều:

1. M1 ≥ 0,90 trên cả pooled OOF lẫn private test;
2. M3 ≥ 2 (khoảng tin cậy bootstrap không chứa 1);
3. M2: đồng thuận với text-only cao hơn đồng thuận với image-only, khoảng tin cậy của hiệu số không chứa 0.

**Nếu không đạt** (ví dụ M3 ≈ 1, hoặc fusion đồng thuận với image-only ngang text-only): bác bỏ claim "phụ thuộc text", đổi khung thành "shortcut đơn modality" và giữ nguyên RQ2, RQ3.

## 5. Giả thuyết nguyên nhân (RQ2, RQ3)

| ID | Giả thuyết | Dự đoán nếu đúng | Ablation | Điều kiện bác bỏ |
|---|---|---|---|---|
| **H1** | Encoder ảnh không đọc được chữ, meme tiếng Việt | thay encoder mạnh hơn hoặc thêm text từ ảnh thì M1 giảm, M3 giảm | (a) C + OCR + mô tả (đã có: M, OD); (b) thay CLIP ViT-B/32 bằng encoder độ phân giải cao hơn hoặc đa ngôn ngữ (SigLIP đa ngôn ngữ, CLIP ViT-B/16; ViCLIP-OT nếu tích hợp kịp) | M1, M3 không đổi qua các encoder |
| **H2** | Tương quan text–nhãn theo nguồn làm đường tắt rẻ, và không tổng quát | baseline chỉ-shortcut đạt tỉ lệ đáng kể của T; điểm giảm khi test trên nguồn chưa thấy; xoá thẻ khi train giảm M1 | (a) baseline chỉ-shortcut; (b) train khi xoá thẻ (đã có trên val); (c) split giữ riêng nguồn | baseline shortcut ≈ prior; không giảm khi đổi nguồn |
| **H3** | Cạnh tranh modality: nhánh text hội tụ trước, nhánh ảnh bị bỏ đói | probe nhánh ảnh trong fusion yếu hơn image-only train riêng; tăng drop text khi train làm ảnh được dùng nhiều hơn | (a) linear probe trên vector ảnh sau chiếu của C so với I; (b) text dropout 30% (đối xứng với modality dropout ảnh hiện có); (c) khởi tạo nhánh ảnh từ I | probe ảnh trong fusion ≈ image-only; text dropout không đổi M3 |
| **H4** | Nhãn các lớp cần ảnh (`image`, `multi`) không nhất quán | mẫu `image` tập trung ở vùng mơ hồ hoặc khó học; nhiều ảnh trùng mang nhãn khác | (a) nhóm ảnh trùng (md5, pHash) và caption trùng có nhãn mâu thuẫn, tính cận trên macro-F1; (b) dataset cartography trên C + B1 | `image` không khác các lớp khác về phân bố cartography |
| **H5** | Quá ít mẫu cần ảnh để học | F1 `image` tăng đều theo lượng dữ liệu; embedding đóng băng tách được `image` khỏi `multi` | (a) learning curve 25/50/75/100% train; (b) probe khả tách (logistic, kNN) trên embedding CLIP và PhoBERT | đường cong phẳng *và* probe không tách được thì đổ cho H4 |

H1–H5 không loại trừ nhau. Kết quả cuối là **bảng phân rã**: mỗi giả thuyết được ủng hộ, không ủng hộ, hay chưa kết luận, kèm độ lớn hiệu ứng.

### 5.1. Chi tiết H2: suy ra nguồn đăng

Không có metadata nguồn. Suy ra tự động, theo thứ tự ưu tiên:

1. thẻ trang trong caption (`interpool*`, `voz*`, `wmo`, `worldmemeorganization`, `quytacungxutx`, …), dùng `strip_source_tags` / `check_source_tags.py`;
2. chữ ký admin một dòng (`- Kem`, `- Elsa`);
3. chuỗi watermark trong OCR (`facebook.com/trollbongda`, `trollxe`, `youtube.com/...`).

Mẫu không suy ra được nguồn xếp vào nhóm `unknown`, giữ ở train, không dùng làm test nguồn. Split giữ riêng nguồn: chọn các nguồn đủ lớn làm test sao cho mỗi lớp có support báo cáo được; giữ nguyên ràng buộc nhóm caption/ảnh của Phase 10. Báo cáo độ phủ (bao nhiêu % mẫu suy ra được nguồn) là một kết quả riêng; nếu độ phủ quá thấp, H2c chỉ là phân tích trên tập con.

### 5.2. Chi tiết H2a: baseline chỉ-shortcut

Logistic regression (class weight `sqrt_inverse`), đặc trưng **không chứa nội dung từ vựng của caption**: thẻ nguồn (one-hot), có hashtag, số và loại emoji, emoticon (`=))`, `:v`), dấu câu lặp, độ dài, có URL, có chữ ký admin, có xuống dòng. Báo cáo macro-F1 trên pooled OOF và private test, và tỉ lệ so với T.

## 6. Giao thức

| Mục | Giá trị |
|---|---|
| Dữ liệu | train 10.805; public test 1.413; private test 1.504; dữ liệu gốc, không lọc nhiễu |
| Chọn mô hình, checkpoint | 3 fold nội bộ chia theo nhóm (`folds_k3_s42.json`), pooled OOF; checkpoint theo macro-F1 bỏ `text-sarcasm` trên phần đánh giá của fold |
| Benchmark | private test: chấm các mô hình chính để so với Vu et al. và Phase 4/10; không dùng để chọn hay chỉnh |
| Held-out mới | split giữ riêng nguồn (H2c), chốt trước khi train |
| Encoder | PhoBERT-base, CLIP ViT-B/32 đóng băng (trừ ablation H1) |
| Seed | 1, 2, 42 cho mọi so sánh chính; screening dùng seed 42 |
| Chỉ số | macro-F1 4 lớp (chính), macro-F1 3 lớp, F1/P/R từng lớp, support |
| Thống kê | paired bootstrap 2.000 lần, group bootstrap, Holm theo họ contrast khai báo trước (`scripts/empirical_stats.py`) |
| Luật quyết định | như `10_1_empirical_plan.md` §7; thêm tiêu chí RQ1 ở §4 |

### 6.1. Bộ mô hình

| ID | Config | Vai trò |
|---|---|---|
| T, T + B1 | `text_only_frozen(_cw)` | text-only |
| I, I + B1 | `image_only_frozen` (+ class weight) | image-only, đối chứng đối xứng |
| C, C + B1 | `fusion_concat_frozen_reg(_cw)` | fusion chính |
| X | `fusion_cross_attn_frozen` (+ class weight) | fusion có tương tác token |
| M | `fusion_multistream_clip_frozen` | thêm OCR + mô tả (H1a) |
| S | baseline chỉ-shortcut | H2a |

## 7. Bằng chứng đã có (dùng lại, ghi rõ nguồn)

| Kết quả | Nguồn | Dùng cho |
|---|---|---|
| T/C trên private test, 3 seed | Phase 4 (`4_1`), Phase 10 (`10_6`) | M1 |
| Concat sửa 123 / làm hỏng 131 mẫu của text-only; cross-attention net +55,7 | `4_1` §3.2, `5_1` §2.3 | M2 sơ bộ |
| Tráo ảnh: C đổi 19,9% dự đoán, mẫu `image` 36% | `10_3` | M3, M5 (vế ảnh) |
| Xoá thẻ lúc suy luận đổi khoảng 25%; train không thẻ: C 0.399 → 0.385 (val) | `10_5`, `9_1` | H2b |
| OCR, mô tả không thêm gì đo được | `3_2`, `10_2` | H1a |
| 123 nhóm caption trùng khác nhãn; 3/5 ví dụ hướng dẫn khác nhãn | `13_1`, `reports/guides/annotation_guideline.md` | H4 |
| OOF 3 fold, F1 `image` 0,17 trên 442 mẫu | `10_4` | H5 |
| Integrated Gradients: thẻ nguồn, dấu câu đứng đầu | `9_2` | H2 minh hoạ |

Phần lớn số liệu cũ đo trên val. Trong paper, các thước đo chính được tính lại trên pooled OOF và private test theo §6.

## 8. Việc cần làm

### 8.1. Code

| # | Việc | File | Ghi chú |
|---|---|---|---|
| 1 | Can thiệp `caption`: tráo caption giữ ảnh, OCR, mô tả | `analysis/interventions.py`, `scripts/diagnose_inputs.py` | caption lấy từ cache đã tiền xử lý của donor |
| 2 | Policy `cross_label`, `same_source`, `cross_source`; thêm M2, M4 vào tổng hợp | `analysis/interventions.py` | |
| 3 | Chạy chẩn đoán trên fold OOF và private test (hiện chỉ val) | `scripts/diagnose_inputs.py` | không train lại |
| 4 | Suy ra nguồn, dựng split giữ riêng nguồn | script mới `scripts/build_source_splits.py`, `data/splits.py` | xuất manifest + bảng độ phủ |
| 5 | Baseline chỉ-shortcut | script mới `scripts/shortcut_baseline.py` | CPU |
| 6 | Ghi xác suất từng epoch trên train (cartography) | `training/` | cờ config, tắt mặc định |
| 7 | Learning curve: lấy mẫu con theo nhóm và lớp | `data/splits.py` + plan | |
| 8 | Linear probe trên vector ảnh của C; text dropout | `models/fusion.py`, script probe | H3 |
| 9 | Encoder ảnh thay thế | `models/` image encoder | kiểm tra tương thích SigLIP tuần 1; ViCLIP-OT chỉ khi kịp |
| 10 | Cận trên do nhãn mâu thuẫn | script phân tích | CPU |

### 8.2. Ngân sách run (Kaggle T4, đóng băng)

| Nhóm | Run | Ước lượng |
|---|---:|---|
| RQ1: T, I, C, C + B1, X, X + B1 trên 3 fold × seed 42, rồi xác nhận 3 seed các mô hình chính | khoảng 24 | T khoảng 3 phút; fusion 14–21 phút |
| H1b: 2 encoder × C + B1 × 3 seed | 6 | |
| H2b–c: T, C + B1 không thẻ và trên split nguồn × 3 seed | 12 | |
| H3: C + B1 với text dropout, khởi tạo nhánh ảnh × 3 seed | 6 | |
| H4b: cartography (C + B1 có log epoch) × 3 seed | 3 | |
| H5a: 3 mức dữ liệu × 3 fold, seed 42 | 9 | |
| **Tổng** | **khoảng 60** | khoảng 15–20 GPU-giờ, trong quota Kaggle |

Chẩn đoán tráo (M2–M5), probe, baseline shortcut, cận trên nhãn: suy luận hoặc CPU.

## 9. Lịch

| Tuần | Việc | Điều kiện đi tiếp |
|---|---|---|
| **1** (17–23/11) | Chốt venue; đọc ViMMSarc-Fine, Vu et al.; code #1–#5, #10; chạy baseline shortcut, cận trên nhãn, M3 trên checkpoint có sẵn (val); dựng split nguồn; smoke test encoder thay thế | **Go/no-go:** M3 ≥ 2 trên checkpoint có sẵn. Nếu M3 ≈ 1, đổi khung theo §4 trước khi train |
| **2** (24–30/11) | RQ1 trên 3 fold; xác nhận 3 seed; chẩn đoán M2–M5 trên OOF và private test; H2b–c | split nguồn có support đủ |
| **3** (1–7/12) | H1b, H3, H4b, H5; tổng hợp bảng phân rã; nhóm tự xem 50 mẫu minh hoạ (định tính) | mọi contrast chính đã có CI |
| **4** (8–15/12) | Viết; đọc phản biện nội bộ; chuẩn bị phát hành code và manifest | |

## 10. Dàn ý bài

1. **Giới thiệu:** benchmark 4 lớp theo nguồn châm biếm; trần khoảng 0,40 (Vu et al., cuộc thi, nhóm); câu hỏi: mô hình có thật sự dùng ảnh?
2. **Liên quan:** benchmark bias (MMSD2.0, MMSD3.0), nhãn theo modality (SarcNet, MuSaG, ViMMSarc-Fine), học đa phương thức tham lam.
3. **ViMMSD và lập luận khả năng xác định nhãn** (§2, Hình 1 slide 6).
4. **Thiết lập** (§6).
5. **RQ1: mức phụ thuộc text** (Bảng M1–M5, Hình: flip rate tráo caption so với tráo ảnh theo lớp).
6. **RQ2: phụ thuộc text là shortcut** (baseline shortcut, train không thẻ, split nguồn).
7. **RQ3: vì sao** (H1, H3, H4, H5; bảng phân rã).
8. **Thảo luận và khuyến nghị:** báo cáo CI và macro-F1 3 lớp; split theo nguồn; kiểm tra tráo đối xứng; nhãn theo modality khi xây dataset.
9. **Giới hạn:** không có nhãn mới; nguồn suy ra tự động; encoder chủ yếu đóng băng; một benchmark.

## 11. Venue

Chưa xác nhận deadline; kiểm tra trong tuần 1.

| Venue | Kỷ yếu | Ghi chú |
|---|---|---|
| KSE 2027 | IEEE | cùng venue với ViMMSarc-Fine |
| SoICT 2026/2027 | ACM / Springer CCIS | |
| RIVF, MAPR | IEEE | |
| ACIIDS, IEA/AIE 2027 | Springer LNCS/LNAI | Vu et al. đăng ở IEA/AIE |
| ACL/EMNLP Student Research Workshop; *Insights from Negative Results* | ACL Anthology | kiểm tra index Scopus của từng năm |

## 12. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| M3 ≈ 1: fusion nhạy với ảnh ngang caption | điểm go/no-go tuần 1; đổi khung thành shortcut đơn modality (§4) |
| Image-only đạt điểm ngang text-only (0.370 so với 0.365) | báo cáo đối xứng; claim dựa trên hành vi của fusion (M2, M3), không trên điểm đơn modality |
| Độ phủ nguồn suy ra thấp | báo cáo độ phủ; H2c chỉ trên tập con; dựa thêm vào H2a, H2b |
| Trùng ý với ViMMSarc-Fine | đọc toàn văn tuần 1; nhấn vào lập luận khả năng xác định và ablation nguyên nhân |
| Đội nhất cuộc thi báo F1 khoảng 0,4475 (metric chưa xác minh) | xác minh; đặt vào khoảng tin cậy; không tuyên bố SOTA |
| Reviewer cho rằng thiếu nhãn người | nêu rõ giới hạn; dùng nhóm mâu thuẫn tự động và 50 mẫu tự xem làm minh hoạ, không làm số liệu chính |
| License ViMMSD | chỉ phát hành id, manifest, code; không phát hành lại ảnh hay caption |
| Thiếu quota hoặc thời gian | thứ tự cắt giảm: H1b encoder thay thế → H5a learning curve → X (cross-attention); giữ RQ1, H2, H3, H4 |

## 13. Ngoài phạm vi

Kiến trúc mới, CoT distillation, augmentation, gán nhãn lại có độ đồng thuận, fine-tune toàn bộ encoder, LLM làm mô hình chính. LLM chỉ xuất hiện ở phần liên quan và có thể là một dòng tham chiếu trong bảng benchmark.

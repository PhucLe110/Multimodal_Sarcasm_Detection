# Literature review: Multimodal Sarcasm Detection (MSD) cho ViMMSD

Ngày tổng hợp: 2026-10-29. Các bài được tải về `papers/`, tên file theo mẫu `<năm>_<tác giả đầu>_<tên ngắn>.pdf` (bảng ở mục 9).

Mục tiêu: tóm tắt các hướng nghiên cứu chính về phát hiện châm biếm từ ảnh + văn bản, các backbone đang dùng trong repo (PhoBERT, CLIP, Vintern, Qwen2.5-VL, PaddleOCR), rồi đối chiếu với những gì project đã làm để chọn thí nghiệm tiếp theo.

**Cách tìm.** Tìm trên arXiv, ACL Anthology, CVF Open Access, IJCAI với các từ khoá: *multimodal sarcasm detection*, *MMSD / MMSD2.0*, *incongruity*, *CLIP sarcasm*, *MLLM sarcasm*, *Vietnamese multimodal sarcasm*, *ViMMSD*. Chọn bài theo ba tiêu chí: (1) là mốc của lĩnh vực (dataset, baseline hay được so sánh), (2) là SOTA 2023–2026, (3) liên quan trực tiếp tới lựa chọn kỹ thuật của project (OCR, mô tả ảnh bằng VLM, mất cân bằng lớp, mất cân bằng modality). Số liệu trong review lấy từ abstract và phần kết quả của chính bài báo. Số nào không kiểm tra được thì không ghi.

---

## 1. Bài toán và đặc điểm của ViMMSD

ViMMSD (UIT Data Science Challenge 2024) gồm các bài đăng mạng xã hội tiếng Việt, mỗi bài có 1 ảnh và 1 caption, gán **4 nhãn**: `not-sarcasm`, `text-sarcasm`, `image-sarcasm`, `multi-sarcasm`. Khác hẳn MMSD/MMSD2.0 (nhị phân, tiếng Anh, Twitter), nhãn ở đây cho biết châm biếm **nằm ở modality nào**. Ý tưởng này gần với SarcNet (Yue et al., 2024), bộ dữ liệu gán nhãn riêng cho ảnh, cho text và cho cặp ảnh-text.

Phân bố nhãn đếm từ bản `phuclkz/vimmsd-private-dataset`:

| Split | Số mẫu | not-sarcasm | multi-sarcasm | image-sarcasm | text-sarcasm |
|---|---:|---:|---:|---:|---:|
| train | 10 805 | 6 062 (56,1%) | 4 224 (39,1%) | 442 (4,1%) | 77 (0,7%) |
| public test (val) | 1 413 | 442 (31,3%) | 942 (66,7%) | 28 (2,0%) | **1** (0,1%) |
| private test (test) | 1 504 | 458 (30,5%) | 999 (66,4%) | 40 (2,7%) | **7** (0,5%) |
| warmup | 101 | 53 | 43 | 4 | 1 |

Có hai điểm ảnh hưởng trực tiếp tới thiết kế thí nghiệm:

1. **Lệch phân bố giữa train và test (label shift).** Trong train, lớp đông nhất là `not-sarcasm`, còn ở public/private test thì `multi-sarcasm` chiếm khoảng 2/3. Model học prior của train sẽ đoán thiếu `multi-sarcasm` khi chạy trên test. Ngưỡng quyết định và class weight chọn bằng cách chia ngẫu nhiên từ train (`data.split=random`) sẽ không chuyển được sang test.
2. **`text-sarcasm` gần như không đo được.** Public test chỉ có 1 mẫu và private test có 7 mẫu thuộc lớp này. Nếu dùng macro-F1 trên public test để chọn checkpoint thì F1 của lớp này chỉ là 0 hoặc 1, nên chiếm tới 1/4 điểm macro-F1 và rất nhiễu. Cần báo cáo thêm F1 từng lớp và cân nhắc dùng metric phụ (weighted-F1, macro-F1 trên 3 lớp còn lại) để chọn model.

Bài duy nhất mình tìm được dùng trực tiếp ViMMSD là *Multimodal Sarcasm Detection on Vietnamese Social Media* (Vu, Hoang, Phan, Nguyen; IEA/AIE 2025, LNCS 15707, DOI 10.1007/978-981-96-8892-0_31). Theo abstract, bài đề xuất ba hướng tiếp cận dựa trên LLM pre-trained, kết hợp đặc trưng ảnh và text cho 4 nhãn trên. Bài này nằm sau paywall Springer nên chưa có trong `papers/` và chưa đọc được chi tiết.

---

## 2. Dataset

| Dataset | Ngôn ngữ / nguồn | Quy mô, nhãn | Ghi chú |
|---|---|---|---|
| MUStARD (Castro et al., 2019) | EN, phim truyền hình | video + audio + text, nhị phân | Dataset MSD đầu tiên, có ngữ cảnh hội thoại. Theo bài gốc, kết hợp đa phương thức giảm tương đối tới 12,9% lỗi (F-score) so với dùng một modality. |
| MMSD (Cai et al., 2019) | EN, Twitter | ảnh + text, nhị phân | Benchmark chuẩn của ảnh-text MSD. |
| MMSD2.0 (Qin et al., 2023) | EN, Twitter | sửa từ MMSD | Bỏ hashtag/emoji làm lộ nhãn, gán lại các mẫu âm tính bất hợp lý. |
| SarcNet (Yue et al., 2024) | EN + ZH | 3 335 cặp, >10 000 nhãn | Gán nhãn riêng cho ảnh, cho text và cho cả cặp. Gần với ViMMSD nhất về triết lý nhãn. |
| MMSD3.0 (Zhao et al., 2025) | EN, tweet + review Amazon | mỗi mẫu nhiều ảnh | Mở rộng sang trường hợp nhiều ảnh. |
| **ViMMSD** (UIT DSC 2024) | VI, mạng xã hội | ~13,7k mẫu, 4 nhãn | Xem mục 1. |

**Bài học từ MMSD2.0.** Trên MMSD, model chỉ dùng text (RoBERTa) đạt kết quả cao hơn model đa phương thức SOTA khi đó (HKE) khoảng 6,6 điểm. Nguyên nhân là hashtag và emoji phân bố lệch giữa hai lớp nên model học được shortcut. Sau khi làm sạch thành MMSD2.0, F1 của RoBERTa giảm 16,24 điểm. MiDRE (Jana et al., 2025) và MMSD3.0 cũng ghi nhận text-only mạnh hơn image-only rõ rệt. Với ViMMSD cũng cần kiểm tra xem hashtag, emoji hay teencode có lộ nhãn không. Pipeline của repo hiện chuẩn hoá emoji (`demojize`) và teencode, nên nên làm một ablation có và không có các tín hiệu này.

---

## 3. Các hướng phương pháp

### 3.1. Fusion sớm và attention (2019–2020)

- **HFM** (Cai et al., 2019) coi text, đặc trưng ảnh và *thuộc tính ảnh* (attribute dự đoán từ ảnh) là ba modality. Attribute được dùng để khởi tạo BiLSTM cho text, sau đó fusion phân cấp. Ý tưởng "đổi ảnh thành từ" để làm cầu nối sang text là tiền thân của cách dùng OCR và caption sau này.
- **D&R Net** (Xu et al., 2020) tách đặc trưng thành phần chung và phần khác biệt giữa ảnh và text (decomposition), đồng thời mô hình hoá liên kết ngữ nghĩa (relation).
- **Pan et al. (2020)** dùng BERT, inter-modality attention để bắt mâu thuẫn giữa ảnh và text, và co-attention để bắt mâu thuẫn *bên trong* text.

Điểm chung của nhóm này là châm biếm được xem như một dạng **incongruity** (mâu thuẫn), và mâu thuẫn có thể nằm giữa các modality hoặc bên trong một modality. Quan niệm này khớp với cách chia `text-sarcasm` / `image-sarcasm` / `multi-sarcasm` của ViMMSD.

### 3.2. Graph và mô hình hoá incongruity (2021–2024)

- **CMGCN** (Liang et al., 2022): phát hiện object trong ảnh kèm mô tả, dùng mô tả object làm cầu nối để dựng một graph liên kết object với từ trong text cho từng mẫu, rồi chạy GCN.
- **HKE** (Liu et al., 2022): mô hình hoá congruity theo hai cấp. Cấp atomic dùng cross-attention, cấp composition dùng GNN. Bài còn đưa thêm **tri thức ngoài như caption ảnh**. Đây là một trong những bài đầu cho thấy dùng caption ảnh làm modality phụ có lợi.
- **DIP** (Wen et al., CVPR 2023) có hai nhánh. Nhánh factual dùng channel reweighting và một phân phối Gaussian sinh từ memory bank để mô hình hoá độ tương đồng bất định. Nhánh affective dùng siamese layer và contrastive loss dựa trên polarity cảm xúc.
- **MICL** (Guo et al., 2024) học incongruity từ ba góc nhìn (token–patch, entity–object, sentiment) bằng contrastive learning. Bài chỉ ra rằng các model MSD dựa vào *spurious correlation*: chúng thường bỏ qua ảnh mà vẫn đoán đúng.
- **Dual generative processes** (Ma et al., IJCAI 2024) đưa khái niệm modality mạnh và yếu vào model, ước lượng đóng góp của từng modality theo khả năng dự đoán của nó rồi tự điều chỉnh trọng số.

### 3.3. Backbone CLIP (2023–2024)

- **Multi-view CLIP** (Qin et al., 2023, cùng bài với MMSD2.0) dùng CLIP để mã hoá cả text và ảnh, rồi kết hợp ba view: text, image và text-image interaction. Model đạt Acc 85,64% trên MMSD2.0. Ablation cho thấy bỏ view nào cũng làm giảm kết quả, và bỏ view tương tác giảm nhiều nhất (–3,2 điểm Acc).
- **InterCLIP-MEP** (Chen et al., 2024) đưa thông tin của modality kia vào thẳng trong encoder (Interactive CLIP), với số tham số train được ít hơn khoảng 20,6 lần so với SOTA trước đó. Bài thêm bộ nhớ hai kênh dùng như một classifier phi tham số lúc test. Trên MMSD2.0, model tăng 1,08 điểm Acc và 1,51 điểm F1. Khi phân phối dữ liệu thay đổi (distribution shift), Acc đạt 73,96%, cao hơn bản không có memory gần 10 điểm.

Các bài này cho thấy **CLIP đóng băng (frozen) cùng một head nhẹ đã là baseline mạnh**. Nhưng CLIP gốc chỉ hiểu tiếng Anh, nên nhánh text tiếng Việt phải dùng PhoBERT. Hệ quả là không còn tận dụng được không gian embedding chung giữa ảnh và text của CLIP. Đây là khác biệt lớn so với Multi-view CLIP. Có thể thử CLIP đa ngôn ngữ, hoặc dùng CLIP cho mô tả ảnh (tiếng Anh) để so khớp với ảnh.

### 3.4. Mô tả ảnh do MLLM sinh ra làm modality phụ (2024–2026)

- **GDCNet** (Zhang et al., 2026) dùng MLLM sinh **mô tả khách quan** của ảnh làm "mốc ngữ nghĩa", rồi tính độ lệch ngữ nghĩa và cảm xúc giữa mô tả và caption gốc, cùng độ khớp giữa ảnh và text. Các đặc trưng độ lệch này được fuse qua cổng (gated fusion) với vector ảnh và text. Model đạt SOTA trên MMSD2.0, và bỏ module tính độ lệch thì Acc giảm 2,96 điểm, F1 giảm 4,15 điểm. Bài cũng chỉ ra rằng nếu để LLM tự sinh "gợi ý châm biếm" thì kết quả nhiễu vì tính chủ quan, còn mô tả khách quan thì ổn định hơn.
- **MiDRE** (Jana et al., 2025) có một expert suy luận nội tại (incongruity trong cặp ảnh-text) và một expert dùng rationale do LVLM sinh bằng chain-of-thought. Một cổng thích nghi quyết định khi nào tin vào rationale, để giảm rủi ro LVLM bịa nội dung (hallucination).

Đây là hướng **gần nhất với project**. Repo đã sinh OCR (PaddleOCR / VietOCR / Vintern) và mô tả ảnh (Qwen2.5-VL-3B / Vintern-1B) cho mọi ảnh, rồi đưa vào ba cách: ghép thành segment 2 của PhoBERT, `text_aux`, và `multistream`. Thiết kế `multistream` (đặc trưng tương tác `t*s` và `|t-s|`, cờ có/không, modality dropout) giống tinh thần GDCNet: đo độ lệch giữa caption và nội dung ảnh, chứ không chỉ nối vector lại.

### 3.5. MLLM zero-shot và few-shot (2025)

- **Commander-GPT** (Zhang et al., 2025) chia bài toán thành 6 sub-task, có một "commander" phân từng sub-task cho MLLM phù hợp nhất rồi tổng hợp kết quả. Theo bài, F1 trên MMSD/MMSD2.0 tăng 19,3 điểm so với các cách prompting khác mà không cần fine-tune.
- **IRONIC** (Ramakrishnan et al., COLM 2025) là in-context learning dựa trên *coherence relations* giữa ảnh và text, đạt SOTA trong thiết lập zero-shot.
- **Basnet et al. (2025)** đánh giá 7 VLM mã nguồn mở (BLIP-2, InstructBLIP, OpenFlamingo, LLaVA, PaliGemma, Gemma 3, Qwen-VL) trên Muse, MMSD2.0 và SarcNet. Kết luận: các model chỉ đạt mức trung bình ở bài toán nhị phân và chưa tự sinh được lời giải thích tốt nếu không fine-tune.

Với ViMMSD có 4 nhãn tiếng Việt và hai lớp rất hiếm, khó kỳ vọng MLLM zero-shot thắng model đã fine-tune. Tuy vậy, MLLM vẫn dùng tốt làm **bộ sinh đặc trưng** như mục 3.4, hoặc làm một baseline đối chiếu (notebook 07 của repo đang so sánh OCR/LLM).

### 3.6. Độ bền vững và độ tin cậy của từng modality (2025–2026)

- **URMF** (Wang et al., 2026) biểu diễn text, ảnh và tương tác như các phân phối Gaussian để ước lượng độ bất định của từng modality, rồi giảm trọng số modality nhiễu khi fuse. Model được train với information bottleneck và contrastive loss có tính đến độ bất định.
- **MMSD3.0 / CIRM** (Zhao et al., 2025) xử lý trường hợp mỗi mẫu có nhiều ảnh. Với ViMMSD (1 ảnh mỗi mẫu) bài chủ yếu có giá trị tham khảo về benchmark.

---

## 4. Backbone và công cụ tiền xử lý đang dùng trong repo

| Thành phần | Bài | Vai trò trong project |
|---|---|---|
| PhoBERT (Nguyen & Nguyen, 2020) | LM đơn ngữ tiếng Việt đầu tiên quy mô lớn, cần tách từ (VnCoreNLP) | Encoder text chính (`vinai/phobert-base`, `word_segment: vncorenlp`). |
| ViSoBERT (Nguyen et al., 2023) | LM tiếng Việt pre-train trên mạng xã hội (kiến trúc XLM-R), vượt PhoBERT ở nhiều tác vụ mạng xã hội | Phương án thay PhoBERT (`word_segment: none`). Caption ViMMSD là văn mạng xã hội nên đáng thử. Chỉ dùng cho mục đích nghiên cứu. |
| CLIP (Radford et al., 2021) | Encoder ảnh-text học tương phản trên 400 triệu cặp | Encoder ảnh (`openai/clip-vit-base-patch32`). |
| Vintern-1B (Doan et al., 2024) | MLLM 1B tiếng Việt (InternViT-300M + Qwen2-0.5B), mạnh về OCR và VQA tiếng Việt | OCR / mô tả ảnh bằng tiếng Việt. |
| Qwen2.5-VL (Qwen Team, 2025) | VLM có độ phân giải động, mạnh về đọc tài liệu và định vị | Sinh mô tả ảnh (3B-Instruct). |
| PP-OCR (Du et al., 2020) | Hệ OCR nhẹ của PaddleOCR | Phát hiện vùng chữ. Nhận dạng chữ tiếng Việt kết hợp VietOCR. |
| Focal loss (Lin et al., 2017) | Giảm trọng số mẫu dễ, tập trung vào mẫu khó | `loss.name: focal` (`fusion_cross_attn_focal*`). |
| OGM-GE (Peng et al., 2022) | Điều chỉnh gradient từng modality theo đóng góp của nó, chống tình trạng một modality lấn át | Chưa dùng. Là phương án mạnh hơn modality dropout. |

---

## 5. Đối chiếu với project

| Vấn đề | Tài liệu nói gì | Project đang làm | Nên làm thêm |
|---|---|---|---|
| Text lấn át ảnh | MMSD2.0, MICL, MiDRE: text-only rất mạnh, model bỏ qua ảnh | `text_only*`, `image_only*` làm baseline đơn modality. Modality dropout và dropout trên vector ảnh trong `fusion_concat_frozen_reg` và `multistream` | So sánh text-only với fusion **theo từng lớp**, nhất là `image-sarcasm`. Thử OGM-GE, hoặc loss phụ riêng cho từng nhánh (unimodal head). |
| Shortcut từ hashtag / emoji | MMSD2.0 | `demojize`, chuẩn hoá teencode | Ablation có và không có hashtag/emoji. Kiểm tra tần suất hashtag theo nhãn. |
| Incongruity giữa ảnh và text | Toàn bộ mục 3.1–3.4 | `cross_attn`, các đặc trưng tương tác `t*s` và `\|t-s\|` trong `multistream` | Thêm nhánh độ lệch cảm xúc kiểu DIP/GDCNet giữa caption và mô tả ảnh. |
| Đưa ảnh về dạng chữ | HFM (attribute), HKE (caption), GDCNet, MiDRE (mô tả/rationale từ MLLM) | OCR + mô tả VLM, ghép segment / `text_aux` / `multistream` (4 config ablation) | Prompt VLM sinh mô tả **khách quan** (GDCNet cho thấy ổn định hơn gợi ý châm biếm). Đo lượng hallucination trên một mẫu nhỏ. |
| Mất cân bằng lớp | Focal loss. SarcNet, ViMMSD có lớp theo modality rất hiếm | `class_weight` (inverse / sqrt_inverse / effective), focal, weighted sampler | Hiệu chỉnh theo label shift: điều chỉnh logit theo prior của val (logit adjustment). Lưu ý `text-sarcasm` có 1 mẫu trong val. |
| Không gian embedding chung | Multi-view CLIP, InterCLIP-MEP dùng chung CLIP cho ảnh và text | PhoBERT (VI) và CLIP (EN) tách rời | Thử CLIP đa ngôn ngữ, hoặc tính độ tương đồng CLIP giữa ảnh và mô tả ảnh (tiếng Anh) làm đặc trưng. |
| Model chỉ chạy được bằng prompting | Commander-GPT, IRONIC, Basnet et al. | Notebook 07 so sánh OCR/LLM | Một baseline zero-shot (Qwen2.5-VL / Vintern) với 4 nhãn để đặt mốc tham chiếu trong báo cáo. |

---

## 6. Đề xuất thứ tự thí nghiệm

1. **Chốt cách đánh giá trước.** Với `data.split=official`, báo cáo macro-F1, F1 từng lớp và confusion matrix trên private test. Chọn checkpoint theo một metric ít nhiễu hơn macro-F1 4 lớp trên public test (vì lớp `text-sarcasm` chỉ có 1 mẫu trong val).
2. **Đo ảnh hưởng của label shift.** Chạy lại baseline tốt nhất, có và không có logit adjustment theo prior của val. Đây là thay đổi rẻ, có thể tác động lớn khi `multi-sarcasm` chiếm 66% test.
3. **Ablation các modality phụ.** Chạy 4 config `fusion_multistream_*` (caption + OCR + mô tả, có/không CLIP) trên split mới.
4. **Thử ViSoBERT thay PhoBERT**, giữ nguyên head tốt nhất.
5. **Thử đặc trưng độ lệch kiểu GDCNet**: độ lệch ngữ nghĩa và cảm xúc giữa caption và mô tả ảnh, đưa vào head qua cổng.

---

## 7. Hạn chế của review này

- Chưa đọc được toàn văn bài ViMMSD ở IEA/AIE 2025 (paywall). Nếu nhóm truy cập được qua thư viện trường thì nên bổ sung vào `papers/`.
- Chưa có toàn văn MILNet (Qiao et al., AAAI 2023) và CofiPara, hai bài được nhiều bài khác trích dẫn.
- Các số liệu SOTA trên MMSD/MMSD2.0 không so sánh trực tiếp được với ViMMSD (khác ngôn ngữ, khác số lớp). Bảng trong review chỉ dùng để chọn hướng tiếp cận, không dùng để đặt mục tiêu điểm số.
- Repo GitHub của các đội thi DSC 2024 (ví dụ `BFCmath/DataScienceChallenge-UIT2024`, `dinhthienan33/DSC2024_WEBUFF`) chưa được đọc kỹ. Nên xem lời giải của họ trước khi chốt pipeline.

---

## 8. Tài liệu tham khảo

| File trong `papers/` | Trích dẫn | Nguồn |
|---|---|---|
| `2017_lin_focal_loss_iccv.pdf` | Lin, Goyal, Girshick, He, Dollár. *Focal Loss for Dense Object Detection.* ICCV 2017. | [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) |
| `2019_cai_hfm_acl.pdf` | Cai, Cai, Wan. *Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion Model.* ACL 2019. | [P19-1239](https://aclanthology.org/P19-1239/) |
| `2019_castro_mustard_acl.pdf` | Castro et al. *Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper).* ACL 2019. | [arXiv:1906.01815](https://arxiv.org/abs/1906.01815) |
| `2020_du_pp_ocr.pdf` | Du et al. *PP-OCR: A Practical Ultra Lightweight OCR System.* 2020. | [arXiv:2009.09941](https://arxiv.org/abs/2009.09941) |
| `2020_nguyen_phobert_findings.pdf` | Nguyen, Nguyen. *PhoBERT: Pre-trained language models for Vietnamese.* Findings of EMNLP 2020. | [arXiv:2003.00744](https://arxiv.org/abs/2003.00744) |
| `2020_pan_intra_inter_incongruity_findings.pdf` | Pan et al. *Modeling Intra and Inter-modality Incongruity for Multi-Modal Sarcasm Detection.* Findings of EMNLP 2020. | [2020.findings-emnlp.124](https://aclanthology.org/2020.findings-emnlp.124/) |
| `2020_xu_dnr_net_acl.pdf` | Xu, Zeng, Mao. *Reasoning with Multimodal Sarcastic Tweets via Modeling Cross-Modality Contrast and Semantic Association.* ACL 2020. | [2020.acl-main.349](https://aclanthology.org/2020.acl-main.349/) |
| `2021_radford_clip_icml.pdf` | Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| `2022_liang_cmgcn_acl.pdf` | Liang et al. *Multi-Modal Sarcasm Detection via Cross-Modal Graph Convolutional Network.* ACL 2022. | [2022.acl-long.124](https://aclanthology.org/2022.acl-long.124/) |
| `2022_liu_hke_emnlp.pdf` | Liu, Wang, Li. *Towards Multi-Modal Sarcasm Detection via Hierarchical Congruity Modeling with Knowledge Enhancement.* EMNLP 2022. | [arXiv:2210.03501](https://arxiv.org/abs/2210.03501) |
| `2022_peng_ogm_ge_cvpr.pdf` | Peng et al. *Balanced Multimodal Learning via On-the-fly Gradient Modulation.* CVPR 2022. | [arXiv:2203.15332](https://arxiv.org/abs/2203.15332) |
| `2023_nguyen_visobert_emnlp.pdf` | Nguyen et al. *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing.* EMNLP 2023. | [arXiv:2310.11166](https://arxiv.org/abs/2310.11166) |
| `2023_qin_mmsd2_findings.pdf` | Qin et al. *MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System.* Findings of ACL 2023. | [arXiv:2307.07135](https://arxiv.org/abs/2307.07135) |
| `2023_wen_dip_cvpr.pdf` | Wen, Jia, Yang. *DIP: Dual Incongruity Perceiving Network for Sarcasm Detection.* CVPR 2023. | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_DIP_Dual_Incongruity_Perceiving_Network_for_Sarcasm_Detection_CVPR_2023_paper.pdf) |
| `2024_chen_interclip_mep.pdf` | Chen et al. *InterCLIP-MEP: Interactive CLIP and Memory-Enhanced Predictor for Multi-modal Sarcasm Detection.* 2024. | [arXiv:2406.16464](https://arxiv.org/abs/2406.16464) |
| `2024_doan_vintern_1b.pdf` | Doan et al. *Vintern-1B: An Efficient Multimodal Large Language Model for Vietnamese.* 2024. | [arXiv:2408.12480](https://arxiv.org/abs/2408.12480) |
| `2024_farabi_survey_ijcai.pdf` | Farabi, Ranasinghe, Kanojia, Kong, Zampieri. *A Survey of Multimodal Sarcasm Detection.* IJCAI 2024. | [arXiv:2410.18882](https://arxiv.org/abs/2410.18882) |
| `2024_guo_multiview_incongruity.pdf` | Guo et al. *Multi-View Incongruity Learning for Multimodal Sarcasm Detection* (MICL). 2024. | [arXiv:2412.00756](https://arxiv.org/abs/2412.00756) |
| `2024_ma_dual_generative_ijcai.pdf` | Ma, He, Wang, Jin, Ge, Wang. *Multi-Modal Sarcasm Detection Based on Dual Generative Processes.* IJCAI 2024. | [IJCAI 2024/252](https://www.ijcai.org/proceedings/2024/252) |
| `2024_yue_sarcnet_lrec.pdf` | Yue, Shi, Mao, Hu, Cambria. *SarcNet: A Multilingual Multimodal Sarcasm Detection Dataset.* LREC-COLING 2024. | [2024.lrec-main.1248](https://aclanthology.org/2024.lrec-main.1248/) |
| `2025_basnet_open_vlms_sarcasm.pdf` | Basnet, Farabi, Ranasinghe, Kanojia, Zampieri. *Evaluating Open-Source Vision-Language Models for Multimodal Sarcasm Detection.* 2025. | [arXiv:2510.11852](https://arxiv.org/abs/2510.11852) |
| `2025_jana_midre.pdf` | Jana, Kundu, Singh. *Think Twice Before You Judge: Mixture of Dual Reasoning Experts for Multimodal Sarcasm Detection.* 2025. | [arXiv:2507.04458](https://arxiv.org/abs/2507.04458) |
| `2025_qwen_team_qwen2_5_vl.pdf` | Qwen Team. *Qwen2.5-VL Technical Report.* 2025. | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) |
| `2025_ramakrishnan_ironic_colm.pdf` | Ramakrishnan, Ramakrishnan, Lee. *IRONIC: Coherence-Aware Reasoning Chains for Multi-Modal Sarcasm Detection.* COLM 2025. | [arXiv:2505.16258](https://arxiv.org/abs/2505.16258) |
| `2025_zhang_commander_gpt.pdf` | Zhang, Zou, Wang, Qin. *Commander-GPT: Fully Unleashing the Sarcasm Detection Capability of Multi-Modal Large Language Models.* 2025. | [arXiv:2503.18681](https://arxiv.org/abs/2503.18681) |
| `2025_zhao_mmsd3.pdf` | Zhao et al. *MMSD3.0: A Multi-Image Benchmark for Real-World Multimodal Sarcasm Detection.* 2025. | [arXiv:2510.23299](https://arxiv.org/abs/2510.23299) |
| `2026_wang_urmf.pdf` | Wang et al. *URMF: Uncertainty-aware Robust Multimodal Fusion for Multimodal Sarcasm Detection.* 2026. | [arXiv:2604.06728](https://arxiv.org/abs/2604.06728) |
| `2026_zhang_gdcnet.pdf` | Zhang, Lian, Yu, Xu, Ao. *GDCNet: Generative Discrepancy Comparison Network for Multimodal Sarcasm Detection.* 2026. | [arXiv:2601.20618](https://arxiv.org/abs/2601.20618) |
| (chưa tải, paywall) | Vu, Hoang, Phan, Nguyen. *Multimodal Sarcasm Detection on Vietnamese Social Media.* IEA/AIE 2025, LNCS 15707. | [Springer](https://link.springer.com/chapter/10.1007/978-981-96-8892-0_31) |

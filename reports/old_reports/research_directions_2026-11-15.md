# Hướng nghiên cứu tiếp theo cho ViMMSD

Ngày rà soát: 15/11/2026. Đây là rà soát có mục tiêu để lựa chọn hướng nghiên cứu, không phải systematic review toàn diện. Chưa chạy thí nghiệm mới. Chi phí và lịch dưới đây là ước lượng lập kế hoạch, chưa benchmark.

## 1. Khuyến nghị

Ưu tiên hai tuyến: (A) kiểm tra chất lượng nhãn và khả năng khái quát; (B) kiểm tra giới hạn của frozen encoders rồi mới thử fusion thích nghi. Không mở lại CoT distillation đại trà và không tiếp tục tuning trên private test đã xem.

Câu hỏi trung tâm đề xuất: **Khi nào thông tin ảnh thực sự giúp phát hiện mỉa mai tiếng Việt, và lợi ích đó còn tồn tại khi thay đổi nguồn dữ liệu hoặc kiểm soát chất lượng nhãn không?**

Đây là câu hỏi có thể trả lời cả khi mô hình mới không tăng F1, phù hợp với hạ tầng và kết quả hiện có.

## 2. Bằng chứng từ project

Nguồn: [báo cáo thực nghiệm](empirical_analysis.md), [hướng dẫn nhãn](../docs/annotation_guideline.md), các module `models/freezing.py`, `models/fusion.py`, `analysis/interventions.py`, `data/splits.py`.

- Nghiên cứu mới chạy 60 lượt train với encoder đóng băng. C + B1 đạt private macro-F1 0,380 ± 0,007; mọi contrast cuối chưa đạt quy tắc bằng chứng đã chốt.
- F1 text bằng 0; F1 image tốt nhất 0,057 trên private test. Validation chỉ có 1 text và 28 image, không thích hợp để chọn phương pháp cho lớp hiếm.
- OCR/mô tả không có lợi ích phát hiện được trong cấu hình đã thử. Điều này không chứng minh mọi OCR hay mọi cách fusion đều vô ích.
- Thay ảnh làm đổi khoảng 20–22% dự đoán của concat. Đây là độ nhạy dưới input bị phá, không phải bằng chứng suy luận đúng.
- Ba trong năm ví dụ hướng dẫn tìm thấy khác nhãn dataset. Đây là tín hiệu cần audit, không phải ước lượng tỷ lệ nhiễu toàn tập.
- Emoji, hashtag, dấu câu có thể là bằng chứng ngữ dụng hợp lệ theo hướng dẫn. Chỉ nên gọi một cue là shortcut khi có bằng chứng về phụ thuộc gây hại hoặc không khái quát; không đồng nhất mọi cue bề mặt với nhiễu.
- Code đã hỗ trợ mở block cuối, cross-attention, grouped folds, can thiệp input và thống kê. Chưa có phép so sánh fine-tuning được kiểm soát trong nghiên cứu frozen hiện tại.

## 3. Tổng hợp literature theo vấn đề

| Vấn đề | Nguồn gốc đã kiểm tra | Bài học áp dụng | Giới hạn chuyển giao |
|---|---|---|---|
| Chất lượng benchmark | MMSD2.0, Findings ACL 2023 [1] | Sửa cue lộ nhãn và nhãn không hợp lý có thể thay đổi đánh giá mô hình | Tiếng Anh, nhị phân; không suy ra ViMMSD có cùng cơ chế nhiễu |
| Nhãn từng modality | SarcNet, LREC-COLING 2024 [2] | Gán nhãn text, image, pair riêng để đánh giá đúng vai trò input | Anh/Trung; không tự ánh xạ bốn nhãn ViMMSD thành hai bit |
| Khái quát và text bias | Debiasing MSD, AAAI 2024 [3]; MICL, COLING 2025 [4] | Cần đánh giá OOD và các tương tác có ích, không chỉ fusion phức tạp | Contrastive loss/hard negatives có thể sai nếu tự giả định nhãn sau khi sửa input |
| Chất lượng giải thích VLM | Basnet et al., ICDMW 2025 [5] | Nhận diện nhị phân và sinh giải thích tốt là hai năng lực khác nhau | Không phải thử nghiệm trực tiếp trên ViMMSD |
| External reasoning và gate | MiDRE [6], arXiv v2 2025 | Chọn lọc reasoning có thể hợp lý hơn luôn ghép vào | Không phải chứng minh cho distillation trong repo; chi phí và teacher khác |
| Fusion theo độ tin cậy | URMF [7], arXiv, ghi nhận accepted ICIC 2026 | Độ tin cậy của từng nguồn có thể khác theo mẫu | Kiến trúc nhiều loss; không nên tái lập toàn bộ ngay |
| Căn chỉnh tiếng Việt–ảnh | ViCLIP-OT [8], preprint 2026; checkpoint chính chủ [9] | Có backbone được huấn luyện cho ảnh–chữ tiếng Việt | Kết quả retrieval không chứng minh khả năng nhận biết mỉa mai |
| Dữ liệu mới cùng ngôn ngữ | ViMMSarc-Fine, thông báo UIT 28/10/2026 [10] | Cơ hội đánh giá nhãn T/I/M và chuyển nền tảng | Mới kiểm tra thông báo và chương trình KSE [11], chưa toàn văn hay schema tải về |

ViMMSarc-Fine được UIT giới thiệu gồm 7.355 bài Facebook/Threads, nhãn T/I/M và OCR. Gói dẫn xuất ẩn danh được mô tả là truy cập có kiểm soát, nội dung gốc không phân phối lại. Chưa xác minh gói có đủ ảnh/embedding để chạy model của project. Không coi đây là dataset mở đã sẵn sàng sử dụng.

## 4. Hướng A — Audit nhãn và tập kiểm tra cần ảnh

**Ưu tiên cao nhất; chi phí GPU thấp, cần công sức người gán nhãn.**

Giả thuyết: một phần lỗi lớp hiếm đến từ ranh giới nhãn hoặc ngữ cảnh cần ảnh, bên cạnh mất cân bằng.

1. Hoàn thành 44 mẫu phân tích định tính và 40 mẫu audit logo đã lên danh sách.
2. Pilot 40–60 mẫu train với hai người gán nhãn độc lập; chưa cho xem dự đoán của model.
3. Nếu rubric dùng được, mở rộng khoảng 200–400 mẫu từ train: tách một tầng lấy mẫu xác suất để mô tả bất đồng, và các tầng chủ đích tập trung lớp hiếm/caption trùng khác nhãn/model bất đồng để tìm cơ chế lỗi. Lưu xác suất chọn mẫu; không dùng tỷ lệ lỗi của tầng chủ đích để ước lượng nhiễu toàn bộ dataset.
4. Thu riêng: sarcasm trong caption đứng riêng, ảnh đứng riêng (bao gồm chữ trong ảnh), cả cặp; bằng chứng cần để quyết định; mức chắc chắn; nhãn không xác định. Trình bày từng modality trong lượt riêng, có kiểm soát thứ tự để giảm ảnh hưởng khi đã xem cả cặp.
5. Báo cáo agreement và support theo từng câu hỏi/lớp; giữ cả nhãn ban đầu và nhãn adjudicated. Không lấy đồng thuận đa số làm bằng chứng chân lý tuyệt đối.

Đầu ra: taxonomy lỗi, manifest audit, rubric có ví dụ, tập thách thức cần ảnh. Nếu lấy mẫu từ train, phải loại cả group liên quan khỏi train của mọi model đánh giá trên tập đó; checkpoint đã thấy các mẫu này chỉ dùng mô tả, không đánh giá tổng quát hóa. Muốn có holdout độc lập mạnh hơn cần dữ liệu mới.

Điểm dừng: nếu người gán nhãn vẫn bất đồng lớn sau pilot, tập trung vào giới hạn nhãn và đánh giá nhị phân bổ sung; không ép thành nhãn bốn lớp sạch giả tạo. Phân loại nhị phân không được thay thế âm thầm nhiệm vụ bốn lớp.

## 5. Hướng B — Kiểm tra giới hạn đóng băng encoder

**Ưu tiên cao; khả thi nhất nếu mục tiêu là cải thiện mô hình trong code hiện tại.**

Câu hỏi: fusion ít hiệu quả vì thiếu thông tin ảnh, hay biểu diễn pretrained chưa thích nghi với meme tiếng Việt?

Ma trận screening trên 3 grouped folds, seed 42, class weighting B1 giữ cố định:

| ID | Input | PhoBERT | CLIP |
|---|---|---|---|
| B-T0 | Caption | Frozen | Không dùng |
| B-C0 | Caption + ảnh | Frozen | Frozen |
| B-T1 | Caption | Mở 2 block cuối | Không dùng |
| B-C1 | Caption + ảnh | Mở 2 block cuối | Frozen |
| B-C2 | Caption + ảnh | Mở 2 block cuối | Mở 2 block cuối |

Tổng 15 run screening; nếu cần tiết kiệm, chưa chạy B-C2. Giữ preprocessing, nhóm/fold, số update tối đa, cách chọn checkpoint và ngân sách tuning nhất quán; không yêu cầu cùng learning rate giữa head và encoder. Chỉ xác nhận phương án tốt nhất và đối chứng liên quan bằng seeds 1, 2, 42. Nếu tái sử dụng run, audit resolved config trước.

Đọc kết quả: B-C1 − B-T1 đo lợi ích thêm ảnh sau khi text được thích nghi; B-C2 − B-C1 đo lợi ích mở vision encoder. Chạy trên 12 GB là giả định lập kế hoạch từ server đã ghi trong báo cáo, phải smoke test VRAM; giảm microbatch và dùng gradient accumulation khi cần. Chưa có số GPU-hour đã đo.

Code đã có `unfreeze_last_n`; không cần triển khai LoRA ngay. Chọn theo pooled OOF macro-F1 bốn lớp, theo dõi F1/PR lớp hiếm. Đăng ký trước mức hiệu ứng thực tế, ví dụ +0,01 macro-F1 là mục tiêu, không phải lời hứa tăng điểm. Nếu chỉ text fine-tuning giúp, đây vẫn là kết quả phân biệt được giả thuyết.

## 6. Hướng C — Backbone ảnh–chữ tiếng Việt có căn chỉnh

**Ưu tiên trung bình–cao; có checkpoint để thử.**

ViCLIP-OT có model card chính chủ công bố 221M tham số, text Vietnamese-SBERT và image DINOv3 ViT-B/16 [9]. Dùng cả hai tower/projector phù hợp của checkpoint để lấy embedding đã căn chỉnh; thay riêng vision encoder không kiểm tra được giả thuyết căn chỉnh tiếng Việt.

Pilot frozen feature extraction rồi train cùng loại head bốn lớp, so sánh text-only của backbone đó và hai modality của nó với PhoBERT+CLIP. Có thể thêm similarity, tích và độ chênh embedding làm feature, nhưng cosine thấp không tự động đồng nghĩa sarcasm.

Đây là phép so sánh thay cả backbone; không được quy mọi chênh lệch cho alignment vì dữ liệu pretrain, pooling và vision architecture cũng khác. Muốn kết luận cơ chế phải có ablation đối chứng bổ sung. Kiểm tra processor và adapter: wrapper `ImageEncoder` hiện tại không thể mặc định tương thích với custom dual encoder này. Tách môi trường nếu dependency khác.

Điểm dừng: nếu ba fold không cho tín hiệu tốt hơn baseline và lợi ích lớp hiếm không ổn định, không chuyển sang pretrain một CLIP tiếng Việt từ đầu.

## 7. Hướng D — Fusion có gate và nhánh hiệu chỉnh từ ảnh

**Ưu tiên sau khi hoàn thành đối chứng fine-tuning; đóng góp phương pháp khả thi.**

Quan sát trong project: ảnh sửa được một số lỗi text nhưng cũng làm hỏng nhiều mẫu; thêm mọi nguồn chưa tốt hơn concat. Đề xuất tối giản, lấy cảm hứng từ reliability-aware fusion [7]:

`logits = text_logits + sigmoid(g(features)) * image_text_residual`

Nhánh residual học từ feature text/ảnh được chiếu và tương tác của chúng. Gate chỉ dùng input hoặc đầu ra model tại suy luận, không dùng nhãn đúng/nhãn "ảnh cần thiết" lấy từ test. Khởi đầu với caption + CLIP; chỉ đưa OCR vào khi audit cho thấy OCR đáng tin và hữu ích. Không cần teacher.

Đối chứng tối thiểu: text-only, concat, residual gate cố định, residual gate học được; giữ encoder/budget giống nhau. Đo macro-F1, calibration nếu gate dùng confidence, số mẫu sửa/hỏng và độ nhạy với ảnh. Giá trị gate không phải bằng chứng giải thích nhân quả.

Rủi ro: gate có thể bỏ ảnh hoàn toàn hoặc học source cue. Muốn khẳng định dùng ảnh tốt hơn, phải tốt hơn trên tập cần ảnh có nhãn được kiểm tra và/hoặc nguồn mới. Không tối ưu mục tiêu "đổi dự đoán nhiều hơn khi tráo ảnh".

## 8. Hướng E — Chuyển nguồn và counterfactual có gán nhãn

**Giá trị nghiên cứu cao, phụ thuộc dữ liệu/annotation.**

- Tạo source-held-out split nếu có metadata đủ tin cậy. Không tự coi hashtag là source ID hoàn chỉnh; bài không xác định nguồn để riêng. Duy trì group caption/ảnh xuyên suốt split và kiểm tra support lớp.
- Nếu truy cập được ViMMSarc-Fine: trước hết kiểm tra schema, overlap và input thực sự được cấp; định nghĩa nhiệm vụ chung, có thể là sarcasm của cả cặp dạng nhị phân. Không so trực tiếp macro-F1 bốn lớp ViMMSD với F1 nhị phân của dataset mới.
- Phân biệt transfer không thích nghi với few-shot adaptation; dữ liệu đích để tune phải tách khỏi test đích.
- Tạo pilot 50–100 cặp sửa caption/đổi ảnh có chủ đích, gồm cả sửa giữ nhãn và đổi nhãn; hai người xác nhận ý nghĩa và nhãn mới. Tách theo bài gốc, tránh cặp cùng gốc vào train và test. Bắt đầu ở train hoặc nguồn mới, không sửa test cũ để tune.
- Random image swapping hiện tại vẫn là disruption diagnostic. Không tự gán mọi cặp tráo ảnh là negative và không ép giữ nhãn cho mọi thay đổi ảnh.

Mục tiêu khả thi: kiểm tra độ chính xác trên cặp có nhãn mới, consistency của cặp giữ nhãn, khả năng thay đổi đúng trên cặp đổi nhãn, và worst-group performance. Các con số này bổ sung benchmark tự nhiên, không thay thế nó.

## 9. Các hướng chưa nên ưu tiên

- Sinh CoT toàn tập hoặc distill lại từ teacher hiện tại: chất lượng teacher là nút thắt đã có bằng chứng. Chỉ mở lại khi pilot mù nhãn, có người chấm, cho thấy reasoning đáng tin trên các lớp khó.
- Chạy lại focal/sampling/stacking/kNN không thay đổi giả thuyết: project đã có kết quả hạn chế; không đáng mở grid lớn.
- Augmentation/back-translation đại trà: có thể mất chơi chữ và sắc thái. Có thể dùng sau audit, với duyệt và provenance parent/fold đầy đủ.
- Tái lập toàn bộ graph/causal/uncertainty architecture nhiều loss: khó xác định thành phần có ích với dữ liệu hiện tại.
- Tuyên bố nhãn multi tương đương T=1,I=1: SarcNet và ViMMSarc-Fine cho thấy cần nhãn cả cặp độc lập; dự án cũng đã ghi nhận multi có nhiều trường hợp.

## 10. Kế hoạch triển khai đề xuất

Ước lượng cho nhóm nhỏ, một GPU loại 12 GB và hai người có thể duyệt nhãn; chưa kiểm tra tài nguyên hiện tại.

| Giai đoạn | Công việc | Điều kiện đi tiếp |
|---|---|---|
| Tuần 1 | Audit pilot, hoàn tất 44/40 mẫu, chốt protocol và smoke test fine-tuning | Rubric dùng được; split không rò rỉ; run vừa VRAM |
| Tuần 2 | Screening B trên grouped folds; pilot C nếu đủ thời gian | Tín hiệu OOF ổn định hoặc xác định rõ kết quả âm |
| Tuần 3 | Xác nhận phương án và baseline ba seed; thử gate chỉ khi có động cơ từ lỗi | Hiệu ứng đủ lớn và không chỉ do một fold/seed |
| Tuần 4 | Đánh giá challenge/source holdout đã chốt; viết báo cáo | Tách rõ exploratory/confirmatory và hạn chế dữ liệu |

Nếu chỉ còn 1–2 tuần: làm A + phiên bản nhỏ của B, chưa thêm gate/backbone mới. Nếu mục tiêu bài báo: thêm E hoặc tập audit có annotation chất lượng; chỉ thêm một gate không đủ chứng minh novelty.

Protocol đợt mới: chọn mô hình trên train grouped OOF; chỉ dùng official public/private đã xem như historical benchmark/descriptive follow-up. Đóng băng holdout mới trước khi train hoặc điều chỉnh phương pháp. Khoảng tin cậy bootstrap theo ví dụ/nhóm không bao gồm toàn bộ biến thiên huấn luyện; báo riêng variability qua seed. Giữ macro-F1 bốn lớp và support; chỉ số nhị phân là phân tích bổ sung.

## 11. Search log và độ phủ

Nguồn tìm kiếm: web discovery giới hạn nguồn chính chủ, ACL Anthology, arXiv, trang nhà xuất bản, model card/repository của tác giả, UIT và chương trình KSE. Đã kiểm tra toàn văn có thể trích xuất của MMSD2.0, MICL, SarcNet, MiDRE và ViCLIP-OT ở các phần liên quan; các mục còn lại dựa trên abstract/model card/thông báo như đã ghi. Chưa kiểm định độc lập mã nguồn, pretrained weights hay số liệu của bài ngoài project. Không tuyên bố tìm hết literature hay xác lập SOTA.

Các chuỗi tìm kiếm đã sử dụng ngày 15/11/2026:

```text
site.aclanthology.org Vietnamese multimodal sarcasm detection ViMMSD 2025 2026
site.aclanthology.org multimodal sarcasm detection MMSD2.0 bias incongruity 2024 2025
site.arxiv.org multimodal sarcasm detection 2025 2026 counterfactual MiDRE
"Vietnamese" "Multimodal Sarcasm" ViMMSD
site.aclanthology.org SarcNet multilingual multimodal sarcasm dataset 2024
"sarcasm" "ViMMSD" 2026
"Think Twice Before You Judge" MiDRE
"ViMMSarc-Fine"
"Think Twice Before You Judge" sarcasm
"Evaluating Open-Source Vision-Language Models" sarcasm
multilingual CLIP Vietnamese ViCLIP image text alignment model
"ViMMSarc-Fine" github huggingface dataset
"ViCLIP-OT" github huggingface
```

## 12. Tài liệu và nguồn xác minh

1. Qin et al. (2023). *MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System*. Findings ACL. https://aclanthology.org/2023.findings-acl.689/
2. Yue et al. (2024). *SarcNet: A Multilingual Multimodal Sarcasm Detection Dataset*. LREC-COLING. https://aclanthology.org/2024.lrec-main.1248/
3. Jia, Xie, Jing (2024). *Debiasing Multimodal Sarcasm Detection with Contrastive Learning*. AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/29795
4. Guo et al. (2025). *Multi-View Incongruity Learning for Multimodal Sarcasm Detection*. COLING. https://aclanthology.org/2025.coling-main.119/
5. Basnet et al. (2025). *Evaluating Open-Source Vision-Language Models for Multimodal Sarcasm Detection*. ICDMW. https://doi.org/10.1109/ICDMW69685.2025.00172 ; preprint https://arxiv.org/abs/2510.11852
6. Jana, Kundu, Singh (2025). *Think Twice Before You Judge: Mixture of Dual Reasoning Experts for Multimodal Sarcasm Detection*. arXiv, bản cập nhật 29/10/2025. https://arxiv.org/abs/2507.04458
7. Wang et al. (2026). *URMF: Uncertainty-aware Robust Multimodal Fusion for Multimodal Sarcasm Detection*. arXiv, bản cập nhật 03/05/2026, trang ghi accepted ICIC 2026. https://arxiv.org/abs/2604.06728
8. Tran, Nguyen, Pham (2026). *ViCLIP-OT: The First Foundation Vision-Language Model for Vietnamese Image-Text Retrieval with Optimal Transport*. Preprint. https://arxiv.org/abs/2602.22678
9. ViCLIP-OT official model card và code: https://huggingface.co/minhnguyent546/ViCLIP-OT ; https://github.com/minhnguyent546/ViCLIP-OT
10. UIT (28/10/2026), thông báo ViMMSarc-Fine: https://www.uit.edu.vn/bai-viet/khi-chu-va-hinh-deu-khong-mang-y-cham-biem-nhung-ghep-lai-thi-co-sinh-vien-uit-nghien-cuu-bai-toan-nhan-dien-cham-biem-tren-mang-xa-hoi-bang-ai
11. KSE 2026, program listing, paper 35: https://kse2026.kse-conferences.org/program/
12. Vu et al. (2025). *Multimodal Sarcasm Detection on Vietnamese Social Media*. IEA/AIE. Trang nhà xuất bản và abstract, toàn văn giới hạn truy cập. https://doi.org/10.1007/978-981-96-8892-0_31

Phương pháp hỗ trợ: dùng skills `literature-review` và `scientific-critical-thinking` của Scientific Agent Skills. Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv, bản hiện tại v2 (02/09/2026). https://doi.org/10.48550/arXiv.2609.00065 . Đây là nguồn công cụ hỗ trợ quy trình, không phải bằng chứng cho hiệu quả mô hình sarcasm.

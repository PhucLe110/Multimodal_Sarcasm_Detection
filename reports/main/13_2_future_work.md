# 13.2. Hướng phát triển

- **Phase:** 13, Phân tích lỗi và Future work
- **Nguồn:** `reports/plan/research_directions_2026-11-15.md` (rà soát tài liệu có mục tiêu, 15/11/2026; chưa chạy thí nghiệm mới), các report Phase 3–13.1
- Chi phí và lịch dưới đây là ước lượng lập kế hoạch, chưa đo.

---

## 1. Câu hỏi trung tâm đề xuất

**Khi nào thông tin ảnh thực sự giúp phát hiện châm biếm tiếng Việt, và lợi ích đó còn không khi đổi nguồn dữ liệu hoặc kiểm soát chất lượng nhãn?**

Câu hỏi này trả lời được cả khi mô hình mới không tăng F1, và dùng lại được hạ tầng hiện có.

## 2. Điểm xuất phát

- 60 lượt train với encoder đóng băng (Phase 10); C + B1 đạt private test macro-F1 0,380 ± 0,007; không contrast cuối nào đạt luật bằng chứng đã chốt.
- F1 `text` = 0; F1 `image` tốt nhất 0,057 trên private test (benchmark chốt trước). Val chỉ có 1 `text` và 28 `image`, không dùng được để chọn phương pháp cho lớp hiếm.
- OCR/mô tả không có lợi ích đo được *trong cấu hình đã thử*; điều này không chứng minh mọi cách dùng OCR hay mọi cách fusion đều vô ích.
- Tráo ảnh đổi khoảng 20–22% dự đoán của concat: đo độ nhạy, không phải bằng chứng suy luận đúng.
- 3/5 ví dụ của hướng dẫn tìm thấy trong dữ liệu khác nhãn: tín hiệu cần audit nhãn, không phải ước lượng tỉ lệ nhiễu toàn tập.
- Emoji, hashtag, dấu câu có thể là bằng chứng ngữ dụng hợp lệ theo hướng dẫn; chỉ gọi là shortcut khi có bằng chứng phụ thuộc gây hại hoặc không tổng quát.
- Code đã hỗ trợ mở block cuối (`unfreeze_last_n`), cross-attention, fold theo nhóm, can thiệp đầu vào, thống kê.

## 3. Các hướng, theo thứ tự ưu tiên

### A. Audit nhãn và tập kiểm tra cần ảnh (ưu tiên cao nhất; ít GPU, cần người gán nhãn)

Giả thuyết: một phần lỗi lớp hiếm đến từ ranh giới nhãn hoặc từ các mẫu cần ảnh, không chỉ từ mất cân bằng.

1. Hoàn thành 44 mẫu định tính và 40 mẫu kiểm tra logo đã có danh sách.
2. Pilot 40–60 mẫu train, hai người gán độc lập, chưa xem dự đoán của mô hình.
3. Nếu rubric dùng được, mở rộng 200–400 mẫu: một tầng lấy mẫu xác suất (để mô tả bất đồng) và các tầng chủ đích (lớp hiếm, caption trùng khác nhãn, mô hình bất đồng); lưu xác suất chọn mẫu.
4. Thu riêng: caption đứng riêng có châm biếm không, ảnh đứng riêng (kể cả chữ) có không, cả cặp có không; bằng chứng; mức chắc chắn. Trình bày từng modality ở lượt riêng.
5. Báo agreement theo từng câu hỏi/lớp; giữ cả nhãn ban đầu và nhãn sau thống nhất.

Đầu ra: phân loại lỗi, rubric có ví dụ, tập thách thức "cần ảnh". Điểm dừng: nếu bất đồng vẫn lớn sau pilot, tập trung vào giới hạn của nhãn và thêm đánh giá nhị phân, không ép thành nhãn 4 lớp "sạch" giả tạo.

### B. Kiểm tra giới hạn của encoder đóng băng (ưu tiên cao; khả thi nhất để cải thiện mô hình)

Câu hỏi: fusion ít hiệu quả vì thiếu thông tin ảnh, hay vì biểu diễn pretrained chưa thích nghi với meme tiếng Việt?

| ID | Đầu vào | PhoBERT | CLIP |
|---|---|---|---|
| B-T0 | caption | đóng băng | – |
| B-C0 | caption + ảnh | đóng băng | đóng băng |
| B-T1 | caption | mở 2 block cuối | – |
| B-C1 | caption + ảnh | mở 2 block cuối | đóng băng |
| B-C2 | caption + ảnh | mở 2 block cuối | mở 2 block cuối |

3 fold nội bộ, seed 42, B1 cố định: 15 run screening; xác nhận phương án tốt nhất với seed 1, 2, 42. B-C1 − B-T1 đo lợi ích của ảnh khi text đã thích nghi; B-C2 − B-C1 đo lợi ích của mở vision encoder. Cần smoke test VRAM trên GPU 12 GB.

Lưu ý: Phase 4 cho thấy `ftlast` (mở 2 block cả hai encoder, CE) không giúp; hướng này kiểm tra lại có class weight, theo OOF, và tách từng encoder. Phase 3.1 cho thấy fine-tune toàn bộ PhoBERT hơn bản đóng băng khoảng 2 điểm trên private test.

### C. Backbone ảnh–chữ tiếng Việt đã căn chỉnh (ưu tiên trung bình–cao)

ViCLIP-OT (Tran et al., 2026, preprint): 221M tham số, text Vietnamese-SBERT, ảnh DINOv3 ViT-B/16, có model card chính chủ. Dùng cả hai tower đã căn chỉnh, trích feature đóng băng, train cùng head 4 lớp, so text-only và hai modality của backbone này với PhoBERT + CLIP. Đây là phép thay cả backbone: không quy mọi chênh lệch cho việc căn chỉnh tiếng Việt. Điểm dừng: 3 fold không cho tín hiệu tốt hơn thì không pretrain CLIP tiếng Việt từ đầu.

### D. Fusion có cổng và nhánh hiệu chỉnh từ ảnh (sau B; đóng góp phương pháp khả thi)

Quan sát: ảnh sửa được một số lỗi của caption nhưng cũng làm hỏng nhiều mẫu (94 so với 80). Đề xuất tối giản, lấy cảm hứng từ fusion có độ tin cậy (URMF, Wang et al., 2026):

```
logits = text_logits + sigmoid(g(features)) · image_text_residual
```

Cổng chỉ dùng input hoặc đầu ra của mô hình lúc suy luận. Đối chứng: text-only, concat, cổng cố định, cổng học được; cùng encoder và ngân sách. Đo macro-F1, số mẫu sửa/hỏng, độ nhạy với ảnh. Rủi ro: cổng có thể bỏ hẳn ảnh hoặc học tín hiệu nguồn. Không tối ưu theo mục tiêu "đổi dự đoán nhiều hơn khi tráo ảnh".

### E. Đổi nguồn và counterfactual có nhãn (giá trị nghiên cứu cao; phụ thuộc dữ liệu)

- Split giữ riêng theo nguồn đăng (nếu metadata đủ tin cậy), giữ nhóm caption/ảnh xuyên suốt.
- **ViMMSarc-Fine** (UIT, công bố 28/10/2026): 7.355 bài Facebook/Threads, nhãn T/I/M, OCR. Cần kiểm tra schema, quyền truy cập, việc gói dữ liệu có đủ ảnh hay không trước khi dùng; không so trực tiếp macro-F1 4 lớp của ViMMSD với F1 nhị phân của bộ mới.
- Pilot 50–100 cặp sửa caption/đổi ảnh có chủ đích (cả giữ nhãn và đổi nhãn), hai người xác nhận nhãn mới. Không sửa test cũ để tune.
- Kết hợp với Phase 12 (dữ liệu thật).

### Hoàn tất các phase còn dở

| Việc | Phase |
|---|---|
| Cross-attention với class weight `sqrt_inverse` (bản CE đã chạy: 0.380 ± 0.001, F1 `image` = 0); ablation dung lượng head | 5 |
| Attention map cho mẫu `image`/mẫu sai, chiều ảnh → text, Grad-CAM (đã có 6 mẫu text → ảnh) | 9.2 |
| Chạy thêm 2 seed cho Model 3.2 (`text_only_imgtext`, mới có seed 42: 0.3893) | 3.2 |
| Chạy Gemini/Qwen trên private test | 11 |
| Thu thập và gán nhãn dữ liệu thật | 12 |
| Web demo: gán nhãn đánh giá của người, thêm dữ liệu ngoài | 14 |

## 4. Các hướng chưa nên ưu tiên

| Hướng | Lý do |
|---|---|
| Sinh CoT cho cả tập / distill lại từ teacher hiện tại | chất lượng teacher là nút thắt đã có bằng chứng (Phase 6.2); chỉ mở lại khi pilot mù nhãn có người chấm cho thấy lập luận đáng tin ở các lớp khó |
| Chạy lại focal / sampling / stacking / kNN không đổi giả thuyết | đã có kết quả âm tính hoặc hạn chế |
| Augmentation back-translation đại trà | mất chơi chữ và sắc thái (Phase 8); chỉ dùng sau audit, có duyệt và lưu nguồn gốc |
| Tái lập kiến trúc graph/causal/uncertainty nhiều loss | khó xác định thành phần có ích với dữ liệu hiện tại |
| Coi nhãn `multi` tương đương (T = 1, I = 1) | SarcNet và ViMMSarc-Fine cho thấy cần nhãn cả cặp độc lập; Phase 6.1 cũng chỉ ra `multi` gồm 4 trường hợp |

## 5. Lịch đề xuất (một GPU 12 GB, hai người gán nhãn)

| Tuần | Việc | Điều kiện đi tiếp |
|---|---|---|
| 1 | Audit pilot, hoàn tất 44/40 mẫu, chốt giao thức, smoke test fine-tune | rubric dùng được; split không rò rỉ; vừa VRAM |
| 2 | Screening B trên fold nhóm; pilot C nếu kịp | tín hiệu OOF ổn định hoặc kết quả âm rõ |
| 3 | Xác nhận phương án và baseline 3 seed; thử cổng nếu có động cơ từ lỗi | hiệu ứng đủ lớn, không chỉ do một fold/seed |
| 4 | Đánh giá tập thách thức / giữ riêng nguồn đã chốt; viết báo cáo | tách rõ thăm dò và xác nhận |

Chỉ còn 1–2 tuần: làm A + bản nhỏ của B. Mục tiêu bài báo: thêm E hoặc tập audit có annotation chất lượng; chỉ thêm một cổng fusion không đủ chứng minh tính mới.

**Giao thức đợt mới:** chọn mô hình theo OOF trên fold nhóm của train; public/private test giữ làm benchmark để so với các phase trước; nếu thu thêm dữ liệu thì chốt holdout mới trước khi train.

## 6. Tài liệu tham khảo

1. Qin et al. (2023). *MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System*. Findings ACL. https://aclanthology.org/2023.findings-acl.689/
2. Yue et al. (2024). *SarcNet: A Multilingual Multimodal Sarcasm Detection Dataset*. LREC-COLING. https://aclanthology.org/2024.lrec-main.1248/
3. Jia, Xie, Jing (2024). *Debiasing Multimodal Sarcasm Detection with Contrastive Learning*. AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/29795
4. Guo et al. (2025). *Multi-View Incongruity Learning for Multimodal Sarcasm Detection*. COLING. https://aclanthology.org/2025.coling-main.119/
5. Basnet et al. (2025). *Evaluating Open-Source Vision-Language Models for Multimodal Sarcasm Detection*. ICDMW. https://doi.org/10.1109/ICDMW69685.2025.00172
6. Jana, Kundu, Singh (2025). *Think Twice Before You Judge: Mixture of Dual Reasoning Experts for Multimodal Sarcasm Detection*. arXiv. https://arxiv.org/abs/2507.04458
7. Wang et al. (2026). *URMF: Uncertainty-aware Robust Multimodal Fusion for Multimodal Sarcasm Detection*. arXiv. https://arxiv.org/abs/2604.06728
8. Tran, Nguyen, Pham (2026). *ViCLIP-OT*. Preprint. https://arxiv.org/abs/2602.22678 ; model card: https://huggingface.co/minhnguyent546/ViCLIP-OT
9. UIT (28/10/2026), thông báo ViMMSarc-Fine: https://www.uit.edu.vn/bai-viet/khi-chu-va-hinh-deu-khong-mang-y-cham-biem-nhung-ghep-lai-thi-co-sinh-vien-uit-nghien-cuu-bai-toan-nhan-dien-cham-biem-tren-mang-xa-hoi-bang-ai
10. Vu et al. (2025). *Multimodal Sarcasm Detection on Vietnamese Social Media*. IEA/AIE. https://doi.org/10.1007/978-981-96-8892-0_31

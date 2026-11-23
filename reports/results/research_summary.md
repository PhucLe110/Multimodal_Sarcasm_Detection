# Khi nào ảnh giúp phát hiện châm biếm tiếng Việt? Tổng hợp thí nghiệm trên ViMMSD

## Tóm tắt

Nghiên cứu đánh giá vai trò của ảnh trong phát hiện châm biếm tiếng Việt và xem xét cách mất cân bằng lớp, tiêu chí đánh giá và chất lượng nhãn ảnh hưởng đến kết luận. Trên private test ViMMSD, khi cả hai nhánh dùng Logit Adjustment (LA), thêm ảnh qua cross-attention tăng accuracy trung bình từ **0,722 lên 0,755**. Macro-F1 bốn lớp tăng từ **0,390 lên 0,404**, với chênh lệch tính từ số chưa làm tròn là **+0,015** và khoảng tin cậy 95% **[−0,004; +0,032]**. Phần cải thiện ròng chủ yếu đến từ lớp `multi`. Trong tám mẫu pilot có đồng thuận rằng ảnh là cần thiết, nhánh đa phương thức dự đoán có châm biếm ở **21/24 lượt**, so với **10/24** của nhánh caption-only. Số lượt khớp nhãn benchmark bốn lớp lần lượt là **11/24** và **12/24**. [Đối chứng trên private test](phase11_la_controlled.md#7-follow-up-caption-only--la-vs-cross-attention--la-full-private-test) · [Đối chiếu theo ID audit](phase11_label_audit/image_needed_model_comparison.csv)

## 1. Bài toán và dữ liệu

ViMMSD phân loại cặp caption–ảnh thành bốn lớp `not`, `text`, `image` và `multi`. Split chính thức gồm **10.805 mẫu train**, **1.413 mẫu public test dùng làm validation** và **1.504 mẫu private test**. Train chỉ có 77 mẫu `text` và 442 mẫu `image`. Private test có 458 `not`, 7 `text`, 40 `image` và 999 `multi`. Nhóm phân tích phân bố nhãn, caption trùng và tín hiệu nguồn, đồng thời xây pipeline chuẩn hóa và tách từ tiếng Việt, tiền xử lý ảnh, OCR và mô tả ảnh. [EDA](../main/2_1_eda.md) · [Tiền xử lý caption](../main/2_2_text_preprocessing.md) · [Tiền xử lý ảnh](../main/2_3_image_preprocessing.md)

Macro-F1 bốn lớp là chỉ số chính. Accuracy, macro-F1 ba lớp bỏ `text`, chỉ số theo lớp và thay đổi dự đoán theo ID giúp giải thích tác động của mất cân bằng. Các điểm giữa những đợt chạy cần đọc cùng split, số seed và tiêu chí chọn checkpoint tương ứng. [Giao thức phân tích thực nghiệm](../main/10_1_empirical_plan.md) · [Nhật ký run](results.md)

## 2. Các thí nghiệm đã chạy

| Nhánh | Thí nghiệm và kết quả |
|---|---|
| **Caption-only** | PhoBERT fine-tune đạt private-test macro-F1 **0,386 ± 0,009**. Trong đợt Kaggle, PhoBERT đóng băng đạt **0,365 ± 0,006**, bản thêm class weight đạt **0,381 ± 0,010**. [Báo cáo caption-only](../main/3_1_text_only_caption.md) |
| **Caption + text từ ảnh** | Đã thử OCR, mô tả ảnh và cả hai. Các cấu hình đa luồng với encoder đóng băng đạt **0,377–0,382** trên private test. Bản fine-tune ba luồng đạt **0,3893** ở một seed. Thí nghiệm Phase 10 chưa xác lập mức cải thiện ổn định nhờ OCR hoặc mô tả trong các cấu hình đã thử. [Báo cáo text nhiều luồng](../main/3_2_text_only_mix.md) · [Experiment A](../main/10_2_exp_a_modality_contributions.md) |
| **Image-only** | CLIP đóng băng + MLP đạt **0,370 ± 0,011**, CLIP fine-tune đạt **0,346 ± 0,009** và ViT fine-tune đạt **0,3313** ở một seed. Vintern zero-shot đạt **0,0453** trong cấu hình ảnh và **0,0151** khi thêm OCR/mô tả theo báo cáo. [Báo cáo image-only](../main/3_3_image_only.md) |
| **Concat fusion** | Phase 4 chạy **11 cấu hình × 3 seed**, gồm các baseline, concat PhoBERT + CLIP, mở block cuối, regularization và class weight. Concat đóng băng đạt **0,380 ± 0,033**. Concat regularized + class weight đạt **0,402 ± 0,015** trong giao thức Phase 4. [Báo cáo concat](../main/4_1_fusion_concat.md) |
| **Cross-attention dùng CE** | So sánh text → ảnh, ảnh → text và hai chiều. Macro-F1 private test lần lượt khoảng **0,372**, **0,367** và **0,380**. Bản hai chiều ngang concat CE về macro-F1 trung bình và đạt accuracy **0,778 so với 0,736**. F1 `image` của bản hai chiều bằng 0. [Báo cáo cross-attention](../main/5_1_cross_attention.md) |
| **CoT và VLM teacher** | Đã chạy baseline head phẳng, phân cấp mềm và cascade, cùng các pilot Gemini Pro, Gemini Flash Lite, Qwen2.5-VL và Qwen3-VL. Baseline phẳng tốt nhất đạt **0,391 ± 0,011** trên validation. Flash Lite v8 đạt **0,339** trên validation. Đã thử sinh giải thích khi cung cấp nhãn trên 100 mẫu. Nhóm dừng trước khi sinh rationale toàn tập và train student distillation do chất lượng giải thích và bất đồng nhãn. [Báo cáo teacher và baseline](../main/6_2_baseline_and_teacher.md) · [So sánh LLM](../main/11_1_llm_comparison.md) |
| **kNN, stacking và hiệu chỉnh prior** | Đã thử retrieval bằng kNN, trung bình xác suất, stacking và hiệu chỉnh prior lúc suy luận. Stacking đạt macro-F1 ba lớp **0,503** trên validation, so với **0,507** của mô hình đa phương thức trong cùng bảng. Hiệu chỉnh prior giảm khoảng **2–4 điểm macro-F1** trong nhánh baseline CoT. [Các thí nghiệm phụ](../main/6_2_baseline_and_teacher.md#4-các-hướng-phụ-đã-chạy) |
| **Lọc nhiễu** | Chạy **2 backbone × 3 phiên bản train × 3 seed**. Bản gốc, lọc cơ bản và lọc thêm mẫu nghi sai nhãn cho kết quả caption-only **0,386 / 0,376 / 0,380** và image-only **0,346 / 0,342 / 0,351** trên private test. Lợi ích chưa ổn định qua backbone và seed. [Báo cáo lọc nhiễu](../main/7_2_noise_filtering_results.md) |
| **Imbalance cơ bản** | Đã thử CE, class weight căn nghịch tần suất (B1), oversampling (B2), focal loss (B3), back-translation và back-translation + class weight. Kaggle validation một seed cho B1 **0,4141** và B2 **0,4151**. Trên dự đoán ngoài fold của ba fold theo nhóm, macro-F1 là **0,389** cho B1, **0,385** cho B2, **0,381** cho CE và **0,371** cho focal. B1 được chọn vào benchmark finalist. Back-translation chưa qua người duyệt cho kết quả thấp hơn đối chứng tương ứng trên validation. [Báo cáo imbalance](../main/8_1_imbalance.md) · [Experiment C](../main/10_4_exp_c_imbalance_augmentation.md) |
| **Can thiệp đầu vào và shortcut** | Đã thay pixel, OCR và mô tả, sửa caption khi suy luận, và train lại sau khi bỏ thẻ nguồn. Thay pixel làm concat đổi khoảng **20% dự đoán** trong Experiment B. Bỏ thẻ nguồn khi train lại làm giảm điểm validation trung bình ở caption và concat, với khoảng tin cậy chứa 0. Đã vẽ attention text → ảnh và chạy Integrated Gradients cho caption trên các mẫu định tính. [Experiment B](../main/10_3_exp_b_input_dependence.md) · [Experiment D](../main/10_5_exp_d_source_surface_cues.md) · [Shortcut](../main/9_1_shortcut_learning.md) · [Diễn giải](../main/9_2_interpretability.md) |
| **Benchmark finalist Phase 10** | Sau sàng lọc đầu vào và imbalance, bốn finalist đạt private-test macro-F1 **0,359** (caption CE), **0,366** (caption + B1), **0,369** (concat CE) và **0,380** (concat + B1). Chênh lệch giữa các finalist chưa đạt ngưỡng bằng chứng đã chốt. Phase 10 ghi nhận **60 lượt train**, bao gồm các thí nghiệm A–E. [Giao thức](../main/10_1_empirical_plan.md) · [Experiment E](../main/10_6_exp_e_targeted_improvement.md) |
| **Cross-attention + imbalance, đợt Kaggle** | Báo cáo Word ghi macro-F1 private test **0,373** cho CE, **0,419** cho LA và **0,368** cho Manifold Mixup qua ba seed. Hiệu chỉnh theo lớp nâng **0,4236 → 0,4288** ở seed 42. Cấu hình mặc định CE và LA khác ở bốn tham số huấn luyện. Config và predictions gốc của Kaggle chưa được xác minh. [Kiểm tra provenance Phase 11](phase11_la_controlled.md#1-audit-of-the-original-phase-11-runs) |

## 3. Đối chứng Phase 11 và phân tích lỗi

### Tác động của Logit Adjustment khi giữ cùng thông số train

Cross-attention + CE và Cross-attention + LA được chạy lại qua seed 42, 1 và 2 với cùng backbone, dữ liệu và thông số train. LA với hệ số 1 cộng log tần suất lớp vào logit khi tính loss, sau đó dùng logit thô lúc suy luận. Private-test macro-F1 tăng từ **0,390 ± 0,023 lên 0,404 ± 0,013**, với chênh lệch từng seed **+0,039 / +0,031 / −0,028**. CI bootstrap ghép cặp theo mẫu là **[+0,001; +0,028]**, có điều kiện trên ba seed. Recall `image` tăng **0,117 → 0,183**, trong khi precision giảm **0,123 → 0,101**. Cả sáu run đều đúng **0/7 mẫu `text`**. [Báo cáo CE–LA](phase11_la_controlled.md#3-results-private-test-n--1504-class-counts-458--7--40--999)

Đối chiếu theo ID cho thấy LA sửa đúng **413 lượt** và làm sai thêm **313 lượt** qua ba seed. Chênh lệch ròng theo lớp là **+202 `multi`**, **−110 `not`**, **+8 `image`** và **0 `text`**. Có 520 ID đổi trạng thái đúng/sai ở ít nhất một seed. [CSV phân tích từng mẫu](phase11_la_error_flips.csv)

### Tác động của nhánh ảnh khi cả hai mô hình dùng LA

Caption-only + LA và caption + ảnh qua cross-attention + LA dùng cùng split, seed, loss và tiêu chí chọn checkpoint. Predictions ghép đủ **1.504 ID** ở mỗi seed. [Báo cáo đối chứng đầu vào](phase11_la_controlled.md#7-follow-up-caption-only--la-vs-cross-attention--la-full-private-test) · [Gói artifacts caption-only](../../results/phase11_caption_la.20261122T154548Z.tar.gz)

| Private test, trung bình ± SD qua 3 seed | Caption-only + LA | Caption + ảnh + LA |
|---|---:|---:|
| Macro-F1 4 lớp | 0,390 ± 0,006 | 0,404 ± 0,013 |
| Macro-F1 3 lớp | 0,519 ± 0,008 | 0,539 ± 0,018 |
| Accuracy | 0,722 ± 0,033 | 0,755 ± 0,009 |
| F1 `image` | 0,097 | 0,126 |
| Mẫu `text` đúng mỗi seed | 0/7 | 0/7 |

Chênh lệch accuracy là **+0,033**, CI bootstrap 95% **[+0,019; +0,047]**. Chênh lệch macro-F1 bốn lớp là **+0,015**, CI **[−0,004; +0,032]**. Nhánh ảnh nhận đúng thêm trung bình **1,3/40 mẫu `image` mỗi seed**. Phần tăng ròng chủ yếu đến từ `multi`. Phép đối chứng đo tác động của việc thêm ảnh **qua cross-attention** trong thiết lập LA, đồng thời thay đổi cách kết hợp biểu diễn. [Thống kê ghép cặp](phase11_la_controlled.md#7-follow-up-caption-only--la-vs-cross-attention--la-full-private-test)

## 4. Audit nhãn và các mẫu cần ảnh

Pilot gồm **40 mẫu** được chọn có chủ đích từ private test. Một người và hai lượt AI chấm riêng caption, ảnh, cả cặp và đóng góp của ảnh. Mức khớp giữa người chấm và hai AI là **82,5–85%** với caption, **67,5–72,5%** với ảnh riêng, **80–85%** với cả cặp và **82,5%** với vai trò của ảnh. Các phiếu coi `khong_chac` là một nhãn riêng. [Giao thức audit](phase11_label_audit/README.md) · [Các phiếu audit](phase11_label_audit/)

Có **8 mẫu** cả ba lượt chấm đều đánh giá caption riêng không châm biếm, cả cặp có châm biếm và ảnh cần thiết. Trên 8 mẫu × 3 seed, caption-only dự đoán có châm biếm **10/24 lượt**, còn nhánh có ảnh dự đoán có châm biếm **21/24 lượt**. Chấm theo nhãn benchmark bốn lớp cho kết quả **12/24** và **11/24** tương ứng. Ba trong tám mẫu mang nhãn benchmark `not` dù cả ba lượt audit đánh giá cặp có châm biếm. Các số này mô tả nhóm ca đã chọn và có các dự đoán lặp lại trên cùng mẫu qua nhiều seed. [Đối chiếu theo ID audit](phase11_label_audit/image_needed_model_comparison.csv) · [Nhãn benchmark theo ID](phase11_label_audit/coordinator_key.csv)

## 5. Đóng góp của nghiên cứu

1. **Đo lợi ích của việc thêm ảnh dưới cùng phương pháp xử lý mất cân bằng.** Đối chứng caption-only và caption + ảnh dùng LA cung cấp bằng chứng về mức tăng accuracy trên benchmark, đồng thời định lượng mức bất định của macro-F1 và giới hạn của cải thiện ở lớp hiếm. [Đối chứng đầu vào](phase11_la_controlled.md#7-follow-up-caption-only--la-vs-cross-attention--la-full-private-test)
2. **Định lượng tác động của LA trong một đối chứng đồng bộ tham số huấn luyện.** Mức tăng macro-F1 trung bình **1,4 điểm** đi kèm biến thiên theo seed. Phân tích theo ID cho thấy thay đổi recall và precision ở `image`, cùng sự dịch chuyển giữa `not` và `multi`. [Đối chứng CE–LA](phase11_la_controlled.md) · [Dự đoán đổi đúng/sai](phase11_la_error_flips.csv)
3. **Xây dựng quy trình audit theo từng đầu vào để giải thích điểm benchmark.** Việc chấm riêng caption, ảnh và cả cặp giúp xác định các ca cần bằng chứng thị giác. Trên pilot, phát hiện có châm biếm và khớp nhãn bốn lớp cho hai cách đọc khác nhau về lợi ích của ảnh. [Giao thức audit](phase11_label_audit/README.md) · [Kết quả theo mẫu](phase11_label_audit/image_needed_model_comparison.csv)

## 6. Phạm vi kết luận và các phần còn mở

Private test đã được dùng ở các phase trước. CI bootstrap phản ánh biến thiên theo mẫu test với ba seed đã huấn luyện, còn độ ổn định trên các seed mới và nguồn dữ liệu khác cần thêm bằng chứng. Lớp `text` chỉ có bảy mẫu private test. Audit là pilot chọn có chủ đích, với một người chấm và hai lượt AI đã lưu. Độ đồng thuận giữa hai người độc lập và kết quả tổng quát hóa trên dữ liệu ngoài chưa được xác lập trong các artifacts hiện có. Các hạng mục chưa hoàn tất gồm chạy Gemini/Qwen trên toàn bộ private test, đánh giá trên tập dữ liệu ngoài và Grad-CAM theo kế hoạch. Web demo đã được triển khai để trình diễn mô hình. [Giới hạn đối chứng](phase11_la_controlled.md) · [So sánh LLM](../main/11_1_llm_comparison.md) · [Dữ liệu ngoài](../main/12_1_real_world_data.md) · [Diễn giải](../main/9_2_interpretability.md) · [Web demo](../main/14_1_web_demo.md)

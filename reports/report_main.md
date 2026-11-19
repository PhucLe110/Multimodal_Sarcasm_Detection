## Report main

### Data
- train.csv: dùng cho train
- public_test (val): dùng cho validation
- private_test (test): dùng cho test. Toàn bộ thông số đánh giá các mô hình phải là test trên bộ này.

### Phase 1. Setup dự án
- Task:
    - Tìm hiểu thông tin cơ bản về dự án
- Report
    - [ ] main/1_1_project_informations.md

### Phase 2. EDA và tiền xử lý dữ liệu
- Task:
    - Phân tích dữ liệu
    - Tiền xử lý text
    - Tiền xử lý ảnh
- Report:
    - [ ] main/2_1_eda.md
    - [ ] main/2_2_text_preprocessing.md
    - [ ] main/2_3_image_preprocessing.md

### Phase 3. Baseline đơn modality 
- Task:
    - Fine-tune PhoBERT cho text-only
        - Text-only chỉ dùng caption
        - Text-only dùng toàn bộ text
    - Fine-tune cho Image-only
        - CLIP
        - ViT Image Encoder
        - VLM với metadata
- Report:
    - [ ] main/3_1_text_only_caption.md
    - [ ] main/3_2_text_only_mix.md
    - [ ] main/3_3_image_only.md

### Phase 4. Multimodal fusion cơ bản
- Thành viên: Phúc, Nhi
- Task:
    - Concatenate embedding text và ảnh, đưa qua MLP classifier cho 4 lớp
    - Training freeze cả 2 encoder
    - Fine-tune nhẹ layer cuối và training
    - So sánh hiệu quả của mô hình fusion với text-only/image-only
- Report:
    - [ ] main/4_1_fusion_concat.md

### Phase 5. Cross-attention Fusion
- Thành viên: Mai
- Task:
    - Code cross-attention giữa text embedding và image embedding
        - Text attend vào ảnh
        - Ảnh attend vào text
    - Training và evaluation cho cả 2 chiều attention
    - So sánh hiệu quả cross-attention với concat baseline của Phase 4
- Report:
    - [ ] main/5_1_cross_attention.md

### Phase 6. Chain-of-Thought + Knowledge Distillation + Hierarchical Prompting
- Thành viên: Trúc
- Task:
    - Mô tả phương pháp
    - Train baseline, lấy điểm tham chiếu cột (a)/(b)
    - Teacher inference
        - Qwen3-VL-8B
        - Gemini Flash Lite
- Report:
    - [ ] main/6_1_cot_distillation_method.md
    - [ ] main/6_2_baseline_and_teacher.md

### Phase 7. Lọc dữ liệu nhiễu
- Thành viên: Trúc
- Task:
    - Lọc dữ liệu nhiễu
        - Loại trừ duplicate
        - Loại trừ mislabel
    - Chạy lại trên dữ liệu đã lọc
        - Text-only
        - Image-only
    - Kết luận
- Report:
    - [ ] main/7_1_noise_filtering.md
    - [ ] main/7_2_noise_filtering_results.md

### Phase 8. Xử lý mất cân bằng dữ liệu
- Thành viên: Phúc, Nhi, Mai
- Task:
    - Thử nghiệm các kỹ thuật xử lý mất cân bằng lớp
        - Class weighting (Phúc, Nhi)
        - Focal loss (Phúc, Nhi)
        - Oversampling (Mai)
        - Data augmentation cho text (Mai)
    - So sánh các phương pháp, chọn kỹ thuật tốt nhất dựa trên macro-F1
- Report:
    - [ ] main/8_1_imbalance.md

### Phase 9. Shortcut learning và Interpretability
- Thành viên:
- Task:
    - Shortcut learning check
        - Lấy mẫu `multi-sarcasm`/`image-sarcasm`, tráo ảnh giữa các mẫu (giữ nguyên text), xem dự đoán có đổi theo không
        - Phân tích định lượng
    - Interpretability
        - Vẽ attention map từ cross-attention lên ảnh gốc
        - Highlight các từ được chú ý nhiều trong caption
- Report:
    - [ ] main/9_1_shortcut_learning.md
    - [ ] main/9_2_interpretability.md

### Phase 10. Empirical Analysis
- Thành viên: Trúc
- Task:
    - Empirical analysis plan: tìm các yếu tố ảnh hưởng đến performance, thử các intervention để tăng khả năng tổng quát hoá và khả năng nhận diện lớp thiểu số
    - Experiment A: modality contributions
    - Experiment B: input dependence
    - Experiment C: imbalance and augmentation
    - Experiment D: source and surface cues
    - Experiment E: targeted improvement and behavior
- Report:
    - [ ] main/10_1_empirical_plan.md
    - [ ] main/10_2_exp_a_modality_contributions.md
    - [ ] main/10_3_exp_b_input_dependence.md
    - [ ] main/10_4_exp_c_imbalance_augmentation.md
    - [ ] main/10_5_exp_d_source_surface_cues.md
    - [ ] main/10_6_exp_e_targeted_improvement.md

### Phase 11. So sánh với LLM đa modal
- Thành viên: Trúc
- Task:
    - Viết prompt zero-shot và few-shot
    - Tạo tập test
    - So sánh kết quả LLM với mô hình fine-tune riêng, phân tích LLM sai ở đâu, mô hình fine-tune sai ở đâu
- Report:
    - [ ] main/11_1_llm_comparison.md

### Phase 12. Thu thập và test với dữ liệu thật
- Thành viên:
- Task:
    - Thu thập dữ liệu thật
    - Inference và đánh giá trên dữ liệu thật
- Report:
    - [ ] main/12_1_real_world_data.md

### Phase 13. Phân tích lỗi và Future work
- Thành viên:
- Task:
    - Phân tích lỗi
        - Trên bộ dữ liệu train (tách test ra)
        - Trên bộ dữ liệu thật
    - Future work
- Report:
    - [ ] main/13_1_error_analysis.md
    - [ ] main/13_2_future_work.md

### Phase 14. Web visualize
- Thành viên: Trúc
- Task:
    - Code base
    - Tích hợp từng model vào inference
    - Label human-round evaluation trên dữ liệu đã inference
    - Thêm dữ liệu ngoài đời để inference và kiểm tra
- Report:
    - [ ] main/14_1_web_demo.md

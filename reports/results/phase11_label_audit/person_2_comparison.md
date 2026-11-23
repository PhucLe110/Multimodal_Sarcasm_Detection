# Phase 11 — Đối chứng sau khi bổ sung Person 2

## Chuẩn hóa dữ liệu

Ba phiếu Person 2 được lấy từ `origin/main` (merge `7b9f4f3`). Phiếu caption có 39 dòng chứa ba ô trống dư trước nhãn. Hai dòng phiếu ảnh có dấu phẩy trong phần căn cứ chưa được quote. Đã chuẩn hóa mỗi phiếu theo schema và thứ tự của mẫu gốc, ghép theo `audit_id`, và kiểm tra đủ 40 ID cùng nội dung caption hoặc đường dẫn ảnh. Giữ nguyên đánh giá gốc, bao gồm dòng **A023** chưa được Person 2 chấm ở lượt cả cặp. [Caption](01_caption_person_2.csv) · [Ảnh](02_image_person_2.csv) · [Cả cặp](03_pair_person_2.csv)

## Độ đồng thuận

Tỷ lệ khớp chính xác và Cohen’s κ tính trên các mẫu có nhãn của cả hai bên. `khong_chac` được tính là một nhãn riêng. A023 được loại khỏi hai phép tính của lượt 03.

| Lượt và trường | Person 2 ↔ Person 1 | Person 2 ↔ AI 1 | Person 2 ↔ AI 2 |
|---|---:|---:|---:|
| Caption — châm biếm | 32/40 · κ 0,619 | 31/40 · κ 0,595 | 30/40 · κ 0,562 |
| Ảnh — châm biếm | 32/40 · κ 0,643 | 31/40 · κ 0,568 | 32/40 · κ 0,649 |
| Cả cặp — châm biếm | 33/39 · κ 0,678 | 29/39 · κ 0,463 | 29/39 · κ 0,533 |
| Cả cặp — vai trò ảnh | 21/39 · κ 0,222 | 22/39 · κ 0,247 | 19/39 · κ 0,202 |

Hai người bất đồng về nhãn châm biếm của cả cặp ở **A004, A012, A019, A024, A034, A037**. Person 2 ghi vai trò ảnh là `can` hoặc `bo_sung` ở **14 cặp** mà chính Person 2 đánh giá `khong` về châm biếm. Điều này gợi ý hai người có thể hiểu phạm vi của cột vai trò ảnh khác nhau. Khi chỉ xét **23 cặp cả hai người đánh giá có châm biếm**, họ đồng ý về vai trò ảnh ở **18/23**, trong đó **17/23** cùng chọn `can`. Đây là phân tích điều kiện trên các cặp được cả hai chấm dương tính.

## Đối chứng mô hình trên nhóm hai người đồng thuận cần ảnh

Tiêu chí lấy mẫu là cả **Person 1 và Person 2** cùng đánh giá caption riêng `khong`, cả cặp `co` và vai trò ảnh `can`. Có **11 mẫu**, gồm A008, A011, A013, A016, A017, A020, A027, A028, A036, A039 và A040. Trong đó, 8 mẫu mang nhãn benchmark `image` và 3 mẫu mang nhãn benchmark `not`. Tất cả 8 mẫu trong nhóm đồng thuận chặt của pilot trước vẫn thỏa tiêu chí khi thêm Person 2. [Nhãn gốc theo ID](coordinator_key.csv)

| Mô hình | Dự đoán có châm biếm theo audit | Khớp nhãn benchmark 4 lớp |
|---|---:|---:|
| Caption-only + LA | 17/33 | 15/33 |
| Cross-attention + LA | 30/33 | 15/33 |
| Cross-attention + CE | 17/33 | 17/33 |

So sánh chính ghép predictions theo ID và seed giữa hai nhánh cùng dùng LA. Nhánh có ảnh chuyển **15 lượt** từ dự đoán không châm biếm sang có châm biếm, và **2 lượt** theo chiều ngược lại. Hai nhánh khớp benchmark bốn lớp ở cùng số lượt. Những con số trên mô tả **11 mẫu × 3 seed** trong pilot chọn có chủ đích. Ba dự đoán của một ID là các quan sát lặp trên cùng mẫu. Các trường hợp khác nhau giữa audit và benchmark cần xem xét trước khi dùng nhóm này như tập đánh giá chính thức. [Predictions ghép theo mẫu và seed](image_needed_human_model_comparison.csv) · [Đối chứng toàn bộ private test](../phase11_la_controlled.md)

# Checklist nhóm tự kiểm tra

- [ ] Điền nguồn/link dataset, giấy phép và tên/mã sinh viên.
- [ ] Kiểm tra đường dẫn CSV trên máy hoặc quyền Drive của từng người.
- [ ] Xác nhận giữ cách chia theo thời gian khoảng 64/16/20.
- [ ] Kiểm tra 1.1 chia tập trước; bảng chia chỉ có số dòng và thời gian, không có fraud/fraud_pct của validation/test.
- [ ] Kiểm tra 1.2–1.9 chỉ dùng train; không đọc sample_all44.parquet hoặc các bảng *_full.csv.
- [ ] Kiểm tra 44 cột được mô tả; unique/missing/statistics tính lại trên toàn bộ train.
- [ ] Kiểm tra các biểu đồ nặng dùng tối đa 200.000 dòng train; bảng đếm/tỷ lệ dùng toàn bộ train.
- [ ] Kiểm tra state vocabulary, median và scaler chỉ fit train.
- [ ] Đọc phần giới hạn CV fit preprocessing trước fold; quyết định có cần sửa phương pháp này riêng hay không.
- [ ] So sánh code mô hình với docs/model_code_preservation.json; các thay đổi gồm bảo vệ test và sửa việc xóa biến lặp.
- [ ] Để RUN_FINAL_TEST=False khi còn review/tinh chỉnh; chỉ bật sau khi chốt mô hình.
- [ ] Không mô tả test cũ là chưa từng được xem: các phiên bản trước đã audit test.
- [ ] Chỉ push notebook, tài liệu, requirements, metadata mô tả và kết quả tổng hợp nhỏ. Không push data.csv, data/processed hay thông tin truy cập.

Chưa có thao tác git init, commit, tạo remote hoặc push trong đợt chuẩn bị này.

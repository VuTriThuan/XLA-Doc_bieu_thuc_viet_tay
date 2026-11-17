# Ảnh chụp trang giấy thật

Đặt ảnh chụp (JPG/PNG) vào thư mục này và khai báo nhãn **trước khi chạy nhận dạng** trong `nhan.json`:

```json
{
  "trang_01.jpg": ["127+45=", "36 1024 7"],
  "trang_02.jpg": ["8×7=56"]
}
```

Mỗi phần tử là một dòng, theo thứ tự đọc: hết cột trái (trên → dưới) rồi mới sang cột phải. Viết đúng ký tự trên giấy
(`+ - × ÷ =`); dấu cách chỉ có ý nghĩa trong dòng dãy số. Phần bị gạch xóa trên giấy không ghi vào nhãn. Sau đó chạy
`python3 danh_gia.py --trang`; kết quả ở `ket_qua/trang_that/`.

Ảnh ở đây chỉ để đánh giá, không dùng để huấn luyện.

| Tệp | Nội dung | Người viết |
|---|---|---|
| trang_01.jpg | 32 dòng, hai cột (biểu thức có/không có đáp số, dãy số dài, chia cho 0, một chỗ gạch xóa); giấy in mặt sau thấm qua | Vũ Trí Thuận |

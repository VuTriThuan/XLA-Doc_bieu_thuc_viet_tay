# Nguồn bộ slide (Nhóm 13)

- `index.html`: 17 slide, lời thoại nằm trong `<aside class="notes">` của từng slide.
- `style.css`: giao diện riêng (theme corporate-clean của skill html-ppt).
- `hinh/`: hình dùng trong slide.

Sửa xong thì gộp lại thành file offline và xuất PDF (chạy trong thư mục `slide/`):

```bash
python3 nguon/gop_slide.py nguon "Slide BTL.html"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer --virtual-time-budget=4000 --print-to-pdf="Slide BTL.pdf" "file://$PWD/Slide BTL.html"
```

`index.html` lấy CSS/JS từ `~/.claude/skills/html-ppt/assets/` nên chỉ mở được trên máy này;
file gửi đi máy khác là `Slide BTL.html` (đã nhúng hết, không cần mạng).

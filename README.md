# Đọc dãy số và biểu thức viết tay từ ảnh chụp trang giấy

Bài tập lớn môn Xử lý ảnh, Nhóm 13. Chụp một tờ giấy có các dòng như `127 + 45 =`, hệ thống tách từng dòng, từng ký tự,
nhận dạng bằng CNN rồi tính kết quả (**172**). Nếu sau dấu `=` có đáp số viết tay, hệ thống cho biết đáp số đó đúng hay
sai. Dòng chỉ có số (vd. `36 1024 7`) được đọc thành dãy số.

Ký tự hỗ trợ: `0–9 + − × ÷ =`. Phần lớn công việc là xử lý ảnh tự cài bằng NumPy (chuẩn hóa nền, chỉnh nghiêng, xóa
dòng kẻ vở, thành phần liên thông, chiếu ngang/dọc, tách chữ dính, chuẩn hóa độ dày nét); nhận dạng bằng CNN, mô hình cuối ghép một CNN nhỏ và một CNN
lớn, cả hai có thêm lớp "rác" cho những mảnh cắt không phải ký tự.

## Chạy nhanh

Cần Python 3.10+; cài thư viện một lần:

```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 ung_dung.py                          # ứng dụng web: http://127.0.0.1:8781 (tải ảnh chụp hoặc viết trên khung)
python3 doc_trang.py anh.jpg                 # đọc một ảnh, in từng dòng và kết quả (--json, --luu-buoc thu_muc/)
python3 doc_trang.py anh.jpg --chi-chu-so    # trang chỉ có dãy số
python3 doc_trang.py anh.jpg --cat-dong ra/  # cắt mỗi phép tính thành một ảnh riêng (ra/dong_01.png, ...)
python3 doc_trang.py anh.jpg --khong-dong-hinh-thai   # tắt phép đóng 3×3 (có thể lấp khe của ÷ và =)
python3 danh_gia.py                          # đánh giá: CNN, CVL Digit Strings, trang tổng hợp, ảnh chụp thật
python3 danh_gia.py --thu-nghiem             # bật/tắt từng bước xử lý ảnh để so sánh
python3 danh_gia.py --cvl --so-sanh-cat-thang  # từng cấu hình trên CVL test, có và không có drop-fall
python3 danh_gia.py --quet-duong-cat          # so các cách tạo đường cắt trên dữ liệu kiểm định
python3 huan_luyen.py                        # huấn luyện lại (--chuan-bi: tạo bộ dữ liệu 15 lớp trước)
python3 -m pytest                            # bộ kiểm thử
```

Mô hình đã huấn luyện có sẵn trong `mo_hinh/`, nên đọc ảnh, chạy ứng dụng web và đánh giá trên ảnh chụp thật hay trang
tổng hợp không cần huấn luyện lại. Mỗi lệnh có `--help`.

Ví dụ (đầu ra thật trên `du_lieu/trang_tong_hop/trang_003.jpg`; dòng 1 trên giấy là `204+54`, chữ `4` cuối bị đọc
nhầm thành `+` nên hệ thống báo không tính được thay vì đưa ra số sai):

```
$ python3 doc_trang.py du_lieu/trang_tong_hop/trang_003.jpg
Dòng 1: 204+5+                   không tính được: Biểu thức kết thúc giữa chừng (thiếu số sau phép toán)
Dòng 2: 4                        dãy số: 4
Dòng 3: 0                        dãy số: 0
Dòng 4: 67÷6+60=                 = 71,166667… (= 427/6)
Dòng 5: 7-15-1                   = -9
Dòng 6: 717-4=                   = 713
```

## Kết quả

| Đánh giá | Kết quả | Ghi chú |
|---|---|---|
| CNN, tập test ký tự | chữ số **9770/10000 (97,70%)**; ký hiệu + − × ÷ = **694/701 (99,00%)** | MNIST test; HASYv2 + CROHME+ test (người viết khác train) |
| **Ảnh chụp thật** của nhóm: 1 trang, 32 dòng, hai cột | dòng đúng **28/32**, biểu thức ra đúng giá trị **23/27**, lỗi ký tự 1,35% | Một người viết; trang này đã được dùng khi sửa khâu tách nên số liệu lạc quan. Dòng `4÷0=` không có giá trị nên đếm riêng (hệ thống báo đúng lỗi chia cho 0) |
| Đọc dãy số viết tay thật — CVL Digit Strings, 744 ảnh / 75 người lạ | đúng cả dãy **467/744 (62,77%)**, lỗi ký tự 10,0% | Ban đầu 31,32%; quy tắc chọn mô hình đã sửa một lần sau khi thấy điểm test (xem thực nghiệm), nên con số có thể hơi lạc quan |
| Trang tổng hợp, 60 trang / 360 dòng | dòng đúng hoàn toàn **279/360 (77,50%)**, biểu thức ra đúng giá trị **177/244 (72,54%)** | Ghép từ ký tự tập test trên nền giấy loang, dòng kẻ, xoay nhẹ |

Mọi số liệu trên được đo lại ngày 18/11/2026 sau khi sửa lỗi ở khâu tách chữ dính (mảnh đầu tiên của một nhóm có đường cắt
bị thay bằng ảnh của cả nhóm); mô hình được chọn không đổi.

Phát hiện chính (chi tiết và bảng đầy đủ trong [bao_cao/thuc_nghiem.md](bao_cao/thuc_nghiem.md)):
- **Chuẩn hóa nền** là bước quan trọng nhất với ảnh chụp: thay bằng Otsu toàn cục, lỗi ký tự trên trang tổng hợp tăng từ
  5,37% lên 86,22% và trên ảnh chụp thật từ 1,35% lên 80,41% (0/32 dòng đúng); Sauvola làm đứt nét mực nhạt.
- **Tách chữ dính** là bước quan trọng nhất với chữ viết liền nét: tắt nó, 90 dãy CVL của 9 người hoàn toàn mới giảm từ
  40 xuống 19 dãy đúng.
- **Tách cột** (chiếu dọc) là bắt buộc với trang viết hai cột: thiếu nó, ảnh chụp thật của nhóm không tính được dòng nào.
- **Nâng cấp nhận dạng trên chữ viết thật** (CVL test, người lạ): 31,32% → 51,75% khi học thêm chữ số viết tay của
  người châu Âu (số 1 có mũ, 7 có gạch) → 56,99% khi thêm lớp **"rác"** giúp tách chữ dính đúng hơn → 61,56% với **CNN lớn
  + EMNIST** → **62,77%** khi ghép hai CNN có lớp rác. Đường cắt **drop-fall** (cắt men theo khe giữa hai chữ nghiêng
  thay vì cắt thẳng) tăng thêm 0,7–2,4 điểm cho mỗi CNN.
- Biến thể kiểu viết sinh tự động và tăng cường dữ liệu tốt trên validation ký tự nhưng **kém trên chữ viết thật**,
  nên mô hình luôn được chọn bằng chữ viết thật chứ không bằng validation ký tự.

## Luồng xử lý

```
ảnh chụp ─ ảnh xám ─ chuẩn hóa nền + Otsu ─ chỉnh nghiêng (chiếu ngang) ─ xóa dòng kẻ vở ─ đóng hình thái học
        ─ thành phần liên thông ─ lọc nhiễu/mép bàn ─ chiếu dọc → các cột ─ chiếu ngang → các dòng
        ─ ghép nét cùng ký tự (= ÷ 5 7) ─ bỏ vết gạch xóa ─ ghép/tách chữ dính: đường cắt thẳng + drop-fall,
          quy hoạch động chấm bằng CNN có lớp "rác"
        ─ chuẩn hóa ký tự 28×28 (độ dày nét, giữ tỉ lệ) ─ CNN 15 lớp ─ ghép số, phân tích cú pháp, tính bằng phân số
```

Chi tiết: [bao_cao/thiet_ke.md](bao_cao/thiet_ke.md). Số liệu: [bao_cao/thuc_nghiem.md](bao_cao/thuc_nghiem.md).

## Cấu trúc

```
BTL_v2/
├── huan_luyen.py, doc_trang.py, danh_gia.py, ung_dung.py     các lệnh chạy
├── tien_xu_ly/     xử lý ảnh tự cài: anh, nguong, nguong_thich_nghi, nghieng, hinh_thai, lien_thong, thu_phong,
│                   trang (tách nền, dòng, ký tự), chuan_hoa (ký tự → 28×28)
├── loi/            cau_hinh, cnn, tang_cuong, huan_luyen, du_doan, doc_trang (tách chữ dính + đọc), bieu_thuc, danh_gia
├── bo_du_lieu/     mnist, emnist, ky_hieu (HASYv2, CROHME+), kieu_viet, cvl (dãy số), cvl_chu_so (chữ số),
│                   rac (lớp "rác"), tap_ky_tu, trang_tong_hop
├── giao_dien/      máy chủ web + index.html
├── mo_hinh/        ky_tu_{co_so,cai_tien,co_cvl,nho_rac,lon_rac}.pth, lua_chon.json (mô hình cuối: nho_rac + lon_rac)
├── du_lieu/        mnist/, ky_hieu/, ky_tu/, cvl/, trang_tong_hop/, trang_that/
├── ket_qua/        phan_loai/, cvl/, trang_tong_hop/, trang_that/, thu_nghiem_tien_xu_ly.csv
├── kiem_thu/       bộ kiểm thử
└── bao_cao/        thiet_ke.md, thuc_nghiem.md
```

## Dữ liệu và nguồn

| Bộ | Dùng để | Giấy phép |
|---|---|---|
| MNIST (LeCun, Cortes) | chữ số 0–9: huấn luyện / validation / test | CC BY-SA 3.0 |
| EMNIST Digits (Cohen và cs., 2017; NIST) | 60.000 chữ số thêm vào huấn luyện CNN lớn | NIST |
| HASYv2 (M. Thoma, 2017; zenodo.org/records/259444) | ký hiệu + − × ÷ | ODbL 1.0 |
| CROHME+ (zenodo.org/records/14968570; dựa trên CROHME 2023) | ký hiệu = + − × ÷ | CC BY-NC-SA 3.0 |
| CVL Single Digits (zenodo.org/records/1492173) | chữ số kiểu châu Âu: huấn luyện + kiểm định (chỉ người viết KHÔNG thuộc nửa test của CVL Strings) | CC BY-NC 4.0 |
| CVL Digit Strings (ICDAR 2013; zenodo.org/records/1492173) | **không huấn luyện**: nửa người viết để chọn tham số tách chữ dính, nửa còn lại để báo cáo | CC BY-NC 4.0 |

`du_lieu/ky_hieu/ky_hieu.npz` là ảnh ký hiệu đã trích (tệp gốc HASYv2 34 MB và CROHME+ 206 MB không lưu trong dự án;
lệnh trích lại: `python3 huan_luyen.py --trich-ky-hieu <thư mục HASYv2> <CROHME+.zip>`).

### Dữ liệu không có trong repo

Các bộ dữ liệu tải từ nguồn ngoài không được đưa lên repo (dung lượng lớn, và CVL chỉ cho phép dùng phi thương mại):

| Tệp | Cách lấy lại |
|---|---|
| `du_lieu/mnist/mnist.npz` | tự tải khi chạy lệnh cần đến (`bo_du_lieu/mnist.py`) |
| `du_lieu/cvl/cvl-strings-train.zip`, `cvl-single-digits-train-validation.zip` | tự tải từ Zenodo khi chạy (`bo_du_lieu/cvl.py`, `bo_du_lieu/cvl_chu_so.py`) |
| `du_lieu/emnist/emnist_chu_so_60k.npz` | làm theo hướng dẫn đầu tệp `bo_du_lieu/emnist.py` (chỉ cần 4 tệp `emnist-digits-*.gz`) |
| `du_lieu/ky_tu/ky_tu.npz` | `python3 huan_luyen.py --chuan-bi` (sau khi đã có các bộ trên) |

Chỉ cần các tệp này khi huấn luyện lại hoặc khi đánh giá CNN / CVL.

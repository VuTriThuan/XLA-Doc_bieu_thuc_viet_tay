# Thiết kế hệ thống đọc dãy số và biểu thức viết tay

## 1. Bài toán

Đầu vào: ảnh chụp (điện thoại) một trang giấy có một hoặc nhiều dòng viết tay, mỗi dòng là
- một **biểu thức** gồm số nguyên và các phép `+ − × ÷`, có thể kết thúc bằng `=` và có thể kèm đáp số viết tay
  (vd. `127+45=`, `8×7=56`), hoặc
- một **dãy số** cách nhau bằng khoảng trắng (vd. `36 1024 7`).

Đầu ra: với từng dòng, chuỗi ký tự đọc được, giá trị của biểu thức (hoặc danh sách số), và — nếu có đáp số viết sau dấu
`=` — đáp số đó đúng hay sai. Kèm theo: hộp bao, ảnh 28×28 và điểm softmax của từng ký tự để người dùng kiểm tra.

So với bài toán nhận dạng từng chữ số đã cắt sẵn (mỗi ảnh một chữ số), phần thêm vào gần như toàn bộ là **xử lý ảnh**: tách nền trên ảnh chụp,
chỉnh nghiêng, xóa dòng kẻ, tách dòng, tách ký tự, xử lý chữ dính và chuẩn hóa ký tự. CNN giữ nguyên kiến trúc, chỉ đổi
số lớp từ 10 thành 15.

## 2. Luồng xử lý

```
ảnh chụp ─► giới hạn cạnh 1600 px ─► ảnh xám
        ─► chuẩn hóa nền (lọc cực đại + lọc trung bình, chia ảnh cho nền) ─► Otsu          [tien_xu_ly/nguong_thich_nghi.py]
        ─► ước lượng góc nghiêng bằng chiếu ngang, trượt dọc để chỉnh                     [tien_xu_ly/nghieng.py]
        ─► xóa dòng kẻ vở (đoạn chạy ngang ≥ 25% bề rộng = phép mở với SE 1×L)            [tien_xu_ly/trang.py]
        ─► đóng hình thái học 3×3 (nối nét đứt)                                             [tien_xu_ly/hinh_thai.py]
        ─► gán nhãn thành phần liên thông (hai lượt trên đoạn chạy + hợp–tìm)              [tien_xu_ly/lien_thong.py]
        ─► bỏ nhiễu: chấm quá nhỏ, mảng tối đặc chạm mép ảnh (bàn, bóng), vết quá lớn
        ─► chiếu dọc → các cột (dải trống thẳng đứng, mỗi bên ≥ 2 dòng)
        ─► trong mỗi cột: chiếu ngang → các dải dòng; dải cao bất thường được chia tại cực tiểu
        ─► mỗi dòng: ghép nét của cùng một ký tự (= ÷, gạch đầu số 5, gạch ngang số 7)
                     gộp mẩu nét nhỏ ở mép trên dòng (mũ số 1) vào ký tự bên cạnh
                     bỏ vết gạch xóa (≥ 6 lỗ kín)
        ─► hai nhóm sát nhau mà ghép lại vẫn hẹp: ghép nếu CNN chấm ký tự ghép cao hơn (số 4 viết hai nét rời)
        ─► nhóm rộng: chỗ cắt ứng viên = cực tiểu chiếu dọc; mỗi chỗ sinh đường cắt thẳng và   [loi/doc_trang.py]
           đường drop-fall (men theo khe giữa hai chữ); quy hoạch động chọn cách cắt
           có tổng log-điểm CNN lớn nhất (nhóm rộng > 1,6 lần chiều cao chữ bắt buộc cắt)
        ─► chuẩn hóa từng ký tự → 28×28 ─► CNN 15 lớp                                        [tien_xu_ly/chuan_hoa.py]
        ─► ghép chữ số liền nhau thành số, phân tích cú pháp, tính bằng phân số              [loi/bieu_thuc.py]
```

### 2.1 Tách nền

Ảnh chụp có ánh sáng không đều (một góc tối, bóng tay), nên ngưỡng Otsu toàn cục thường coi cả vùng giấy tối là nét.
Hệ thống có ba cách (chọn bằng `--nguong`):

| Cách | Ý tưởng | Ghi chú |
|---|---|---|
| `nen_otsu` (mặc định) | Ước lượng nền bằng lọc cực đại cửa sổ k×k (xóa nét tối mảnh hơn k), làm mượt bằng lọc trung bình, chia ảnh cho nền rồi mới Otsu | Giấy thành ~255 ở mọi chỗ; giữ được mực nhạt |
| `sauvola` | Ngưỡng riêng từng pixel T = m·(1 + k·(s/R − 1)), m, s là trung bình và độ lệch chuẩn trong cửa sổ, tính bằng ảnh tích phân | Hay làm rỗng ruột nét đậm và đứt nét mực nhạt |
| `otsu` | Một ngưỡng cho cả ảnh | Để so sánh |

Kích thước cửa sổ mặc định ≈ 1/25 cạnh ngắn của ảnh, trong [15, 101].

### 2.2 Chỉnh nghiêng

Khi dòng chữ nằm ngang, biểu đồ chiếu ngang có đỉnh cao và hẹp. Thử các góc trong ±5° (bước 0,25°), với mỗi góc dịch
từng cột một đoạn x·tan(góc) và tính tổng bình phương của biểu đồ chiếu; chọn góc làm tổng lớn nhất. Dải 6% sát mép ảnh
bị bỏ khi ước lượng vì mảng tối của bàn/góc ảnh làm lệch biểu đồ. Với góc nhỏ, phép xoay được xấp xỉ bằng phép trượt
dọc nên ảnh nhị phân không bị nội suy.

### 2.3 Thành phần liên thông

Mỗi hàng được nén thành các đoạn chạy; đoạn ở hàng y nối với đoạn ở hàng y−1 nếu chồng nhau khi nới mỗi bên 1 cột
(8-liên thông); nhãn tương đương gộp bằng hợp–tìm có nén đường đi. Số đoạn chạy nhỏ hơn số pixel hàng chục lần nên
gán nhãn ảnh 1600×1200 bằng Python thuần vẫn nhanh. Kiểm thử so sánh với BFS từng pixel trên 200 ảnh ngẫu nhiên.

### 2.4 Tách cột, tách dòng và tách ký tự

- **Cột**: chiếu dọc của cả trang; một dải cột liên tiếp KHÔNG có pixel nét, rộng ít nhất một chiều cao chữ, nằm giữa
  trang, là ranh giới cột nếu mỗi bên có ít nhất hai dòng chữ (đếm bằng chiếu ngang của riêng bên đó). Điều kiện "hai
  dòng" để khoảng trắng lớn giữa hai số trên cùng một dòng không bị tách thành hai cột. Kết quả đọc theo thứ tự hết cột
  trái rồi sang cột phải. Thêm bước này sau khi thử ảnh chụp thật của nhóm (trang hai cột): không tách cột thì mỗi dòng
  dính hai biểu thức.
- **Dòng**: trong mỗi cột, chiếu ngang của mặt nạ; các dải hàng có nét cách nhau dưới 0,15 chiều cao chữ được nối; dải cao hơn
  1,8 lần chiều cao chữ (hai dòng viết sát) được chia tại cực tiểu của chiếu ngang đã làm mượt. Mỗi thành phần được
  gán cho dải chứa tâm của nó.
- **Ghép nét của cùng một ký tự**: hai nhóm chồng nhau ≥ 50% bề rộng của nhóm hẹp hơn VÀ (nằm trên – dưới nhau, chồng
  dọc < 50% chiều cao nhóm thấp hơn, HOẶC một nhóm rất dẹt, thấp hơn 35% nhóm kia) được ghép — đây là cách dấu `=`,
  `÷`, gạch đầu số 5, gạch ngang của số 7 trở thành một ký tự. Điều kiện thứ hai tránh gộp số 1 viết nghiêng vào chữ
  số cao ngang nó bên cạnh (lỗi thật gặp trên ảnh chụp: "×10" thành "×0").
- **Ghép bằng CNN**: hai nhóm cạnh nhau (khe < 0,1 chiều cao chữ), đều cao hơn nửa chiều cao chữ, ghép lại vẫn hẹp hơn
  0,9 chiều cao chữ, được ghép nếu log-điểm CNN của ký tự ghép lớn hơn tổng log-điểm hai ký tự rời (số 4 viết hai nét
  rời đứng cạnh nhau: rời thì thành "6" và "1").
- **Đường cắt drop-fall**: thả một "giọt" từ mép trên/dưới tại chỗ cắt ứng viên; giọt đi thẳng xuống qua nền, đi chéo
  hoặc trượt ngang tối đa 3 cột khi gặp nét, chỉ cắt xuyên nét khi không còn khe (`tien_xu_ly/trang.py: duong_roi`).
  Mỗi chỗ cắt có tới 5 đường (thẳng; từ trên/dưới × ưu tiên trái/phải); đoạn giữa hai đường là các pixel nằm giữa
  chúng trên từng hàng. Có thêm tùy chọn lấy chỗ cắt từ thung lũng đường viền trên/dưới (`thung_lung_duong_vien`),
  mặc định tắt vì không cộng thêm khi đã có drop-fall.
- **Vết gạch xóa**: nét tô đi tô lại tạo rất nhiều vùng nền kín; nhóm có ≥ 6 lỗ (chữ số nhiều nhất 2, số 8 tô đậm 3)
  bị bỏ.
- **Khoảng trắng**: khe giữa hai ký tự lớn hơn 0,6 chiều cao chữ được coi là dấu cách (chỉ có nghĩa trong dòng dãy số).
- **Chữ dính**: xem đầu tệp `loi/doc_trang.py`. Nhóm rộng hơn 1,6 lần chiều cao chữ gần như không thể là một ký tự
  (`−` và `=` thấp nên không bị xét) nên bắt buộc cắt — CNN chỉ biết 15 lớp và vẫn cho điểm cao cho một khối nhiều chữ
  số dính nhau. Nhóm rộng 1,1–1,6 lần được cắt nếu tổng log-điểm các đoạn lớn hơn giữ nguyên. Hai ngưỡng này chọn bằng
  quét trên CVL validation và trang tổng hợp (ngưỡng 1,3 cắt nhầm chữ số đơn lẻ viết rộng).

### 2.5 Chuẩn hóa ký tự (dùng chung lúc huấn luyện và lúc đọc)

`tien_xu_ly/chuan_hoa.py`: nhị phân hóa (ảnh nhỏ được phóng to trước) → cắt hộp bao → **chuẩn hóa độ dày nét**
(ước lượng độ dày w ≈ 2·diện tích / số pixel biên; giãn hoặc co bằng hình thái học để nét sau khi thu về 20 px dày
≈ 2,7 px, bằng trung vị đo trên MNIST) → thu nhỏ giữ tỉ lệ bằng trung bình theo diện tích vào ô 20×20 → căn giữa canvas
28×28 → kéo giãn độ sáng. Mọi ảnh huấn luyện (MNIST, HASYv2, CROHME+) cũng đi qua đúng hàm này, nên CNN học trên cùng
kiểu đầu vào mà nó gặp khi đọc trang (phiên bản đầu vào `ky-tu-hop-bao-net-1` được ghi vào checkpoint và kiểm tra khi nạp).

## 3. CNN và dữ liệu

Hai kiến trúc (`loi/cnn.py`):
- **nhỏ** (57 nghìn tham số): Conv(1→16,3×3) – ReLU – MaxPool – Conv(16→32,3×3) – ReLU –
  MaxPool – Dropout 0,25 – Linear(800→64) – ReLU – Linear(64→số lớp).
- **lớn** (0,47 triệu tham số): hai khối [Conv 3×3 có đệm – BatchNorm – ReLU – Conv 3×3 – BatchNorm – ReLU – MaxPool]
  với 32 rồi 64 bộ lọc – Dropout 0,4 – Linear(3136→128) – ReLU – Dropout 0,4 – Linear(128→số lớp).

Lớp: `0–9 + − × ÷ =` (15) và lớp thứ 16 **"rác"** (không phải một ký tự) cho các cấu hình mới.

| Nguồn | Lớp | Chia tập | Giấy phép |
|---|---|---|---|
| MNIST | 0–9 | 54.000 / 6.000 / 10.000 (chia phân tầng, hạt cố định; test chính thức giữ nguyên) | CC BY-SA 3.0 |
| HASYv2 (Thoma, 2017) | + − × ÷ | theo **người viết**, gần 70/15/15 cho từng lớp; tối đa 80 mẫu/người/lớp vì một người viết 1.396 dấu × | ODbL 1.0 |
| CROHME+ (mở rộng CROHME 2023) | = + − × ÷ | khử trùng lặp nét, chia theo băm của **mã tờ viết gốc** 70/15/15 | CC BY-NC-SA 3.0 |
| EMNIST Digits (Cohen và cs., 2017) | 0–9 | 60.000 ảnh train chọn ngẫu nhiên (chỉ huấn luyện) | NIST, dùng tự do |
| Lớp "rác" (tự sinh, `bo_du_lieu/rac.py`) | rác | hai chữ số dính nhau / mảnh cắt của một chữ số, sinh từ chữ số của đúng tập train / validation / test: 8.000 / 1.500 / 1.500 | — |
| CVL Single Digits | 0–9 (chữ người châu Âu) | chỉ 52 người thuộc nửa validation của CVL Strings: 43 người huấn luyện (4.771 ảnh), 9 người kiểm định (969 ảnh); 75 người của nửa test bị loại | CC BY-NC 4.0 |
| CVL Digit Strings | (dãy số) | **không huấn luyện**; chia theo người viết: validation 518 ảnh / 52 người (đã dùng để chọn ngưỡng tách chữ dính, TRƯỚC khi chữ số của những người này vào huấn luyện), test 744 ảnh / 75 người (chỉ báo cáo) | CC BY-NC 4.0 |

HASYv2 không có dấu `=`, nên `=` chỉ đến từ CROHME+. CROHME+ lưu nét bút (InkML); ký hiệu được dựng lại thành ảnh từ
nét. Phần dữ liệu "ghép" của CROHME+ tái sử dụng nét viết tay thật trong nhiều biểu thức, nên phải khử trùng lặp trước
khi chia tập; mã tờ viết (vd. `2011_IVC_depart_F039`) dùng làm xấp xỉ người viết.

Năm cấu hình (`loi/huan_luyen.py`, `BIEN_THE_CAU_HINH`):
- `co_so`: kiến trúc nhỏ, MNIST + ký hiệu, không tăng cường.
- `cai_tien`: như `co_so` + tăng cường (xoay ±10° — không hơn, vì xoay 45° biến `+` thành `×` —, dịch ±2 px, co giãn
  [0,85; 1,1], giãn/co nét 3×3) + biến thể **số 1 có mũ** và **số 7 có gạch** sinh từ MNIST train.
- `co_cvl`: như `co_so` + chữ số viết tay thật của 43 người châu Âu (CVL Single Digits), chiếm 35% số lần lấy mẫu
  của mỗi lớp chữ số.
- `nho_rac`: như `co_cvl` + lớp "rác".
- `lon_rac`: kiến trúc lớn, dữ liệu của `nho_rac` + 60.000 chữ số EMNIST.

Tất cả lấy mẫu cân bằng lớp (ký hiệu chỉ có vài trăm đến hơn nghìn mẫu so với hàng chục nghìn chữ số). Trong một lần
huấn luyện, checkpoint được chọn theo **accuracy cân bằng** (trung bình recall các lớp) trên validation ký tự.

**Lớp "rác" và tách chữ dính.** Khi tách một nhóm rộng, mỗi đoạn được chấm bằng log xác suất của *ký tự thật* tốt nhất
(không tính lớp rác). Một đoạn là nửa chữ số hay hai chữ số dính nhau sẽ được CNN gán phần lớn xác suất cho lớp rác,
nên điểm thấp và quy hoạch động tránh chọn nó. Không có lớp rác, CNN 15 lớp buộc phải gán đoạn đó cho một ký tự và
thường rất tự tin. Ảnh rác: hai chữ số đặt sát/chồng nhau, và mảnh cắt dọc của một chữ số; mảnh nào bị mô hình 15 lớp
có sẵn đọc chắc chắn là một chữ số thì bỏ vì mơ hồ.

**Chọn mô hình cuối bằng cả luồng đọc trên chữ viết thật.** Validation ký tự cùng phân phối với dữ liệu huấn luyện
nên không cho biết cấu hình nào đọc chữ viết thật tốt hơn. Mô hình cuối được chọn trong 5 cấu hình đơn và 10 cặp ghép
(ghép = trung bình softmax; mô hình 15 lớp coi xác suất rác bằng 0) bằng cách cho mỗi ứng viên đọc 90 dãy CVL của
9 người kiểm định; ứng viên kém tốt nhất không quá 2 dãy coi là ngang nhau, trong đó chọn CER thấp nhất. Kết quả là
`nho_rac + lon_rac`. Ngưỡng "2 dãy" được thêm sau khi quy tắc so tuyệt đối chọn một mô hình hơn 1/90 dãy nhưng kém
rõ trên test (ghi trong thuc_nghiem.md mục 2). `PhanLoaiKyTu` đọc `mo_hinh/lua_chon.json`, kiểm tra SHA-256 từng
checkpoint rồi nạp các thành phần.

Ghi chú về quy trình: lỗi "1 có mũ", "7 có gạch" được phát hiện khi soi 158 ảnh CVL (mỗi ảnh thứ 8, lẫn cả hai nửa)
trước khi tách CVL thành validation/test; ý tưởng thêm chữ số châu Âu vì vậy đến từ việc nhìn thấy một phần ảnh test
(dữ liệu và lựa chọn thì không dùng người viết của nửa test).

## 4. Biểu thức

`loi/bieu_thuc.py`: chữ số liền nhau ghép thành số (trong dòng dãy số, khoảng trắng tách số); phân tích cú pháp đệ quy
xuống (× ÷ ưu tiên hơn + −, cho phép dấu trừ một ngôi); tính bằng `fractions.Fraction` nên `1÷3×3` ra đúng `1`; không
dùng `eval`. Hiển thị số lẻ theo kiểu Việt Nam (`3,5`), kèm phân số. Chia cho 0, thiếu số, hai dấu `=` → báo lỗi rõ ràng.

## 5. Đánh giá

- CNN: tập test ký tự (MNIST test + ký hiệu test).
- CVL Digit Strings, nửa test (744 ảnh, 75 người viết không dùng để chọn gì): đúng cả dãy và tỉ lệ lỗi ký tự (CER,
  khoảng cách sửa / độ dài nhãn). Đọc ở chế độ "chỉ chữ số" (trang chỉ có dãy số) và ở chế độ đủ 15 lớp.
- Trang tổng hợp: 60 trang ghép từ ký tự TẬP TEST trên nền giấy có ánh sáng loang, dòng kẻ, nhiễu, xoay ±1,5°; đo tỉ lệ
  dòng đọc đúng hoàn toàn, tỉ lệ biểu thức ra đúng giá trị, CER.
- Ảnh chụp thật: `du_lieu/trang_that/` với nhãn khai báo trước; mỗi thư mục con có `nhan.json` riêng là một bộ riêng
  (vd. `trang_01_cac_dong/`: từng phép tính cắt từ `trang_01.jpg` bằng `doc_trang.py --cat-dong`).
- Thí nghiệm: tắt/đổi từng bước (ngưỡng, chỉnh nghiêng, xóa dòng kẻ, chuẩn hóa nét, tách chữ dính), đo trên CVL
  validation và trang tổng hợp.

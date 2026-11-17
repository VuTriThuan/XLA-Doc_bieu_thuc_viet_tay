# Thực nghiệm

Mọi số liệu dưới đây lấy từ các tệp trong `ket_qua/` và `mo_hinh/lua_chon.json`; chạy lại bằng `python3 danh_gia.py`
và `python3 danh_gia.py --thu-nghiem`. Khoảng tin cậy 95% theo Wilson. Môi trường: macOS, Python 3.10, torch 2.2.2, CPU;
mỗi cấu hình chỉ huấn luyện với một hạt (42).

## 1. CNN trên tập test ký tự (`ket_qua/phan_loai/test_metrics.json`)

Mô hình cuối: **ghép `nho_rac` + `lon_rac`** (trung bình softmax; xem mục 2). Lớp: 15 ký tự + lớp "rác".

| Phần | Đúng / tổng | Accuracy | KTC 95% |
|---|---|---|---|
| Chữ số (MNIST test) | 9770 / 10000 | 97,70% | 97,39–97,98% |
| Ký hiệu (HASYv2 + CROHME+ test) | 694 / 701 | 99,00% | 97,95–99,52% |
| Lớp "rác" (1.500 ảnh sinh từ chữ số TEST) | 1478 / 1500 nhận ra là rác | 98,53% | — |

Accuracy cân bằng (16 lớp): 98,09%. Có 80 trong 10.701 ký tự thật bị gọi nhầm là rác (khi đọc trang, nhãn luôn lấy
từ 15 ký tự thật nên việc này chỉ làm điểm của ký tự đó thấp đi). Recall từng ký hiệu: `+` 244/245, `−` 230/233, `×`
124/124, `÷` 29/29, `=` 67/70.

So với BTL trước (99,15% trên MNIST, 10 lớp), chữ số giảm vì (a) thêm 6 lớp, trong đó có lớp dễ lẫn với chữ số (`4`/`+`,
`1`/`−`, rác), (b) MNIST nay đi qua cùng bước chuẩn hóa ký tự như ảnh chụp, và (c) mô hình học thêm chữ số kiểu châu
Âu (số 1 có mũ giống số 7 kiểu Mỹ) — đổi lại là đọc chữ viết thật tốt hơn nhiều (mục 3).

Dữ liệu ký hiệu có nhiễu nhãn: 41/516 dấu `=` train và 8/70 dấu `=` test của CROHME+ có gạch trên chỉ là một chấm
(nét bút bị cắt khi tách ký hiệu), 13/210 dấu `÷` train thiếu chấm. Không lọc bỏ để giữ nguyên dữ liệu gốc.

## 2. Các cấu hình và cách chọn mô hình (`mo_hinh/lua_chon.json`)

Năm cấu hình (cùng 12 epoch, hạt 42, lấy mẫu cân bằng lớp):

| Cấu hình | Kiến trúc | Dữ liệu huấn luyện | Lớp |
|---|---|---|---|
| `co_so` | nhỏ (57 nghìn tham số, như BTL trước) | MNIST + ký hiệu | 15 |
| `cai_tien` | nhỏ | + tăng cường (xoay, dịch, co giãn, đổi nét) + biến thể "1 có mũ / 7 có gạch" sinh tự động | 15 |
| `co_cvl` | nhỏ | MNIST + ký hiệu + 4.771 chữ số CVL thật của 43 người châu Âu | 15 |
| `nho_rac` | nhỏ | như `co_cvl` + 8.000 ảnh "rác" | 16 |
| `lon_rac` | lớn (4 lớp tích chập + BatchNorm, 0,47 triệu tham số) | như `nho_rac` + 60.000 chữ số EMNIST | 16 |

**Người viết của CVL được chia tách hoàn toàn.** CVL Single Digits được cắt từ chính các dãy của CVL Strings (cùng 127
người). Chữ số của 75 người thuộc nửa test KHÔNG có trong huấn luyện; trong 52 người còn lại, 43 người dùng để huấn
luyện, **9 người để kiểm định** (969 chữ số, 90 dãy số).

**Lớp "rác"** (`bo_du_lieu/rac.py`): ảnh hai chữ số đặt sát/chồng lên nhau (60%) và mảnh cắt dọc của một chữ số (40%),
sinh từ chữ số đã chuẩn hóa của đúng tập tương ứng; mảnh nào bị hai mô hình 15 lớp có sẵn đọc chắc chắn là một chữ số
(> 0,8) bị bỏ vì mơ hồ. Khi tách chữ dính, điểm của một đoạn là log xác suất của ký tự thật tốt nhất, nên đoạn bị CNN
coi là rác có điểm thấp và quy hoạch động tránh chọn nó.

**Chọn mô hình bằng cả luồng đọc.** Mỗi ứng viên (5 cấu hình đơn và 10 cặp ghép) đọc 90 dãy CVL của 9 người kiểm định
(chế độ chỉ chữ số, luồng đọc hiện tại có đường cắt drop-fall — mục 5). Ứng viên đọc đúng ít hơn ứng viên tốt nhất
không quá **2/90 dãy** được coi là ngang nhau; trong nhóm đó chọn CER thấp nhất, rồi accuracy chữ số CVL của 9 người đó.
Không dùng tập test ký tự, không dùng CVL test.

| Ứng viên (ngang nhau: đúng ≥ 39/90) | Dãy đúng / 90 | CER | Chữ số CVL kiểm định (969) | MNIST validation |
|---|---|---|---|---|
| `nho_rac` | 41 | 16,31% | 91,43% | 96,70% |
| `co_cvl + nho_rac` | 40 | 17,92% | 91,54% | 97,22% |
| `co_cvl + lon_rac` | 40 | 15,95% | 92,78% | 97,90% |
| **`nho_rac + lon_rac`** (được chọn: CER thấp nhất) | 39 | **15,77%** | 92,36% | 97,85% |
| các ứng viên khác (`lon_rac` 38, `co_cvl` 34, …) | ≤ 38 | — | — | — |

**Ghi chú trung thực về quy trình chọn.** Quy tắc ban đầu chọn ứng viên có số dãy đúng cao nhất tuyệt đối. Sau khi thêm
đường cắt drop-fall, quy tắc đó chọn `nho_rac` (hơn 1/90 dãy) — và trên tập test `nho_rac` kém rõ (56,45% so với
59,54% của mô hình trước; trang ảnh thật 26/32 so với 29/32). Chênh 1/90 dãy là nhiễu, nên quy tắc được sửa thành
"chênh không quá 2 dãy coi là ngang nhau, rồi so CER". Việc sửa này được làm SAU khi đã thấy điểm test của lựa chọn
cũ; tổ hợp được chọn theo quy tắc mới (`nho_rac + lon_rac`) thì chưa từng được đo trên test trước đó. Con số test ở
mục 3 vì vậy có thể lạc quan nhẹ. Cách khắc phục đúng là một tập kiểm định lớn hơn (vd. chữ của nhiều người trong lớp).

Bài học:
- **Validation cùng nguồn với huấn luyện đánh lừa**: `cai_tien` ngang `co_so` trên validation ký tự nhưng kém hẳn trên
  chữ viết thật. Biến thể kiểu viết sinh tự động không thay được chữ thật.
- **Dữ liệu thật đúng kiểu chữ** (chữ số CVL) là bước tăng lớn nhất; **lớp rác** và **mạng lớn + EMNIST** mỗi bước
  tăng thêm vài điểm (mục 3).

## 3. Đọc dãy số viết tay thật: CVL Digit Strings, nửa test (`ket_qua/cvl/test/`)

744 ảnh của 75 người viết không có trong huấn luyện và không dùng để chọn gì. Mỗi ảnh là một dãy 5–7 chữ số.

Luồng đọc hiện tại (có đường cắt drop-fall):

| Mô hình (chế độ chỉ chữ số) | Đúng cả dãy | Tỉ lệ | KTC 95% | CER |
|---|---|---|---|---|
| `co_so` (MNIST + ký hiệu) | 234 / 744 | 31,45% | 28,22–34,88% | 22,12% |
| `cai_tien` | 191 / 744 | 25,67% | 22,66–28,93% | 24,14% |
| `co_cvl` (+ chữ số CVL) | 382 / 744 | 51,34% | 47,75–54,92% | 15,34% |
| `nho_rac` (+ lớp rác) | 420 / 744 | 56,45% | 52,86–59,97% | 12,48% |
| `lon_rac` (+ mạng lớn, EMNIST) | 452 / 744 | 60,75% | 57,20–64,20% | 10,29% |
| **Ghép `nho_rac + lon_rac`** (được chọn) | **463 / 744** | **62,23%** | 58,69–65,64% | **10,10%** |
| Ghép `nho_rac + lon_rac`, đủ 15 lớp | 450 / 744 | 60,48% | 56,93–63,93% | 10,96% |

Mỗi bước nâng cấp đều tăng trên tập test độc lập: thêm chữ số CVL +19,9 điểm, thêm lớp rác +5,1 điểm, thêm mạng lớn và
EMNIST +4,3 điểm, ghép hai CNN có lớp rác +1,5 điểm. So với luồng cắt thẳng (mục 5), đường cắt drop-fall tăng từng CNN
0,5–1,9 điểm.

Theo từng dãy (mô hình được chọn): 41–52/75 với mọi dãy trừ `25000` (35/73 — ba số 0 viết liền nét thành một khối).

Phân tích lỗi (281 ảnh sai): sai độ dài (thừa 77, thiếu 35), thay ký tự `7 → 2` (13), `9 → 8` (13), `9 → 3` (11),
`9 → 4` (10), `4 → 6` (10). Số ảnh sai độ dài: 180 (trước lớp rác) → 119 (có lớp rác) → 112 (thêm drop-fall).

Lưu ý quy trình: lỗi "1 có mũ / 7 có gạch" được phát hiện khi soi 158 ảnh CVL (mỗi ảnh thứ 8, lẫn cả hai nửa) TRƯỚC khi
tách CVL thành validation/test. Ý tưởng dùng chữ số châu Âu vì vậy đến từ việc nhìn thấy một phần ảnh test; dữ liệu
huấn luyện và mọi lựa chọn đều không dùng người viết của nửa test. Các ngưỡng tách chữ dính được chọn trên nửa
validation của CVL Strings khi người viết của nửa đó CHƯA có trong huấn luyện.

## 4. Trang tổng hợp (`ket_qua/trang_tong_hop/mac_dinh.json`)

60 trang, 360 dòng (244 dòng là biểu thức), ký tự lấy từ tập TEST (MNIST test, ký hiệu test), nền giấy ánh sáng loang,
50% có dòng kẻ vở, nhiễu Gauss, xoay ±1,5°.

| Chỉ số | Kết quả | KTC 95% |
|---|---|---|
| Dòng đọc đúng hoàn toàn | 280 / 360 = **77,78%** | 73,20–81,77% |
| Biểu thức ra đúng giá trị | 177 / 244 = **72,54%** | 66,62–77,76% |
| CER toàn trang | 5,37% | — |

Lỗi chính (dòng cùng độ dài): `= → −` (17, phần lớn do mẫu `=` dị dạng của CROHME+ — gạch trên chỉ là một chấm, bị
lọc như nhiễu), `÷ → +` (12), `7 → 1` (4); 19 dòng sai độ dài. Đây là ảnh ghép: các ký tự không bao giờ dính nhau và
mỗi ký tự là một mẫu viết tay khác người; nó đo luồng tách nền/tách dòng/tách ký tự, không thay được ảnh chụp thật.

## 5. Bật/tắt từng bước xử lý ảnh (`ket_qua/thu_nghiem_tien_xu_ly.csv`)

**Đường cắt khi tách chữ dính** (`ket_qua/quet_duong_cat.csv`, mô hình `co_cvl + lon_rac`, chỉ dữ liệu kiểm định):

| Đường cắt drop-fall | Chỗ cắt từ thung lũng đường viền | 90 dãy kiểm định | 518 dãy nửa validation CVL | Trang tổng hợp: dòng đúng |
|---|---|---|---|---|
| không (chỉ cắt thẳng đứng) | không | 40 (CER 16,13%) | 318 = 61,39% (CER 10,43%) | 79,44% |
| không | có | 40 (CER 15,23%) | 325 = 62,74% (CER 9,69%) | 79,44% |
| **có** | **không** (được chọn) | 40 (CER 15,95%) | **347 = 66,99%** (CER 8,66%) | 78,89% |
| có | có | 39 (CER 15,95%) | 347 = 66,99% (CER 8,60%) | 78,89% |

Drop-fall: thả một "giọt" từ mép trên (hoặc dưới) xuống qua nhóm nét, đi thẳng khi ô dưới là nền, đi chéo hoặc trượt
ngang tìm khe khi bị chặn, chỉ cắt xuyên nét khi không còn khe — nên tách được hai chữ viết nghiêng chồng nhau theo cột
mà đường thẳng đứng phải cắt ngang cả hai (`ket_qua/minh_hoa_duong_cat.png`). Mỗi chỗ cắt ứng viên sinh tối đa 5 đường
(thẳng; từ trên/dưới, ưu tiên trái/phải), quy hoạch động chọn. Trên 518 dãy, drop-fall tăng +29 dãy; trên 90 dãy gần
như không đổi (bộ nhỏ). Chỗ cắt từ đường viền không cộng thêm khi đã có drop-fall nên để tắt. Lưu ý: chữ số của 43/52
người viết của nửa validation có trong huấn luyện, nên con số tuyệt đối ở cột 518 dãy lạc quan; chỉ đọc chênh lệch.
Trên tập test, drop-fall tăng từng CNN 0,5–1,9 điểm (mục 3) — ít hơn mức +5,6 điểm thấy trên nửa validation.

**Các bước khác** (bảng dưới):

**Bảng bật/tắt dưới đây đo với mô hình của phiên bản trước (`co_so + co_cvl`), chưa chạy lại với mô hình cuối** (mỗi
lần chạy mất khoảng 40 phút); kết luận về các bước xử lý ảnh không phụ thuộc mô hình nhưng con số tuyệt đối sẽ khác.
Bảng quét ngưỡng tách chữ dính dưới đây chạy với mô hình `co_so`, TRƯỚC khi thêm tách cột, các luật ghép mới (mục 6)
và chữ số CVL. Bảng bật/tắt chạy với mã và mô hình hiện tại; vì chữ số của 43/52 người viết thuộc CVL validation nay
có trong huấn luyện, cột "CVL val" của bảng này lạc quan về mức tuyệt đối — chỉ nên đọc chênh lệch giữa các dòng.

Đo trên CVL **validation** (518 ảnh, chế độ chỉ chữ số) và 60 trang tổng hợp.

| Cấu hình | CVL val: đúng cả dãy | CVL val: CER | Trang: dòng đúng | Trang: biểu thức đúng | Trang: CER |
|---|---|---|---|---|---|
| **Mặc định** | 48,65% | 15,48% | 76,94% | 70,49% | 5,82% |
| Otsu toàn cục thay cho chuẩn hóa nền | 48,65% | 15,48% | 70,56% | 67,21% | **84,13%** |
| Sauvola thay cho chuẩn hóa nền | 37,84% | 35,16% | 68,33% | 64,34% | 8,94% |
| Không tách chữ dính | 34,94% | 21,68% | 78,33% | 72,13% | 5,13% |
| Không chuẩn hóa độ dày nét | 48,07% | 15,26% | 78,89% | 72,95% | 5,46% |
| Không xóa dòng kẻ | 48,46% | 15,51% | 76,94% | 70,49% | 5,78% |
| Không chỉnh nghiêng | 51,74% | 14,29% | 76,11% | 70,90% | 5,62% |

Nhận xét:
- **Chuẩn hóa nền là bước quan trọng nhất trên ảnh chụp**: Otsu toàn cục làm CER trang tăng từ 5,8% lên 84,1% (các vùng
  giấy tối bị coi là nét, sinh hàng loạt ký tự giả). Trên CVL (ảnh quét nền đều) hai cách như nhau. Sauvola làm đứt nét
  mực nhạt và rỗng ruột nét đậm nên kém cả hai bộ.
- **Tách chữ dính** tăng CVL từ 34,9% lên 48,7% nhưng giảm trang tổng hợp 1,4 điểm, vì trang tổng hợp không có chữ dính
  nên mọi lần cắt đều là cắt nhầm (chữ số đơn lẻ viết rộng). Ngưỡng được chọn bằng quét
  (`ket_qua/quet_nguong_tach_dinh.csv`, `quet_nguong_tach_dinh_2.csv`, mô hình `co_so`, CVL validation lúc chưa có
  trong huấn luyện):

  | Bắt buộc cắt khi rộng > | Biên (nat) | CVL val đúng cả dãy | Trang: dòng đúng |
  |---|---|---|---|
  | 1,3 × chiều cao chữ | 0,7 | 31,66% | 72,78% |
  | 1,3 | 0 | 33,01% | 71,67% |
  | **1,6** | **0** | **33,40%** | 76,11% |
  | 1,6 | 0,3 | 30,69% | 78,06% |
  | 2,0 | 0 | 33,01% | 76,67% |
  | 2,0 | 0,7 | 26,25% | 78,61% |
- **Chuẩn hóa độ dày nét, xóa dòng kẻ, chỉnh nghiêng**: không cải thiện đo được trên hai bộ này. Tắt chỉnh nghiêng
  còn nhỉnh hơn trên CVL (51,7% so với 48,7%) vì ảnh CVL đã thẳng và ước lượng góc trên một dòng ngắn có thể lệch;
  tắt chuẩn hóa nét nhỉnh hơn trên trang tổng hợp (78,9% so với 76,9%). Các chênh lệch này nằm trong khoảng tin cậy
  (khoảng ±4 điểm với 518 ảnh, ±4,5 điểm với 360 dòng). Các bước được giữ vì nhắm vào ảnh chụp điện thoại thật (giấy
  nghiêng nhiều, bút to/nhỏ, vở kẻ đậm) — cần thêm ảnh chụp thật để kiểm chứng; nếu không có bằng chứng, có thể cân
  nhắc tắt chỉnh nghiêng mặc định cho ảnh một dòng.

## 6. Ảnh chụp thật (`ket_qua/trang_that/`)

`du_lieu/trang_that/trang_01.jpg`: một trang giấy A4 chụp bằng điện thoại, **một người viết** (Vũ Trí Thuận), 32 dòng
xếp **hai cột**, giấy có chữ in mặt sau thấm qua, một chỗ gạch xóa. Nhãn 32 dòng ghi vào `nhan.json` trước khi chạy.
`trang_01_cac_dong/` là 32 ảnh từng phép tính do `doc_trang.py --cat-dong` cắt từ chính trang đó.

| Bộ | Dòng đọc đúng hoàn toàn | KTC 95% | Biểu thức ra đúng giá trị | CER |
|---|---|---|---|---|
| Cả trang (tách cột, dòng, ký tự) | **28 / 32 (87,50%)** | 71,93–95,03% | **23 / 28 (82,14%)** | 1,69% |
| 32 ảnh cắt từng phép tính | 27 / 32 (84,38%) | 68,25–93,14% | 22 / 28 (78,57%) | 3,40% |

Bốn dòng sai (cả trang):

| Trên giấy | Máy đọc | Nguyên nhân |
|---|---|---|
| `90347×10=9` | `90347×0=9` | số 1 dính vào số 0 nhưng nhóm chưa đủ rộng để xét tách |
| `4953×5349=7624513` | `4953×534927624513` | dấu `=` bị đọc thành `2` |
| `261794×3=` | `264734×3=` | 1 → 4, 9 → 3 |
| `69-20÷10+6=` | `69-20÷1046=` | `+` → 4 |

Qua các phiên bản mô hình, trang này đọc đúng: `co_so` 27/32, `co_cvl` 26/32, `co_so + co_cvl` 28/32,
`co_cvl + lon_rac` (cắt thẳng) 29/32, `nho_rac` 26/32, mô hình cuối 28/32. Một trang 32 dòng của một người viết dao
động 1–3 dòng giữa các mô hình; không nên kết luận từ chênh lệch nhỏ ở đây.

**Cảnh báo về độ lạc quan**: trang này được dùng để phát hiện và sửa ba lỗi của khâu tách (tách cột, luật ghép số 1
nghiêng, ghép số 4 hai nét, bỏ vết gạch xóa) trước khi ra con số trên. Lần chạy đầu tiên, trước các sửa đổi đó, mỗi
dòng dính hai biểu thức của hai cột nên không dòng nào tính được. Con số 27/32 vì vậy là kết quả trên dữ liệu đã nhìn
thấy khi phát triển, chỉ có một người viết, không phải kết quả kiểm định độc lập. Cần thêm trang mới (lý tưởng là của
người khác) chưa từng dùng khi sửa mã. Đo với mô hình `co_so` lúc đó, các sửa đổi này không làm giảm kết quả trên CVL
validation (33,4% → 33,2%) hay trang tổng hợp (76,1% → 76,9%); CVL test giảm 6 ảnh (31,45% → 30,65%), trong khoảng
tin cậy.

## 7. Hạn chế

- Đọc đúng cả dãy trên chữ viết thật của người lạ: 62,23% với dãy 5–7 chữ số (CER 10,1%), còn thấp hơn các hệ thống
  chuyên dụng (đọc cả dòng bằng CRNN). Nguồn lỗi lớn nhất vẫn là tách chữ viết liền nét (sai độ dài), sau đó là nhầm
  7/2, 9/8.
- Tập kiểm định dùng để chọn mô hình chỉ có 90 dãy của 9 người; quy tắc chọn đã phải sửa một lần sau khi thấy điểm test
  (mục 2).
- Ký hiệu `÷` (210 mẫu train) và `=` (516 mẫu train, ~8% dị dạng) ít dữ liệu nhất và là nguồn lỗi chính trên trang.
- Mô hình cuối ghép hai CNN (một nhỏ, một lớn) nên đọc chậm hơn khoảng hai lần so với một CNN nhỏ.
- Không hỗ trợ ngoặc, số thập phân, phân số viết dọc, chỉ số trên/dưới; biểu thức phải nằm trên một dòng.
- Điểm softmax không được hiệu chuẩn; ngưỡng "nghi ngờ" 0,6 chỉ để gợi ý người dùng kiểm tra.
- Mỗi cấu hình chỉ huấn luyện một lần (một hạt), chưa đo độ dao động giữa các lần chạy.
- Ảnh chụp thật mới có một trang của một người viết, và trang đó đã được dùng khi sửa khâu tách (mục 6).

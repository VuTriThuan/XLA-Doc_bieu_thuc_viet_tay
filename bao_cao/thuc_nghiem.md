# Thực nghiệm

Mọi số liệu dưới đây lấy từ các tệp trong `ket_qua/` và `mo_hinh/lua_chon.json`; chạy lại bằng
`python3 huan_luyen.py --chi-chon`, `python3 danh_gia.py --cvl --so-sanh-cat-thang`, `python3 danh_gia.py --trang`,
`python3 danh_gia.py --thu-nghiem`, `python3 danh_gia.py --quet-duong-cat` và `python3 danh_gia.py --phan-tich-loi`.

**Đo lại ngày 18/11/2026** sau khi sửa lỗi ở khâu tách chữ dính: ảnh của cả nhóm và mảnh từ mép trái tới đường cắt đầu
tiên dùng chung một khóa trong bộ nhớ đệm, nên khi nhóm có đường cắt, mảnh đầu tiên bị thay bằng ảnh cả nhóm. Lỗi chỉ ảnh
hưởng luồng có đường cắt drop-fall (kết quả chỉ cắt thẳng trùng khớp với trước khi có lỗi). Sau khi sửa, mọi bảng dưới
đây được chạy lại với cùng một mã và mô hình; quy tắc chọn mô hình không đổi và vẫn chọn `nho_rac + lon_rac`. Khoảng tin cậy 95% theo Wilson. Môi trường: macOS, Python 3.10, torch 2.2.2, CPU;
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

So với một CNN chỉ phân loại 10 chữ số (khoảng 99% trên MNIST), chữ số giảm vì (a) thêm 6 lớp, trong đó có lớp dễ lẫn với chữ số (`4`/`+`,
`1`/`−`, rác), (b) MNIST nay đi qua cùng bước chuẩn hóa ký tự như ảnh chụp, và (c) mô hình học thêm chữ số kiểu châu
Âu (số 1 có mũ giống số 7 kiểu Mỹ) — đổi lại là đọc chữ viết thật tốt hơn nhiều (mục 3).

Dữ liệu ký hiệu có nhiễu nhãn: 41/516 dấu `=` train và 8/70 dấu `=` test của CROHME+ có gạch trên chỉ là một chấm
(nét bút bị cắt khi tách ký hiệu), 13/210 dấu `÷` train thiếu chấm. Không lọc bỏ để giữ nguyên dữ liệu gốc.

## 2. Các cấu hình và cách chọn mô hình (`mo_hinh/lua_chon.json`)

Năm cấu hình (cùng 12 epoch, hạt 42, lấy mẫu cân bằng lớp):

| Cấu hình | Kiến trúc | Dữ liệu huấn luyện | Lớp |
|---|---|---|---|
| `co_so` | nhỏ (57 nghìn tham số) | MNIST + ký hiệu | 15 |
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
Điểm test không dùng để chấm ứng viên (nhưng đã ảnh hưởng tới quy tắc, xem ghi chú bên dưới).

| Ứng viên (ngang nhau: đúng ≥ 39/90) | Dãy đúng / 90 | CER | Chữ số CVL kiểm định (969) | MNIST validation |
|---|---|---|---|---|
| `nho_rac` | 41 | 16,31% | 91,43% | 96,70% |
| `co_cvl + nho_rac` | 40 | 17,74% | 91,54% | 97,22% |
| `co_cvl + lon_rac` | 40 | 15,59% | 92,78% | 97,90% |
| **`nho_rac + lon_rac`** (được chọn: CER thấp nhất) | 40 | **15,23%** | 92,36% | 97,85% |
| `lon_rac` | 39 | 16,13% | 92,88% | 97,72% |
| các ứng viên khác (`co_cvl` 34, `co_so + lon_rac` 34, …) | ≤ 34 | — | — | — |

**Ghi chú trung thực về quy trình chọn.** Quy tắc ban đầu chọn ứng viên có số dãy đúng cao nhất tuyệt đối. Sau khi thêm
đường cắt drop-fall, quy tắc đó chọn `nho_rac` (hơn 1/90 dãy) — và trên tập test `nho_rac` kém rõ (khi đó 56,45% so với
59,54% của mô hình trước; trang ảnh thật 26/32 so với 29/32). Chênh 1/90 dãy là nhiễu, nên quy tắc được sửa thành
"chênh không quá 2 dãy coi là ngang nhau, rồi so CER". Việc sửa này được làm SAU khi đã thấy điểm test của lựa chọn
cũ; tổ hợp được chọn theo quy tắc mới (`nho_rac + lon_rac`) thì chưa từng được đo trên test trước đó. Con số test ở
mục 3 vì vậy có thể lạc quan nhẹ. `mo_hinh/lua_chon.json` ghi rõ hai ý này: `test_dung_de_cham_ung_vien: false`,
`diem_test_anh_huong_quy_tac_chon: true`. Cách khắc phục đúng là một tập kiểm định lớn hơn (vd. chữ của nhiều người trong lớp).

Bài học:
- **Validation cùng nguồn với huấn luyện đánh lừa**: `cai_tien` ngang `co_so` trên validation ký tự nhưng kém hẳn trên
  chữ viết thật. Biến thể kiểu viết sinh tự động không thay được chữ thật.
- **Dữ liệu thật đúng kiểu chữ** (chữ số CVL) là bước tăng lớn nhất; **lớp rác** và **mạng lớn + EMNIST** mỗi bước
  tăng thêm vài điểm (mục 3).

## 3. Đọc dãy số viết tay thật: CVL Digit Strings, nửa test (`ket_qua/cvl/test/`)

744 ảnh của 75 người viết không có trong huấn luyện. Mỗi ảnh là một dãy 5–7 chữ số. Điểm test không dùng để chấm ứng
viên, nhưng quy tắc chọn đã sửa một lần sau khi thấy điểm test (mục 2).

| Mô hình (chế độ chỉ chữ số) | Chỉ cắt thẳng | Có drop-fall (mặc định) | KTC 95% | CER |
|---|---|---|---|---|
| `co_so` (MNIST + ký hiệu) | 228 (30,65%) | 233 / 744 = 31,32% | 28,09–34,74% | 22,21% |
| `cai_tien` | 189 (25,40%) | 193 / 744 = 25,94% | 22,92–29,21% | 24,36% |
| `co_cvl` (+ chữ số CVL) | 374 (50,27%) | 385 / 744 = 51,75% | 48,16–55,32% | 15,28% |
| `nho_rac` (+ lớp rác) | 406 (54,57%) | 424 / 744 = 56,99% | 53,40–60,50% | 12,44% |
| `lon_rac` (+ mạng lớn, EMNIST) | 448 (60,22%) | 458 / 744 = 61,56% | 58,01–64,99% | 10,10% |
| **Ghép `nho_rac + lon_rac`** (được chọn) | 450 (60,48%) | **467 / 744 = 62,77%** | 59,24–66,17% | **10,03%** |
| Ghép `nho_rac + lon_rac`, đủ 15 lớp | — | 455 / 744 = 61,16% | 57,60–64,59% | 10,90% |

Mỗi bước nâng cấp đều tăng trên tập test: thêm chữ số CVL +20,4 điểm, thêm lớp rác +5,2 điểm, thêm mạng lớn và EMNIST
+4,6 điểm, ghép hai CNN có lớp rác +1,2 điểm. Đường cắt drop-fall tăng từng CNN 0,5–2,4 điểm so với chỉ cắt thẳng
(mô hình cuối +17 dãy).

Theo từng dãy (mô hình được chọn): 42–52/75 với mọi dãy trừ `25000` (36/73 — ba số 0 viết liền nét thành một khối).

Phân tích lỗi (`ket_qua/cvl/test/phan_tich_loi.csv`):

| Mô hình | Dãy sai | Thừa ký tự | Thiếu ký tự | Sai đúng 1 chữ số | Ba cặp nhầm nhiều nhất |
|---|---|---|---|---|---|
| `co_so` | 511 | 120 | 66 | 197 | 1→4 (76), 1→7 (55), 4→6 (46) |
| `cai_tien` | 551 | 131 | 64 | 187 | 1→4 (146), 4→6 (51), 9→3 (49) |
| `co_cvl` | 359 | 100 | 71 | 128 | 9→8 (29), 4→6 (22), 9→3 (16) |
| `nho_rac` | 320 | 93 | 28 | 143 | 9→8 (21), 4→6 (21), 5→8 (11) |
| `lon_rac` | 286 | 81 | 29 | 142 | 7→2 (23), 9→4 (15), 9→3 (12) |
| Ghép (cuối) | 277 | 82 | 29 | 132 | 7→2 (14), 9→8 (13), 9→3 (10) |

Lưu ý quy trình: lỗi "1 có mũ / 7 có gạch" được phát hiện khi soi 158 ảnh CVL (mỗi ảnh thứ 8, lẫn cả hai nửa) TRƯỚC khi
tách CVL thành validation/test. Ý tưởng dùng chữ số châu Âu vì vậy đến từ việc nhìn thấy một phần ảnh test; dữ liệu
huấn luyện và mọi lựa chọn đều không dùng người viết của nửa test. Các ngưỡng tách chữ dính được chọn trên nửa
validation của CVL Strings khi người viết của nửa đó CHƯA có trong huấn luyện.

## 4. Trang tổng hợp (`ket_qua/trang_tong_hop/mac_dinh.json`)

60 trang, 360 dòng (244 dòng là biểu thức), ký tự lấy từ tập TEST (MNIST test, ký hiệu test), nền giấy ánh sáng loang,
50% có dòng kẻ vở, nhiễu Gauss, xoay ±1,5°. Không có dòng chia cho 0.

| Chỉ số | Kết quả | KTC 95% |
|---|---|---|
| Dòng đọc đúng hoàn toàn | 279 / 360 = **77,50%** | 72,91–81,51% |
| Biểu thức ra đúng giá trị | 177 / 244 = **72,54%** | 66,62–77,76% |
| CER toàn trang | 5,37% | — |

Lỗi chính (thay ký tự trong đoạn cùng độ dài): `= → −` (15, phần lớn do mẫu `=` dị dạng của CROHME+ — gạch trên chỉ là
một chấm, bị lọc như nhiễu), `÷ → +` (12); 22 dòng sai độ dài. Đây là ảnh ghép: các ký tự không bao giờ dính nhau và
mỗi ký tự là một mẫu viết tay khác người; nó đo luồng tách nền/tách dòng/tách ký tự, không thay được ảnh chụp thật.

## 5. Bật/tắt từng bước xử lý ảnh (`ket_qua/thu_nghiem_tien_xu_ly.csv`)

Mã và mô hình cuối, đo lại 18/11/2026. Cột "CVL val" là 518 dãy nửa validation: chữ số của 43/52 người viết của tập này
có trong huấn luyện, nên mức tuyệt đối lạc quan — chỉ đọc chênh lệch giữa các dòng. Cột "90 KĐ" là 9 người hoàn toàn
mới (sạch nhưng nhỏ, chênh dưới 3 dãy là nhiễu). Cột "Ảnh thật" là trang duy nhất của nhóm, đã dùng khi phát triển.

| Cấu hình | CVL val: đúng cả dãy | CVL val: CER | 90 KĐ | Trang TH: dòng đúng | Trang TH: CER | Ảnh thật: dòng | Ảnh thật: CER |
|---|---|---|---|---|---|---|---|
| **Mặc định** | 354 (68,34%) | 7,82% | 40 | 279 (77,50%) | 5,37% | 28 | 1,35% |
| Otsu toàn cục thay chuẩn hóa nền | 354 (68,34%) | 7,82% | 40 | 254 (70,56%) | **86,22%** | **0** | 80,41% |
| Sauvola thay chuẩn hóa nền | 262 (50,58%) | 28,90% | 30 | 246 (68,33%) | 8,90% | 1 | 3,38% |
| Không chỉnh nghiêng | 358 (69,11%) | 7,75% | 40 | 274 (76,11%) | 5,17% | 28 | 1,35% |
| Không xóa dòng kẻ | 354 (68,34%) | 7,88% | 40 | 279 (77,50%) | 5,33% | 28 | 1,35% |
| Không phép đóng 3×3 | 345 (66,60%) | 8,47% | 41 | 292 (81,11%) | 4,43% | 30 | 0,68% |
| Không tách cột | 354 (68,34%) | 7,82% | 40 | 274 (76,11%) | 5,33% | **0** | 68,24% |
| Không bỏ vết gạch xóa | 355 (68,53%) | 7,75% | 41 | 279 (77,50%) | 5,37% | 27 | 2,70% |
| Không gộp mẩu nét đầu dòng | 343 (66,22%) | 8,56% | 37 | 274 (76,11%) | 5,78% | 28 | 1,35% |
| Không ghép nét bằng CNN | 349 (67,37%) | 8,28% | 41 | 279 (77,50%) | 5,37% | 27 | 2,03% |
| Chỉ cắt thẳng (không drop-fall) | 335 (64,67%) | 8,81% | 37 | 281 (78,06%) | 5,29% | 28 | 1,35% |
| Không tách chữ dính | 216 (41,70%) | 19,53% | 19 | 285 (79,17%) | 4,84% | 25 | 3,72% |
| Không chuẩn hóa độ dày nét | 339 (65,44%) | 8,69% | 36 | 277 (76,94%) | 5,50% | 28 | 1,35% |

Nhận xét:
- **Chuẩn hóa nền** quyết định ảnh chụp có đọc được hay không (Otsu toàn cục: 0/32 dòng, CER trang tổng hợp 86%). Trên
  CVL (ảnh quét nền đều) hai cách như nhau. Sauvola làm đứt nét mực nhạt và rỗng ruột nét đậm.
- **Tách chữ dính** quan trọng nhất với chữ viết liền nét (90 dãy sạch: 40 → 19 khi tắt), nhưng trang tổng hợp không có
  chữ dính nên tắt nó lại được thêm 6 dòng. **Drop-fall** thêm 19 dãy CVL val và 3/90 dãy sạch so với cắt thẳng; trên
  ảnh thật hai cách như nhau.
- **Tách cột** bắt buộc với trang hai cột (0/32 khi tắt), không ảnh hưởng CVL.
- **Phép đóng 3×3**: có lợi trên CVL val (−9 dãy khi tắt) nhưng lấp khe của `÷` (thành `+`; 9/16 dòng trang tổng hợp
  được sửa khi tắt) và `=`; trên ảnh thật tắt nó được 30/32. Ảnh thật đã dùng khi phát triển nên chưa đủ để đổi mặc
  định; nay bật/tắt được bằng `--khong-dong-hinh-thai` và trên giao diện để thử trên ảnh mới.
- **Chỉnh nghiêng**: giúp trang tổng hợp (+5 dòng) nhưng hơi hại ảnh một dòng của CVL (tắt: sửa 17, hỏng 13 dãy) vì ước
  lượng góc trên một dòng ngắn không ổn định.
- **Chuẩn hóa độ dày nét** (+15 dãy CVL val, +4/90), **gộp mẩu nét** (+11 dãy, cả 11 dãy hỏng khi tắt đều do đọc thừa
  một chữ số), **ghép bằng CNN** (+5 dãy, sửa `13×4=52`), **bỏ vết gạch xóa** (sửa `743260`): mỗi bước vài dãy/dòng.
- **Xóa dòng kẻ**: chưa đo được tác dụng (dòng kẻ của trang tổng hợp nhạt, đã bị loại ở bước phân ngưỡng).

**Đường cắt khi tách chữ dính** (`ket_qua/quet_duong_cat.csv`, mô hình cuối, chỉ dữ liệu kiểm định):

| Đường cắt drop-fall | Chỗ cắt từ thung lũng đường viền | 90 dãy kiểm định | 518 dãy nửa validation CVL | Trang tổng hợp: dòng đúng |
|---|---|---|---|---|
| không (chỉ cắt thẳng đứng) | không | 37 (CER 15,95%) | 335 = 64,67% (CER 8,81%) | 78,06% |
| không | có | 39 (CER 15,41%) | 344 = 66,41% (CER 8,35%) | 78,06% |
| **có** | **không** (mặc định) | 40 (CER 15,23%) | **354 = 68,34%** (CER 7,82%) | 77,50% |
| có | có | 40 (CER 14,70%) | 353 = 68,15% (CER 7,79%) | 77,50% |

Drop-fall: thả một "giọt" từ mép trên (hoặc dưới) xuống qua nhóm nét, đi thẳng khi ô dưới là nền, đi chéo hoặc trượt
ngang tìm khe khi bị chặn, chỉ cắt xuyên nét khi không còn khe — nên tách được hai chữ viết nghiêng chồng nhau theo cột
mà đường thẳng đứng phải cắt ngang cả hai (`ket_qua/minh_hoa_duong_cat.png`). Mỗi chỗ cắt ứng viên sinh tối đa 5 đường
(thẳng; từ trên/dưới, ưu tiên trái/phải), quy hoạch động chọn. Thêm chỗ cắt từ thung lũng đường viền khi đã có
drop-fall: 90 dãy ngang nhau (40/40, CER giảm nhẹ), 518 dãy kém 1 dãy — không đủ để bật, nên giữ tắt (mặc định chọn
trước khi đo lại, không đổi sau khi đo).

**Ngưỡng bắt buộc tách** (`ket_qua/quet_nguong_tach_dinh.csv`, `quet_nguong_tach_dinh_2.csv`): quét ở giai đoạn đầu với
mô hình `co_so`, chỉ cắt thẳng (không bị lỗi khóa nói ở đầu tệp), khi người viết của nửa validation chưa có trong huấn
luyện; số liệu này là lịch sử và không chạy lại:

| Bắt buộc cắt khi rộng > | Biên (nat) | CVL val đúng cả dãy | Trang: dòng đúng |
|---|---|---|---|
| 1,3 × chiều cao chữ | 0,7 | 31,66% | 72,78% |
| 1,3 | 0 | 33,01% | 71,67% |
| **1,6** | **0** | **33,40%** | 76,11% |
| 1,6 | 0,3 | 30,69% | 78,06% |
| 2,0 | 0 | 33,01% | 76,67% |
| 2,0 | 0,7 | 26,25% | 78,61% |

## 6. Ảnh chụp thật (`ket_qua/trang_that/`)

`du_lieu/trang_that/trang_01.jpg`: một trang giấy A4 chụp bằng điện thoại, **một người viết** (Vũ Trí Thuận), 32 dòng
xếp **hai cột**, giấy có chữ in mặt sau thấm qua, một chỗ gạch xóa. Nhãn 32 dòng ghi vào `nhan.json` trước khi chạy.
`trang_01_cac_dong/` là 32 ảnh từng phép tính do `doc_trang.py --cat-dong` cắt từ chính trang đó.

| Bộ | Dòng đọc đúng hoàn toàn | KTC 95% | Biểu thức ra đúng giá trị | Chia cho 0 báo lỗi đúng | CER |
|---|---|---|---|---|---|
| Cả trang (tách cột, dòng, ký tự) | **28 / 32 (87,50%)** | 71,93–95,03% | **23 / 27 (85,19%)** | 1 / 1 | 1,35% |
| 32 ảnh cắt từng phép tính | 28 / 32 (87,50%) | 71,93–95,03% | 23 / 27 (85,19%) | 1 / 1 | 2,26% |

Biểu thức `4÷0=` không có giá trị nên không nằm trong mẫu số "ra đúng giá trị" (trước đây được tính vào mẫu số 28 và
không bao giờ có thể đúng); nó được đếm riêng và đúng khi hệ thống đọc đúng dòng và báo lỗi chia cho 0.

Bốn dòng sai (cả trang): ba dòng nhầm đúng một ký tự, một dòng mất một chữ số.

| Trên giấy | Máy đọc | Nguyên nhân |
|---|---|---|
| `90347×10=9` | `90347×0=9` | số 1 dính vào số 0 (phép đóng nối hai số) nên mất số 1 |
| `4953×5349=7624513` | `4953×534927624513` | dấu `=` bị đọc thành `2` (phép đóng nối hai gạch) |
| `261794×3=` | `264794×3=` | 1 → 4 |
| `69-20÷10+6=` | `69-20÷1046=` | `+` → 4 |

Qua các phiên bản mô hình, trang này đọc đúng: `co_so` 27/32, `co_cvl` 26/32, `co_so + co_cvl` 28/32,
`co_cvl + lon_rac` (cắt thẳng) 29/32, `nho_rac` 26/32 (các số này đo trước khi sửa lỗi khóa), mô hình cuối 28/32. Một trang 32 dòng của một người viết dao
động 1–3 dòng giữa các mô hình; không nên kết luận từ chênh lệch nhỏ ở đây.

**Cảnh báo về độ lạc quan**: trang này được dùng để phát hiện và sửa ba lỗi của khâu tách (tách cột, luật ghép số 1
nghiêng, ghép số 4 hai nét, bỏ vết gạch xóa) trước khi ra con số trên. Lần chạy đầu tiên, trước các sửa đổi đó, mỗi
dòng dính hai biểu thức của hai cột nên không dòng nào tính được. Con số 27/32 vì vậy là kết quả trên dữ liệu đã nhìn
thấy khi phát triển, chỉ có một người viết, không phải kết quả kiểm định độc lập. Cần thêm trang mới (lý tưởng là của
người khác) chưa từng dùng khi sửa mã. Đo với mô hình `co_so` lúc đó, các sửa đổi này không làm giảm kết quả trên CVL
validation (33,4% → 33,2%) hay trang tổng hợp (76,1% → 76,9%); CVL test giảm 6 ảnh (31,45% → 30,65%), trong khoảng
tin cậy.

## 7. Hạn chế

- Đọc đúng cả dãy trên chữ viết thật của người lạ: 62,77% với dãy 5–7 chữ số (CER 10,0%), còn thấp hơn các hệ thống
  chuyên dụng (đọc cả dòng bằng CRNN). Nguồn lỗi lớn nhất vẫn là tách chữ viết liền nét (111/277 dãy sai do thừa hoặc
  thiếu ký tự), sau đó là nhầm 7/2, 9/8.
- Tập kiểm định dùng để chọn mô hình chỉ có 90 dãy của 9 người; quy tắc chọn đã phải sửa một lần sau khi thấy điểm test
  (mục 2).
- Ký hiệu `÷` (210 mẫu train) và `=` (516 mẫu train, ~8% dị dạng) ít dữ liệu nhất và là nguồn lỗi chính trên trang.
- Mô hình cuối ghép hai CNN (một nhỏ, một lớn) nên đọc chậm hơn khoảng hai lần so với một CNN nhỏ.
- Không hỗ trợ ngoặc, số thập phân, phân số viết dọc, chỉ số trên/dưới; biểu thức phải nằm trên một dòng.
- Điểm softmax không được hiệu chuẩn; ngưỡng "nghi ngờ" 0,6 chỉ để gợi ý người dùng kiểm tra.
- Mỗi cấu hình chỉ huấn luyện một lần (một hạt), chưa đo độ dao động giữa các lần chạy.
- Ảnh chụp thật mới có một trang của một người viết, và trang đó đã được dùng khi sửa khâu tách (mục 6). Khoảng tin
  cậy Wilson coi mỗi dãy/dòng là độc lập, trong khi các dãy của cùng người viết và các dòng của cùng trang liên quan nhau,
  nên khoảng thật rộng hơn. Muốn đo lần cuối cần khóa mã, mô hình và quy tắc, rồi thu trang mới của người khác và không
  chỉnh mã sau khi xem kết quả trên các trang đó.

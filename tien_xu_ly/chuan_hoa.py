"""Chuẩn hóa MỘT ký tự (mặt nạ nhị phân hoặc ảnh mức nét) thành đầu vào CNN 28×28.

Dùng chung cho mọi nguồn: ảnh MNIST, ký hiệu HASYv2/CROHME khi tạo bộ huấn luyện, và từng ký tự cắt ra từ ảnh chụp
trang giấy khi đọc. Nhờ vậy ảnh lúc huấn luyện và lúc dùng đi qua cùng một phép biến đổi.

Các bước: nhị phân hóa (phóng to trước nếu ảnh nhỏ) → cắt theo hộp bao → chuẩn hóa độ dày nét bằng giãn/co hình
thái học → thu nhỏ giữ tỉ lệ (trung bình theo diện tích) vào ô 20×20 → căn giữa canvas 28×28 → kéo giãn độ sáng.

Chuẩn hóa độ dày là cần thiết vì cùng một chữ số, chụp ở 80 px chiều cao bằng bút bi (nét ~3 px) sẽ chỉ còn nét
~0,75 px sau khi thu về 20 px, trong khi MNIST có nét ~2–3 px; ký hiệu HASYv2 vẽ bằng chuột lại rất mảnh.
"""
import math

import numpy as np

from ._kiem_tra import KhongCoDoiTuong, LoiXuLyAnh, ma_tran
from .hinh_thai import hinh_thai_nhi_phan
from .thu_phong import noi_suy_song_tuyen_tinh, thu_phong

KICH_THUOC = 28
NOI_DUNG = 20                 # cạnh dài của ký tự sau khi thu nhỏ (lề 4 px như MNIST)
DO_DAY_DICH = 2.7             # độ dày nét mong muốn (px) trong ô 20×20 = trung vị đo trên 2.000 ảnh MNIST train
CANH_LAM_VIEC = 56            # ảnh nhỏ hơn được phóng to tới cỡ này trước khi nhị phân hóa


def uoc_luong_do_day(mat_na):
    """Độ dày nét ≈ 2·diện tích / số pixel biên (nét dài L, dày w: diện tích ≈ wL, biên ≈ 2L).

    Pixel biên là pixel nét có ít nhất một láng giềng 4 là nền.
    """
    m = ma_tran(mat_na).astype(bool)
    dien_tich = int(m.sum())
    if not dien_tich:
        return 0.0
    p = np.pad(m, 1)
    trong = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    bien = dien_tich - int(trong.sum())
    return 2 * dien_tich / max(bien, 1)


def cat_hop_bao(m):
    ys, xs = np.nonzero(m)
    if not len(xs):
        raise KhongCoDoiTuong("Ký tự không có pixel nét")
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def chuan_do_day(m, do_day_dich_px):
    """Giãn (nét quá mảnh) hoặc co (nét quá đậm) để độ dày ước lượng gần do_day_dich_px. Trả về mặt nạ đã cắt lại."""
    w = uoc_luong_do_day(m)
    if w < 0.75 * do_day_dich_px:
        k = 2 * max(1, round((do_day_dich_px - w) / 2)) + 1          # giãn với SE k×k làm nét dày thêm k−1
        m = cat_hop_bao(hinh_thai_nhi_phan(np.pad(m, k), "gian", k))
    elif w > 1.6 * do_day_dich_px:
        k = 2 * max(1, round((w - do_day_dich_px) / 2)) + 1
        co = hinh_thai_nhi_phan(m, "co", k)
        if co.sum() >= 0.3 * m.sum():                                # không co nếu ký tự bị xóa gần hết
            m = cat_hop_bao(co)
    return m


def chuan_hoa_ky_tu(anh, nguong=0.5, chuan_net=True):
    """anh: mảng 2D, nét SÁNG (bool, hoặc số thực trong [0,1]). Trả về float32 28×28 trong [0,1].

    chuan_net=False bỏ bước chuẩn hóa độ dày (để làm thí nghiệm so sánh).
    """
    a = ma_tran(anh)
    if a.dtype == bool:
        a = a.astype(np.float64)
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise LoiXuLyAnh("Ảnh ký tự phải là bool hoặc số thực trong [0,1], nét sáng")
    m = cat_hop_bao(a > nguong)
    canh = max(m.shape)
    if canh < CANH_LAM_VIEC:                       # phóng to song tuyến tính rồi mới nhị phân để biên mượt hơn
        f = math.ceil(CANH_LAM_VIEC / canh)
        ys, xs = np.nonzero(a > nguong)
        vung = np.pad(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 1)
        m = cat_hop_bao(noi_suy_song_tuyen_tinh(vung, vung.shape[0] * f, vung.shape[1] * f) > nguong)
        canh = max(m.shape)
    if chuan_net:
        m = chuan_do_day(m, DO_DAY_DICH * canh / NOI_DUNG)
    ti_le = NOI_DUNG / max(m.shape)
    cao, rong = max(1, round(m.shape[0] * ti_le)), max(1, round(m.shape[1] * ti_le))
    nho = thu_phong(m.astype(np.float64), cao, rong)
    canvas = np.zeros((KICH_THUOC, KICH_THUOC), np.float32)
    y, x = (KICH_THUOC - cao) // 2, (KICH_THUOC - rong) // 2
    canvas[y:y + cao, x:x + rong] = nho
    if canvas.max() <= 0:
        raise KhongCoDoiTuong("Ký tự biến mất khi thu nhỏ")
    return np.clip(canvas / canvas.max(), 0, 1).astype(np.float32)

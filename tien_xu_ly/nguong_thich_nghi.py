"""Phân ngưỡng thích nghi cho ảnh chụp trang giấy (ánh sáng không đều, bóng tay, giấy ngả màu).

Otsu chọn MỘT ngưỡng cho cả ảnh nên hỏng khi một góc tối hơn góc kia. Sauvola chọn ngưỡng riêng cho từng pixel
từ trung bình m và độ lệch chuẩn s của cửa sổ quanh nó:  T = m · (1 + k · (s / R − 1)).
Vùng giấy trơn (s nhỏ) có T thấp hơn m nhiều nên nhiễu nền không thành nét; vùng có chữ (s lớn) có T gần m.
m và s tính bằng ảnh tích phân (bảng tổng tiền tố) nên mỗi pixel chỉ tốn O(1) dù cửa sổ lớn.
"""
import numpy as np

from ._kiem_tra import LoiXuLyAnh, ma_tran, so_nguyen_duong


def anh_tich_phan(a):
    """I[y, x] = tổng a[:y, :x] (có thêm một hàng/cột 0 ở đầu để tính tổng hình chữ nhật không cần xét biên)."""
    a = np.asarray(a, dtype=np.float64)
    kq = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    kq[1:, 1:] = a.cumsum(0).cumsum(1)
    return kq


def trung_binh_cua_so(a, cua_so):
    """Trung bình trong cửa sổ vuông cạnh `cua_so` (lẻ) tâm tại mỗi pixel; ở biên chỉ lấy phần nằm trong ảnh."""
    a = ma_tran(a)
    so_nguyen_duong(cua_so, "Cửa sổ", le=True)
    h, w = a.shape
    r = cua_so // 2
    y0, y1 = np.clip(np.arange(h) - r, 0, h), np.clip(np.arange(h) + r + 1, 0, h)
    x0, x1 = np.clip(np.arange(w) - r, 0, w), np.clip(np.arange(w) + r + 1, 0, w)
    tp = anh_tich_phan(a)
    tong = tp[y1][:, x1] - tp[y0][:, x1] - tp[y1][:, x0] + tp[y0][:, x0]
    dien_tich = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return tong / dien_tich


def cua_so_mac_dinh(hinh):
    """Khoảng 1/25 cạnh ngắn (đủ phủ vài nét chữ), lẻ, trong [15, 101]."""
    c = int(min(hinh) / 25) | 1
    return int(np.clip(c, 15, 101)) | 1


def nguong_sauvola(xam, cua_so=None, k=0.2, R=128.0):
    """Trả về (mặt nạ nét tối, ảnh ngưỡng T). xam: uint8, nét TỐI trên nền sáng.

    k ∈ [0,2; 0,5] trong tài liệu gốc (Sauvola & Pietikäinen, 2000); R = 128 là độ lệch chuẩn tối đa của ảnh 8 bit.
    """
    xam = ma_tran(xam)
    if xam.dtype != np.uint8:
        raise LoiXuLyAnh("Sauvola cần ảnh uint8")
    cua_so = cua_so or cua_so_mac_dinh(xam.shape)
    a = xam.astype(np.float64)
    m = trung_binh_cua_so(a, cua_so)
    phuong_sai = np.maximum(trung_binh_cua_so(a * a, cua_so) - m * m, 0)
    T = m * (1 + k * (np.sqrt(phuong_sai) / R - 1))
    return a <= T, T


def loc_max(a, k):
    """Lọc cực đại cửa sổ k×k (= phép giãn mức xám), tách thành hai lượt 1D theo hàng và theo cột."""
    a = np.asarray(a)
    r = k // 2
    hang = np.lib.stride_tricks.sliding_window_view(np.pad(a, ((0, 0), (r, r)), mode="edge"), k, axis=1).max(-1)
    return np.lib.stride_tricks.sliding_window_view(np.pad(hang, ((r, r), (0, 0)), mode="edge"), k, axis=0).max(-1)


def chuan_hoa_nen(xam, cua_so=None):
    """Ước lượng nền giấy bằng lọc cực đại (xóa nét tối mảnh hơn cửa sổ) rồi làm mượt bằng lọc trung bình, sau đó
    chia ảnh cho nền: giấy thành ~255 ở mọi chỗ, bất kể chỗ đó sáng hay tối. Trả về (ảnh đã chuẩn hóa uint8, nền)."""
    xam = ma_tran(xam)
    cua_so = cua_so or cua_so_mac_dinh(xam.shape)
    nen = trung_binh_cua_so(loc_max(xam, cua_so).astype(np.float64), cua_so)
    chuan = np.clip(xam.astype(np.float64) / np.maximum(nen, 1) * 255, 0, 255)
    return np.rint(chuan).astype(np.uint8), nen

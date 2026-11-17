"""Ước lượng và chỉnh độ nghiêng của trang bằng chiếu ngang.

Khi các dòng chữ nằm ngang, biểu đồ chiếu ngang (số pixel nét trên mỗi hàng) có các đỉnh cao, hẹp xen khe trống; khi
trang nghiêng, các đỉnh bị trải rộng ra. Thử các góc trong [−góc tối đa, +góc tối đa], với mỗi góc dịch từng cột theo
x·tan(góc) rồi tính tổng bình phương biểu đồ chiếu: góc làm tổng này lớn nhất là góc làm các dòng thẳng nhất.

Với góc nhỏ (vài độ), phép xoay được xấp xỉ bằng phép trượt dọc (mỗi cột dịch lên/xuống), không cần nội suy và giữ
nguyên ảnh nhị phân.
"""
import numpy as np


def uoc_luong_goc(mat_na, goc_toi_da=5.0, buoc=0.25):
    """Trả về góc (độ) cần trượt để dòng chữ nằm ngang; dương nghĩa là dòng đi xuống khi sang phải."""
    h, w = mat_na.shape
    ly, lx = int(0.06 * h), int(0.06 * w)          # bỏ dải mép: mảng tối của bàn / góc ảnh làm lệch biểu đồ
    ys, xs = np.nonzero(mat_na[ly:h - ly, lx:w - lx])
    xs = xs + lx
    if len(xs) < 50:
        return 0.0
    if len(xs) > 200_000:                                   # lấy mẫu cho nhanh, kết quả gần như không đổi
        chon = np.random.default_rng(0).choice(len(xs), 200_000, replace=False)
        ys, xs = ys[chon], xs[chon]
    xc = xs - w / 2
    tot, goc_tot = -1.0, 0.0
    for goc in np.arange(-goc_toi_da, goc_toi_da + 1e-9, buoc):
        r = ys - np.rint(xc * np.tan(np.radians(goc))).astype(int)
        hist = np.bincount(r - r.min())
        diem = float((hist.astype(np.float64) ** 2).sum())
        if diem > tot * (1 + 1e-9):
            tot, goc_tot = diem, float(goc)
    return goc_tot


def truot_doc(anh, goc, nen=0):
    """Dịch cột x lên một đoạn (x − W/2)·tan(góc) để dòng nghiêng thành nằm ngang; chỗ trống điền giá trị nền."""
    if abs(goc) < 1e-9:
        return anh
    a = np.asarray(anh)
    h, w = a.shape[:2]
    dich = np.rint((np.arange(w) - w / 2) * np.tan(np.radians(goc))).astype(int)
    y_nguon = np.arange(h)[:, None] + dich[None, :]
    hop_le = (y_nguon >= 0) & (y_nguon < h)
    kq = a[np.clip(y_nguon, 0, h - 1), np.arange(w)[None, :]]
    kq[~hop_le] = nen
    return kq

"""Biến thể kiểu viết không có (hoặc rất hiếm) trong MNIST, sinh từ ảnh MNIST TRAIN:

- số 1 có nét mũ (nét xiên ở đầu, như cách viết phổ biến ở Việt Nam và châu Âu),
- số 7 có gạch ngang ở giữa thân.

MNIST thu từ người Mỹ nên gần như toàn số 1 một nét thẳng và số 7 không gạch; trên CVL Digit Strings (người viết ở Áo)
và trên chữ viết của nhóm ở BTL trước, đây là hai nguồn lỗi lớn (1 → 4/7, 7 có gạch → 2). Chỉ dùng cho cấu hình
cai_tien; validation và test giữ nguyên ảnh MNIST gốc.
"""
import numpy as np


def ve_doan(a, p0, p1, day=2.0):
    """Vẽ đoạn thẳng nét sáng (giá trị 1) dày `day` px lên ảnh float bằng cách lấy mẫu dày dọc đoạn."""
    h, w = a.shape
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) * 3) + 2
    yy, xx = np.mgrid[0:h, 0:w]
    for t in np.linspace(0, 1, n):
        y, x = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
        d = np.hypot(yy - y, xx - x)
        a[:] = np.maximum(a, np.clip(day / 2 + 0.5 - d, 0, 1))
    return a


def them_mu_so_1(anh, rng):
    """anh: float 28×28 nét sáng. Nét mũ đi từ đỉnh thân xuống bên trái, dài 25–45% chiều cao, góc 20–55°."""
    a = anh.copy()
    ys, xs = np.nonzero(a > 0.5)
    tren = ys.min()
    x_dinh = xs[ys <= tren + 1].mean()
    cao = ys.max() - tren + 1
    dai = cao * rng.uniform(0.25, 0.45)
    goc = np.radians(rng.uniform(20, 55))
    return ve_doan(a, (tren + 0.5, x_dinh), (tren + dai * np.sin(goc), x_dinh - dai * np.cos(goc)),
                   rng.uniform(1.6, 2.6))


def them_gach_so_7(anh, rng):
    """Gạch ngang cắt thân số 7 ở khoảng 45–65% chiều cao, dài 50–80% bề rộng, nghiêng ±10°."""
    a = anh.copy()
    ys, xs = np.nonzero(a > 0.5)
    tren, duoi = ys.min(), ys.max()
    y = int(tren + (duoi - tren) * rng.uniform(0.45, 0.65))
    hang = np.nonzero(a[y] > 0.5)[0]
    if not len(hang):
        return None
    xc = hang.mean()
    dai = (xs.max() - xs.min() + 1) * rng.uniform(0.5, 0.8)
    lech = np.tan(np.radians(rng.uniform(-10, 10))) * dai / 2
    return ve_doan(a, (y + lech, xc - dai / 2), (y - lech, xc + dai / 2), rng.uniform(1.6, 2.6))


def sinh_bien_the(x, y, ti_le=0.5, hat=20261119):
    """x uint8 N×28×28 MNIST train, y nhãn. Trả về (ảnh float [0,1], nhãn) cho ti_le số ảnh 1 và 7."""
    rng = np.random.default_rng(hat)
    anh, nhan = [], []
    for so, ham in ((1, them_mu_so_1), (7, them_gach_so_7)):
        chi_so = np.flatnonzero(y == so)
        for i in rng.choice(chi_so, int(len(chi_so) * ti_le), replace=False):
            kq = ham(x[i] / 255.0, rng)
            if kq is not None:
                anh.append(kq)
                nhan.append(so)
    return anh, nhan

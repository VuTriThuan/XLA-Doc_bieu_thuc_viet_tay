"""Sinh ảnh "chụp trang giấy" tổng hợp có nhãn để đo toàn bộ luồng đọc trang một cách định lượng.

Mỗi trang: giấy ngả màu, ánh sáng loang (gradient + bóng tròn), có thể có dòng kẻ ngang của vở, nhiễu Gauss, xoay nhẹ;
3–6 dòng, mỗi dòng là một biểu thức (vd. 127+45=) hoặc một dãy số. Ký tự lấy từ TẬP TEST (MNIST test, ký hiệu test của
HASYv2/CROHME+) nên không trùng ảnh nào CNN đã thấy khi huấn luyện. Đây vẫn là ảnh ghép, không thay được ảnh chụp thật.
"""
from pathlib import Path

import numpy as np

from loi.cau_hinh import DU_LIEU, ghi_json
from tien_xu_ly.nguong_thich_nghi import loc_max
from .ky_hieu import tai_ky_hieu
from .mnist import tai_mnist_tho

THU_MUC = DU_LIEU / "trang_tong_hop"
HAT = 20261118
TI_LE_KY_HIEU = {"+": 0.6, "-": 0.55, "×": 0.55, "÷": 0.6, "=": 0.55}   # cạnh dài của ký hiệu / chiều cao chữ số


def sinh_noi_dung(rng):
    """Một dòng: 70% biểu thức, 30% dãy số."""
    def so():
        return str(int(rng.integers(0, 10 ** int(rng.integers(1, 4)))))
    if rng.random() < 0.3:
        return " ".join(so() for _ in range(int(rng.integers(1, 4))))
    n = int(rng.integers(2, 4))
    s = so()
    for _ in range(n - 1):
        p = str(rng.choice(["+", "-", "×", "÷"]))
        b = so()
        if p == "÷" and int(b) == 0:
            b = str(int(rng.integers(1, 10)))
        s += p + b
    return s + ("=" if rng.random() < 0.7 else "")


class KhoKyTu:
    """Lấy ngẫu nhiên ảnh một ký tự (nét sáng, float [0,1], đã cắt hộp bao) từ các tập test."""

    def __init__(self, rng):
        self.rng = rng
        x, y = tai_mnist_tho("test")
        self.chu_so = {str(d): x[y == d] for d in range(10)}
        kh = tai_ky_hieu()
        chon = kh["tap"] == "test"
        self.ky_hieu = {str(c): kh["anh"][chon & (kh["nhan"] == c)] for c in TI_LE_KY_HIEU}

    def lay(self, c):
        kho = self.chu_so[c] if c.isdigit() else self.ky_hieu[c]
        a = kho[int(self.rng.integers(len(kho)))].astype(np.float64) / 255
        if not c.isdigit():          # ký hiệu dựng lại từ nét bút điện tử rất mảnh; làm dày như nét bút thật
            a = loc_max(a, 5)
        ys, xs = np.nonzero(a > 0.2)
        return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _thu_phong(a, cao, rong):
    from PIL import Image
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((max(1, rong), max(1, cao)), Image.BILINEAR),
                      np.float64) / 255


def sinh_trang(rng, kho, rong=1200, cao=1500):
    from PIL import Image
    h_chu = int(rng.integers(45, 85))
    giay = np.array([245, 242, 232]) - rng.integers(0, 25, 3)
    anh = np.ones((cao, rong, 3)) * giay
    yy, xx = np.mgrid[0:cao, 0:rong] / max(cao, rong)
    sang = 1 - 0.35 * rng.random() * (yy * np.cos(rng.uniform(0, 6.3)) + xx * np.sin(rng.uniform(0, 6.3)) + 1) / 2
    cy, cx, r = rng.uniform(0, 1.2), rng.uniform(0, 1), rng.uniform(0.2, 0.6)
    sang *= 1 - 0.25 * rng.random() * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / r ** 2)
    khoang_dong = int(h_chu * rng.uniform(1.9, 2.6))
    if rng.random() < 0.5:                       # dòng kẻ ngang của vở
        for y in range(int(rng.integers(20, 60)), cao, khoang_dong // 2 if rng.random() < 0.4 else khoang_dong):
            anh[y:y + 2] = anh[y:y + 2] * 0.55 + np.array([150, 170, 215]) * 0.45
    muc = np.array([20, 20, 30]) if rng.random() < 0.6 else np.array([25, 40, 140])
    nhan = []
    y = int(rng.integers(60, 140))
    while y + 2 * h_chu < cao - 60 and len(nhan) < 6:
        dong = sinh_noi_dung(rng)
        x = int(rng.integers(50, 160))
        giua = y + h_chu // 2
        ok = True
        for c in dong:
            if c == " ":
                x += int(h_chu * rng.uniform(0.9, 1.4))
                continue
            g = kho.lay(c)
            if c.isdigit():
                ch = int(h_chu * rng.uniform(0.85, 1.1))
                cr = max(2, int(g.shape[1] * ch / g.shape[0]))
            else:
                canh = h_chu * TI_LE_KY_HIEU[c] * rng.uniform(0.85, 1.15)
                k = canh / max(g.shape)
                ch, cr = max(2, int(g.shape[0] * k)), max(2, int(g.shape[1] * k))
            g = _thu_phong(g, ch, cr)
            ty = int(giua - ch / 2 + rng.normal(0, h_chu * 0.05)) if not c.isdigit() else int(y + (h_chu - ch) + rng.normal(0, h_chu * 0.04))
            if x + cr > rong - 40:
                ok = False
                break
            vung = anh[ty:ty + ch, x:x + cr]
            anh[ty:ty + ch, x:x + cr] = vung * (1 - g[..., None] * 0.9) + muc * g[..., None] * 0.9
            x += cr + int(h_chu * rng.uniform(0.12, 0.35))
        if ok:
            nhan.append(dong)
        else:
            anh[y - 5:y + h_chu + 10, :] = (np.ones((1, rong, 3)) * giay)   # dòng tràn lề: xóa và bỏ
        y += khoang_dong
    anh = anh * sang[..., None] + rng.normal(0, rng.uniform(2, 8), anh.shape)
    im = Image.fromarray(np.clip(anh, 0, 255).astype(np.uint8))
    goc = rng.uniform(-1.5, 1.5)
    im = im.rotate(goc, resample=Image.BILINEAR, expand=False, fillcolor=tuple(int(v) for v in giay * 0.7))
    return np.array(im), nhan, {"cao_chu": h_chu, "goc_xoay": round(goc, 2)}


def sinh_bo_trang(so_trang=60, thu_muc=THU_MUC, hat=HAT):
    from PIL import Image
    rng = np.random.default_rng(hat)
    kho = KhoKyTu(rng)
    thu_muc = Path(thu_muc)
    thu_muc.mkdir(parents=True, exist_ok=True)
    nhan = {}
    for i in range(so_trang):
        anh, dong, thong_tin = sinh_trang(rng, kho)
        ten = f"trang_{i:03d}.jpg"
        Image.fromarray(anh).save(thu_muc / ten, quality=88)
        nhan[ten] = {"dong": dong, **thong_tin}
    ghi_json(thu_muc / "nhan.json", {"hat": hat, "nguon_ky_tu": "MNIST test + ký hiệu test (HASYv2, CROHME+)",
                                     "trang": nhan})
    return nhan

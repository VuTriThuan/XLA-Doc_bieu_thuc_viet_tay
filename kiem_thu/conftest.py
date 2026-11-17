"""Cấu hình pytest dùng chung: đưa thư mục gốc dự án vào sys.path và không tạo __pycache__ trong dự án."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
GOC = Path(__file__).resolve().parents[1]
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def bo_doc():
    """Nạp mô hình một lần cho cả phiên kiểm thử."""
    from loi.doc_trang import DocTrang
    return DocTrang()


def hop_le(c, anh):
    """Ký hiệu mẫu có hình dạng chuẩn: '=' gồm hai gạch dài gần bằng nhau, '÷' có gạch và hai chấm."""
    from tien_xu_ly.lien_thong import gan_nhan
    tp = sorted(gan_nhan(anh > 50)[1], key=lambda t: -t.rong)
    if c == "=":
        return len(tp) == 2 and tp[1].rong >= 0.6 * tp[0].rong
    if c == "÷":
        return len(tp) == 3
    return len(tp) <= 2


def ve_dong(chuoi, cao=60, nen=235, muc=30, khoang=14):
    """Ảnh xám một dòng ghép từ ảnh MNIST test (chữ số) và ký hiệu test; trả về ảnh uint8 nét tối."""
    from bo_du_lieu.ky_hieu import tai_ky_hieu
    from bo_du_lieu.mnist import tai_mnist_tho
    from PIL import Image
    x, y = tai_mnist_tho("test")
    kh = tai_ky_hieu()
    rng = np.random.default_rng(0)
    manh = []
    for c in chuoi:
        if c.isdigit():
            a = x[np.flatnonzero(y == int(c))[int(rng.integers(50))]]
            h = cao
        else:
            chon = [i for i in np.flatnonzero((kh["tap"] == "test") & (kh["nhan"] == c) & (kh["nguon"] == "crohme"))
                    if hop_le(c, kh["anh"][i])]
            a = kh["anh"][chon[int(rng.integers(min(20, len(chon))))]]
            from tien_xu_ly.nguong_thich_nghi import loc_max
            a = loc_max(a, 5)
            h = int(cao * 0.55)
        ys, xs = np.nonzero(a > 50)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        k = h / max(a.shape) if not c.isdigit() else h / a.shape[0]
        im = Image.fromarray(a).resize((max(2, int(a.shape[1] * k)), max(2, int(a.shape[0] * k))), Image.BILINEAR)
        manh.append((c, np.asarray(im, float) / 255))
    rong = sum(m.shape[1] + khoang for _, m in manh) + 60
    anh = np.full((cao + 60, rong), float(nen))
    xx = 30
    for c, m in manh:
        ty = 30 + (cao - m.shape[0]) // 2 if not c.isdigit() else 30 + cao - m.shape[0]
        vung = anh[ty:ty + m.shape[0], xx:xx + m.shape[1]]
        anh[ty:ty + m.shape[0], xx:xx + m.shape[1]] = vung * (1 - m) + muc * m
        xx += m.shape[1] + khoang
    return anh.astype(np.uint8)

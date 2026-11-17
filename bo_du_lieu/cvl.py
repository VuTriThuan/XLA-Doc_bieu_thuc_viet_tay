"""CVL Digit Strings (ICDAR 2013 HDSRC): 1.262 ảnh dãy số viết tay thật của khoảng 120 người, 10 dãy khác nhau.

Nguồn: https://zenodo.org/records/1492173 (tệp cvl-strings-train.zip), giấy phép CC BY-NC 4.0. Dự án KHÔNG huấn luyện trên
bộ này, chỉ dùng để đo khả năng đọc cả dãy số của luồng tách ký tự + CNN trên chữ viết của nhiều người. Nhãn là phần tên
tệp trước dấu '-' đầu tiên (vd. 135579-0001-10.png là dãy 135579 của người viết 0001).

Chia theo NGƯỜI VIẾT thành hai nửa (băm SHA-1 của mã người viết):
- validation: dùng để chọn mô hình cuối và các lựa chọn tiền xử lý (chữ viết tay thật, khác phân phối với MNIST);
- test: chỉ dùng để báo cáo kết quả cuối.
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from loi.cau_hinh import DU_LIEU, sha256

THU_MUC = DU_LIEU / "cvl"
TEP = THU_MUC / "cvl-strings-train.zip"
URL = "https://zenodo.org/records/1492173/files/cvl-strings-train.zip?download=1"
SHA256_DUNG = "a51b98570afc33dff7e46df830f6b146ac936f8f6680ad0586bc7d3930c3a325"


def dam_bao_cvl(tep=TEP):
    tep = Path(tep)
    if not tep.exists():
        tep.parent.mkdir(parents=True, exist_ok=True)
        tam = tep.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=120) as r, tam.open("wb") as f:
            for khoi in iter(lambda: r.read(1024 * 1024), b""):
                f.write(khoi)
        tam.replace(tep)
    if sha256(tep) != SHA256_DUNG:
        raise ValueError("SHA-256 của CVL Digit Strings không khớp")
    return tep


def tap_cua(nguoi_viet):
    return "validation" if int(hashlib.sha1(nguoi_viet.encode()).hexdigest()[:8], 16) % 2 == 0 else "test"


def nguoi_kiem_dinh(nguoi_viet):
    """9 người viết (thuộc nửa validation) mà chữ số KHÔNG đưa vào huấn luyện: dùng để chọn mô hình."""
    from .cvl_chu_so import tap_huan_luyen_cua
    return tap_cua(nguoi_viet) == "validation" and tap_huan_luyen_cua(nguoi_viet) == "validation"


def doc_cvl(tep=TEP, gioi_han=None, tap=None):
    """Sinh (tên tệp, nhãn chuỗi số, mã người viết, ảnh RGB uint8).

    tap = validation | test | kiem_dinh (90 dãy của 9 người viết không có trong huấn luyện) | None (tất cả)."""
    from PIL import Image
    with zipfile.ZipFile(dam_bao_cvl(tep)) as z:
        def chon(nguoi):
            return tap is None or (nguoi_kiem_dinh(nguoi) if tap == "kiem_dinh" else tap_cua(nguoi) == tap)
        ten = sorted(n for n in z.namelist() if n.endswith(".png") and chon(Path(n).name.split("-")[1]))
        for n in ten[:gioi_han]:
            goc = Path(n).name
            nhan, nguoi = goc.split("-")[0], goc.split("-")[1]
            with Image.open(io.BytesIO(z.read(n))) as im:
                yield goc, nhan, nguoi, np.array(im.convert("RGB"))

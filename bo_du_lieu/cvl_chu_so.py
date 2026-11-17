"""CVL Single Digits: chữ số viết tay của người châu Âu (số 1 có mũ, số 7 có gạch...) để bổ sung cho MNIST.

Nguồn: https://zenodo.org/records/1492173 (cvl-single-digits-train-validation.zip, 14.000 ảnh), CC BY-NC 4.0.
Các chữ số này được cắt ra từ chính các dãy của CVL Digit Strings, cùng 127 người viết (mã người viết là trường thứ hai
của tên tệp, vd. 2-0202-21-04.png là chữ số 2 của người 0202). Vì vậy để nửa TEST của CVL Strings (75 người) vẫn là
người lạ hoàn toàn với mô hình, chỉ lấy chữ số của 52 người thuộc nửa validation (bo_du_lieu.cvl.tap_cua). Trong số
đó lại chia theo người viết: ~80% để huấn luyện, ~20% để chọn checkpoint và chọn cấu hình.

Hệ quả: nửa validation của CVL Strings không còn dùng để chọn được nữa (người viết đã có trong huấn luyện).
Tệp zip gốc lưu ở du_lieu/cvl/ (tự tải nếu thiếu, kiểm tra SHA-256); ảnh đã chuẩn hóa nằm trong
du_lieu/ky_tu/ky_tu.npz (nguồn "cvl_chu_so"); danh sách tệp được dùng ở du_lieu/cvl/chu_so_dung.csv.
"""
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np

from loi.cau_hinh import DU_LIEU, ghi_csv
from tien_xu_ly.anh import chuyen_xam
from tien_xu_ly.chuan_hoa import chuan_hoa_ky_tu
from tien_xu_ly.lien_thong import gan_nhan
from tien_xu_ly.nguong import nguong_otsu
from .cvl import tap_cua

URL = "https://zenodo.org/records/1492173/files/cvl-single-digits-train-validation.zip?download=1"
TEP = DU_LIEU / "cvl" / "cvl-single-digits-train-validation.zip"
SHA256_DUNG = "c7b3e9ed8ea21d103440760aebc5fab9e18a970e85b7f5871e18859b9279376e"


def dam_bao_cvl_chu_so(tep=TEP):
    """Tải tệp zip nếu thiếu và kiểm tra SHA-256."""
    import urllib.request
    from loi.cau_hinh import sha256
    tep = Path(tep)
    if not tep.exists():
        tep.parent.mkdir(parents=True, exist_ok=True)
        tam = tep.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=120) as r, tam.open("wb") as f:
            for khoi in iter(lambda: r.read(1024 * 1024), b""):
                f.write(khoi)
        tam.replace(tep)
    if sha256(tep) != SHA256_DUNG:
        raise ValueError("SHA-256 của CVL Single Digits không khớp")
    return tep


def tap_huan_luyen_cua(nguoi_viet):
    """Trong nửa validation của CVL: ~80% người viết để huấn luyện, ~20% để kiểm định (băm có muối khác tap_cua)."""
    return "validation" if int(hashlib.sha1(("chu_so-" + nguoi_viet).encode()).hexdigest()[:8], 16) % 5 == 0 else "train"


def mat_na_chu_so(rgb):
    """Ảnh cắt nhỏ một chữ số (nét tối, nền sáng) -> mặt nạ nét: Otsu, bỏ mẩu < 8% diện tích (nét của chữ bên cạnh)."""
    xam = chuyen_xam(rgb)
    m = xam <= nguong_otsu(xam)
    nhan, tp = gan_nhan(m)
    tong = m.sum()
    return np.isin(nhan, [t.nhan for t in tp if t.dien_tich >= 0.08 * tong])


def doc_cvl_chu_so(tep_zip):
    """Trả về (ảnh 28×28 float, nhãn, tập, danh sách dòng mô tả). Bỏ mọi chữ số của người viết thuộc nửa test."""
    from PIL import Image
    anh, nhan, tap, dong = [], [], [], []
    with zipfile.ZipFile(tep_zip) as z:
        for n in sorted(x for x in z.namelist() if x.endswith(".png")):
            ten = Path(n).name
            truong = ten.split("-")
            so, nguoi = int(truong[0]), truong[1]
            if tap_cua(nguoi) == "test":
                continue
            with Image.open(io.BytesIO(z.read(n))) as im:
                m = mat_na_chu_so(np.array(im.convert("RGB")))
            if not m.any():
                continue
            t = tap_huan_luyen_cua(nguoi)
            anh.append(chuan_hoa_ky_tu(m))
            nhan.append(so)
            tap.append(t)
            dong.append({"tep": n, "chu_so": so, "nguoi_viet": nguoi, "tap": t})
    ghi_csv(DU_LIEU / "cvl" / "chu_so_dung.csv", dong)
    return anh, nhan, tap, dong

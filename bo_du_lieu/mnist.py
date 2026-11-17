"""MNIST chính thức (Yann LeCun, Corinna Cortes; CC BY-SA 3.0).

Chỉ số chia 54.000 train / 6.000 validation lấy lại từ BTL trước (`du_lieu/mnist/splits.npz`, chia phân tầng có hạt);
10.000 ảnh test chính thức giữ nguyên và không dùng để chọn mô hình.
"""
import urllib.request
from pathlib import Path

import numpy as np

from loi.cau_hinh import DU_LIEU, sha256

THU_MUC = DU_LIEU / "mnist"
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256_DUNG = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def dam_bao_mnist(thu_muc=THU_MUC):
    """Tải MNIST nếu thiếu và kiểm tra SHA-256."""
    tep = Path(thu_muc) / "mnist.npz"
    if not tep.exists():
        tep.parent.mkdir(parents=True, exist_ok=True)
        tam = tep.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=90) as r, tam.open("wb") as f:
            for khoi in iter(lambda: r.read(1024 * 1024), b""):
                f.write(khoi)
        tam.replace(tep)
    if sha256(tep) != SHA256_DUNG:
        raise ValueError("SHA-256 của MNIST không khớp")
    return tep


def tai_mnist_tho(tap, thu_muc=THU_MUC):
    """(x uint8 N×28×28, y int64) của tập train | validation | test."""
    thu_muc = Path(thu_muc)
    with np.load(dam_bao_mnist(thu_muc), allow_pickle=False) as d:
        if tap == "test":
            return d["x_test"], d["y_test"].astype(np.int64)
        if tap not in ("train", "validation"):
            raise ValueError("tap phải là train, validation hoặc test")
        with np.load(thu_muc / "splits.npz", allow_pickle=False) as s:
            i = s[tap]
        return d["x_train"][i], d["y_train"][i].astype(np.int64)

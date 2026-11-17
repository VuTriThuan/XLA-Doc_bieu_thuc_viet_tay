"""EMNIST Digits (Cohen và cs., 2017; NIST Special Database 19): 240.000 chữ số train, 40.000 test, nhiều người viết hơn
MNIST. Nguồn: https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip (561 MB; chỉ cần 4 tệp emnist-digits-*.gz bên trong,
có thể lấy riêng bằng tải từng đoạn HTTP Range).

Dự án chỉ dùng 60.000 ảnh TRAIN chọn ngẫu nhiên (hạt cố định) để thêm vào huấn luyện cấu hình lon_rac, và lưu chúng
ở du_lieu/emnist/emnist_chu_so_60k.npz (ảnh gốc 28×28 đã xoay về đúng chiều) để chạy lại không cần tải. Ảnh EMNIST lưu
theo cột, nên phải chuyển vị.
"""
import gzip
from pathlib import Path

import numpy as np

from loi.cau_hinh import DU_LIEU, ghi_json, sha256

TEP = DU_LIEU / "emnist" / "emnist_chu_so_60k.npz"
SO_LUONG = 60000
HAT = 20261120


def trich_emnist(thu_muc_gz, tep=TEP, so_luong=SO_LUONG, hat=HAT):
    thu_muc_gz = Path(thu_muc_gz)
    anh = thu_muc_gz / "emnist-digits-train-images-idx3-ubyte.gz"
    nhan = thu_muc_gz / "emnist-digits-train-labels-idx1-ubyte.gz"
    x = np.frombuffer(gzip.open(anh).read(), np.uint8, offset=16).reshape(-1, 28, 28).transpose(0, 2, 1)
    y = np.frombuffer(gzip.open(nhan).read(), np.uint8, offset=8).astype(np.int64)
    chon = np.sort(np.random.default_rng(hat).choice(len(y), so_luong, replace=False))
    tep = Path(tep)
    tep.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tep, x=np.ascontiguousarray(x[chon]), y=y[chon], chi_so=chon)
    ghi_json(tep.parent / "nguon.json", {"nguon": "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",
                                         "tep": {p.name: sha256(p) for p in (anh, nhan)},
                                         "so_anh_goc": int(len(y)), "so_anh_chon": so_luong, "hat": hat,
                                         "tham_khao": "Cohen, Afshar, Tapson, van Schaik (2017), EMNIST, arXiv:1702.05373"})
    return tep


def tai_emnist(tep=TEP):
    with np.load(tep, allow_pickle=False) as d:
        return d["x"], d["y"]

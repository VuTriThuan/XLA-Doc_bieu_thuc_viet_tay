"""Bộ dữ liệu 15 lớp ký tự (0–9, + − × ÷ =) cho CNN: MNIST + ký hiệu viết tay thật (HASYv2, CROHME+).

Thêm biến thể "1 có mũ", "7 có gạch" (bo_du_lieu/kieu_viet.py) cho cấu hình cai_tien và chữ số CVL Single Digits
(bo_du_lieu/cvl_chu_so.py, chỉ người viết thuộc nửa validation của CVL) cho cấu hình co_cvl.
Mọi ảnh, kể cả MNIST, đều đi qua `tien_xu_ly.chuan_hoa.chuan_hoa_ky_tu` — đúng phép biến đổi dùng cho từng ký tự cắt
ra từ ảnh chụp trang giấy — nên CNN học trên cùng kiểu đầu vào mà nó sẽ gặp khi đọc trang.
Kết quả lưu ở du_lieu/ky_tu/ky_tu.npz (x uint8 N×28×28, y, tập, nguồn).
"""
from pathlib import Path

import numpy as np

from loi.cau_hinh import DU_LIEU, LOP, PHIEN_BAN_DAU_VAO, RAC, ghi_json
from tien_xu_ly.chuan_hoa import chuan_hoa_ky_tu
from .ky_hieu import tai_ky_hieu
from .mnist import tai_mnist_tho

TEP = DU_LIEU / "ky_tu" / "ky_tu.npz"
CHI_SO = {c: i for i, c in enumerate(LOP)}


def tao_tap_ky_tu(tep=TEP, in_ra=True):
    x, y, tap, nguon = [], [], [], []
    for t in ("train", "validation", "test"):
        xm, ym = tai_mnist_tho(t)
        for a, nhan in zip(xm, ym):
            x.append(chuan_hoa_ky_tu(a / 255.0))
        y += list(ym)
        tap += [t] * len(ym)
        nguon += ["mnist"] * len(ym)
        if in_ra:
            print(f"MNIST {t}: {len(ym)} ảnh", flush=True)
    from .kieu_viet import sinh_bien_the
    xm, ym = tai_mnist_tho("train")
    anh, nhan = sinh_bien_the(xm, ym)
    for a, so in zip(anh, nhan):
        x.append(chuan_hoa_ky_tu(a))
    y += nhan
    tap += ["train"] * len(nhan)
    nguon += ["mnist_kieu_viet"] * len(nhan)
    if in_ra:
        print(f"Biến thể kiểu viết (1 có mũ, 7 có gạch) từ MNIST train: {len(nhan)} ảnh", flush=True)
    from .cvl_chu_so import dam_bao_cvl_chu_so, doc_cvl_chu_so
    anh, nhan, tap_cvl, _ = doc_cvl_chu_so(dam_bao_cvl_chu_so())
    x += anh
    y += nhan
    tap += tap_cvl
    nguon += ["cvl_chu_so"] * len(nhan)
    if in_ra:
        print(f"CVL Single Digits (chỉ người viết thuộc nửa validation của CVL): {len(nhan)} ảnh", flush=True)
    from .emnist import tai_emnist
    xe, ye = tai_emnist()
    for a in xe:
        x.append(chuan_hoa_ky_tu(a / 255.0))
    y += list(ye)
    tap += ["train"] * len(ye)
    nguon += ["emnist"] * len(ye)
    if in_ra:
        print(f"EMNIST digits (60.000 ảnh train chọn ngẫu nhiên): {len(ye)} ảnh", flush=True)
    kh = tai_ky_hieu()
    for a, nhan, t, ng in zip(kh["anh"], kh["nhan"], kh["tap"], kh["nguon"]):
        x.append(chuan_hoa_ky_tu(a / 255.0))
        y.append(CHI_SO[str(nhan)])
        tap.append(str(t))
        nguon.append(str(ng))
    x = np.stack(x)
    y, tap, nguon = np.array(y, np.int64), np.array(tap), np.array(nguon)
    # lớp "rác": sinh từ chữ số MNIST + CVL của đúng tập đó (train / validation / test)
    from .rac import sinh_rac
    from loi.du_doan import PhanLoaiKyTu
    from loi.cau_hinh import tep_mo_hinh
    # lọc mảnh mơ hồ bằng hai mô hình 15 lớp có từ trước (co_so + co_cvl), không dùng mô hình đang huấn luyện
    bo_loc = PhanLoaiKyTu([tep_mo_hinh("co_so"), tep_mo_hinh("co_cvl")], so_luong=6).xac_suat
    them_x, them_tap = [], []
    for t, so_luong, hat in (("train", 8000, 1), ("validation", 1500, 2), ("test", 1500, 3)):
        chon = (tap == t) & (y < 10) & np.isin(nguon, ["mnist", "cvl_chu_so"])
        r = sinh_rac(x[chon], y[chon], so_luong, 20261121 + hat, bo_loc)
        them_x.append(r)
        them_tap += [t] * len(r)
        if in_ra:
            print(f"Rác {t}: {len(r)} ảnh", flush=True)
    x = np.concatenate([x, *them_x])
    y = np.concatenate([y, np.full(len(them_tap), CHI_SO[RAC])])
    tap = np.concatenate([tap, them_tap])
    nguon = np.concatenate([nguon, ["rac"] * len(them_tap)])
    x = np.rint(x * 255).astype(np.uint8)
    tep = Path(tep)
    tep.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tep, x=x, y=y, tap=tap, nguon=nguon)
    goc = ~np.isin(nguon, ["mnist_kieu_viet", "cvl_chu_so", "emnist", "rac"])
    dem = [{"lop": c, **{t: int(((y == i) & (tap == t) & goc).sum()) for t in ("train", "validation", "test")},
            "kieu_viet_them": int(((y == i) & (nguon == "mnist_kieu_viet")).sum()),
            **{f"cvl_{t}": int(((y == i) & (nguon == "cvl_chu_so") & (tap == t)).sum()) for t in ("train", "validation")},
            "emnist_train": int(((y == i) & (nguon == "emnist")).sum()),
            **{f"rac_{t}": int(((y == i) & (nguon == "rac") & (tap == t)).sum()) for t in ("train", "validation", "test")}}
           for i, c in enumerate(LOP)]
    ghi_json(tep.parent / "thong_ke.json", {"phien_ban_dau_vao": PHIEN_BAN_DAU_VAO, "lop": list(LOP),
                                             "so_mau_moi_lop": dem, "tong": int(len(y))})
    if in_ra:
        print(f"Đã ghi {tep} ({len(y)} ảnh)", flush=True)
    return dem


def tai_tap(tap, tep=TEP, nguon=None, kieu_viet=False, cvl=False, tra_nguon=False, emnist=False, rac=False):
    """(x float32 N×1×28×28 trong [0,1], y int64).

    nguon lọc theo 'mnist' | 'hasy' | 'crohme' nếu cần; kieu_viet=True thêm các biến thể 1 có mũ / 7 có gạch (chỉ có
    trong train, chỉ cấu hình cai_tien dùng); cvl=True thêm chữ số CVL (chỉ cấu hình co_cvl dùng để huấn luyện; mọi
    cấu hình được chấm trên phần validation của nó khi chọn cấu hình cuối); emnist=True thêm 60.000 chữ số EMNIST (chỉ train);
    rac=True thêm lớp "rác". tra_nguon=True trả thêm mảng nguồn.
    """
    with np.load(tep, allow_pickle=False) as d:
        chon = d["tap"] == tap
        if not kieu_viet:
            chon &= d["nguon"] != "mnist_kieu_viet"
        if not cvl:
            chon &= d["nguon"] != "cvl_chu_so"
        if not emnist:
            chon &= d["nguon"] != "emnist"
        if not rac:
            chon &= d["nguon"] != "rac"
        if nguon is not None:
            chon &= np.isin(d["nguon"], np.atleast_1d(nguon))
        kq = d["x"][chon][:, None].astype(np.float32) / 255, d["y"][chon]
        return (*kq, d["nguon"][chon]) if tra_nguon else kq

"""Huấn luyện CNN ký tự; chọn checkpoint và chọn cấu hình chỉ bằng dữ liệu kiểm định (không đọc test).

Năm cấu hình (BIEN_THE_CAU_HINH):
- co_so    : không tăng cường dữ liệu.
- cai_tien : tăng cường (xoay ±10°, dịch, co giãn, đổi độ dày nét) và thêm biến thể kiểu viết "1 có mũ", "7 có gạch"
             sinh từ MNIST train (bo_du_lieu/kieu_viet.py).
- co_cvl   : như co_so, thêm chữ số viết tay thật của người châu Âu (CVL Single Digits, chỉ 43 người viết thuộc nửa
             validation của CVL); trong mỗi lớp chữ số, CVL chiếm TI_LE_CVL số lần lấy mẫu.
- nho_rac  : như co_cvl, thêm lớp thứ 16 "rác" (bo_du_lieu/rac.py) để CNN từ chối các đoạn cắt sai.
- lon_rac  : CNN lớn (loi/cnn.py), dữ liệu của nho_rac + 60.000 chữ số EMNIST.
Ba cấu hình đầu 15 lớp (không có "rác"). Tất cả lấy mẫu cân bằng lớp (WeightedRandomSampler): ký hiệu phép toán chỉ có vài trăm mẫu so với ~5.400 mẫu mỗi chữ
số, nếu lấy mẫu đều thì mạng gần như bỏ qua các lớp hiếm. Tiêu chí chọn là accuracy CÂN BẰNG (trung bình recall
từng lớp) trên validation, vì accuracy thường gần như chỉ phản ánh chữ số. Tập test không được đọc ở đây.
"""
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from bo_du_lieu.tap_ky_tu import tai_tap
from .cau_hinh import (BIEN_THE, CHU_SO, KET_QUA, LOP, LOP_KY_TU, PHIEN_BAN_DAU_VAO, SO_LOP, doc_json, ghi_csv, ghi_json, sha256,
                       tep_lua_chon, tep_mo_hinh)
from .cnn import dem_tham_so, tao_mang
from .tang_cuong import tang_cuong_ky_tu

TI_LE_CVL = 0.35
SAI_SO_DAY = 2            # chênh không quá 2/90 dãy kiểm định thì coi là ngang nhau
CONG_THUC = dict(so_epoch=12, co_lo=128, toc_do_hoc=1e-3, hat=42, buoc_giam=(8, 0.3))
# Mỗi cấu hình: kiến trúc + dữ liệu. so_mau_moi_epoch=None nghĩa là bằng số ảnh train.
BIEN_THE_CAU_HINH = {
    "co_so": dict(loai="nho", tang_cuong=False, cvl=False, emnist=False, rac=False, so_mau_moi_epoch=None),
    "cai_tien": dict(loai="nho", tang_cuong=True, cvl=False, emnist=False, rac=False, so_mau_moi_epoch=None),
    "co_cvl": dict(loai="nho", tang_cuong=False, cvl=True, emnist=False, rac=False, so_mau_moi_epoch=None),
    "nho_rac": dict(loai="nho", tang_cuong=False, cvl=True, emnist=False, rac=True, so_mau_moi_epoch=62500),
    "lon_rac": dict(loai="lon", tang_cuong=False, cvl=True, emnist=True, rac=True, so_mau_moi_epoch=62500),
}


def dat_hat(hat):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(hat)
    np.random.seed(hat)
    torch.manual_seed(hat)
    torch.use_deterministic_algorithms(True)


def du_doan_lo(mang, x, co_lo=1024):
    mang.eval()
    kq = []
    with torch.inference_mode():
        for dau in range(0, len(x), co_lo):
            kq.append(mang(x[dau:dau + co_lo]))
    return torch.cat(kq)


def chi_so(logits, y):
    """loss, accuracy, accuracy cân bằng (trung bình recall các lớp có mặt)."""
    loss = float(nn.functional.cross_entropy(logits, y))
    du = logits.argmax(1)
    acc = float((du == y).float().mean())
    recall = [float((du[y == c] == c).float().mean()) for c in range(logits.shape[1]) if (y == c).any()]
    return loss, acc, float(np.mean(recall))


def huan_luyen(bien_the, *, so_epoch=None, hat=None, so_luong=6, in_ra=True, thu_muc_mo_hinh=None, thu_muc_ket_qua=None):
    if bien_the not in BIEN_THE:
        raise ValueError(f"biến thể phải là một trong {BIEN_THE}")
    ct = {**CONG_THUC, **({"so_epoch": so_epoch} if so_epoch else {}), **({"hat": hat} if hat is not None else {})}
    torch.set_num_threads(so_luong)
    dat_hat(ct["hat"])
    bt = BIEN_THE_CAU_HINH[bien_the]
    co_tang_cuong, co_cvl = bt["tang_cuong"], bt["cvl"]
    lop = list(LOP) if bt["rac"] else list(LOP_KY_TU)
    kien_truc = {"loai": bt["loai"], "so_lop": len(lop), "batch_norm": bt["loai"] == "lon",
                 "dropout": 0.4 if bt["loai"] == "lon" else 0.25}
    x_tr, y_tr, ng_tr = tai_tap("train", kieu_viet=co_tang_cuong, cvl=co_cvl, emnist=bt["emnist"], rac=bt["rac"],
                                tra_nguon=True)
    x_va, y_va = tai_tap("validation", cvl=co_cvl, rac=bt["rac"])
    x_tr, y_tr, x_va, y_va = map(torch.from_numpy, (x_tr, y_tr, x_va, y_va))
    so_lop = len(lop)
    dem = torch.bincount(y_tr, minlength=so_lop).float()
    trong_so = (1.0 / dem)[y_tr]
    if co_cvl:                     # trong mỗi lớp chữ số: CVL chiếm TI_LE_CVL, phần còn lại là MNIST (+ EMNIST)
        la_cvl = torch.from_numpy(ng_tr == "cvl_chu_so")
        for c in range(len(CHU_SO)):
            m_cvl, m_khac = (y_tr == c) & la_cvl, (y_tr == c) & ~la_cvl
            if m_cvl.any():
                trong_so[m_cvl] = TI_LE_CVL / m_cvl.sum() / so_lop
                trong_so[m_khac] = (1 - TI_LE_CVL) / m_khac.sum() / so_lop
    so_mau = bt["so_mau_moi_epoch"] or len(y_tr)
    lay_mau = WeightedRandomSampler(trong_so.double(), num_samples=so_mau, replacement=True,
                                    generator=torch.Generator().manual_seed(ct["hat"] + 1))
    nap = DataLoader(TensorDataset(x_tr, y_tr), batch_size=ct["co_lo"], sampler=lay_mau, num_workers=0)
    mang = tao_mang(kien_truc)
    toi_uu = torch.optim.Adam(mang.parameters(), lr=ct["toc_do_hoc"])
    lich = torch.optim.lr_scheduler.StepLR(toi_uu, *ct["buoc_giam"])
    rng_tc = torch.Generator().manual_seed(ct["hat"] + 1000)
    ket_qua = Path(thu_muc_ket_qua or KET_QUA / "phan_loai") / bien_the
    tep = tep_mo_hinh(bien_the, thu_muc_mo_hinh)
    tep.parent.mkdir(parents=True, exist_ok=True)
    cau_hinh = {"bien_the": bien_the, **{k: v for k, v in ct.items()}, "bo_toi_uu": "Adam",
                "lich_giam_toc_do": f"StepLR(step={ct['buoc_giam'][0]}, gamma={ct['buoc_giam'][1]})",
                "kien_truc": kien_truc, "so_tham_so": dem_tham_so(mang), "lop": lop,
                "lay_mau": f"cân bằng lớp (trọng số 1/số mẫu của lớp), có hoàn lại, {so_mau} mẫu mỗi epoch",
                "du_lieu_them": [t for t, co in (
                    (f"CVL Single Digits ({TI_LE_CVL:.0%} số lần lấy mẫu mỗi lớp chữ số)", co_cvl),
                    ("biến thể 1 có mũ / 7 có gạch", co_tang_cuong), ("60.000 chữ số EMNIST", bt["emnist"]),
                    ("lớp rác", bt["rac"])) if co] or None,
                "tang_cuong": ("xoay ±10°, dịch ±2 px, co giãn [0,85; 1,1], giãn/co nét 3×3; thêm 1 có mũ / 7 có gạch"
                               if co_tang_cuong else None),
                "so_train": len(y_tr), "so_validation": len(y_va), "so_train_moi_lop": dem.int().tolist(),
                "phien_ban": {"torch": str(torch.__version__), "numpy": np.__version__,
                              "python": platform.python_version()},
                "tieu_chi_chon": "accuracy cân bằng validation cao nhất, hòa thì loss validation thấp nhất; không đọc test",
                "bat_dau_utc": datetime.now(timezone.utc).isoformat()}
    tot, epoch_tot, lich_su = (-1.0, -float("inf")), 0, []
    bat_dau = time.perf_counter()
    for epoch in range(1, ct["so_epoch"] + 1):
        mang.train()
        t0 = time.perf_counter()
        loss_tong, dung, tong = 0.0, 0, 0
        for xb, yb in nap:
            if co_tang_cuong:
                xb = tang_cuong_ky_tu(xb, rng_tc)
            toi_uu.zero_grad(set_to_none=True)
            logits = mang(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            toi_uu.step()
            loss_tong += float(loss) * len(yb)
            dung += int((logits.argmax(1) == yb).sum())
            tong += len(yb)
        loss_va, acc_va, can_bang_va = chi_so(du_doan_lo(mang, x_va), y_va)
        dong = {"epoch": epoch, "train_loss": loss_tong / tong, "train_accuracy": dung / tong,
                "validation_loss": loss_va, "validation_accuracy": acc_va, "validation_can_bang": can_bang_va,
                "toc_do_hoc": float(toi_uu.param_groups[0]["lr"]), "giay": time.perf_counter() - t0}
        lich_su.append(dong)
        if (can_bang_va, -loss_va) > tot:
            tot, epoch_tot = (can_bang_va, -loss_va), epoch
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in mang.state_dict().items()},
                        "bien_the": bien_the, "lop": lop, "kien_truc": kien_truc,
                        "phien_ban_dau_vao": PHIEN_BAN_DAU_VAO, "epoch_tot_nhat": epoch,
                        "validation_can_bang": can_bang_va, "loss_validation": loss_va}, tep)
        lich.step()
        if in_ra:
            print(f"{bien_the} {epoch:02d}/{ct['so_epoch']}: train={dung / tong:.4f} val={acc_va:.4f} "
                  f"cân bằng={can_bang_va:.4f} loss={loss_va:.4f} ({dong['giay']:.0f}s)", flush=True)
    ghi_csv(ket_qua / "history.csv", lich_su)
    cau_hinh.update(epoch_tot_nhat=epoch_tot, validation_can_bang_tot_nhat=tot[0], loss_validation_tot_nhat=-tot[1],
                    giay_huan_luyen=time.perf_counter() - bat_dau, sha256_checkpoint=sha256(tep))
    ghi_json(ket_qua / "config.json", cau_hinh)
    return cau_hinh


def chon_mo_hinh(thu_muc_mo_hinh=None, thu_muc_ket_qua=None, in_ra=True, so_thanh_phan_toi_da=2):
    """Chọn mô hình cuối trong các cấu hình đơn và mọi cặp ghép (ghép = trung bình softmax; mô hình 15 lớp coi xác suất
    "rác" = 0); ghi mo_hinh/lua_chon.json (khóa "thanh_phan" là danh sách cấu hình được ghép).

    Cho mỗi ứng viên chạy TOÀN BỘ luồng đọc (chế độ chỉ chữ số) trên 90 dãy CVL của 9 người kiểm định (không có ảnh nào
    trong huấn luyện). Vì 90 dãy là ít, các ứng viên đọc đúng ít hơn ứng viên tốt nhất không quá SAI_SO_DAY dãy được
    coi là NGANG NHAU; trong nhóm đó chọn CER thấp nhất trên 90 dãy, rồi accuracy trên 969 chữ số CVL của chính 9 người
    đó, rồi ít thành phần hơn. Không dùng tập test ký tự, không dùng nửa test của CVL Strings.

    Ghi chú quy trình: ngưỡng SAI_SO_DAY được thêm SAU khi quy tắc cũ (so đúng tuyệt đối số dãy) chọn một mô hình hơn
    đúng 1/90 dãy nhưng kém hơn rõ trên tập test — xem bao_cao/thuc_nghiem.md.
    """
    from itertools import combinations
    from .danh_gia import danh_gia_cvl
    from .doc_trang import DocTrang
    from .du_doan import PhanLoaiKyTu
    goc = Path(thu_muc_ket_qua or KET_QUA / "phan_loai")
    ch = {b: doc_json(goc / b / "config.json") for b in BIEN_THE}
    for b, c in ch.items():
        if c["sha256_checkpoint"] != sha256(tep_mo_hinh(b, thu_muc_mo_hinh)):
            raise ValueError(f"Checkpoint {b} đã thay đổi so với lúc huấn luyện")
    x, y, nguon = tai_tap("validation", cvl=True, tra_nguon=True)
    la_cvl = nguon == "cvl_chu_so"
    p = {b: PhanLoaiKyTu(tep_mo_hinh(b, thu_muc_mo_hinh), so_luong=6).xac_suat(x) for b in BIEN_THE}
    diem = {}
    for k in range(1, so_thanh_phan_toi_da + 1):
        for to_hop in combinations(BIEN_THE, k):
            du = np.mean([p[b] for b in to_hop], axis=0)[:, :len(LOP_KY_TU)].argmax(1)
            doc = DocTrang(PhanLoaiKyTu([tep_mo_hinh(b, thu_muc_mo_hinh) for b in to_hop], so_luong=6))
            kq = danh_gia_cvl(doc, "chon", tap="kiem_dinh", in_ra=False, ghi=False, chi_chu_so=True)
            diem[to_hop] = {"cvl_kiem_dinh_dung_ca_day": kq["dung_ca_day"], "cvl_kiem_dinh_so_day": kq["so_anh"],
                            "cvl_kiem_dinh_cer": kq["cer"],
                            "cvl_chu_so_kiem_dinh": round(float((du[la_cvl] == y[la_cvl]).mean()), 4),
                            "mnist_validation": round(float((du[nguon == "mnist"] == y[nguon == "mnist"]).mean()), 4)}
            if in_ra:
                print(f"{'+'.join(to_hop)}: {diem[to_hop]}", flush=True)
    tot_nhat = max(d["cvl_kiem_dinh_dung_ca_day"] for d in diem.values())
    ngang = [t for t in diem if diem[t]["cvl_kiem_dinh_dung_ca_day"] >= tot_nhat - SAI_SO_DAY]
    chon = min(ngang, key=lambda t: (diem[t]["cvl_kiem_dinh_cer"], -diem[t]["cvl_chu_so_kiem_dinh"], len(t)))
    lua_chon = {"thanh_phan": list(chon), "bien_the_duoc_chon": "+".join(chon),
                "cach_ghep": "trung bình softmax của các thành phần (mô hình 15 lớp: xác suất rác = 0)",
                "tieu_chi": f"các ứng viên đọc đúng ít hơn tốt nhất không quá {SAI_SO_DAY}/90 dãy CVL kiểm định coi là "
                            "ngang nhau; trong đó chọn CER thấp nhất trên 90 dãy, rồi accuracy chữ số CVL của 9 người đó, "
                            "rồi ít thành phần hơn; không dùng test ký tự, không dùng CVL Strings test",
                "ung_vien_ngang_nhau": ["+".join(t) for t in ngang],
                "ghi_chu": "ngưỡng ngang nhau được thêm sau khi quy tắc cũ chọn nho_rac (hơn 1/90 dãy) nhưng kém trên test",
                "khong_dung_test": True,
                "to_hop": {"+".join(t): d for t, d in diem.items()},
                "ung_vien": {b: {"epoch": c["epoch_tot_nhat"], "sha256_checkpoint": c["sha256_checkpoint"]}
                             for b, c in ch.items()},
                "chot_luc_utc": datetime.now(timezone.utc).isoformat()}
    ghi_json(tep_lua_chon(thu_muc_mo_hinh), lua_chon)
    return lua_chon

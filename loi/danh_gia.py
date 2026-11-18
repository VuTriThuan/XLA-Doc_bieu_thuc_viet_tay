"""Đánh giá: (1) CNN trên tập test ký tự, (2) đọc dãy số viết tay thật CVL, (3) đọc trang tổng hợp, (4) ảnh chụp thật.

Mọi số liệu ghi ra ket_qua/; báo cáo chỉ được lấy số từ các tệp này.
"""
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from .bieu_thuc import tach_tu_to, tinh, LoiBieuThuc
from .cau_hinh import DU_LIEU, KET_QUA, LOP, LOP_KY_TU, SO_LOP, doc_json, ghi_csv, ghi_json


def khoang_cach_sua(a, b):
    """Khoảng cách Levenshtein (chèn/xóa/thay một ký tự)."""
    truoc = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        hien = [i]
        for j, cb in enumerate(b, 1):
            hien.append(min(truoc[j] + 1, hien[j - 1] + 1, truoc[j - 1] + (ca != cb)))
        truoc = hien
    return truoc[-1]


def wilson(dung, n, z=1.96):
    """Khoảng tin cậy 95% Wilson cho một tỉ lệ."""
    if n == 0:
        return (0.0, 0.0)
    p = dung / n
    t = z * z / n
    giua = (p + t / 2) / (1 + t)
    nua = z * np.sqrt(p * (1 - p) / n + t / (4 * n)) / (1 + t)
    return (round(float(giua - nua), 4), round(float(giua + nua), 4))


# ------------------------------------------------------------------------------------------------ (1) CNN
def danh_gia_phan_loai(phan_loai, thu_muc=None):
    from bo_du_lieu.tap_ky_tu import TEP
    with np.load(TEP, allow_pickle=False) as d:
        chon = d["tap"] == "test"
        x, y, nguon = d["x"][chon].astype(np.float32) / 255, d["y"][chon], d["nguon"][chon]
    du = phan_loai.xac_suat(x).argmax(1)
    ma_tran = np.zeros((SO_LOP, SO_LOP), int)
    np.add.at(ma_tran, (y, du), 1)
    theo_lop = [{"lop": c, "so_mau": int(ma_tran[i].sum()), "dung": int(ma_tran[i, i]),
                 "recall": round(float(ma_tran[i, i] / max(1, ma_tran[i].sum())), 4),
                 "precision": round(float(ma_tran[i, i] / max(1, ma_tran[:, i].sum())), 4)} for i, c in enumerate(LOP)]
    theo_nguon = {}
    for ng in sorted(set(nguon)):
        m = nguon == ng
        theo_nguon[ng] = {"so_mau": int(m.sum()), "dung": int((du[m] == y[m]).sum()),
                          "accuracy": round(float((du[m] == y[m]).mean()), 4),
                          "khoang_tin_cay_95": wilson(int((du[m] == y[m]).sum()), int(m.sum()))}
    ky_hieu = (y >= 10) & (y < len(LOP_KY_TU))
    la_rac = y == len(LOP_KY_TU)
    kq = {"bien_the": phan_loai.bien_the, "so_mau": int(len(y)), "dung": int((du == y).sum()),
          "accuracy": round(float((du == y).mean()), 4),
          "accuracy_can_bang": round(float(np.mean([r["recall"] for r in theo_lop if r["so_mau"]])), 4),
          "chu_so_mnist": theo_nguon.get("mnist"),
          "ky_hieu": {"so_mau": int(ky_hieu.sum()), "dung": int((du[ky_hieu] == y[ky_hieu]).sum()),
                      "accuracy": round(float((du[ky_hieu] == y[ky_hieu]).mean()), 4),
                      "khoang_tin_cay_95": wilson(int((du[ky_hieu] == y[ky_hieu]).sum()), int(ky_hieu.sum()))},
          "rac": ({"so_mau": int(la_rac.sum()), "nhan_ra_la_rac": int((du[la_rac] == y[la_rac]).sum()),
                   "ky_tu_that_bi_goi_la_rac": int(((du == len(LOP_KY_TU)) & ~la_rac).sum())} if la_rac.any() else None),
          "theo_nguon": theo_nguon, "theo_lop": theo_lop}
    thu_muc = Path(thu_muc or KET_QUA / "phan_loai")
    ghi_json(thu_muc / "test_metrics.json", kq)
    ghi_csv(thu_muc / "confusion_matrix.csv", [{"that\\du_doan": LOP[i], **{LOP[j]: int(ma_tran[i, j]) for j in range(SO_LOP)}}
                                               for i in range(SO_LOP)])
    ve_ma_tran(ma_tran, thu_muc / "confusion_matrix.png", f"Ma trận nhầm lẫn trên tập test ({phan_loai.bien_the})")
    return kq


def ve_ma_tran(ma_tran, duong_dan, tieu_de):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chuan = ma_tran / np.maximum(ma_tran.sum(1, keepdims=True), 1)
    hinh, truc = plt.subplots(figsize=(7.5, 6.5))
    truc.imshow(chuan, cmap="Blues", vmin=0, vmax=1)
    for i in range(SO_LOP):
        for j in range(SO_LOP):
            if ma_tran[i, j]:
                truc.text(j, i, str(ma_tran[i, j]), ha="center", va="center", fontsize=6.5,
                          color="white" if chuan[i, j] > 0.5 else "black")
    truc.set_xticks(range(SO_LOP), LOP)
    truc.set_yticks(range(SO_LOP), LOP)
    truc.set_xlabel("Dự đoán")
    truc.set_ylabel("Nhãn thật")
    truc.set_title(tieu_de)
    hinh.tight_layout()
    hinh.savefig(duong_dan, dpi=150)
    plt.close(hinh)


# ------------------------------------------------------------------------------------------------ (2) CVL
def danh_gia_cvl(doc_trang, ten_thi_nghiem="mac_dinh", gioi_han=None, thu_muc=None, in_ra=True, tap="test",
                 ghi=True, **tuy_chon):
    """Đọc từng ảnh dãy số CVL của nửa `tap` (theo người viết); bỏ dấu cách, so cả chuỗi.

    Chỉ số: tỉ lệ đọc đúng cả dãy, tỉ lệ lỗi ký tự (CER = tổng khoảng cách sửa / tổng độ dài nhãn).
    """
    from bo_du_lieu.cvl import doc_cvl
    dong, sai = [], []
    tong_sua, tong_ky_tu, dung = 0, 0, 0
    bat_dau = time.perf_counter()
    for i, (ten, nhan, nguoi, anh) in enumerate(doc_cvl(gioi_han=gioi_han, tap=tap)):
        try:
            kq = doc_trang.doc(anh, **tuy_chon)
            doc = "".join(k.ky_tu for d in kq.dong for k in d.ky_tu)
        except ValueError:
            doc = ""
        s = khoang_cach_sua(doc, nhan)
        tong_sua += s
        tong_ky_tu += len(nhan)
        dung += doc == nhan
        dong.append({"tep": ten, "nhan": nhan, "doc_duoc": doc, "dung": doc == nhan, "so_loi_ky_tu": s,
                     "nguoi_viet": nguoi})
        if in_ra and (i + 1) % 200 == 0:
            print(f"  CVL {ten_thi_nghiem}: {i + 1} ảnh, đúng cả dãy {dung}/{i + 1}", flush=True)
    n = len(dong)
    if tap == "validation":       # tập con: 90 dãy của 9 người kiểm định (không có ảnh nào trong huấn luyện)
        from bo_du_lieu.cvl import nguoi_kiem_dinh
        kd = [r for r in dong if nguoi_kiem_dinh(r["nguoi_viet"])]
        kq_kd = {"so_anh": len(kd), "dung_ca_day": sum(r["dung"] for r in kd),
                 "cer": round(sum(r["so_loi_ky_tu"] for r in kd) / max(1, sum(len(r["nhan"]) for r in kd)), 4)}
    else:
        kq_kd = None
    theo_day = {}
    for r in dong:
        t = theo_day.setdefault(r["nhan"], [0, 0])
        t[0] += r["dung"]
        t[1] += 1
    kq = {"thi_nghiem": ten_thi_nghiem, "tap": tap, "so_nguoi_viet": len({r["nguoi_viet"] for r in dong}),
          "tuy_chon": {k: v for k, v in tuy_chon.items()}, "so_anh": n,
          "dung_ca_day": dung, "ti_le_dung_ca_day": round(dung / n, 4), "khoang_tin_cay_95": wilson(dung, n),
          "cer": round(tong_sua / tong_ky_tu, 4), "so_ky_tu": tong_ky_tu,
          "theo_day": {k: {"dung": v[0], "so_anh": v[1]} for k, v in sorted(theo_day.items())},
          "giay": round(time.perf_counter() - bat_dau, 1), "kiem_dinh_90": kq_kd}
    if ghi:
        thu_muc = Path(thu_muc or KET_QUA / "cvl" / tap)
        ghi_json(thu_muc / f"{ten_thi_nghiem}.json", kq)
        ghi_csv(thu_muc / f"{ten_thi_nghiem}_chi_tiet.csv", dong)
    return kq


def phan_tich_loi_cvl(ten_thi_nghiem, tap="test"):
    """Phân loại các dãy đọc sai trong ket_qua/cvl/<tap>/<tên>_chi_tiet.csv: thừa ký tự, thiếu ký tự, cùng độ dài nhưng
    sai đúng một chữ số; và các cặp nhầm (nhãn → đọc) đếm trên những dãy đọc đủ độ dài."""
    import csv
    from collections import Counter
    tep = KET_QUA / "cvl" / tap / f"{ten_thi_nghiem}_chi_tiet.csv"
    with open(tep, encoding="utf-8-sig") as f:
        dong = list(csv.DictReader(f))
    sai = [r for r in dong if r["dung"] != "True"]
    cap = Counter()
    for r in sai:
        if len(r["doc_duoc"]) == len(r["nhan"]):
            cap.update(f"{a}→{b}" for a, b in zip(r["nhan"], r["doc_duoc"]) if a != b)
    return {"thi_nghiem": ten_thi_nghiem, "so_day": len(dong), "day_sai": len(sai),
            "thua_ky_tu": sum(len(r["doc_duoc"]) > len(r["nhan"]) for r in sai),
            "thieu_ky_tu": sum(len(r["doc_duoc"]) < len(r["nhan"]) for r in sai),
            "sai_dung_1_chu_so": sum(len(r["doc_duoc"]) == len(r["nhan"]) and
                                     sum(a != b for a, b in zip(r["nhan"], r["doc_duoc"])) == 1 for r in sai),
            "cap_nham_nhieu_nhat": "; ".join(f"{k} ({v})" for k, v in cap.most_common(3))}


# ------------------------------------------------------------------------------------------------ (3)(4) trang
def so_trang(doc_trang, thu_muc_anh, nhan_trang, ten_thi_nghiem, thu_muc_kq, **tuy_chon):
    """So từng trang với nhãn (danh sách dòng). Dòng ghép theo thứ tự trên → dưới.

    Chỉ số: CER trên toàn trang (các dòng nối bằng xuống dòng, bỏ dấu cách), tỉ lệ dòng đọc đúng hoàn toàn,
    và với dòng là biểu thức:
    - biểu thức có giá trị xác định (vd. 13×4): tỉ lệ ra ĐÚNG GIÁ TRỊ (vế trái tính được và bằng giá trị của nhãn);
    - biểu thức không xác định (vd. 4÷0): đếm riêng, đúng khi dòng đọc đúng và hệ thống báo lỗi thay vì ra giá trị.
    Biểu thức không xác định không nằm trong mẫu số của tỉ lệ đúng giá trị (nó không thể "ra đúng giá trị").
    """
    from tien_xu_ly.anh import doc_anh_tu_tep
    chi_tiet = []
    tong_sua = tong_ky_tu = dong_dung = so_dong = bt_dung = so_bt = kxd_dung = so_kxd = 0
    for ten, nhan in nhan_trang.items():
        kq = doc_trang.doc(doc_anh_tu_tep(Path(thu_muc_anh) / ten), **tuy_chon)
        doc = [d.ket_qua.chuoi.replace(" ", "") for d in kq.dong]
        that = [s.replace(" ", "") for s in nhan]
        s = khoang_cach_sua("\n".join(doc), "\n".join(that))
        tong_sua += s
        tong_ky_tu += len("\n".join(that))
        for i, t in enumerate(that):
            d = doc[i] if i < len(doc) else ""
            so_dong += 1
            dong_dung += d == t
            tu_to = tach_tu_to(list(t))
            if any(c in "+-×÷" for c in t):
                doc_bt = kq.dong[i].ket_qua if i < len(kq.dong) else None
                trai = tu_to[:tu_to.index("=")] if "=" in tu_to else tu_to
                try:
                    gt = tinh(trai)
                except LoiBieuThuc:                     # nhãn không có giá trị (chia cho 0)
                    so_kxd += 1
                    kxd_dung += d == t and doc_bt is not None and doc_bt.gia_tri is None
                    continue
                so_bt += 1
                bt_dung += doc_bt is not None and doc_bt.gia_tri is not None and doc_bt.gia_tri == gt
        chi_tiet.append({"trang": ten, "so_dong_nhan": len(that), "so_dong_doc": len(doc), "so_loi_ky_tu": s,
                         "nhan": " | ".join(that), "doc_duoc": " | ".join(doc)})
    kq = {"thi_nghiem": ten_thi_nghiem, "tuy_chon": tuy_chon, "so_trang": len(nhan_trang), "so_dong": so_dong,
          "dong_dung": dong_dung, "ti_le_dong_dung": round(dong_dung / max(1, so_dong), 4),
          "khoang_tin_cay_dong_95": wilson(dong_dung, so_dong),
          "so_bieu_thuc": so_bt, "bieu_thuc_dung_gia_tri": bt_dung,
          "ti_le_bieu_thuc_dung": round(bt_dung / max(1, so_bt), 4), "khoang_tin_cay_bieu_thuc_95": wilson(bt_dung, so_bt),
          "so_bieu_thuc_khong_xac_dinh": so_kxd, "khong_xac_dinh_bao_loi_dung": kxd_dung,
          "cer": round(tong_sua / max(1, tong_ky_tu), 4)}
    ghi_json(Path(thu_muc_kq) / f"{ten_thi_nghiem}.json", kq)
    ghi_csv(Path(thu_muc_kq) / f"{ten_thi_nghiem}_chi_tiet.csv", chi_tiet)
    return kq


def danh_gia_trang_tong_hop(doc_trang, ten_thi_nghiem="mac_dinh", **tuy_chon):
    goc = DU_LIEU / "trang_tong_hop"
    nhan = {k: v["dong"] for k, v in doc_json(goc / "nhan.json")["trang"].items()}
    return so_trang(doc_trang, goc, nhan, ten_thi_nghiem, KET_QUA / "trang_tong_hop", **tuy_chon)


def danh_gia_trang_that(doc_trang, **tuy_chon):
    """Ảnh chụp thật trong du_lieu/trang_that/ với nhãn khai báo trước trong nhan.json: {"tệp": ["dòng 1", ...]}.

    Mỗi thư mục con có nhan.json riêng (vd. trang_01_cac_dong/: từng phép tính cắt từ trang_01.jpg) được đánh giá thành
    một bộ riêng, không gộp với trang gốc. Trả về {tên bộ: kết quả}; rỗng nếu chưa có ảnh.
    """
    goc = DU_LIEU / "trang_that"
    kq = {}
    for tep in sorted([goc / "nhan.json", *goc.glob("*/nhan.json")]):
        if not tep.exists() or not doc_json(tep):
            continue
        ten = "trang_that" if tep.parent == goc else f"trang_that_{tep.parent.name}"
        kq[ten] = so_trang(doc_trang, tep.parent, doc_json(tep), ten, KET_QUA / "trang_that", **tuy_chon)
    return kq

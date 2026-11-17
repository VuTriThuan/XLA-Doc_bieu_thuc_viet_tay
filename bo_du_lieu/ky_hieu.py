"""Ký hiệu phép toán viết tay thật: HASYv2 (+ − × ÷) và CROHME+ (= + − × ÷).

Hai bộ đều do nhiều người viết trên bảng vẽ/chuột (dữ liệu trực tuyến), nên ảnh ở đây được dựng lại từ nét:
- HASYv2: ảnh PNG 32×32 có sẵn, nét đen nền trắng.
- CROHME+: tệp InkML lưu tọa độ nét của cả biểu thức và nhóm nét (traceGroup) của từng ký hiệu; ta vẽ lại từng
  ký hiệu thành ảnh 64×64.
Tệp gốc được tải về thư mục tạm; dự án chỉ giữ `du_lieu/ky_hieu/ky_hieu.npz` (ảnh thô đã trích + nhãn + người viết
+ tập) để chạy lại không cần tải. HASYv2 không có dấu '=' nên '=' chỉ đến từ CROHME+.

Chia tập:
- HASYv2: theo NGƯỜI VIẾT (user_id) 70/15/15, để người đã thấy lúc huấn luyện không xuất hiện trong test.
- CROHME+: khử trùng lặp nét (phần dữ liệu ghép dùng lại nét), rồi chia theo băm của mã tờ viết gốc (annotation UI,
  vd. 2011_IVC_depart_F039) 70/15/15 — xấp xỉ chia theo người viết.
"""
import csv
import hashlib
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from loi.cau_hinh import DU_LIEU, ghi_json

THU_MUC = DU_LIEU / "ky_hieu"
TEP = THU_MUC / "ky_hieu.npz"
HAT_CHIA = 20261117
KICH_THUOC_THO = 64
TOI_DA_MOI_NGUOI = 80          # HASYv2: số mẫu tối đa của một người viết cho một lớp

HASY_LOP = {"+": "+", "-": "-", "\\times": "×", "\\div": "÷"}
CROHME_LOP = {"=": "=", "+": "+", "-": "-", "\\times": "×", "\\div": "÷", "\\minus": "-"}
NGUON = {
    "hasy": {"ten": "HASYv2", "url": "https://zenodo.org/records/259444", "giay_phep": "ODbL 1.0",
             "tac_gia": "Martin Thoma (2017), arXiv:1701.08380"},
    "crohme": {"ten": "CROHME+ (mở rộng CROHME 2023)", "url": "https://zenodo.org/records/14968570",
               "giay_phep": "CC BY-NC-SA 3.0", "tac_gia": "CROHME (Mouchère và cs.) + chú thích cấu trúc CROHME+"},
}


def chia_theo_nhom(nhom, nhan, ti_le=(0.7, 0.15, 0.15), so_lan_thu=3000, hat=HAT_CHIA):
    """Gán train/validation/test sao cho mọi mẫu của cùng một người viết chung một tập.

    Số mẫu mỗi người rất lệch (một người viết 1.396 dấu ×), nên thử nhiều thứ tự ngẫu nhiên của người viết, cắt theo
    tỉ lệ tích lũy, và giữ cách chia có tỉ lệ từng lớp gần 70/15/15 nhất (mỗi lớp phải có mặt ở cả ba tập).
    """
    nhom, nhan = list(nhom), list(nhan)
    nguoi = sorted(set(nhom))
    lop = sorted(set(nhan))
    dem = {g: {c: 0 for c in lop} for g in nguoi}
    for g, c in zip(nhom, nhan):
        dem[g][c] += 1
    tong_lop = {c: sum(dem[g][c] for g in nguoi) for c in lop}
    rng = np.random.default_rng(hat)
    tot, tot_diem = None, float("inf")
    for _ in range(so_lan_thu):
        thu_tu = list(rng.permutation(nguoi))
        tap, tich_luy = {}, 0
        for g in thu_tu:
            tap[g] = "train" if tich_luy < ti_le[0] * len(nhom) else \
                     "validation" if tich_luy < (ti_le[0] + ti_le[1]) * len(nhom) else "test"
            tich_luy += sum(dem[g].values())
        diem = 0.0
        for c in lop:
            for t, muc_tieu in zip(("train", "validation", "test"), ti_le):
                n = sum(dem[g][c] for g in nguoi if tap[g] == t)
                if n == 0:
                    diem += 10
                diem += abs(n / tong_lop[c] - muc_tieu)
        if diem < tot_diem:
            tot, tot_diem = tap, diem
    return [tot[g] for g in nhom]


def gioi_han_moi_nguoi(mau, toi_da, hat=HAT_CHIA):
    """Mỗi người viết giữ tối đa `toi_da` mẫu mỗi lớp (chọn ngẫu nhiên có hạt) để một người không chiếm cả lớp."""
    rng = np.random.default_rng(hat)
    theo = {}
    for i, m in enumerate(mau):
        theo.setdefault((m["nguoi_viet"], m["nhan"]), []).append(i)
    giu = sorted(j for v in theo.values() for j in (v if len(v) <= toi_da else rng.choice(v, toi_da, replace=False)))
    return [mau[i] for i in giu]


def doc_hasy(thu_muc):
    """thu_muc: nơi giải nén HASYv2.tar.bz2 (có hasy-data-labels.csv). Trả về list dict."""
    from PIL import Image
    thu_muc = Path(thu_muc)
    mau = []
    with (thu_muc / "hasy-data-labels.csv").open(encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if d["latex"] in HASY_LOP:
                with Image.open(thu_muc / d["path"]) as im:
                    a = np.array(im.convert("L"))
                muc = 255 - a                                   # nét sáng
                mau.append({"anh": phong_vao_o(muc), "nhan": HASY_LOP[d["latex"]], "nguoi_viet": f"hasy-{d['user_id']}",
                            "nguon": "hasy", "tep": d["path"]})
    mau = gioi_han_moi_nguoi(mau, TOI_DA_MOI_NGUOI)
    tap = chia_theo_nhom([m["nguoi_viet"] for m in mau], [m["nhan"] for m in mau])
    for m, t in zip(mau, tap):
        m["tap"] = t
    return mau


def phong_vao_o(muc, canh=KICH_THUOC_THO):
    """Đặt ảnh uint8 (nét sáng) vào ô vuông canh×canh, giữ tỉ lệ (lấy mẫu láng giềng gần nhất)."""
    h, w = muc.shape
    k = canh / max(h, w)
    ch, cw = max(1, round(h * k)), max(1, round(w * k))
    yy = np.minimum((np.arange(ch) / k).astype(int), h - 1)
    xx = np.minimum((np.arange(cw) / k).astype(int), w - 1)
    o = np.zeros((canh, canh), np.uint8)
    o[(canh - ch) // 2:(canh - ch) // 2 + ch, (canh - cw) // 2:(canh - cw) // 2 + cw] = muc[yy][:, xx]
    return o


def ve_net(cac_net, canh=KICH_THUOC_THO, le=4, day=2):
    """Vẽ các nét (list mảng N×2 tọa độ x, y) vào ảnh canh×canh, giữ tỉ lệ, nét dày `day` px (Pillow chỉ để rải pixel)."""
    from PIL import Image, ImageDraw
    diem = np.concatenate(cac_net)
    lo, hi = diem.min(0), diem.max(0)
    co = max(float((hi - lo).max()), 1e-6)
    k = (canh - 2 * le - day) / co
    lech = (canh - (hi - lo) * k) / 2
    im = Image.new("L", (canh, canh), 0)
    ve = ImageDraw.Draw(im)
    for n in cac_net:
        p = [tuple(map(float, (q - lo) * k + lech)) for q in n]
        if len(p) == 1:
            x, y = p[0]
            ve.ellipse((x - day / 2, y - day / 2, x + day / 2, y + day / 2), fill=255)
        else:
            ve.line(p, fill=255, width=day, joint="curve")
            for x, y in (p[0], p[-1]):
                ve.ellipse((x - day / 2, y - day / 2, x + day / 2, y + day / 2), fill=255)
    return np.array(im)


def _bo_ns(tag):
    return tag.split("}")[-1]


def nguon_bieu_thuc(noi_dung):
    """Mã biểu thức gốc (annotation type="UI"), vd. 2011_IVC_depart_F039_E070; bỏ đuôi _E070 để còn mã tờ/người viết."""
    goc = ET.fromstring(noi_dung)
    ui = next((a.text for a in goc if _bo_ns(a.tag) == "annotation" and a.get("type") == "UI"), None) or ""
    return re.sub(r"_E\d+$", "", ui.strip())


def doc_inkml(noi_dung):
    """noi_dung: bytes của một tệp InkML. Trả về list (nhãn LaTeX, list mảng nét N×2) cho từng traceGroup có nhãn."""
    goc = ET.fromstring(noi_dung)
    net = {}
    for t in goc.iter():
        if _bo_ns(t.tag) == "trace":
            diem = []
            for cap in (t.text or "").strip().split(","):
                so = cap.split()
                if len(so) >= 2:
                    diem.append((float(so[0]), float(so[1])))
            if diem:
                net[t.get("id")] = np.array(diem)
    kq = []
    for g in goc.iter():
        if _bo_ns(g.tag) != "traceGroup":
            continue
        nhan = next((c.text for c in g if _bo_ns(c.tag) == "annotation"), None)
        ref = [c.get("traceDataRef") for c in g if _bo_ns(c.tag) == "traceView"]
        if nhan is not None and ref and all(r in net for r in ref):
            kq.append((nhan.strip(), [net[r] for r in ref]))
    return kq


def khoa_net(cac_net):
    """Dấu vân tay của một ký hiệu: tọa độ tương đối (làm tròn) của mọi nét. Dùng để khử trùng lặp, vì phần dữ liệu
    ghép của CROHME+ dùng lại cùng một nét viết tay trong nhiều biểu thức."""
    diem = np.concatenate(cac_net)
    lo = diem.min(0)
    co = max(float((diem.max(0) - lo).max()), 1e-6)
    return hashlib.sha1(b"|".join(np.round((n - lo) / co * 1000).astype(np.int32).tobytes() for n in cac_net)).hexdigest()


def doc_crohme(tep_zip, toi_da_moi_lop=1500):
    """Đọc thẳng từ CROHME+.zip (không giải nén 124.000 tệp): phần người viết thật (real_train, val, test) và phần
    ghép lại từ nét viết tay thật (gen_syntactic_data). Bỏ phần sinh từ LaTeX. Khử trùng lặp theo khoa_net; mỗi lớp
    tối đa toi_da_moi_lop mẫu. Chia tập theo băm của mã tờ viết gốc (annotation UI, xấp xỉ người viết) 70/15/15.
    """
    mau, da_thay = [], set()
    dem = {}
    with zipfile.ZipFile(tep_zip) as z:
        ten = sorted(n for n in z.namelist() if n.endswith(".inkml") and
                     re.search(r"/(real_train|val|test|gen_syntactic_data)/", n))
        for n in ten:
            if all(dem.get(c, 0) >= toi_da_moi_lop for c in set(CROHME_LOP.values())):
                break
            try:
                du_lieu = z.read(n)
                nhom, to_goc = doc_inkml(du_lieu), nguon_bieu_thuc(du_lieu) or n
            except ET.ParseError:
                continue
            for nhan, cac_net in nhom:
                c = CROHME_LOP.get(nhan)
                if c is None or dem.get(c, 0) >= toi_da_moi_lop:
                    continue
                k = khoa_net(cac_net)
                if k in da_thay:
                    continue
                da_thay.add(k)
                dem[c] = dem.get(c, 0) + 1
                v = int(hashlib.sha1(to_goc.encode()).hexdigest()[:8], 16) % 100
                mau.append({"anh": ve_net(cac_net), "nhan": c, "nguoi_viet": f"crohme-{to_goc}", "nguon": "crohme",
                            "tep": n, "tap": "train" if v < 70 else "validation" if v < 85 else "test"})
    return mau


def luu(mau, tep=TEP):
    tep = Path(tep)
    tep.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tep, anh=np.stack([m["anh"] for m in mau]), nhan=np.array([m["nhan"] for m in mau]),
                        nguoi_viet=np.array([m["nguoi_viet"] for m in mau]), nguon=np.array([m["nguon"] for m in mau]),
                        tap=np.array([m["tap"] for m in mau]), tep=np.array([m["tep"] for m in mau]))
    thong_ke = {}
    for m in mau:
        k = thong_ke.setdefault(m["nguon"], {}).setdefault(m["nhan"], {"train": 0, "validation": 0, "test": 0})
        k[m["tap"]] += 1
    ghi_json(tep.parent / "thong_ke.json", {"nguon": NGUON, "so_mau": thong_ke,
                                             "chia_tap": {"hasy": f"theo người viết, gần 70/15/15 cho từng lớp, tối đa {TOI_DA_MOI_NGUOI} mẫu/người/lớp, hạt {HAT_CHIA}",
                                                          "crohme": "khử trùng lặp nét, chia theo băm SHA-1 của mã tờ viết gốc (UI) 70/15/15"},
                                             "ghi_chu": "Ảnh thô 64×64 nét sáng; chuẩn hóa bằng tien_xu_ly.chuan_hoa khi tạo bộ huấn luyện"})
    return thong_ke


def tai_ky_hieu(tep=TEP):
    """Trả về dict mảng: anh (N,64,64) uint8, nhan, nguoi_viet, nguon, tap, tep."""
    with np.load(tep, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}

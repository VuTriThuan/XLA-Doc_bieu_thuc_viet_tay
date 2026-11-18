"""Đọc cả trang: tách ký tự (tien_xu_ly.trang) → tách chữ dính nhờ CNN → nhận dạng → dãy số / biểu thức.

Tách chữ dính theo kiểu "cắt thừa rồi để bộ nhận dạng chọn": với một nhóm rộng bất thường, lấy các cột cắt ứng viên
(cực tiểu của chiếu dọc), xét mọi cách ghép các đoạn giữa hai vị trí cắt, chấm mỗi đoạn bằng log xác suất cao nhất của
CNN rồi dùng quy hoạch động tìm cách chia có tổng điểm lớn nhất.
- Nhóm rộng hơn RONG_BAT_BUOC_TACH × chiều cao chữ gần như không thể là một ký tự (− và = thấp nên không bị xét) nên
  BẮT BUỘC chia. Không thể để CNN tự quyết vì CNN chỉ biết 15 lớp và vẫn trả điểm cao cho một khối hai ba chữ số dính
  nhau (vd. "000" viết liền bị gọi là "=" với điểm 0,95).
- Nhóm hơi rộng (1,1–1,6 lần) được chia nếu tổng log-điểm các đoạn lớn hơn giữ nguyên cộng BIEN_TACH. Ngưỡng 1,3 cắt
  nhầm các chữ số đơn lẻ viết rộng (số 2, số 6); ngưỡng cao hơn bỏ sót chữ dính — xem thực nghiệm.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from tien_xu_ly.chuan_hoa import chuan_hoa_ky_tu
from tien_xu_ly.lien_thong import gan_nhan
from tien_xu_ly.trang import cac_duong_cat, manh_giua, tach_trang, vi_tri_cat_ung_vien
from .bieu_thuc import phan_tich_dong
from .cau_hinh import CHU_SO, LOP, LOP_KY_TU
from .du_doan import PhanLoaiKyTu

# Hai ngưỡng chọn bằng quét trên CVL validation + trang tổng hợp (ket_qua/quet_nguong_tach_dinh*.csv), không dùng CVL test.
BIEN_TACH = 0.0            # nat: nhóm rộng 1,1–1,6 lần chiều cao chữ được chia nếu tổng log-điểm các đoạn lớn hơn giữ nguyên
RONG_BAT_BUOC_TACH = 1.6
BIEN_GHEP = 0.0            # nat: ghép hai nhóm kề nhau nếu log-điểm ký tự ghép > tổng log-điểm hai ký tự rời
RONG_TOI_DA_DOAN = 1.4     # một đoạn sau khi cắt không rộng quá 1,4 lần chiều cao chữ
NGUONG_NGHI_NGO = 0.6


@dataclass
class KyTu:
    ky_tu: str
    do_tin_cay: float
    top3: list
    hop_bao: tuple                 # (trái, trên, phải, dưới) theo pixel ảnh gốc, sau khi chỉnh nghiêng
    anh: np.ndarray                # 28×28 đưa vào CNN
    tu_tach_dinh: bool = False     # được tạo ra khi tách một nhóm chữ dính
    khoang_trang_truoc: bool = False

    def to_dict(self):
        return {"ky_tu": self.ky_tu, "do_tin_cay": round(self.do_tin_cay, 4), "top3": self.top3,
                "hop_bao": list(self.hop_bao), "tu_tach_dinh": self.tu_tach_dinh,
                "nghi_ngo": self.do_tin_cay < NGUONG_NGHI_NGO}


@dataclass
class DongDoc:
    ky_tu: list
    hop_bao: tuple
    ket_qua: object                # bieu_thuc.KetQuaDong

    def to_dict(self):
        return {"chuoi": self.ket_qua.chuoi, "hop_bao": list(self.hop_bao), **self.ket_qua.to_dict(),
                "ky_tu": [k.to_dict() for k in self.ky_tu]}


@dataclass
class KetQuaTrang:
    dong: list
    cac_buoc: dict = field(default_factory=dict)
    ms: float = 0.0
    goc_nghieng: float = 0.0
    rgb: np.ndarray = None           # ảnh màu cỡ làm việc, đã chỉnh nghiêng
    ti_le: float = 1.0               # cỡ làm việc / cỡ gốc

    def to_dict(self):
        return {"dong": [d.to_dict() for d in self.dong], "goc_nghieng": self.goc_nghieng, "ms": self.ms}

    def cat_dong(self, le=0.35):
        """Ảnh RGB riêng của từng dòng (mỗi phép tính), cắt theo hộp bao của dòng, nới thêm `le` × chiều cao dòng."""
        h, w = self.rgb.shape[:2]
        kq = []
        for d in self.dong:
            trai, tren, phai, duoi = (v * self.ti_le for v in d.hop_bao)
            n = le * (duoi - tren)
            kq.append(self.rgb[max(0, int(tren - n)):min(h, int(duoi + n) + 1),
                               max(0, int(trai - n)):min(w, int(phai + n) + 1)].copy())
        return kq


def lam_sach_doan(m):
    """Bỏ mẩu nét vụn (< 8% diện tích) sinh ra khi cắt ngang nét của ký tự bên cạnh."""
    if not m.any():
        return m
    nhan, tp = gan_nhan(m)
    tong = m.sum()
    giu = [t.nhan for t in tp if t.dien_tich >= 0.08 * tong]
    return np.isin(nhan, giu)


class DocTrang:
    def __init__(self, phan_loai=None):
        self.phan_loai = phan_loai or PhanLoaiKyTu()

    def doc(self, anh, phuong_phap="nen_otsu", xoa_dong_ke=True, dong_hinh_thai=True, tach_dinh=True,
            canh_toi_da=1600, chuan_net=True, chinh_nghieng=True, chi_chu_so=False, rong_bat_buoc_tach=None,
            bien_tach=None, tach_cot=True, cat_cong=True, duong_vien=False, gop_manh=True, bo_gach_xoa=True,
            ghep_cnn=True):
        """chi_chu_so=True: trang chỉ có dãy số, CNN chỉ được chọn trong 10 lớp chữ số (bỏ 5 lớp phép toán).
        Các cờ còn lại (mặc định bật) cho phép tắt từng bước để đo đóng góp của nó."""
        self._ghep_cnn = ghep_cnn
        bat_dau = time.perf_counter()
        tt = tach_trang(anh, phuong_phap, canh_toi_da, xoa_dong_ke, dong_hinh_thai, chinh_nghieng, tach_cot, gop_manh,
                        bo_gach_xoa)
        ket_qua = []
        for d in tt.dong:
            ky_tu = self._doc_dong(tt, d, tach_dinh, chuan_net, chi_chu_so, rong_bat_buoc_tach or RONG_BAT_BUOC_TACH,
                                   BIEN_TACH if bien_tach is None else bien_tach, cat_cong, duong_vien)
            if not any(k.ky_tu.isdigit() for k in ky_tu):      # dòng chỉ có gạch: vết dòng kẻ, mép giấy
                continue
            kq = phan_tich_dong([k.ky_tu for k in ky_tu], [k.khoang_trang_truoc for k in ky_tu])
            hop = (min(k.hop_bao[0] for k in ky_tu), min(k.hop_bao[1] for k in ky_tu),
                   max(k.hop_bao[2] for k in ky_tu), max(k.hop_bao[3] for k in ky_tu))
            ket_qua.append(DongDoc(ky_tu, hop, kq))
        cac_buoc = dict(tt.cac_buoc)
        cac_buoc["ket_qua"] = ve_ket_qua(tt.rgb, tt.ti_le, ket_qua)
        return KetQuaTrang(ket_qua, cac_buoc, round((time.perf_counter() - bat_dau) * 1000, 1), tt.goc_nghieng,
                           tt.rgb, tt.ti_le)

    # -------------------------------------------------------------------------------------------------
    def _xac_suat(self, anh, chi_chu_so):
        """Softmax 16 lớp; chế độ chỉ chữ số bỏ 5 lớp phép toán rồi chuẩn hóa lại (GIỮ lớp "rác" trong mẫu số, để một
        đoạn rác vẫn có điểm thấp)."""
        p = self.phan_loai.xac_suat(np.stack(anh))
        if chi_chu_so:
            p[:, len(CHU_SO):len(LOP_KY_TU)] = 0
            p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
        return p

    @staticmethod
    def _log_diem(p):
        """Điểm của một đoạn = log xác suất của KÝ TỰ THẬT tốt nhất (không tính lớp rác)."""
        return np.log(np.clip(p[:, :len(LOP_KY_TU)].max(1), 1e-9, 1))

    def _diem(self, anh, chi_chu_so):
        return self._log_diem(self._xac_suat(anh, chi_chu_so))

    def _ghep_bang_cnn(self, tt, nhom, h, chuan_net, chi_chu_so):
        """Ngược với tách chữ dính: hai nhóm kề nhau (khe < 0,1 chiều cao chữ) mà ghép lại vẫn hẹp (< 0,9 chiều cao
        chữ) và đều cao hơn nửa chiều cao chữ được ghép nếu CNN chấm ký tự ghép cao hơn tổng điểm hai ký tự rời
        (vd. số 4 viết hai nét rời: rời thì thành "6" và "1", ghép thì thành "4" với điểm cao hơn)."""
        ung_vien = [i for i in range(len(nhom) - 1)
                    if nhom[i + 1].trai - nhom[i].phai < 0.1 * h
                    and max(nhom[i].phai, nhom[i + 1].phai) - min(nhom[i].trai, nhom[i + 1].trai) < 0.9 * h
                    and min(nhom[i].cao, nhom[i + 1].cao) >= 0.5 * h]
        if not ung_vien:
            return nhom
        anh = []
        for i in ung_vien:
            for g in (nhom[i], nhom[i + 1], nhom[i].hop(nhom[i + 1])):
                anh.append(chuan_hoa_ky_tu(tt.mat_na_nhom(g), chuan_net=chuan_net))
        diem = self._diem(anh, chi_chu_so).reshape(-1, 3)
        kq, bo_qua = list(nhom), set()
        for i, a, b, ghep in sorted(zip(ung_vien, *diem.T), key=lambda x: -(x[3] - x[1] - x[2])):
            if ghep > a + b + BIEN_GHEP and i not in bo_qua and i + 1 not in bo_qua:
                kq[i] = nhom[i].hop(nhom[i + 1])
                kq[i + 1] = None
                bo_qua |= {i - 1, i, i + 1, i + 2}
        return [g for g in kq if g is not None]

    def _doc_dong(self, tt, d, tach_dinh, chuan_net, chi_chu_so=False, rong_bat_buoc_tach=RONG_BAT_BUOC_TACH,
                  bien_tach=BIEN_TACH, cat_cong=True, duong_vien=False):
        """cat_cong: thêm đường cắt drop-fall bên cạnh đường thẳng; duong_vien: thêm chỗ cắt từ thung lũng đường viền.
        Mặc định chọn bằng ket_qua/quet_duong_cat.csv (90 dãy kiểm định, rồi 518 dãy nửa validation CVL): đường cắt cong
        tăng rõ, thung lũng đường viền không cộng thêm khi đã có đường cắt cong nên để tắt."""
        h = max(d.cao_chu, 1.0)
        anh_doan, mat_na_doan, chi_muc = [], [], {}

        def them(khoa, doan):
            if khoa in chi_muc:
                return chi_muc[khoa]
            doan = lam_sach_doan(doan)
            if not doan.any():
                chi_muc[khoa] = None
                return None
            anh_doan.append(chuan_hoa_ky_tu(doan, chuan_net=chuan_net))
            mat_na_doan.append(doan)
            chi_muc[khoa] = len(anh_doan) - 1
            return chi_muc[khoa]

        if getattr(self, "_ghep_cnn", True):
            d.nhom = self._ghep_bang_cnn(tt, d.nhom, h, chuan_net, chi_chu_so)
        d.khoang_trang = [False] + [d.nhom[i].trai - d.nhom[i - 1].phai > 0.6 * h for i in range(1, len(d.nhom))]
        ke_hoach = []
        for gi, g in enumerate(d.nhom):
            m = tt.mat_na_nhom(g)
            hh, w = m.shape
            # các "nút" là đường cắt (một cột x cho mỗi hàng), sắp theo vị trí trung bình; nút đầu/cuối là hai mép nhóm
            nut = [np.zeros(hh, int), np.full(hh, w)]
            # khóa riêng cho cả nhóm: (gi, i, j) chỉ dành cho mảnh giữa hai đường cắt i, j. Trước đây cả nhóm dùng
            # khóa (gi, 0, 1), trùng với mảnh đầu tiên khi có đường cắt, nên mảnh đầu bị thay bằng ảnh cả nhóm.
            them((gi, "ca"), m)
            if tach_dinh and g.cao >= 0.5 * h and w > 1.1 * h:
                giua = [dg for x in vi_tri_cat_ung_vien(m, h, duong_vien=duong_vien)
                        for dg in cac_duong_cat(m, x, cong=cat_cong)]
                nut = [nut[0], *sorted(giua, key=lambda dg: dg.mean()), nut[1]]
                tb = [dg.mean() for dg in nut]
                for i in range(len(nut)):
                    for j in range(i + 1, len(nut)):
                        if (i, j) == (0, len(nut) - 1):
                            continue
                        if 0.15 * h <= tb[j] - tb[i] <= RONG_TOI_DA_DOAN * h and (nut[j] >= nut[i]).all():
                            them((gi, i, j), manh_giua(m, nut[i], nut[j]))
            ke_hoach.append((g, m, nut))
        if not anh_doan:
            return []
        p = self._xac_suat(anh_doan, chi_chu_so)
        logp = self._log_diem(p)

        ky_tu = []
        for gi, (g, m, nut) in enumerate(ke_hoach):
            n = len(nut)
            nguyen = chi_muc.get((gi, "ca"))
            if nguyen is None:
                continue
            chon = [nguyen]
            if n > 2:
                # quy hoạch động trên các đường cắt: tot[j] = điểm tốt nhất để phủ phần bên trái đường j
                tot, truoc = [0.0] + [-np.inf] * (n - 1), [-1] * n
                for j in range(1, n):
                    for i in range(j):
                        k = chi_muc.get((gi, i, j)) if (i, j) != (0, n - 1) else None
                        if k is not None and tot[i] + logp[k] > tot[j]:
                            tot[j], truoc[j] = tot[i] + logp[k], i
                bat_buoc = m.shape[1] > rong_bat_buoc_tach * h
                if np.isfinite(tot[-1]) and (bat_buoc or tot[-1] > logp[nguyen] + bien_tach):
                    chon, j = [], n - 1
                    while j > 0:
                        chon.append(chi_muc[(gi, truoc[j], j)])
                        j = truoc[j]
                    chon.reverse()
            for si, k in enumerate(chon):
                thu_tu = np.argsort(-p[k, :len(LOP_KY_TU)])[:3]       # nhãn luôn là ký tự thật
                cot = np.flatnonzero(mat_na_doan[k].any(0))
                trai, phai = g.trai + int(cot[0]), g.trai + int(cot[-1]) + 1
                hop = tuple(int(round(v / tt.ti_le)) for v in (trai, g.tren, phai, g.duoi))
                ky_tu.append(KyTu(LOP[int(thu_tu[0])], float(p[k, thu_tu[0]]),
                                  [{"ky_tu": LOP[int(c)], "p": round(float(p[k, c]), 4)} for c in thu_tu],
                                  hop, anh_doan[k], len(chon) > 1,
                                  bool(d.khoang_trang[gi]) if si == 0 else False))
        return ky_tu


def phong_chu(co):
    """DejaVu Sans đi kèm matplotlib (có × ÷ ✓ ✗ và tiếng Việt); không có thì dùng phông mặc định của Pillow."""
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), co)
    except Exception:
        try:
            return ImageFont.load_default(size=co)
        except TypeError:
            return ImageFont.load_default()


def ve_ket_qua(rgb, ti_le, dong):
    """Ảnh cỡ làm việc (đã chỉnh nghiêng) với khung từng ký tự, nhãn và kết quả mỗi dòng."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    ve = ImageDraw.Draw(im)
    cao = [(k.hop_bao[3] - k.hop_bao[1]) * ti_le for d in dong for k in d.ky_tu]
    co = int(max(14, 0.45 * np.median(cao))) if cao else 14
    font = phong_chu(co)
    for d in dong:
        for k in d.ky_tu:
            hop = [v * ti_le for v in k.hop_bao]
            mau = (47, 91, 216) if k.do_tin_cay >= NGUONG_NGHI_NGO else (214, 120, 0)
            ve.rectangle(hop, outline=mau, width=max(1, co // 8))
            ve.text((hop[0], max(0, hop[1] - co - 2)), k.ky_tu, fill=mau, font=font)
        kq = d.ket_qua
        if kq.loai == "day_so":
            continue
        if kq.loai == "khong_hop_le":
            nhan, mau = "?", (190, 40, 40)
        elif kq.ve_phai is None:
            nhan, mau = ("" if kq.chuoi.endswith("=") else "= ") + kq.to_dict()["gia_tri"], (20, 130, 60)
        else:
            nhan, mau = ("✓", (20, 130, 60)) if kq.dung else (f"✗ {kq.to_dict()['gia_tri']}", (190, 40, 40))
        lon = phong_chu(int(co * 1.6))
        rong = ve.textlength(nhan, font=lon)
        x, y = d.hop_bao[2] * ti_le + co // 2, d.hop_bao[1] * ti_le
        if x + rong > im.width:                                 # không đủ chỗ bên phải: ghi dưới dòng
            x, y = d.hop_bao[0] * ti_le, d.hop_bao[3] * ti_le + 2
        ve.text((x, y), nhan, fill=mau, font=lon)
    return np.array(im)

"""Tách ký tự từ ảnh chụp một trang giấy: tách nền → tách dòng → nhóm thành ký tự.

    ảnh RGB ─ giới hạn kích thước ─ ảnh xám ─ chuẩn hóa nền + Otsu (hoặc Sauvola / Otsu thuần) ─ chỉnh nghiêng
          ─ xóa dòng kẻ ngang ─ đóng hình thái học
          ─ gán nhãn liên thông ─ bỏ nhiễu / viền giấy ─ chiếu ngang tìm dòng ─ ghép thành phần chồng nhau theo cột
          (dấu =, dấu ÷, nét đứt của 5, 7) ─ bỏ vết gạch xóa (nhiều lỗ kín) ─ danh sách ký tự ứng viên trái → phải

Tách chữ dính nhau cần bộ nhận dạng để chấm điểm các cách cắt, nên nằm ở loi/doc_trang.py; mô-đun này chỉ cung cấp
các vị trí cắt ứng viên (cực tiểu của chiếu dọc).
"""
from dataclasses import dataclass, field

import numpy as np

from ._kiem_tra import AnhDongNhat, LoiXuLyAnh
from .anh import chuyen_xam
from .hinh_thai import hinh_thai_nhi_phan
from .lien_thong import doan_chay, gan_nhan
from .nghieng import truot_doc, uoc_luong_goc
from .nguong import nguong_otsu
from .nguong_thich_nghi import chuan_hoa_nen, nguong_sauvola
from .thu_phong import gioi_han_kich_thuoc

PHUONG_PHAP_NGUONG = ("nen_otsu", "sauvola", "otsu")
LO_GACH_XOA = 6


@dataclass
class Nhom:
    """Một ký tự ứng viên: tập thành phần liên thông được ghép lại."""
    nhan: list                       # các nhãn thành phần trong ảnh nhãn
    trai: int
    tren: int
    phai: int
    duoi: int
    dien_tich: int

    @property
    def rong(self):
        return self.phai - self.trai

    @property
    def cao(self):
        return self.duoi - self.tren

    def hop(self, khac):
        return Nhom(self.nhan + khac.nhan, min(self.trai, khac.trai), min(self.tren, khac.tren),
                    max(self.phai, khac.phai), max(self.duoi, khac.duoi), self.dien_tich + khac.dien_tich)


@dataclass
class Dong:
    tren: int
    duoi: int
    nhom: list = field(default_factory=list)       # Nhom theo thứ tự trái → phải
    cao_chu: float = 0.0                            # chiều cao ký tự tham chiếu của dòng (px)
    khoang_trang: list = field(default_factory=list)   # khoang_trang[i] = True nếu trước nhom[i] có dấu cách
    cot: int = 0                                    # chỉ số cột (trái → phải)


@dataclass
class KetQuaTachTrang:
    xam: np.ndarray
    mat_na: np.ndarray               # mặt nạ nét cuối cùng (bool)
    anh_nhan: np.ndarray             # ảnh nhãn thành phần liên thông
    dong: list                       # các Dong từ trên xuống
    ti_le: float                     # kích thước ảnh làm việc / ảnh gốc
    cac_buoc: dict                   # tên bước -> ảnh uint8 để hiển thị
    cao_chu: float                   # chiều cao ký tự trung vị toàn trang
    goc_nghieng: float = 0.0         # độ, đã chỉnh bằng trượt dọc
    rgb: np.ndarray = None           # ảnh màu ở cỡ làm việc, đã chỉnh nghiêng (để vẽ kết quả)

    def mat_na_nhom(self, nhom, dem=0):
        """Mặt nạ chỉ gồm pixel của các thành phần trong nhóm (không lẫn nét của ký tự bên cạnh), cắt theo hộp bao."""
        t, d = max(0, nhom.tren - dem), min(self.anh_nhan.shape[0], nhom.duoi + dem)
        l, p = max(0, nhom.trai - dem), min(self.anh_nhan.shape[1], nhom.phai + dem)
        return np.isin(self.anh_nhan[t:d, l:p], nhom.nhan)


# ------------------------------------------------------------------------------------------------ tách nền
def xoa_dong_ke_ngang(mat_na, do_dai_toi_thieu):
    """Bỏ các đoạn chạy ngang dài >= do_dai_toi_thieu (dòng kẻ của vở).

    Giữ đoạn chạy dài >= L chính là phép MỞ với phần tử cấu trúc ngang 1×L; trừ kết quả đó khỏi mặt nạ sẽ xóa dòng kẻ
    mà không đụng tới nét chữ ngắn hơn L. Nét chữ cắt ngang dòng kẻ bị khuyết vài pixel, được phép đóng phía sau nối lại.
    """
    kq = mat_na.copy()
    so_doan = 0
    for y in range(mat_na.shape[0]):
        bd, kt = doan_chay(mat_na[y])
        for s, e in zip(bd, kt):
            if e - s >= do_dai_toi_thieu:
                kq[y, s:e] = False
                so_doan += 1
    return kq, so_doan


def uoc_luong_cao_chu(thanh_phan):
    """Trung vị chiều cao các thành phần 'đáng kể' (bỏ chấm nhỏ); 0 nếu không có."""
    if not thanh_phan:
        return 0.0
    cao = np.array([t.cao for t in thanh_phan], float)
    dt = np.array([t.dien_tich for t in thanh_phan], float)
    lon = cao[dt >= max(12, 0.15 * np.median(dt))]
    lon = lon[lon >= 0.4 * np.percentile(lon, 90)] if len(lon) else lon
    return float(np.median(lon)) if len(lon) else float(np.median(cao))


def tach_nen(anh, phuong_phap="nen_otsu", canh_toi_da=1600, xoa_dong_ke=True, dong_hinh_thai=True, chinh_nghieng=True):
    """Ảnh trang (RGB/xám) -> (xám, mặt nạ nét, các bước, tỉ lệ thu nhỏ, góc nghiêng đã chỉnh, ảnh RGB đã chỉnh)."""
    if phuong_phap not in PHUONG_PHAP_NGUONG:
        raise LoiXuLyAnh(f"phương pháp ngưỡng phải là một trong {PHUONG_PHAP_NGUONG}")
    anh = np.asarray(anh)
    h0 = anh.shape[0]
    if canh_toi_da:
        anh = gioi_han_kich_thuoc(anh, canh_toi_da)
    xam = chuyen_xam(anh)
    if int(xam.max()) - int(xam.min()) < 10:
        raise AnhDongNhat("Ảnh gần như một màu, không có nét để đọc")
    cac_buoc = {"xam": xam}
    if phuong_phap == "sauvola":
        mat_na, _ = nguong_sauvola(xam)
    elif phuong_phap == "otsu":
        mat_na = xam <= nguong_otsu(xam)
    else:                                             # chuẩn hóa nền rồi Otsu trên ảnh đã phẳng ánh sáng
        chuan, _ = chuan_hoa_nen(xam)
        cac_buoc["chuan_hoa_nen"] = chuan
        mat_na = chuan <= min(nguong_otsu(chuan), 235)
    cac_buoc["nguong"] = (~mat_na).astype(np.uint8) * 255
    goc = uoc_luong_goc(mat_na) if chinh_nghieng else 0.0
    rgb = anh if anh.ndim == 3 else np.stack([anh] * 3, 2)
    rgb = np.asarray(rgb[:, :, :3], dtype=np.uint8)
    if goc:
        mat_na = truot_doc(mat_na, goc, False)
        xam = truot_doc(xam, goc, 255)
        rgb = truot_doc(rgb, goc, 255)
    if xoa_dong_ke:
        mat_na, _ = xoa_dong_ke_ngang(mat_na, max(40, int(0.25 * mat_na.shape[1])))
    if dong_hinh_thai:
        mat_na = hinh_thai_nhi_phan(mat_na, "dong", 3)
    return xam, mat_na, cac_buoc, xam.shape[0] / h0, goc, rgb


def loc_thanh_phan(thanh_phan, hinh):
    """Bỏ mảng tối ở mép ảnh (bàn, bóng, mép giấy), vết quá lớn, và chấm nhiễu cực nhỏ.

    Mảng tối ở mép được nhận ra vì vừa chạm mép ảnh, vừa ĐẶC (diện tích > 50% hộp bao) và lớn hơn nhiều một nét chữ;
    chỉ xét chạm mép thì sẽ xóa nhầm chữ số khi ảnh được cắt sát (như ảnh của CVL Digit Strings).
    """
    H, W = hinh
    cao_chu = uoc_luong_cao_chu([t for t in thanh_phan if t.dien_tich >= 12])
    giu = []
    for t in thanh_phan:
        cham_mep = t.trai == 0 or t.tren == 0 or t.phai == W or t.duoi == H
        dac = t.dien_tich > 0.5 * t.rong * t.cao
        if cham_mep and dac and t.dien_tich > 0.5 * max(cao_chu, 8) ** 2:
            continue
        if t.dien_tich > 0.25 * H * W or (cao_chu and t.cao > 4 * cao_chu):
            continue
        if max(t.rong, t.cao) < max(3, 0.08 * cao_chu):
            continue
        giu.append(t)
    return giu, uoc_luong_cao_chu(giu)


# ------------------------------------------------------------------------------------------------ tách dòng
def chieu_ngang(mat_na):
    return mat_na.sum(axis=1)


def tim_dai_dong(chieu, cao_chu):
    """Các dải hàng [trên, dưới) có nét; dải sát nhau (khe < 0,15 chiều cao chữ) được nối, dải quá cao bị chia đôi
    tại cực tiểu của chiếu ngang (hai dòng viết sát nhau)."""
    co = chieu > 0
    bd, kt = doan_chay(co)
    dai = []
    for s, e in zip(bd, kt):
        if dai and s - dai[-1][1] < 0.15 * cao_chu:
            dai[-1][1] = e
        else:
            dai.append([int(s), int(e)])
    kq = []
    ngan_xep = list(reversed(dai))
    while ngan_xep:
        s, e = ngan_xep.pop()
        if cao_chu and e - s > 1.8 * cao_chu:
            k = max(1, int(cao_chu / 8))
            muot = np.convolve(chieu[s:e].astype(float), np.ones(k) / k, mode="same")
            a, b = int(0.35 * cao_chu), (e - s) - int(0.35 * cao_chu)
            if b > a:
                c = a + int(np.argmin(muot[a:b]))
                if muot[c] < 0.3 * muot.max():
                    ngan_xep += [[s + c, e], [s, s + c]]
                    continue
        kq.append((s, e))
    return kq


# ------------------------------------------------------------------------------------------------ tách cột
def tim_cot(mat_na, cao_chu, so_dong_toi_thieu=2):
    """Chia trang thành các cột tại dải dọc TRỐNG (chiếu dọc bằng 0) rộng >= 1 chiều cao chữ.

    Một dải trống chỉ được coi là ranh giới cột nếu mỗi bên đều có ít nhất `so_dong_toi_thieu` dòng chữ; nhờ vậy
    khoảng trắng lớn giữa hai số trên cùng một dòng (vd. "1355   79") không bị coi là hai cột. Trả về [(x0, x1), ...].
    """
    h, w = mat_na.shape
    chieu = mat_na.sum(axis=0)
    co_net = np.flatnonzero(chieu > 0)
    if not len(co_net) or not cao_chu:
        return [(0, w)]
    bd, kt = doan_chay(chieu == 0)
    ranh_gioi = []
    for s, e in zip(bd, kt):
        if s <= co_net[0] or e > co_net[-1] or e - s < cao_chu:
            continue                                        # dải trống ở lề trái/phải hoặc quá hẹp
        x = (s + e) // 2
        trai = len(tim_dai_dong(chieu_ngang(mat_na[:, :x]), cao_chu))
        phai = len(tim_dai_dong(chieu_ngang(mat_na[:, x:]), cao_chu))
        if trai >= so_dong_toi_thieu and phai >= so_dong_toi_thieu:
            ranh_gioi.append(x)
    bien = [0] + ranh_gioi + [w]
    return [(bien[i], bien[i + 1]) for i in range(len(bien) - 1)]


# ------------------------------------------------------------------------------------------------ ghép ký tự
def chong_cot(a, b):
    """Tỉ lệ chồng lấn theo trục x so với nhóm hẹp hơn."""
    giao = min(a.phai, b.phai) - max(a.trai, b.trai)
    return giao / max(1, min(a.rong, b.rong))


def chong_dong(a, b):
    """Tỉ lệ chồng lấn theo trục y so với nhóm thấp hơn."""
    giao = min(a.duoi, b.duoi) - max(a.tren, b.tren)
    return giao / max(1, min(a.cao, b.cao))


def nen_ghep(a, b, nguong=0.5):
    """Hai nhóm là các nét của CÙNG một ký tự khi chồng nhau theo cột và
    - nằm trên – dưới nhau (hai gạch của '=', chấm và gạch của '÷', gạch đầu số 5), hoặc
    - một nhóm rất dẹt so với nhóm kia (gạch ngang cắt thân số 7).
    Số 1 nghiêng chồng lên cột của chữ số bên cạnh có chiều cao tương đương nên KHÔNG bị ghép."""
    if chong_cot(a, b) < nguong:
        return False
    return chong_dong(a, b) < 0.5 or min(a.cao, b.cao) < 0.35 * max(a.cao, b.cao)


def ghep_theo_cot(nhom, nguong=0.5):
    """Ghép các nhóm là nét của cùng một ký tự (xem nen_ghep)."""
    nhom = sorted(nhom, key=lambda g: g.trai)
    doi = True
    while doi:
        doi = False
        for i in range(len(nhom)):
            for j in range(i + 1, len(nhom)):
                if nhom[j].trai >= nhom[i].phai:
                    break
                if nen_ghep(nhom[i], nhom[j], nguong):
                    nhom[i] = nhom[i].hop(nhom[j])
                    del nhom[j]
                    doi = True
                    break
            if doi:
                break
    return sorted(nhom, key=lambda g: g.trai)


def dem_lo(m, dien_tich_toi_thieu=4):
    """Số lỗ (vùng nền kín) trong mặt nạ một nhóm: 0 → 0, 8 → 2, vết gạch xóa → rất nhiều."""
    nen = np.pad(~m, 1, constant_values=True)
    _, tp = gan_nhan(nen)
    H, W = nen.shape
    return sum(1 for t in tp if t.trai > 0 and t.tren > 0 and t.phai < W and t.duoi < H
               and t.dien_tich >= dien_tich_toi_thieu)


def gop_manh_tren(nhom, cao_chu, tren, duoi):
    """Gộp mẩu nét nhỏ nằm ở phần trên của dòng (gạch đầu số 5, đuôi số 1, 7) vào nhóm gần nhất bên cạnh.

    Dấu − và = có kích thước tương tự nhưng nằm giữa dòng nên không bị gộp.
    """
    nhom = list(nhom)
    i = 0
    while i < len(nhom):
        g = nhom[i]
        if len(nhom) > 1 and g.cao < 0.4 * cao_chu and (g.tren + g.duoi) / 2 < tren + 0.3 * (duoi - tren):
            khe = [(max(nhom[j].trai - g.phai, g.trai - nhom[j].phai), j) for j in range(len(nhom))
                   if j != i and nhom[j].cao >= 0.5 * cao_chu]
            if khe and min(khe)[0] < 0.3 * cao_chu:
                j = min(khe)[1]
                nhom[j] = nhom[j].hop(g)
                del nhom[i]
                i = 0
                continue
        i += 1
    return sorted(nhom, key=lambda g: g.trai)


def cao_tham_chieu(nhom):
    """Chiều cao ký tự của một dòng: trung vị chiều cao các nhóm cao >= một nửa nhóm cao nhất (bỏ dấu -, =)."""
    if not nhom:
        return 0.0
    cao = np.array([g.cao for g in nhom], float)
    return float(np.median(cao[cao >= 0.5 * cao.max()]))


def tach_trang(anh, phuong_phap="nen_otsu", canh_toi_da=1600, xoa_dong_ke=True, dong_hinh_thai=True,
               chinh_nghieng=True, tach_cot=True, gop_manh=True, bo_gach_xoa=True):
    """Các cờ (mặc định bật) cho phép tắt từng bước để đo đóng góp của nó: tach_cot, gop_manh (gộp mẩu nét nhỏ ở
    mép trên dòng), bo_gach_xoa (bỏ nhóm nét nhiều lỗ kín)."""
    xam, mat_na, cac_buoc, ti_le, goc, rgb = tach_nen(anh, phuong_phap, canh_toi_da, xoa_dong_ke, dong_hinh_thai,
                                                      chinh_nghieng)
    anh_nhan, thanh_phan = gan_nhan(mat_na)
    thanh_phan, cao_chu = loc_thanh_phan(thanh_phan, mat_na.shape)
    giu = np.zeros(int(anh_nhan.max()) + 1, bool)
    for t in thanh_phan:
        giu[t.nhan] = True
    mat_na = giu[anh_nhan]
    anh_nhan = np.where(mat_na, anh_nhan, 0)
    cac_buoc["mat_na"] = (~mat_na).astype(np.uint8) * 255

    cot = tim_cot(mat_na, cao_chu) if tach_cot else [(0, mat_na.shape[1])]
    ket_qua = []
    for ci, (x0, x1) in enumerate(cot):
        tp_cot = [t for t in thanh_phan if x0 <= t.tam_x < x1]
        mat_na_cot = np.zeros_like(mat_na)
        mat_na_cot[:, x0:x1] = mat_na[:, x0:x1]
        dong = [Dong(s, e, cot=ci) for s, e in tim_dai_dong(chieu_ngang(mat_na_cot), cao_chu)]
        for t in tp_cot:                      # gán thành phần cho dải chứa tâm, nếu không thì dải gần nhất
            if not dong:
                break
            kc = [0 if d.tren <= t.tam_y < d.duoi else min(abs(t.tam_y - d.tren), abs(t.tam_y - d.duoi)) for d in dong]
            dong[int(np.argmin(kc))].nhom.append(Nhom([t.nhan], t.trai, t.tren, t.phai, t.duoi, t.dien_tich))
        for d in dong:
            if not d.nhom:
                continue
            d.nhom = ghep_theo_cot(d.nhom)
            d.cao_chu = cao_tham_chieu(d.nhom)
            if gop_manh:
                d.nhom = gop_manh_tren(d.nhom, d.cao_chu, min(g.tren for g in d.nhom), max(g.duoi for g in d.nhom))
            # bỏ chấm lẻ còn sót sau khi ghép (nhỏ hơn 1/4 chiều cao chữ theo cả hai chiều) và vết gạch xóa
            # (nét tô đi tô lại tạo nhiều lỗ kín; chữ số nhiều nhất 2 lỗ, số 8 tô đậm 3)
            d.nhom = [g for g in d.nhom if max(g.rong, g.cao) >= 0.25 * d.cao_chu
                      and (not bo_gach_xoa or dem_lo(np.isin(anh_nhan[g.tren:g.duoi, g.trai:g.phai], g.nhan)) < LO_GACH_XOA)]
            if not d.nhom:
                continue
            d.tren, d.duoi = min(g.tren for g in d.nhom), max(g.duoi for g in d.nhom)
            d.khoang_trang = [False] + [d.nhom[i].trai - d.nhom[i - 1].phai > 0.6 * d.cao_chu
                                        for i in range(1, len(d.nhom))]
            ket_qua.append(d)
    chieu = chieu_ngang(mat_na)
    cac_buoc["chieu_ngang"] = ve_chieu_ngang(mat_na, chieu, ket_qua, [x for x, _ in cot[1:]])
    return KetQuaTachTrang(xam, mat_na, anh_nhan, ket_qua, ti_le, cac_buoc, cao_chu, goc, rgb)


# ------------------------------------------------------------------------------------------------ cắt chữ dính
def _cuc_tieu_chieu_doc(mat_na, cao_chu):
    """(cột, điểm) tại cực tiểu địa phương của chiếu dọc đã làm mượt; điểm nhỏ = ít nét = chỗ cắt tốt."""
    chieu = np.convolve(mat_na.sum(axis=0).astype(float), np.ones(3) / 3, mode="same")
    w = len(chieu)
    bien = max(2, int(0.25 * cao_chu))
    return [(x, chieu[x] / max(1.0, chieu.max())) for x in range(bien, w - bien)
            if chieu[x] <= chieu[x - 1] and chieu[x] <= chieu[x + 1]]


def thung_lung_duong_vien(mat_na, cao_chu, do_sau_toi_thieu=0.15):
    """Chỗ cắt ứng viên từ ĐƯỜNG VIỀN: với mỗi cột, đường viền trên = hàng nét đầu tiên, đường viền dưới = hàng nét
    cuối cùng. Hai chữ số viết liền (vd. "00" nối ở giữa thân) không tạo cực tiểu rõ trên chiếu dọc, nhưng đường viền
    trên lõm sâu xuống (và đường viền dưới lồi lên) ngay chỗ nối. Trả về (cột, điểm) với điểm = −độ sâu / chiều cao chữ.
    """
    h, w = mat_na.shape
    co = mat_na.any(axis=0)
    tren = np.where(co, mat_na.argmax(axis=0), h).astype(float)
    duoi = np.where(co, h - 1 - mat_na[::-1].argmax(axis=0), -1).astype(float)
    bien = max(2, int(0.25 * cao_chu))
    r = max(2, int(0.3 * cao_chu))
    kq = []
    for x in range(bien, w - bien):
        trai, phai = slice(max(0, x - r), x), slice(x + 1, min(w, x + r + 1))
        # lõm từ trên: đường viền trên ở x thấp hơn (số hàng lớn hơn) cả hai bên
        sau_tren = min(tren[x] - tren[trai].min(), tren[x] - tren[phai].min())
        sau_duoi = min(duoi[trai].max() - duoi[x], duoi[phai].max() - duoi[x])
        sau = max(sau_tren, sau_duoi) / max(1.0, cao_chu)
        dinh_tren = sau_tren >= sau_duoi and tren[x] >= tren[x - 1] and tren[x] >= tren[x + 1]
        dinh_duoi = sau_duoi > sau_tren and duoi[x] <= duoi[x - 1] and duoi[x] <= duoi[x + 1]
        if sau >= do_sau_toi_thieu and (dinh_tren or dinh_duoi):
            kq.append((x, -sau))
    return kq


def vi_tri_cat_ung_vien(mat_na, cao_chu, toi_da=6, duong_vien=True):
    """Cột cắt ứng viên trong một nhóm rộng, cách mép và cách nhau ít nhất 0,25 chiều cao chữ.

    Nguồn: cực tiểu chiếu dọc (ưu tiên cột ít nét) và — nếu duong_vien — thung lũng của đường viền trên/dưới (ưu tiên
    thung lũng sâu). Lấy luân phiên từ hai nguồn để nguồn nào cũng có mặt.
    """
    bien = max(2, int(0.25 * cao_chu))
    nguon = [sorted(_cuc_tieu_chieu_doc(mat_na, cao_chu), key=lambda t: (t[1], t[0]))]
    if duong_vien:
        nguon.append(sorted(thung_lung_duong_vien(mat_na, cao_chu), key=lambda t: (t[1], t[0])))
    chon = []
    for i in range(max(len(n) for n in nguon) if nguon else 0):
        for n in nguon:
            if i < len(n) and all(abs(n[i][0] - c) >= bien for c in chon) and len(chon) < toi_da:
                chon.append(n[i][0])
    return sorted(chon)


def duong_roi(mat_na, x0, tu_duoi=False, uu_tien_trai=True, lech_toi_da=None):
    """Đường cắt "giọt nước rơi" (drop-fall): thả một giọt từ cột x0 ở mép trên (hoặc mép dưới) xuống qua nhóm nét.

    Ở mỗi hàng, giọt đi thẳng xuống nếu ô dưới là nền; nếu không thì đi chéo xuống (ưu tiên trái hoặc phải); nếu cả ba ô
    dưới đều là nét thì trượt ngang tối đa vài cột tìm khe nền; không có khe thì cắt xuyên nét. Kết quả là đường cắt
    men theo khe giữa hai chữ nghiêng hoặc chồng nhau, thay vì một đường thẳng đứng cắt ngang cả hai.
    Trả về mảng cột x cho từng hàng (pixel x < đường[y] thuộc phần bên trái).
    """
    m = mat_na[::-1] if tu_duoi else mat_na
    h, w = m.shape
    lech_toi_da = lech_toi_da or max(2, w // 4)
    huong = (-1, 1) if uu_tien_trai else (1, -1)
    x = int(x0)
    duong = np.empty(h, int)
    for y in range(h):
        duong[y] = x
        if y == h - 1:
            break
        if not m[y + 1, x]:
            continue
        di = None
        for dx in huong:
            nx = x + dx
            if 0 < nx < w and abs(nx - x0) <= lech_toi_da and not m[y + 1, nx]:
                di = nx
                break
        if di is None:
            for buoc in range(2, 4):
                for dx in huong:
                    nx = x + dx * buoc
                    if 0 < nx < w and abs(nx - x0) <= lech_toi_da and not m[y + 1, nx] and not m[y, x:nx:dx].any():
                        di = nx
                        break
                if di is not None:
                    break
        x = di if di is not None else x
    return duong[::-1].copy() if tu_duoi else duong


def cac_duong_cat(mat_na, x0, cong=True):
    """Các đường cắt khác nhau qua cột x0: thẳng đứng, và (nếu cong) drop-fall từ trên/dưới, ưu tiên trái/phải."""
    h = mat_na.shape[0]
    duong = [np.full(h, int(x0))]
    if cong:
        for tu_duoi in (False, True):
            for trai in (True, False):
                duong.append(duong_roi(mat_na, x0, tu_duoi, trai))
    kq = []
    for d in duong:
        if not any(np.array_equal(d, k) for k in kq):
            kq.append(d)
    return kq


def manh_giua(mat_na, duong_trai, duong_phai):
    """Phần nét nằm giữa hai đường cắt (x >= đường trái và x < đường phải trên từng hàng)."""
    cot = np.arange(mat_na.shape[1])[None, :]
    return mat_na & (cot >= duong_trai[:, None]) & (cot < duong_phai[:, None])


def ve_chieu_ngang(mat_na, chieu, dong, ranh_gioi_cot=()):
    """Ảnh minh họa: mặt nạ nét (trái) + biểu đồ chiếu ngang (phải), các dòng tô màu nhạt, ranh giới cột màu đỏ."""
    h, w = mat_na.shape
    rong_bd = max(60, w // 4)
    anh = np.full((h, w + rong_bd, 3), 255, np.uint8)
    for d in dong:
        x0 = min(g.trai for g in d.nhom)
        x1 = max(g.phai for g in d.nhom)
        anh[d.tren:d.duoi, x0:x1] = (225, 235, 255)
        anh[d.tren:d.duoi, w:] = (225, 235, 255)
    anh[:, :w][mat_na] = (30, 30, 30)
    for x in ranh_gioi_cot:
        anh[:, max(0, x - 1):x + 2] = (214, 60, 60)
    dai = (chieu / max(1, chieu.max()) * (rong_bd - 4)).astype(int)
    for y in range(h):
        anh[y, w:w + dai[y]] = (47, 91, 216)
    anh[:, w] = (150, 150, 150)
    return anh

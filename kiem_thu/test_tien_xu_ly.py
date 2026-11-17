"""Các bước xử lý ảnh tự cài."""
from collections import deque

import numpy as np
import pytest

from tien_xu_ly.chuan_hoa import DO_DAY_DICH, NOI_DUNG, chuan_hoa_ky_tu, uoc_luong_do_day
from tien_xu_ly.lien_thong import gan_nhan
from tien_xu_ly.nghieng import truot_doc, uoc_luong_goc
from tien_xu_ly.nguong_thich_nghi import anh_tich_phan, chuan_hoa_nen, loc_max, nguong_sauvola, trung_binh_cua_so
from tien_xu_ly.trang import chong_cot, ghep_theo_cot, Nhom, tach_trang, tim_dai_dong, xoa_dong_ke_ngang


def bfs(m):
    h, w = m.shape
    lab, n = np.zeros((h, w), int), 0
    for y, x in zip(*np.nonzero(m)):
        if lab[y, x]:
            continue
        n += 1
        q = deque([(y, x)])
        lab[y, x] = n
        while q:
            cy, cx = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = n
                        q.append((ny, nx))
    return lab, n


def test_gan_nhan_khop_bfs():
    rng = np.random.default_rng(1)
    for _ in range(200):
        m = rng.random((rng.integers(1, 25), rng.integers(1, 25))) < rng.uniform(.1, .6)
        a, tp = gan_nhan(m)
        b, n = bfs(m)
        assert len(tp) == n and len(set(zip(a[m], b[m]))) == n
        for t in tp:
            ys, xs = np.nonzero(a == t.nhan)
            assert (t.trai, t.tren, t.phai, t.duoi, t.dien_tich) == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, len(xs))


def test_gan_nhan_lien_thong_cheo():
    m = np.eye(5, dtype=bool)
    assert len(gan_nhan(m)[1]) == 1


def test_trung_binh_cua_so_khop_vong_lap():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 255, (9, 11)).astype(float)
    kq = trung_binh_cua_so(a, 5)
    for y in range(9):
        for x in range(11):
            assert kq[y, x] == pytest.approx(a[max(0, y - 2):y + 3, max(0, x - 2):x + 3].mean())
    assert anh_tich_phan(a)[-1, -1] == pytest.approx(a.sum())


def test_loc_max():
    a = np.zeros((7, 7))
    a[3, 3] = 5
    assert (loc_max(a, 3) == 5).sum() == 9


def anh_loang():
    """Giấy sáng dần từ trái sang phải (60 -> 230) có hai nét chữ tối hơn nền xung quanh 50 mức."""
    nen = np.tile(np.linspace(60, 230, 200), (80, 1))
    anh = nen.copy()
    anh[20:60, 30:34] -= 50
    anh[20:60, 160:164] -= 50
    return anh.astype(np.uint8)


def test_otsu_hong_khi_loang_nhung_chuan_hoa_nen_va_sauvola_dung():
    from tien_xu_ly.nguong import nguong_otsu
    anh = anh_loang()
    otsu = anh <= nguong_otsu(anh)
    assert otsu[:, :20].mean() > 0.9                 # Otsu coi cả vùng giấy tối là nét
    chuan, _ = chuan_hoa_nen(anh, 15)
    for m in (chuan <= 200, nguong_sauvola(anh, 15)[0]):
        assert m[20:60, 30:34].all() and m[20:60, 160:164].all()
        assert m[:, 60:140].sum() == 0


def test_xoa_dong_ke_giu_net_ngan():
    m = np.zeros((40, 300), bool)
    m[20, :] = True                     # dòng kẻ
    m[5:35, 100:104] = True             # nét chữ cắt ngang dòng kẻ
    m[10, 200:230] = True               # dấu trừ ngắn
    kq, so = xoa_dong_ke_ngang(m, 100)
    assert so == 1 and not kq[20, :100].any() and kq[10, 200:230].all() and kq[5:20, 100:104].all()


def test_uoc_luong_goc_va_truot():
    m = np.zeros((300, 400), bool)
    for y0 in (60, 140, 220):
        for x in range(40, 360):
            m[y0 + int(round((x - 200) * np.tan(np.radians(2)))), x] = True
    goc = uoc_luong_goc(m)
    assert abs(goc - 2) <= 0.25
    thang = truot_doc(m, goc)
    assert (thang.sum(1) > 0).sum() <= 9


def test_do_day_va_chuan_hoa_ky_tu():
    m = np.zeros((100, 100), bool)
    m[10:90, 48:52] = True
    assert uoc_luong_do_day(m) == pytest.approx(4, abs=0.6)
    x = chuan_hoa_ky_tu(m)
    assert x.shape == (28, 28) and x.dtype == np.float32 and x.max() == pytest.approx(1)
    ys, xs = np.nonzero(x > 0.5)
    assert ys.max() - ys.min() + 1 == NOI_DUNG                    # cạnh dài đúng 20 px
    assert xs.max() - xs.min() + 1 == pytest.approx(DO_DAY_DICH, abs=1.5)   # nét mảnh được làm dày
    # dấu trừ giữ tỉ lệ (không bị kéo thành ô vuông)
    tru = np.zeros((30, 120), bool)
    tru[12:18, 10:110] = True
    ys, xs = np.nonzero(chuan_hoa_ky_tu(tru) > 0.5)
    assert xs.max() - xs.min() + 1 == NOI_DUNG and ys.max() - ys.min() + 1 <= 4


def test_ghep_theo_cot_gom_dau_bang():
    tren, duoi, ben = Nhom([1], 10, 10, 40, 14, 120), Nhom([2], 12, 24, 41, 28, 116), Nhom([3], 60, 0, 80, 40, 200)
    kq = ghep_theo_cot([ben, duoi, tren])
    assert len(kq) == 2 and sorted(kq[0].nhan) == [1, 2]
    assert chong_cot(tren, ben) <= 0


def test_tim_dai_dong_tach_hai_dong_sat():
    chieu = np.zeros(200)
    chieu[20:60] = 30
    chieu[66:104] = 30
    chieu[60:66] = 1                   # hai dòng viết sát nhau, chỉ dính bởi vài nét
    (a, b), (c, d) = tim_dai_dong(chieu, 40)
    assert (a, d) == (20, 104) and b == c and 60 <= b <= 66      # cắt trong khe giữa hai dòng
    chieu[60:66] = 0
    assert tim_dai_dong(chieu, 40) == [(20, 60), (66, 104)]


def test_tach_trang_dem_dung_ky_tu():
    from conftest import ve_dong
    anh = np.concatenate([ve_dong("127+45="), np.full((40, ve_dong("127+45=").shape[1]), 235, np.uint8),
                          ve_dong("127+45=")])
    tt = tach_trang(anh)
    assert len(tt.dong) == 2
    assert [len(d.nhom) for d in tt.dong] == [7, 7]


def test_anh_mot_mau_bao_loi():
    from tien_xu_ly._kiem_tra import AnhDongNhat
    with pytest.raises(AnhDongNhat):
        tach_trang(np.full((50, 50), 200, np.uint8))


def test_tim_cot_hai_cot_va_khong_tach_khoang_trang_trong_dong():
    from tien_xu_ly.trang import tim_cot
    m = np.zeros((300, 400), bool)
    for y in (20, 100, 180):
        m[y:y + 40, 20:150] = True           # cột trái: 3 dòng
        m[y + 20:y + 60, 230:380] = True     # cột phải: 3 dòng, lệch hàng với cột trái
    assert tim_cot(m, 40) == [(0, 190), (190, 400)]
    mot_dong = np.zeros((60, 400), bool)
    mot_dong[10:50, 20:150] = True
    mot_dong[10:50, 230:380] = True          # khoảng trắng lớn trong MỘT dòng: không phải hai cột
    assert tim_cot(mot_dong, 40) == [(0, 400)]


def test_nen_ghep_khong_gop_so_1_nghieng_vao_so_ben_canh():
    from tien_xu_ly.trang import nen_ghep
    so_1, so_0 = Nhom([1], 100, 10, 112, 60, 300), Nhom([2], 106, 12, 140, 60, 600)
    gach_tren, gach_duoi = Nhom([3], 10, 20, 50, 25, 150), Nhom([4], 12, 40, 52, 45, 150)
    than_7, gach_7 = Nhom([5], 200, 10, 240, 60, 400), Nhom([6], 205, 33, 235, 37, 90)
    assert not nen_ghep(so_1, so_0)
    assert nen_ghep(gach_tren, gach_duoi) and nen_ghep(than_7, gach_7)


def test_dem_lo():
    from tien_xu_ly.trang import dem_lo
    o = np.zeros((30, 30), bool)
    o[5:25, 5:25] = True
    o[10:20, 10:20] = False
    assert dem_lo(o) == 1
    luoi = np.ones((40, 40), bool)
    for y in range(4, 36, 8):
        for x in range(4, 36, 8):
            luoi[y:y + 4, x:x + 4] = False
    assert dem_lo(luoi) == 16


def test_drop_fall_men_theo_khe_giua_hai_net_nghieng():
    from tien_xu_ly.trang import duong_roi, manh_giua
    net1, net2 = np.zeros((40, 60), bool), np.zeros((40, 60), bool)
    for y in range(40):
        net1[y, 25 - y // 2:28 - y // 2] = True     # hai nét nghiêng song song, chồng nhau theo cột
        net2[y, 45 - y // 2:48 - y // 2] = True
    m = net1 | net2
    d = duong_roi(m, 30)
    assert not m[np.arange(40), d].any()            # không cắt xuyên pixel nét nào
    assert m[:, 30].any()                            # còn đường thẳng đứng thì cắt xuyên
    assert (manh_giua(m, np.zeros(40, int), d) == net1).all()
    assert (manh_giua(m, d, np.full(40, 60)) == net2).all()


def test_thung_lung_duong_vien_tim_cho_noi():
    from tien_xu_ly.trang import thung_lung_duong_vien
    h, w = 40, 70
    yy, xx = np.mgrid[0:h, 0:w]
    hai_so_0 = (np.abs(np.hypot(yy - 20, xx - 18) - 15) < 2.5) | (np.abs(np.hypot(yy - 20, xx - 50) - 15) < 2.5)
    hai_so_0[18:23, 30:38] = True                    # nối dày ở giữa thân: chiếu dọc không còn cực tiểu rõ
    xs = [x for x, _ in thung_lung_duong_vien(hai_so_0, 40)]
    assert any(30 <= x <= 38 for x in xs)

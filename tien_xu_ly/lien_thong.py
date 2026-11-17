"""Thành phần liên thông (8 láng giềng).

`gan_nhan` dùng thuật toán hai lượt trên các ĐOẠN CHẠY (run) thay vì từng pixel: mỗi hàng được nén thành các đoạn
pixel nét liên tiếp; đoạn ở hàng y nối với đoạn ở hàng y−1 nếu chúng chồng nhau khi nới mỗi bên 1 cột (8-liên thông).
Các nhãn tương đương được gộp bằng cấu trúc hợp–tìm (union–find). Số đoạn chạy nhỏ hơn số pixel hàng chục lần nên
chạy nhanh trên ảnh trang giấy cỡ 1600×1200 bằng Python thuần.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from ._kiem_tra import ma_tran


@dataclass
class ThanhPhan:
    nhan: int
    trai: int
    tren: int
    phai: int        # không bao gồm
    duoi: int        # không bao gồm
    dien_tich: int

    @property
    def rong(self):
        return self.phai - self.trai

    @property
    def cao(self):
        return self.duoi - self.tren

    @property
    def tam_y(self):
        return (self.tren + self.duoi) / 2

    @property
    def tam_x(self):
        return (self.trai + self.phai) / 2


def doan_chay(hang):
    """Các đoạn True liên tiếp của một hàng bool: (mảng bắt đầu, mảng kết thúc không bao gồm)."""
    d = np.diff(np.concatenate(([0], hang.view(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def gan_nhan(mat_na):
    """Trả về (ảnh nhãn int32: 0 = nền, 1..n theo thứ tự gặp khi quét từ trên xuống; danh sách ThanhPhan)."""
    mat_na = ma_tran(mat_na).astype(bool)
    cao, rong = mat_na.shape
    cha = [0]

    def tim(a):
        while cha[a] != a:
            cha[a] = cha[cha[a]]
            a = cha[a]
        return a

    cac_doan = []                          # (y, bắt đầu, kết thúc, nhãn tạm)
    truoc = (np.empty(0, int), np.empty(0, int), [])
    for y in range(cao):
        bd, kt = doan_chay(mat_na[y])
        nhan_hang = []
        j = 0
        for s, e in zip(bd, kt):
            nhan = 0
            # đoạn trước [s0, e0) chạm đoạn này nếu s0 <= e và e0 >= s (nới 1 cột cho láng giềng chéo)
            while j < len(truoc[0]) and truoc[1][j] < s:
                j += 1
            k = j
            while k < len(truoc[0]) and truoc[0][k] <= e:
                n2 = tim(truoc[2][k])
                if nhan == 0:
                    nhan = n2
                elif n2 != nhan:
                    cha[max(n2, nhan)] = min(n2, nhan)
                    nhan = min(n2, nhan)
                k += 1
            if nhan == 0:
                nhan = len(cha)
                cha.append(nhan)
            nhan_hang.append(nhan)
            cac_doan.append((y, s, e, nhan))
        truoc = (bd, kt, nhan_hang)

    goc = {}
    anh_nhan = np.zeros((cao, rong), dtype=np.int32)
    thong_ke = {}
    for y, s, e, nhan in cac_doan:
        r = tim(nhan)
        n = goc.setdefault(r, len(goc) + 1)
        anh_nhan[y, s:e] = n
        t = thong_ke.get(n)
        if t is None:
            thong_ke[n] = [s, y, e, y + 1, e - s]
        else:
            t[0], t[2], t[3] = min(t[0], s), max(t[2], e), y + 1
            t[4] += e - s
    return anh_nhan, [ThanhPhan(n, *thong_ke[n]) for n in sorted(thong_ke)]


def thanh_phan_lien_thong_lon_nhat(mat_na):
    """Giữ thành phần nhiều pixel nhất (BFS 8 láng giềng; hòa thì giữ thành phần gặp trước khi quét từ trên xuống)."""
    mat_na = ma_tran(mat_na).astype(bool)
    cao, rong = mat_na.shape
    da_tham = np.zeros_like(mat_na, dtype=bool)
    tot_nhat = []
    for y, x in zip(*np.nonzero(mat_na)):
        if da_tham[y, x]:
            continue
        hang_doi = deque([(y, x)])
        da_tham[y, x] = True
        thanh_phan = []
        while hang_doi:
            cy, cx = hang_doi.popleft()
            thanh_phan.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < cao and 0 <= nx < rong and mat_na[ny, nx] and not da_tham[ny, nx]:
                        da_tham[ny, nx] = True
                        hang_doi.append((ny, nx))
        if len(thanh_phan) > len(tot_nhat):
            tot_nhat = thanh_phan
    kq = np.zeros_like(mat_na, dtype=bool)
    if tot_nhat:
        yy, xx = zip(*tot_nhat)
        kq[yy, xx] = True
    return kq

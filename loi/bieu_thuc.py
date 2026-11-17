"""Từ dãy ký tự đã nhận dạng của một dòng -> dãy số hoặc biểu thức và giá trị của nó.

Không dùng eval: tự phân tích cú pháp đệ quy xuống theo văn phạm
    biểu_thức := hạng (('+' | '-') hạng)*
    hạng      := thừa_số (('×' | '÷') thừa_số)*
    thừa_số   := '-' thừa_số | số
nên × và ÷ được ưu tiên trước + và −. Tính bằng phân số (fractions.Fraction) để 1 ÷ 3 × 3 ra đúng 1.
Một dòng có dấu '=' thì vế phải (nếu có) là đáp số viết tay và được so với kết quả tính được.
"""
from dataclasses import dataclass, field
from fractions import Fraction

PHEP = {"+", "-", "×", "÷"}


class LoiBieuThuc(ValueError):
    pass


@dataclass
class KetQuaDong:
    loai: str                          # day_so | bieu_thuc | khong_hop_le
    chuoi: str                         # ký tự đọc được, dấu cách nơi có khoảng trống lớn
    so: list = field(default_factory=list)      # day_so: các số
    gia_tri: Fraction = None            # bieu_thuc: giá trị vế trái
    ve_phai: str = None                 # đáp số viết sau dấu '=' (nếu có)
    dung: bool = None                   # đáp số viết có khớp giá trị tính được không
    loi: str = None

    def to_dict(self):
        return {"loai": self.loai, "chuoi": self.chuoi, "so": self.so,
                "gia_tri": dinh_dang(self.gia_tri) if self.gia_tri is not None else None,
                "phan_so": (f"{self.gia_tri.numerator}/{self.gia_tri.denominator}"
                            if self.gia_tri is not None and self.gia_tri.denominator != 1 else None),
                "ve_phai": self.ve_phai, "dung": self.dung, "loi": self.loi}


def dinh_dang(x, so_le=6):
    """Fraction -> chuỗi kiểu Việt Nam: số nguyên giữ nguyên, số lẻ dùng dấu phẩy, tối đa so_le chữ số thập phân."""
    if x.denominator == 1:
        return str(x.numerator)
    am = x < 0
    x = abs(x)
    nguyen, du = divmod(x.numerator, x.denominator)
    le = round(Fraction(du, x.denominator) * 10 ** so_le)
    if le == 10 ** so_le:
        nguyen, le = nguyen + 1, 0
    chuoi = str(nguyen) + ("," + str(le).rjust(so_le, "0").rstrip("0") if le else "")
    return ("-" if am else "") + chuoi + ("…" if Fraction(du, x.denominator) * 10 ** so_le != le else "")


def tach_tu_to(ky_tu, khoang_trang=None):
    """['1','2','+','4'] -> ['12', '+', '4']; chữ số liên tiếp ghép thành một số trừ khi có khoảng trắng chen giữa."""
    khoang_trang = khoang_trang or [False] * len(ky_tu)
    tu_to = []
    for c, cach in zip(ky_tu, khoang_trang):
        if c.isdigit() and tu_to and tu_to[-1].isdigit() and not cach:
            tu_to[-1] += c
        else:
            tu_to.append(c)
    return tu_to


class _BoPhanTich:
    def __init__(self, tu_to):
        self.t, self.i = tu_to, 0

    def xem(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def lay(self):
        self.i += 1
        return self.t[self.i - 1]

    def bieu_thuc(self):
        v = self.hang()
        while self.xem() in ("+", "-"):
            v = v + self.hang() if self.lay() == "+" else v - self.hang()
        return v

    def hang(self):
        v = self.thua_so()
        while self.xem() in ("×", "÷"):
            p = self.lay()
            r = self.thua_so()
            if p == "÷" and r == 0:
                raise LoiBieuThuc("Chia cho 0")
            v = v * r if p == "×" else v / r
        return v

    def thua_so(self):
        c = self.xem()
        if c == "-":
            self.lay()
            return -self.thua_so()
        if c is None:
            raise LoiBieuThuc("Biểu thức kết thúc giữa chừng (thiếu số sau phép toán)")
        if not c.isdigit():
            raise LoiBieuThuc(f"Cần một số nhưng gặp '{c}'")
        return Fraction(int(self.lay()))


def tinh(tu_to):
    """Giá trị của danh sách từ tố (số và phép toán, không có '=')."""
    if not tu_to:
        raise LoiBieuThuc("Vế trái rỗng")
    bo = _BoPhanTich(list(tu_to))
    v = bo.bieu_thuc()
    if bo.xem() is not None:
        con = bo.xem()
        raise LoiBieuThuc("Hai số đứng cạnh nhau mà không có phép toán" if con.isdigit()
                          else f"Thừa ký tự '{con}'")
    return v


def phan_tich_dong(ky_tu, khoang_trang=None):
    """ky_tu: danh sách ký tự (lớp của CNN) theo thứ tự trái -> phải."""
    ky_tu = list(ky_tu)
    khoang_trang = list(khoang_trang) if khoang_trang else [False] * len(ky_tu)
    chuoi = "".join((" " if cach and i else "") + c for i, (c, cach) in enumerate(zip(ky_tu, khoang_trang)))
    tu_to = tach_tu_to(ky_tu, khoang_trang)
    if not tu_to:
        return KetQuaDong("khong_hop_le", chuoi, loi="Dòng trống")
    if all(t.isdigit() for t in tu_to):
        return KetQuaDong("day_so", chuoi, so=[int(t) if len(t) < 300 else t for t in tu_to])
    tu_to = tach_tu_to(ky_tu)          # trong biểu thức, khoảng trống giữa hai chữ số chỉ là nét viết thưa
    try:
        if tu_to.count("=") > 1:
            raise LoiBieuThuc("Có nhiều hơn một dấu '='")
        if "=" in tu_to:
            k = tu_to.index("=")
            trai, phai = tu_to[:k], tu_to[k + 1:]
        else:
            trai, phai = tu_to, []
        gia_tri = tinh(trai)
    except LoiBieuThuc as loi:
        return KetQuaDong("khong_hop_le", chuoi, loi=str(loi))
    kq = KetQuaDong("bieu_thuc", chuoi, gia_tri=gia_tri)
    if phai:
        kq.ve_phai = "".join(phai)
        try:
            kq.dung = tinh(phai) == gia_tri
        except LoiBieuThuc:
            kq.dung = False
    return kq

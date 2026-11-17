"""Phân tích và tính biểu thức."""
from fractions import Fraction

import pytest

from loi.bieu_thuc import dinh_dang, phan_tich_dong, tach_tu_to, tinh, LoiBieuThuc


@pytest.mark.parametrize("chuoi,gia_tri", [("127+45", 172), ("2+3×4", 14), ("10-2-3", 5), ("8÷4÷2", 1),
                                            ("1÷3×3", 1), ("-5+2", -3), ("7÷2", Fraction(7, 2)), ("2×-3", -6)])
def test_tinh(chuoi, gia_tri):
    assert tinh(tach_tu_to(list(chuoi))) == gia_tri


@pytest.mark.parametrize("tu_to", [["5", "÷", "0"], ["1", "+"], ["+"], ["12", "3"], ["×", "4"], []])
def test_loi(tu_to):
    with pytest.raises(LoiBieuThuc):
        tinh(tu_to)


def test_dong_bieu_thuc_va_dap_so():
    kq = phan_tich_dong(list("127+45="))
    assert kq.loai == "bieu_thuc" and kq.gia_tri == 172 and kq.ve_phai is None
    assert phan_tich_dong(list("127+45=172")).dung is True
    assert phan_tich_dong(list("127+45=171")).dung is False
    assert phan_tich_dong(list("3=4=5")).loai == "khong_hop_le"


def test_day_so_theo_khoang_trang():
    kq = phan_tich_dong(list("12345"), [False, False, True, False, True])
    assert kq.loai == "day_so" and kq.so == [12, 34, 5] and kq.chuoi == "12 34 5"
    # trong biểu thức, khoảng trống giữa các chữ số chỉ là viết thưa
    assert phan_tich_dong(list("12+34"), [False, True, False, False, False]).gia_tri == 46


def test_dinh_dang():
    assert dinh_dang(Fraction(172)) == "172"
    assert dinh_dang(Fraction(7, 2)) == "3,5"
    assert dinh_dang(Fraction(-2, 3)) == "-0,666667…"

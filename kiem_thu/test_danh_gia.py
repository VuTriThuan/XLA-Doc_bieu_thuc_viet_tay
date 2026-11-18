from loi.danh_gia import khoang_cach_sua, wilson


def test_khoang_cach_sua():
    assert khoang_cach_sua("135579", "135579") == 0
    assert khoang_cach_sua("13579", "135579") == 1
    assert khoang_cach_sua("", "abc") == 3
    assert khoang_cach_sua("kitten", "sitting") == 3


def test_wilson():
    lo, hi = wilson(50, 58)
    assert 0.74 < lo < 0.76 and 0.92 < hi < 0.94


def test_so_trang_tach_bieu_thuc_khong_xac_dinh(tmp_path, monkeypatch):
    """4÷0 không có giá trị: không nằm trong mẫu số "đúng giá trị", được đếm riêng và đúng khi hệ thống báo lỗi."""
    import numpy as np
    import tien_xu_ly.anh
    from types import SimpleNamespace
    from loi.bieu_thuc import phan_tich_dong
    from loi.danh_gia import so_trang

    doc_duoc = ["13×4=", "4÷0=", "7+1="]
    monkeypatch.setattr(tien_xu_ly.anh, "doc_anh_tu_tep", lambda p: np.zeros((2, 2), np.uint8))

    class Doc:
        def doc(self, anh, **k):
            return SimpleNamespace(dong=[SimpleNamespace(ket_qua=phan_tich_dong(list(s))) for s in doc_duoc])

    kq = so_trang(Doc(), tmp_path, {"a.png": ["13×4=", "4÷0=", "7+2="]}, "thu", tmp_path)
    assert kq["so_bieu_thuc"] == 2 and kq["bieu_thuc_dung_gia_tri"] == 1        # 13×4 đúng, 7+1 sai
    assert kq["so_bieu_thuc_khong_xac_dinh"] == 1 and kq["khong_xac_dinh_bao_loi_dung"] == 1
    assert kq["dong_dung"] == 2

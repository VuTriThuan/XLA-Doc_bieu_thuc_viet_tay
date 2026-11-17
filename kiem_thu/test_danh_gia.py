from loi.danh_gia import khoang_cach_sua, wilson


def test_khoang_cach_sua():
    assert khoang_cach_sua("135579", "135579") == 0
    assert khoang_cach_sua("13579", "135579") == 1
    assert khoang_cach_sua("", "abc") == 3
    assert khoang_cach_sua("kitten", "sitting") == 3


def test_wilson():
    lo, hi = wilson(50, 58)
    assert 0.74 < lo < 0.76 and 0.92 < hi < 0.94

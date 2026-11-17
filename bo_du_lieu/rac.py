"""Sinh ảnh lớp "rác" (không phải một ký tự) để CNN biết từ chối các đoạn cắt sai khi tách chữ dính.

Hai loại, đều tạo từ ảnh chữ số ĐÃ chuẩn hóa của đúng tập tương ứng (train → rác train, validation → rác validation...):
- hai chữ số đặt sát/chồng lên nhau (chồng 0–35% bề rộng chữ hẹp hơn, lệch dọc ±2 px): khối hai chữ dính chưa tách;
- một mảnh của chữ số rộng (cắt dọc ở 35–65% bề rộng, giữ một bên): đoạn cắt quá tay. Bỏ mảnh hẹp (rộng < 35% cao)
  vì trông như số 1, và không cắt số 1.
Sau đó đi qua đúng tien_xu_ly.chuan_hoa.chuan_hoa_ky_tu như mọi ảnh khác.

Một số mảnh cắt trông giống hệt một chữ số thật (nửa số 8 như số 3, nửa số 6 như chữ "c"/số 6); gán nhãn "rác" cho
chúng sẽ dạy sai. Nếu truyền `bo_loc` (hàm trả xác suất các lớp), mảnh nào bị mô hình hiện có đọc là một chữ số với
xác suất > 0,8 sẽ bị bỏ.
"""
import numpy as np

from tien_xu_ly.chuan_hoa import cat_hop_bao, chuan_hoa_ky_tu


def _ghep_doi(a, b, rng):
    a, b = cat_hop_bao(a > 0.3).astype(float), cat_hop_bao(b > 0.3).astype(float)
    chong = int(rng.uniform(0, 0.35) * min(a.shape[1], b.shape[1]))
    lech = int(rng.integers(-2, 3))
    h = max(a.shape[0], b.shape[0]) + abs(lech) + 2
    w = a.shape[1] + b.shape[1] - chong
    c = np.zeros((h, w))
    ya = (h - a.shape[0]) // 2
    yb = min(max(0, (h - b.shape[0]) // 2 + lech), h - b.shape[0])
    c[ya:ya + a.shape[0], :a.shape[1]] = a
    c[yb:yb + b.shape[0], a.shape[1] - chong:] = np.maximum(c[yb:yb + b.shape[0], a.shape[1] - chong:], b)
    return c


def _manh(a, rng):
    m = cat_hop_bao(a > 0.3)
    h, w = m.shape
    if w < 0.55 * h:
        return None
    x = int(w * rng.uniform(0.35, 0.65))
    manh = m[:, :x] if rng.random() < 0.5 else m[:, x:]
    if not manh.any():
        return None
    manh = cat_hop_bao(manh)
    if manh.shape[1] < 0.35 * manh.shape[0] or manh.sum() < 0.2 * m.sum():
        return None
    return manh


def sinh_rac(x, y, so_luong, hat, bo_loc=None, ti_le_doi=0.6):
    """x: ảnh chữ số đã chuẩn hóa (N×28×28, [0,1]), y: nhãn 0–9. Trả về mảng so_luong×28×28 float32."""
    rng = np.random.default_rng(hat)
    doi, manh = [], []
    so_doi = int(so_luong * ti_le_doi)
    while len(doi) < so_doi:
        i, j = rng.integers(len(x), size=2)
        doi.append(chuan_hoa_ky_tu(_ghep_doi(x[i], x[j], rng)))
    while len(manh) < so_luong - so_doi:
        lo = []
        while len(lo) < 512:
            i = int(rng.integers(len(x)))
            if y[i] == 1:
                continue
            m = _manh(x[i], rng)
            if m is not None:
                lo.append(chuan_hoa_ky_tu(m))
        lo = np.stack(lo)
        if bo_loc is not None:
            lo = lo[bo_loc(lo)[:, :10].max(1) <= 0.8]
        manh += list(lo[:so_luong - so_doi - len(manh)])
    return np.stack(doi + manh).astype(np.float32)

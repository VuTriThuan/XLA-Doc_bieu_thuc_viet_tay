"""Toàn bộ luồng đọc trang với mô hình đã huấn luyện, máy chủ web và các lệnh chạy."""
import base64
import io
import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import GOC, ve_dong


def test_doc_mot_dong_va_tinh(bo_doc):
    kq = bo_doc.doc(ve_dong("127+45="))
    assert len(kq.dong) == 1
    d = kq.dong[0]
    assert d.ket_qua.chuoi == "127+45=" and d.ket_qua.gia_tri == 172
    assert all(k.anh.shape == (28, 28) for k in d.ky_tu)


def test_doc_nhieu_dong_tren_nen_loang(bo_doc):
    a, b = ve_dong("36×2="), ve_dong("905")
    rong = max(a.shape[1], b.shape[1])
    trang = np.full((a.shape[0] + b.shape[0] + 40, rong), 235, np.float64)
    trang[:a.shape[0], :a.shape[1]] = a
    trang[a.shape[0] + 40:, :b.shape[1]] = b
    trang *= np.linspace(0.55, 1.0, rong)[None, :]                    # ánh sáng tối dần về bên trái
    kq = bo_doc.doc(trang.astype(np.uint8))
    assert [d.ket_qua.chuoi for d in kq.dong] == ["36×2=", "905"]
    assert kq.dong[0].ket_qua.gia_tri == 72 and kq.dong[1].ket_qua.so == [905]


def test_tach_chu_dinh(bo_doc):
    dong = ve_dong("4791", khoang=-6)                                 # các chữ số chồng lên nhau vài pixel
    co = bo_doc.doc(dong)
    khong = bo_doc.doc(dong, tach_dinh=False)
    assert len("".join(d.ket_qua.chuoi for d in co.dong)) > len("".join(d.ket_qua.chuoi for d in khong.dong))


def test_tach_chu_dinh_moi_manh_chi_lay_phan_cua_no(bo_doc, monkeypatch):
    """Hai khối nối bằng một cầu mảnh: phải ra hai ký tự, khối trái và khối phải, hộp bao không chồng nhau.

    Thay CNN bằng bộ chấm cố định để chỉ kiểm tra cách tạo mảnh: mảnh hẹp (một khối) được điểm cao, mảnh rộng (cả
    nhóm) bị coi là rác. Lỗi cũ (cả nhóm và mảnh đầu dùng chung khóa) làm mảnh đầu là ảnh cả nhóm."""
    import loi.doc_trang as dt
    from loi.cau_hinh import LOP_KY_TU

    def danh_dau(doan, chuan_net=True):                     # ghi cột trái/phải của mảnh vào ảnh để bộ chấm đọc
        cot = np.flatnonzero(doan.any(0))
        a = np.zeros((28, 28), np.float32)
        a[0, 0], a[0, 1] = cot[0], cot[-1] + 1
        return a

    def cham(anh, chi_chu_so):
        p = np.full((len(anh), len(LOP_KY_TU) + 1), 1e-4)
        for i, a in enumerate(anh):
            if a[0, 1] - a[0, 0] <= 70:
                p[i, 0] = 0.99                              # một khối: chữ số "0" chắc chắn
            else:
                p[i, len(LOP_KY_TU)] = 0.99                 # cả nhóm: rác
        return p / p.sum(1, keepdims=True)

    monkeypatch.setattr(dt, "chuan_hoa_ky_tu", danh_dau)
    monkeypatch.setattr(bo_doc, "_xac_suat", cham)
    # đúng một chỗ cắt, giữa cầu nối: khi đó mảnh trái chính là đoạn (đường 0 → đường 1)
    monkeypatch.setattr(dt, "vi_tri_cat_ung_vien", lambda m, h, duong_vien=False: [m.shape[1] // 2])
    anh = np.full((150, 240), 235, np.uint8)
    anh[40:110, 40:90] = 30
    anh[40:110, 98:148] = 30
    anh[72:78, 90:98] = 30                                  # cầu nối hai khối
    ky_tu = [k for d in bo_doc.doc(anh, xoa_dong_ke=False).dong for k in d.ky_tu]
    assert len(ky_tu) == 2
    (t1, _, p1, _), (t2, _, p2, _) = ky_tu[0].hop_bao, ky_tu[1].hop_bao
    assert p1 <= t2 + 1, "hai mảnh chồng lên nhau"
    assert p1 - t1 <= 60 and p2 - t2 <= 60 and t1 < t2


def test_checkpoint_sai_phien_ban_bi_tu_choi(tmp_path):
    from loi.cau_hinh import tep_mo_hinh
    from loi.du_doan import nap_checkpoint
    tt = torch.load(tep_mo_hinh("co_so"), weights_only=True)
    tt["phien_ban_dau_vao"] = "khac"
    torch.save(tt, tmp_path / "x.pth")
    with pytest.raises(ValueError):
        nap_checkpoint(tmp_path / "x.pth")


def test_may_chu(bo_doc):
    from PIL import Image
    from giao_dien.may_chu import tao_may_chu
    may_chu = tao_may_chu(0, bo_doc)
    threading.Thread(target=may_chu.serve_forever, daemon=True).start()
    goc = f"http://127.0.0.1:{may_chu.server_address[1]}"
    try:
        assert b"<html" in urllib.request.urlopen(goc + "/").read()
        bo = io.BytesIO()
        Image.fromarray(ve_dong("8+5=")).save(bo, format="PNG")
        than = json.dumps({"anh": "data:image/png;base64," + base64.b64encode(bo.getvalue()).decode()}).encode()
        kq = json.loads(urllib.request.urlopen(urllib.request.Request(goc + "/api/doc-trang", than,
                                                                      {"Content-Type": "application/json"})).read())
        assert kq["dong"][0]["chuoi"] == "8+5=" and kq["dong"][0]["gia_tri"] == "13"
        assert kq["cac_buoc"] and kq["dong"][0]["ky_tu"][0]["anh"].startswith("data:image/png")
        # JSON hợp lệ nhưng sai kiểu (không phải object, thiếu/sai ảnh, tùy chọn không phải bool) đều trả 400,
        # và máy chủ vẫn phục vụ yêu cầu tiếp theo
        anh = json.loads(than)["anh"]
        for sai in (b'{"anh": 1}', b"[]", b"null", b"1", b'"chuoi"',
                    json.dumps({"anh": anh, "tach_dinh": "false"}).encode()):
            with pytest.raises(urllib.error.HTTPError) as loi:
                urllib.request.urlopen(urllib.request.Request(goc + "/api/doc-trang", sai,
                                                              {"Content-Type": "application/json"}))
            assert loi.value.code == 400, sai
        than_tat = json.dumps({"anh": anh, "dong_hinh_thai": False, "chinh_nghieng": False}).encode()
        kq = json.loads(urllib.request.urlopen(urllib.request.Request(goc + "/api/doc-trang", than_tat,
                                                                      {"Content-Type": "application/json"})).read())
        assert kq["dong"][0]["chuoi"] == "8+5="
    finally:
        may_chu.shutdown()
        may_chu.server_close()


@pytest.mark.parametrize("tep", ["huan_luyen.py", "doc_trang.py", "danh_gia.py", "ung_dung.py"])
def test_lenh_help(tep):
    r = subprocess.run([sys.executable, str(GOC / tep), "--help"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0 and "python3" in r.stdout


def test_lenh_doc_trang(tmp_path):
    from PIL import Image
    tep = tmp_path / "anh.png"
    Image.fromarray(ve_dong("127+45=")).save(tep)
    r = subprocess.run([sys.executable, str(GOC / "doc_trang.py"), str(tep), "--luu-buoc", str(tmp_path / "b")],
                       capture_output=True, text=True, encoding="utf-8", cwd=tmp_path)
    assert r.returncode == 0 and "127+45=" in r.stdout and "172" in r.stdout
    assert (tmp_path / "b" / "ket_qua.png").exists()


def test_hai_cot_va_cat_dong(bo_doc):
    a, b = ve_dong("12+7="), ve_dong("905")
    h = max(a.shape[0], b.shape[0])
    cot_trai = np.full((4 * h, a.shape[1]), 235, np.uint8)
    cot_phai = np.full((4 * h, b.shape[1]), 235, np.uint8)
    cot_trai[:a.shape[0]], cot_trai[2 * h:2 * h + a.shape[0]] = a, a
    cot_phai[h // 2:h // 2 + b.shape[0]], cot_phai[2 * h + h // 2:2 * h + h // 2 + b.shape[0]] = b, b
    trang = np.concatenate([cot_trai, np.full((4 * h, 120), 235, np.uint8), cot_phai], 1)
    kq = bo_doc.doc(trang)
    assert [d.ket_qua.chuoi for d in kq.dong] == ["12+7=", "12+7=", "905", "905"]
    anh = kq.cat_dong()
    assert len(anh) == 4 and all(x.ndim == 3 and x.shape[0] < trang.shape[0] for x in anh)

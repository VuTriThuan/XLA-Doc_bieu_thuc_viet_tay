"""Máy chủ web: nhận ảnh trang giấy (tải lên hoặc viết trên khung vẽ), trả về từng dòng, từng ký tự và kết quả.

Chỉ lắng nghe 127.0.0.1.
"""
import base64
import binascii
import io
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from loi.doc_trang import DocTrang
from tien_xu_ly._kiem_tra import LoiXuLyAnh
from tien_xu_ly.anh import doc_anh
from tien_xu_ly.trang import PHUONG_PHAP_NGUONG

GIAO_DIEN = Path(__file__).resolve().parent
TOI_DA_BODY = 25 * 1024 * 1024
DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,")
BUOC = (("xam", "Ảnh xám"), ("chuan_hoa_nen", "Chuẩn hóa nền"), ("nguong", "Phân ngưỡng"),
        ("mat_na", "Sau lọc nhiễu, chỉnh nghiêng, xóa dòng kẻ"), ("chieu_ngang", "Chiếu ngang → tách dòng"),
        ("ket_qua", "Kết quả"))


def png_data_url(a, canh_toi_da=None):
    from PIL import Image
    im = Image.fromarray(np.asarray(a, dtype=np.uint8))
    if canh_toi_da and max(im.size) > canh_toi_da:
        k = canh_toi_da / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BILINEAR)
    bo = io.BytesIO()
    im.save(bo, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bo.getvalue()).decode()


def tach_data_url(chuoi):
    if not isinstance(chuoi, str) or not DATA_URL.match(chuoi):
        raise ValueError("ảnh phải là data URL")
    try:
        return base64.b64decode(DATA_URL.sub("", chuoi, count=1), validate=True)
    except binascii.Error as e:
        raise ValueError("base64 không hợp lệ") from e


def dinh_dang(kq):
    dong = []
    for d in kq.dong:
        x = d.to_dict()
        for k, kt in zip(x["ky_tu"], d.ky_tu):
            k["anh"] = png_data_url(kt.anh * 255)
        dong.append(x)
    return {"dong": dong, "ms": kq.ms, "goc_nghieng": kq.goc_nghieng,
            "cac_buoc": [{"ten": ten, "png": png_data_url(kq.cac_buoc[k], 1400 if k == "ket_qua" else 900)}
                         for k, ten in BUOC if k in kq.cac_buoc]}


class XuLyYeuCau(BaseHTTPRequestHandler):
    bo_doc = None
    server_version = "DocTrang/1.0"
    khoa = threading.Lock()

    def _gui(self, ma, noi_dung, loai="application/json; charset=utf-8"):
        than = noi_dung if isinstance(noi_dung, bytes) else json.dumps(noi_dung, ensure_ascii=False).encode("utf-8")
        self.send_response(ma)
        self.send_header("Content-Type", loai)
        self.send_header("Content-Length", str(len(than)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(than)

    def do_HEAD(self):
        self.send_response(200 if urlsplit(self.path).path in ("/", "/index.html") else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        duong = urlsplit(self.path).path
        if duong in ("/", "/index.html"):
            self._gui(200, (GIAO_DIEN / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif duong == "/api/thong-tin":
            self._gui(200, {"bien_the": self.bo_doc.phan_loai.bien_the, "nguong": list(PHUONG_PHAP_NGUONG)})
        else:
            self._gui(404, {"loi": "Không tìm thấy"})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/doc-trang":
            return self._gui(404, {"loi": "Không tìm thấy"})
        try:
            dai = int(self.headers.get("Content-Length", "0"))
            if not 0 < dai <= TOI_DA_BODY:
                raise ValueError
            yc = json.loads(self.rfile.read(dai))
            nguong = yc.get("nguong", "nen_otsu")
            if nguong not in PHUONG_PHAP_NGUONG:
                raise ValueError
            tuy_chon = {"phuong_phap": nguong, "xoa_dong_ke": bool(yc.get("xoa_dong_ke", True)),
                        "tach_dinh": bool(yc.get("tach_dinh", True)), "chinh_nghieng": bool(yc.get("chinh_nghieng", True)),
                        "chi_chu_so": bool(yc.get("chi_chu_so", False))}
            du_lieu = tach_data_url(yc["anh"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return self._gui(400, {"loi": "Yêu cầu không hợp lệ (cần ảnh dạng data URL, tối đa 25 MB)"})
        try:
            rgb = doc_anh(du_lieu)
        except Exception:
            return self._gui(422, {"loi": "Không đọc được ảnh. Hãy dùng PNG hoặc JPEG."})
        try:
            with self.khoa:
                kq = self.bo_doc.doc(rgb, **tuy_chon)
        except LoiXuLyAnh as loi:
            return self._gui(422, {"loi": str(loi)})
        self._gui(200, dinh_dang(kq))

    def log_message(self, dinh_dang_log, *tham_so):
        sys.stderr.write("[ung_dung] " + (dinh_dang_log % tham_so) + "\n")


def tao_may_chu(cong=8781, bo_doc=None):
    XuLyYeuCau.bo_doc = bo_doc or DocTrang()
    return ThreadingHTTPServer(("127.0.0.1", cong), XuLyYeuCau)


def chay(cong=8781):
    may_chu = tao_may_chu(cong)
    print(f"Ứng dụng sẵn sàng: http://127.0.0.1:{may_chu.server_address[1]}  (Ctrl+C để dừng)", flush=True)
    try:
        may_chu.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        may_chu.server_close()

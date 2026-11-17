"""Cấu hình dùng chung: đường dẫn, tập lớp ký tự, tiện ích đọc/ghi.

Mọi đường dẫn tính từ vị trí của file này nên chạy được từ bất kỳ thư mục nào.
"""
import csv
import hashlib
import json
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
DU_LIEU = GOC / "du_lieu"
MO_HINH = GOC / "mo_hinh"
KET_QUA = GOC / "ket_qua"
BAO_CAO = GOC / "bao_cao"

KICH_THUOC_CNN = 28          # CNN nhận ảnh 28x28, một kênh, [0,1], nền đen, nét sáng
LE = 4                       # nội dung tối đa 20x20 trong canvas 28x28 (như MNIST)
# Đầu vào CNN được tạo bằng tien_xu_ly.chuan_hoa.chuan_hoa_ky_tu (cả lúc huấn luyện lẫn lúc đọc trang);
# checkpoint ghi phiên bản này và chỉ được nạp khi khớp.
PHIEN_BAN_DAU_VAO = "ky-tu-hop-bao-net-1"
BIEN_THE = ("co_so", "cai_tien", "co_cvl", "nho_rac", "lon_rac")

# Thứ tự lớp của mạng. Tên lớp là ký tự hiển thị luôn.
CHU_SO = tuple("0123456789")
PHEP_TOAN = ("+", "-", "×", "÷", "=")
LOP_KY_TU = CHU_SO + PHEP_TOAN    # 15 ký tự thật mà hệ thống đọc ra
RAC = "rác"                         # lớp thứ 16: "không phải một ký tự" (nửa chữ số, hai chữ số dính nhau...)
LOP = LOP_KY_TU + (RAC,)
SO_LOP = len(LOP)
TEN_LOP = {**{c: f"Chữ số {c}" for c in CHU_SO}, "+": "Cộng", "-": "Trừ", "×": "Nhân", "÷": "Chia", "=": "Bằng",
           RAC: "Không phải một ký tự"}


def tep_mo_hinh(bien_the, thu_muc=None):
    """mo_hinh/ky_tu_<bien_the>.pth; bien_the = co_so | cai_tien | co_cvl. Lựa chọn cuối ghi trong lua_chon.json."""
    return Path(thu_muc or MO_HINH) / f"ky_tu_{bien_the}.pth"


def tep_lua_chon(thu_muc=None):
    return Path(thu_muc or MO_HINH) / "lua_chon.json"


# ---------------------------------------------------------------------------------- đọc / ghi
def ghi_json(duong_dan, gia_tri):
    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    duong_dan.write_text(json.dumps(gia_tri, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def doc_json(duong_dan):
    return json.loads(Path(duong_dan).read_text(encoding="utf-8"))


def ghi_csv(duong_dan, dong, cot=None):
    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    dong = list(dong)
    with duong_dan.open("w", encoding="utf-8-sig", newline="") as f:
        viet = csv.DictWriter(f, fieldnames=cot or list(dong[0]))
        viet.writeheader()
        viet.writerows(dong)


def sha256(duong_dan):
    h = hashlib.sha256()
    with Path(duong_dan).open("rb") as f:
        for khoi in iter(lambda: f.read(1024 * 1024), b""):
            h.update(khoi)
    return h.hexdigest()

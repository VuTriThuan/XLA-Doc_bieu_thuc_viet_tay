"""Nạp checkpoint và phân loại ảnh ký tự 28×28 đã chuẩn hóa."""
import numpy as np
import torch

from .cau_hinh import KICH_THUOC_CNN, LOP, LOP_KY_TU, PHIEN_BAN_DAU_VAO, doc_json, sha256, tep_lua_chon, tep_mo_hinh
from .cnn import tao_mang


def nap_checkpoint(duong_dan):
    """Chỉ nạp trọng số và kiểu dữ liệu cơ bản (weights_only); kiểm tra thứ tự lớp và phiên bản đầu vào."""
    tt = torch.load(duong_dan, map_location="cpu", weights_only=True)
    if tt["lop"] not in (list(LOP), list(LOP_KY_TU)):     # mô hình cũ 15 lớp (không có lớp "rác") vẫn nạp được
        raise ValueError("Checkpoint có thứ tự lớp khác cấu hình hiện tại")
    if tt["phien_ban_dau_vao"] != PHIEN_BAN_DAU_VAO:
        raise ValueError("Checkpoint được huấn luyện với cách chuẩn hóa ký tự khác phiên bản hiện tại")
    mang = tao_mang(tt["kien_truc"])
    mang.load_state_dict(tt["state_dict"], strict=True)
    mang.eval()
    return mang, tt


def mo_hinh_duoc_chon():
    """Danh sách tệp checkpoint của lựa chọn cuối trong mo_hinh/lua_chon.json (một cấu hình hoặc một tổ hợp),
    sau khi kiểm tra SHA-256 khớp với lúc chọn."""
    lua_chon = doc_json(tep_lua_chon())
    tep = []
    for b in lua_chon["thanh_phan"]:
        t = tep_mo_hinh(b)
        if sha256(t) != lua_chon["ung_vien"][b]["sha256_checkpoint"]:
            raise ValueError(f"Checkpoint {b} đã thay đổi so với lúc chọn mô hình")
        tep.append(t)
    return tep


class PhanLoaiKyTu:
    """Phân loại ký tự 28×28. Nhiều checkpoint thì lấy TRUNG BÌNH softmax (ghép mô hình)."""

    def __init__(self, checkpoint=None, so_luong=2):
        torch.set_num_threads(so_luong)
        ds = checkpoint if isinstance(checkpoint, (list, tuple)) else [checkpoint] if checkpoint else mo_hinh_duoc_chon()
        nap = [nap_checkpoint(c) for c in ds]
        self.cac_mang = [m for m, _ in nap]
        self.bien_the = "+".join(t["bien_the"] for _, t in nap)

    def xac_suat(self, x, co_lo=512):
        """x: N×28×28 hoặc N×1×28×28 float trong [0,1] -> softmax N×16 (15 ký tự + "rác"; chưa hiệu chuẩn)."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[:, None]
        if x.ndim != 4 or x.shape[1:] != (1, KICH_THUOC_CNN, KICH_THUOC_CNN):
            raise ValueError("Cần ảnh N×28×28")
        kq = []
        with torch.inference_mode():
            for dau in range(0, len(x), co_lo):
                lo = torch.from_numpy(np.ascontiguousarray(x[dau:dau + co_lo]))
                p = [m(lo).softmax(1) for m in self.cac_mang]
                # mô hình 15 lớp: xác suất "rác" = 0
                p = [torch.nn.functional.pad(v, (0, len(LOP) - v.shape[1])) for v in p]
                kq.append(torch.stack(p).mean(0).numpy())
        return np.concatenate(kq) if kq else np.zeros((0, len(LOP)), np.float32)

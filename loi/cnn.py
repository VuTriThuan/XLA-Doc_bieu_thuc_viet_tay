"""Hai kiến trúc CNN.

nho (giữ nguyên từ BTL trước, 57 nghìn tham số):
    Conv(1->16, 3x3) -> ReLU -> MaxPool2 -> Conv(16->32, 3x3) -> ReLU -> MaxPool2
    -> Flatten(800) -> Dropout 0,25 -> Linear(800->64) -> ReLU -> Linear(64->so_lop)
lon (0,47 triệu tham số), hai khối "hai tích chập 3x3 có đệm + BatchNorm + ReLU, rồi MaxPool2":
    [Conv(1->32) Conv(32->32)] -> MaxPool2 -> [Conv(32->64) Conv(64->64)] -> MaxPool2
    -> Flatten(3136) -> Dropout 0,4 -> Linear(3136->128) -> ReLU -> Dropout 0,4 -> Linear(128->so_lop)
so_lop = 15 (mô hình cũ) hoặc 16 (thêm lớp "rác").
"""
from torch import nn

from .cau_hinh import KICH_THUOC_CNN


class MangCNN(nn.Module):
    def __init__(self, so_lop, batch_norm=False, dropout=0.25):
        super().__init__()
        lop = []
        for vao, ra in ((1, 16), (16, 32)):
            lop.append(nn.Conv2d(vao, ra, 3))
            if batch_norm:
                lop.append(nn.BatchNorm2d(ra))
            lop += [nn.ReLU(), nn.MaxPool2d(2)]
        self.dac_trung = nn.Sequential(*lop)
        self.phan_loai = nn.Sequential(nn.Flatten(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                                       nn.Linear(32 * 5 * 5, 64), nn.ReLU(), nn.Linear(64, so_lop))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, KICH_THUOC_CNN, KICH_THUOC_CNN):
            raise ValueError("Cần đầu vào N x 1 x 28 x 28")
        return self.phan_loai(self.dac_trung(x))


class MangCNNLon(nn.Module):
    def __init__(self, so_lop, dropout=0.4):
        super().__init__()

        def khoi(vao, ra):
            return [nn.Conv2d(vao, ra, 3, padding=1), nn.BatchNorm2d(ra), nn.ReLU(),
                    nn.Conv2d(ra, ra, 3, padding=1), nn.BatchNorm2d(ra), nn.ReLU(), nn.MaxPool2d(2)]
        self.dac_trung = nn.Sequential(*khoi(1, 32), *khoi(32, 64))
        self.phan_loai = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(128, so_lop))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, KICH_THUOC_CNN, KICH_THUOC_CNN):
            raise ValueError("Cần đầu vào N x 1 x 28 x 28")
        return self.phan_loai(self.dac_trung(x))


def tao_mang(kien_truc):
    """Dựng mạng từ dict kiến trúc lưu trong checkpoint: {loai: nho|lon, so_lop, batch_norm, dropout}."""
    if kien_truc.get("loai", "nho") == "lon":
        return MangCNNLon(kien_truc["so_lop"], kien_truc.get("dropout", 0.4))
    return MangCNN(kien_truc["so_lop"], kien_truc.get("batch_norm", False), kien_truc.get("dropout", 0.25))


def dem_tham_so(mang):
    return sum(p.numel() for p in mang.parameters() if p.requires_grad)

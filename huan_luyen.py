"""Chuẩn bị dữ liệu và huấn luyện CNN 15 lớp ký tự (chạy được từ bất kỳ thư mục nào).

    python3 huan_luyen.py --chuan-bi                      tạo du_lieu/ky_tu/ky_tu.npz từ MNIST + du_lieu/ky_hieu/ky_hieu.npz
    python3 huan_luyen.py --trich-ky-hieu HASY_DIR CROHME_ZIP   (chỉ khi muốn trích lại ký hiệu từ tệp gốc đã tải)
    python3 huan_luyen.py --sinh-trang                    sinh 60 trang tổng hợp có nhãn để đánh giá
    python3 huan_luyen.py                                 huấn luyện mọi cấu hình rồi chọn mô hình cuối
    python3 huan_luyen.py --chi-chon                      chỉ chọn lại cấu hình cuối từ các checkpoint đã có
    python3 huan_luyen.py --bien-the co_cvl               chỉ huấn luyện một cấu hình
    python3 huan_luyen.py --bien-the cai_tien --so-epoch 3

Kết quả: mo_hinh/ky_tu_{co_so,cai_tien,co_cvl}.pth, mo_hinh/lua_chon.json (cấu hình/tổ hợp được chọn), ket_qua/phan_loai/.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _luong in (sys.stdout, sys.stderr):       # Windows in bằng cp1252 khi chuyển hướng đầu ra, làm hỏng tiếng Việt
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8")

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chuan-bi", action="store_true", help="tạo bộ dữ liệu 15 lớp đã chuẩn hóa")
    p.add_argument("--trich-ky-hieu", nargs=2, metavar=("HASY_DIR", "CROHME_ZIP"),
                   help="trích lại ký hiệu từ HASYv2 đã giải nén và CROHME+.zip")
    p.add_argument("--sinh-trang", action="store_true", help="sinh trang tổng hợp để đánh giá")
    p.add_argument("--chi-chon", action="store_true", help="chỉ chọn lại cấu hình cuối")
    from loi.cau_hinh import BIEN_THE
    p.add_argument("--bien-the", choices=[*BIEN_THE, "tat_ca"], default="tat_ca")
    p.add_argument("--so-epoch", type=int)
    p.add_argument("--hat", type=int)
    p.add_argument("--so-luong", type=int, default=6, help="số luồng CPU")
    args = p.parse_args(argv)

    if args.trich_ky_hieu:
        from bo_du_lieu.ky_hieu import doc_crohme, doc_hasy, luu
        print(luu(doc_hasy(args.trich_ky_hieu[0]) + doc_crohme(args.trich_ky_hieu[1])))
        return 0
    if args.chuan_bi:
        from bo_du_lieu.tap_ky_tu import tao_tap_ky_tu
        tao_tap_ky_tu()
        return 0
    if args.sinh_trang:
        from bo_du_lieu.trang_tong_hop import sinh_bo_trang
        print(f"Đã sinh {len(sinh_bo_trang())} trang vào du_lieu/trang_tong_hop/")
        return 0

    from loi.huan_luyen import chon_mo_hinh, huan_luyen
    if args.chi_chon:
        print("Chọn:", chon_mo_hinh()["bien_the_duoc_chon"])
        return 0
    bien_the = BIEN_THE if args.bien_the == "tat_ca" else (args.bien_the,)
    for b in bien_the:
        huan_luyen(b, so_epoch=args.so_epoch, hat=args.hat, so_luong=args.so_luong)
    if args.bien_the == "tat_ca":
        print("Chọn:", chon_mo_hinh()["bien_the_duoc_chon"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

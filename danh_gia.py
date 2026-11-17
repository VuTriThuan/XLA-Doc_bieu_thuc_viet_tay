"""Đánh giá (chạy được từ bất kỳ thư mục nào). Kết quả ghi vào ket_qua/.

    python3 danh_gia.py                     CNN trên tập test + CVL Digit Strings + trang tổng hợp + ảnh chụp thật (nếu có)
    python3 danh_gia.py --phan-loai         chỉ CNN trên tập test ký tự (MNIST test, ký hiệu test)
    python3 danh_gia.py --cvl [--gioi-han 200]
    python3 danh_gia.py --trang             trang tổng hợp và ảnh chụp thật trong du_lieu/trang_that/
    python3 danh_gia.py --thu-nghiem        so sánh các lựa chọn tiền xử lý (ngưỡng, tách chữ dính, chuẩn hóa nét, dòng kẻ,
                                            chỉnh nghiêng) trên nửa VALIDATION của CVL và trên trang tổng hợp

CVL Digit Strings được chia theo người viết: nửa validation dùng để chọn mô hình và các lựa chọn tiền xử lý, nửa test chỉ
để báo cáo (lệnh mặc định và --cvl đọc nửa test).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8")

import argparse

THU_NGHIEM = {                                           # tên -> tùy chọn của DocTrang.doc
    "mac_dinh": {},
    "nguong_otsu": {"phuong_phap": "otsu"},
    "nguong_sauvola": {"phuong_phap": "sauvola"},
    "khong_chinh_nghieng": {"chinh_nghieng": False},
    "khong_xoa_dong_ke": {"xoa_dong_ke": False},
    "khong_dong_hinh_thai": {"dong_hinh_thai": False},
    "khong_tach_cot": {"tach_cot": False},
    "khong_bo_gach_xoa": {"bo_gach_xoa": False},
    "khong_gop_manh": {"gop_manh": False},
    "khong_ghep_cnn": {"ghep_cnn": False},
    "chi_cat_thang": {"cat_cong": False},
    "khong_tach_dinh": {"tach_dinh": False},
    "khong_chuan_net": {"chuan_net": False},
}


def in_kq(ten, kq, *khoa):
    print(f"{ten}: " + ", ".join(f"{k}={kq[k]}" for k in khoa), flush=True)


def chay_thu_nghiem(doc, dg, ten_chay, gioi_han=None):
    """Mỗi thí nghiệm đo trên: nửa validation CVL (518 dãy; trong đó 90 dãy của 9 người kiểm định), 60 trang tổng hợp
    và ảnh chụp thật của nhóm. Ghi ket_qua/thu_nghiem/<tên>.json rồi gộp mọi tệp có sẵn thành thu_nghiem_tien_xu_ly.csv."""
    from loi.cau_hinh import DU_LIEU, KET_QUA, doc_json, ghi_csv, ghi_json
    thu_muc = KET_QUA / "thu_nghiem"
    nhan_that = doc_json(DU_LIEU / "trang_that" / "nhan.json")
    for ten in ten_chay:
        tuy_chon = THU_NGHIEM[ten]
        cvl = dg.danh_gia_cvl(doc, "tn_" + ten, gioi_han=gioi_han, in_ra=False, tap="validation", chi_chu_so=True,
                              **tuy_chon)
        trang = dg.danh_gia_trang_tong_hop(doc, "tn_" + ten, **tuy_chon)
        that = dg.so_trang(doc, DU_LIEU / "trang_that", nhan_that, "tn_" + ten, thu_muc / "trang_that", **tuy_chon)
        dong = {"thi_nghiem": ten,
                "cvl_val_dung": cvl["dung_ca_day"], "cvl_val_so": cvl["so_anh"], "cvl_val_ti_le": cvl["ti_le_dung_ca_day"],
                "cvl_val_cer": cvl["cer"], "kd90_dung": cvl["kiem_dinh_90"]["dung_ca_day"],
                "kd90_cer": cvl["kiem_dinh_90"]["cer"],
                "trang_dong_dung": trang["dong_dung"], "trang_ti_le_dong": trang["ti_le_dong_dung"],
                "trang_bieu_thuc_dung": trang["ti_le_bieu_thuc_dung"], "trang_cer": trang["cer"],
                "that_dong_dung": that["dong_dung"], "that_bieu_thuc_dung": that["bieu_thuc_dung_gia_tri"],
                "that_cer": that["cer"]}
        ghi_json(thu_muc / f"{ten}.json", dong)
        print(dong, flush=True)
    bang = [doc_json(thu_muc / f"{t}.json") for t in THU_NGHIEM if (thu_muc / f"{t}.json").exists()]
    ghi_csv(KET_QUA / "thu_nghiem_tien_xu_ly.csv", bang)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phan-loai", action="store_true")
    p.add_argument("--cvl", action="store_true")
    p.add_argument("--trang", action="store_true")
    p.add_argument("--thu-nghiem", nargs="*", metavar="TEN", help="tắt/thay từng bước; không ghi tên = chạy tất cả")
    p.add_argument("--gioi-han", type=int, help="chỉ đọc N ảnh CVL đầu tiên (chạy thử nhanh)")
    args = p.parse_args(argv)
    tat_ca = not (args.phan_loai or args.cvl or args.trang or args.thu_nghiem is not None)

    from loi import danh_gia as dg
    from loi.doc_trang import DocTrang
    from loi.du_doan import PhanLoaiKyTu
    import os
    phan_loai = PhanLoaiKyTu(so_luong=int(os.environ.get("SO_LUONG", "6")))
    doc = DocTrang(phan_loai)
    if tat_ca or args.phan_loai:
        kq = dg.danh_gia_phan_loai(phan_loai)
        in_kq("CNN trên test", kq, "accuracy", "accuracy_can_bang")
        print("  chữ số MNIST:", kq["chu_so_mnist"]["accuracy"], " ký hiệu:", kq["ky_hieu"]["accuracy"],
              f"({kq['ky_hieu']['dung']}/{kq['ky_hieu']['so_mau']})")
    if tat_ca or args.cvl:
        from loi.cau_hinh import BIEN_THE, tep_mo_hinh
        for ten_mh in BIEN_THE:                                 # so sánh các cấu hình trên CVL test (chỉ để báo cáo)
            kq = dg.danh_gia_cvl(DocTrang(PhanLoaiKyTu(tep_mo_hinh(ten_mh), so_luong=6)), f"mo_hinh_{ten_mh}",
                                 gioi_han=args.gioi_han, in_ra=False, chi_chu_so=True)
            in_kq(f"CVL test, mô hình {ten_mh} (chỉ chữ số)", kq, "dung_ca_day", "so_anh", "ti_le_dung_ca_day", "cer")
        kq = dg.danh_gia_cvl(doc, "chi_chu_so", gioi_han=args.gioi_han, chi_chu_so=True)
        in_kq("CVL test, mô hình được chọn (chỉ chữ số)", kq, "dung_ca_day", "so_anh", "ti_le_dung_ca_day", "cer")
        kq = dg.danh_gia_cvl(doc, "mac_dinh", gioi_han=args.gioi_han)
        in_kq("CVL test, mô hình được chọn (đủ 15 lớp)", kq, "dung_ca_day", "so_anh", "ti_le_dung_ca_day", "cer")
    if tat_ca or args.trang:
        kq = dg.danh_gia_trang_tong_hop(doc)
        in_kq("Trang tổng hợp", kq, "ti_le_dong_dung", "ti_le_bieu_thuc_dung", "cer")
        tat_ca_that = dg.danh_gia_trang_that(doc)
        if not tat_ca_that:
            print("Ảnh chụp thật: chưa có (đặt ảnh và nhan.json vào du_lieu/trang_that/, xem README ở đó)")
        for ten, kq in tat_ca_that.items():
            in_kq(f"Ảnh chụp thật ({ten})", kq, "so_trang", "so_dong", "dong_dung", "ti_le_dong_dung",
                  "bieu_thuc_dung_gia_tri", "so_bieu_thuc", "cer")
    if args.thu_nghiem is not None:
        chay_thu_nghiem(doc, dg, args.thu_nghiem or list(THU_NGHIEM), args.gioi_han)
    return 0


if __name__ == "__main__":
    sys.exit(main())

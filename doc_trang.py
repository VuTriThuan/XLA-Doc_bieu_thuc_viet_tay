"""Đọc dãy số / biểu thức trên ảnh chụp một trang giấy và in kết quả (chạy được từ bất kỳ thư mục nào).

    python3 doc_trang.py anh.jpg                         in từng dòng: chuỗi đọc được và kết quả phép tính
    python3 doc_trang.py anh.jpg --json                  in JSON đầy đủ (từng ký tự, độ tin cậy, hộp bao)
    python3 doc_trang.py anh.jpg --luu-buoc cac_buoc/    lưu ảnh từng bước xử lý và ảnh kết quả có khung
    python3 doc_trang.py anh.jpg --cat-dong cac_dong/          cắt mỗi dòng (mỗi phép tính) thành một ảnh riêng
    python3 doc_trang.py anh.jpg --nguong sauvola --khong-tach-dinh
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _luong in (sys.stdout, sys.stderr):       # Windows in bằng cp1252 khi chuyển hướng đầu ra, làm hỏng tiếng Việt
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8")

import argparse
import json


def mo_ta_dong(d):
    kq = d.ket_qua
    nghi_ngo = [k.ky_tu for k in d.ky_tu if k.do_tin_cay < 0.6]
    them = f"   (ký tự chưa chắc: {' '.join(nghi_ngo)})" if nghi_ngo else ""
    if kq.loai == "day_so":
        return f"{kq.chuoi:<24} dãy số: {', '.join(map(str, kq.so))}{them}"
    if kq.loai == "khong_hop_le":
        return f"{kq.chuoi:<24} không tính được: {kq.loi}{them}"
    gt = kq.to_dict()
    gia_tri = gt["gia_tri"] + (f" (= {gt['phan_so']})" if gt["phan_so"] else "")
    if kq.ve_phai is None:
        return f"{kq.chuoi:<24} = {gia_tri}{them}"
    return f"{kq.chuoi:<24} kết quả đúng là {gia_tri} → đáp số viết {'ĐÚNG' if kq.dung else 'SAI'}{them}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("anh", type=Path)
    p.add_argument("--nguong", choices=["nen_otsu", "sauvola", "otsu"], default="nen_otsu",
                   help="cách tách nét khỏi nền (mặc định: chuẩn hóa nền rồi Otsu)")
    p.add_argument("--khong-xoa-dong-ke", action="store_true", help="tắt bước xóa dòng kẻ ngang của vở")
    p.add_argument("--khong-tach-dinh", action="store_true", help="tắt bước tách chữ dính nhau")
    p.add_argument("--chi-chu-so", action="store_true", help="trang chỉ có dãy số: không xét 5 lớp phép toán")
    p.add_argument("--json", action="store_true")
    p.add_argument("--luu-buoc", type=Path, help="thư mục lưu ảnh từng bước")
    p.add_argument("--cat-dong", type=Path, help="thư mục lưu ảnh riêng của từng dòng (dong_01.png, ...) + danh_sach.csv")
    args = p.parse_args(argv)

    from loi.doc_trang import DocTrang
    from tien_xu_ly._kiem_tra import LoiXuLyAnh
    from tien_xu_ly.anh import doc_anh_tu_tep
    try:
        kq = DocTrang().doc(doc_anh_tu_tep(args.anh), phuong_phap=args.nguong, xoa_dong_ke=not args.khong_xoa_dong_ke,
                            tach_dinh=not args.khong_tach_dinh, chi_chu_so=args.chi_chu_so)
    except (OSError, LoiXuLyAnh) as loi:
        print("Lỗi:", loi, file=sys.stderr)
        return 2
    if args.luu_buoc:
        from PIL import Image
        args.luu_buoc.mkdir(parents=True, exist_ok=True)
        for ten, a in kq.cac_buoc.items():
            Image.fromarray(a).save(args.luu_buoc / f"{ten}.png")
    if args.cat_dong:
        from PIL import Image
        from loi.cau_hinh import ghi_csv
        args.cat_dong.mkdir(parents=True, exist_ok=True)
        dong = []
        for i, (d, a) in enumerate(zip(kq.dong, kq.cat_dong()), 1):
            ten = f"dong_{i:02d}.png"
            Image.fromarray(a).save(args.cat_dong / ten)
            dong.append({"tep": ten, "may_doc": d.ket_qua.chuoi, "ket_qua": mo_ta_dong(d)[len(d.ket_qua.chuoi):].strip()})
        if dong:
            ghi_csv(args.cat_dong / "danh_sach.csv", dong)
        print(f"Đã cắt {len(dong)} dòng vào {args.cat_dong}", file=sys.stderr)
    if args.json:
        print(json.dumps(kq.to_dict(), ensure_ascii=False, indent=2))
    elif not kq.dong:
        print("Không tìm thấy dòng chữ nào.")
    else:
        for i, d in enumerate(kq.dong, 1):
            print(f"Dòng {i}: {mo_ta_dong(d)}")
        print(f"({kq.ms:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

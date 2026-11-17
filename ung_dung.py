"""Mở ứng dụng web đọc biểu thức viết tay (chạy được từ bất kỳ thư mục nào).

    python3 ung_dung.py                      http://127.0.0.1:8781
    python3 ung_dung.py --port 9000 --mo-trinh-duyet
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8")

import argparse
import webbrowser


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8781)
    p.add_argument("--mo-trinh-duyet", action="store_true", help="tự mở trình duyệt")
    args = p.parse_args(argv)
    from giao_dien.may_chu import chay
    if args.mo_trinh_duyet:
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    chay(args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

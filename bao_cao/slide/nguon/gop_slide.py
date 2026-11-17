"""Gộp nguon/index.html thành một file HTML tự chứa (CSS, JS, hình nhúng), mở offline."""
import base64, io, os, re, sys
from PIL import Image

nguon = sys.argv[1]          # thư mục nguon/
dich = sys.argv[2]           # file html đầu ra
html_path = os.path.join(nguon, "index.html")
html = open(html_path, encoding="utf-8").read()

def doc(rel):
    return open(os.path.normpath(os.path.join(nguon, rel)), encoding="utf-8").read()

def an_toan(js):
    # chuỗi "</script" hay "<!--" trong JS sẽ cắt ngang thẻ <script> nhúng
    return js.replace("</script", "<\\/script").replace("<!--", "<\\!--")

# 1. CSS: <link rel="stylesheet" ... href="X"> -> <style>
def thay_css(m):
    tag = m.group(0)
    href = re.search(r'href="([^"]+)"', tag).group(1)
    css = doc(href)
    ten = os.path.basename(href)
    return f"<style data-nguon=\"{ten}\">\n{css}\n</style>"
html, n_css = re.subn(r'<link rel="stylesheet"[^>]*>', thay_css, html)

# bỏ đổi theme (chỉ một theme, nằm sẵn trong file)
html = html.replace(' data-themes="corporate-clean"', '')

# 2. Hình -> data URI (ảnh lớn đổi sang JPEG cho nhẹ)
cache = {}
def data_uri(rel):
    if rel in cache:
        return cache[rel]
    p = os.path.join(nguon, rel)
    raw = open(p, "rb").read()
    mime = "image/png"
    if rel.endswith(".jpg") or rel.endswith(".jpeg"):
        mime = "image/jpeg"
    elif len(raw) > 400_000:
        im = Image.open(p).convert("RGB")
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=88, optimize=True)
        raw, mime = buf.getvalue(), "image/jpeg"
    uri = f"data:{mime};base64," + base64.b64encode(raw).decode()
    cache[rel] = uri
    return uri
html, n_img = re.subn(r'src="(hinh/[^"]+)"', lambda m: f'src="{data_uri(m.group(1))}"', html)

# 3. JS: runtime + fx (nhúng sẵn _util và neural-net, tắt bộ nạp động)
fx_dir = None
def thay_js(m):
    global fx_dir
    src = m.group(1)
    js = doc(src)
    if src.endswith("fx-runtime.js"):
        fx_dir = os.path.join(os.path.dirname(src), "fx")
        js, k = re.subn(r"const FX_LIST = \[[^\]]*\];", "const FX_LIST = [];", js, count=1)
        assert k == 1, "không tìm thấy FX_LIST"
        mods = "".join(f"<script data-nguon=\"{t}.js\">\n{an_toan(doc(os.path.join(fx_dir, t + '.js')))}\n</script>\n"
                       for t in ("_util", "neural-net"))
        return mods + f"<script data-nguon=\"fx-runtime.js\">\n{an_toan(js)}\n</script>"
    return f"<script data-nguon=\"{os.path.basename(src)}\">\n{an_toan(js)}\n</script>"
html, n_js = re.subn(r'<script src="([^"]+)"></script>', thay_js, html)

assert "../" not in re.sub(r"<(style|script)[^>]*>.*?</\1>", "", html, flags=re.S), "còn đường dẫn ngoài"
open(dich, "w", encoding="utf-8").write(html)
print(f"css={n_css} img={n_img} js={n_js} kich_thuoc={os.path.getsize(dich)/1e6:.2f} MB")

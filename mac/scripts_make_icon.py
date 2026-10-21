"""Resources/AppIcon.png · AppIcon.icns 만들기 (StemMixer · ClipStack과 같은 타일, 안에는 음표 막대 + 음높이 선).
사용: python mac/scripts_make_icon.py  (Pillow 필요)"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 4  # 크게 그려서 줄이기 (매끈하게)
W = 1024 * S
out = Path(__file__).parent / "Resources"


def squircle_mask(box, n=5.0):
    x0, y0, x1, y1 = box
    a, b = (x1 - x0) / 2, (y1 - y0) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    yy, xx = np.mgrid[0:W, 0:W]
    v = np.abs((xx - cx) / a) ** n + np.abs((yy - cy) / b) ** n
    return Image.fromarray(((v <= 1) * 255).astype(np.uint8))


def gradient(c0, c1, angle_box):
    yy, xx = np.mgrid[0:W, 0:W] / W
    t = np.clip((xx * 0.35 + yy * 0.65 - angle_box[0]) / (angle_box[1] - angle_box[0]), 0, 1)[..., None]
    rgb = np.array(c0) * (1 - t) + np.array(c1) * t
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
tile = (100 * S, 100 * S, 924 * S, 924 * S)
mask = squircle_mask(tile)
shadow = Image.new("RGBA", (W, W), (0, 0, 0, 0))
shadow.paste((60, 0, 20, 120), mask=mask.filter(ImageFilter.GaussianBlur(18 * S)).point(lambda v: v * 0.6))
img.alpha_composite(shadow, (0, 12 * S))
img.paste(gradient((255, 120, 140), (190, 20, 60), (0.1, 0.95)), mask=mask)

card = Image.new("RGBA", (W, W), (0, 0, 0, 0))
cbox = (252 * S, 250 * S, 772 * S, 774 * S)
d = ImageDraw.Draw(card)
d.rounded_rectangle(cbox, radius=96 * S, fill=(255, 255, 255, 255))
cs = Image.new("RGBA", (W, W), (0, 0, 0, 0))
ImageDraw.Draw(cs).rounded_rectangle(cbox, radius=96 * S, fill=(90, 0, 30, 110))
img.alpha_composite(cs.filter(ImageFilter.GaussianBlur(22 * S)), (0, 14 * S))
img.alpha_composite(card)

d = ImageDraw.Draw(img)
# 음표 막대 (피아노 롤)
bars = [(300, 520, 410, 0.0), (410, 560, 470, 0.0), (470, 610, 600, 0.0), (600, 660, 724, 0.0)]
ys = [600, 525, 445, 545]
xs = [(300, 410), (410, 520), (520, 620), (620, 724)]
for (x0, x1), y in zip(xs, ys):
    d.rounded_rectangle(((x0 + 6) * S, (y - 26) * S, (x1 - 6) * S, (y + 26) * S), radius=26 * S, fill=(255, 205, 214, 255))
# 음높이 선 (사람처럼: 미끄러짐 + 떨림)
pts = []
for x in np.linspace(300, 724, 400):
    k = min(3, int((x - 300) // 106))
    base = ys[k]
    nxt = ys[min(3, k + 1)]
    frac = (x - xs[k][0]) / (xs[k][1] - xs[k][0])
    u = max(0.0, (frac - 0.62) / 0.38)
    y = base + (nxt - base) * (3 * u ** 2 - 2 * u ** 3)
    if k == 3:
        y += 12 * np.sin((x - 640) / 10) * min(1, (x - 640) / 40) if x > 640 else 0
    pts.append((x * S, y * S))
d.line(pts, fill=(214, 30, 72, 255), width=16 * S, joint="curve")
for p in (pts[0], pts[-1]):
    r = 8 * S
    d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(214, 30, 72, 255))

img = img.resize((1024, 1024), Image.LANCZOS)
out.mkdir(exist_ok=True)
img.save(out / "AppIcon.png")
img.save(out / "AppIcon.icns", sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
print("ok")

"""Génère les icônes de Plimb Studio (PNG, ICO et mipmaps Android)."""
from PIL import Image, ImageDraw, ImageFilter
import os

S = 1024

def icon(size=S):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # Dégradé diagonal violet -> turquoise
    grad = Image.new("RGBA", (S, S))
    px = grad.load()
    a, b = (98, 72, 236), (24, 182, 160)
    for y in range(S):
        for x in range(S):
            t = (x + y) / (2 * S)
            px[x, y] = tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3)) + (255,)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([40, 40, S - 40, S - 40], radius=230, fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    w = (255, 255, 255, 255)
    # "p" géométrique : tige + anneau
    d.rounded_rectangle([300, 300, 400, 790], radius=50, fill=w)
    d.ellipse([300, 270, 720, 650], fill=w)
    d.ellipse([400, 370, 620, 550], fill=(0, 0, 0, 0))
    # On re-remplit l'intérieur avec le dégradé
    inner = Image.new("L", (S, S), 0)
    ImageDraw.Draw(inner).ellipse([400, 370, 620, 550], fill=255)
    img.paste(grad, (0, 0), inner)
    # Étincelle
    cx, cy, r = 735, 735, 95
    d.polygon([(cx, cy - r), (cx + 22, cy - 22), (cx + r, cy), (cx + 22, cy + 22),
               (cx, cy + r), (cx - 22, cy + 22), (cx - r, cy), (cx - 22, cy - 22)], fill=w)
    return img.resize((size, size), Image.LANCZOS)

root = os.path.join(os.path.dirname(__file__), "..")
big = icon(512)
big.save(os.path.join(root, "assets", "icon.png"))
icon(256).save(os.path.join(root, "assets", "icon.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
for name, sz in {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}.items():
    d = os.path.join(root, "android", "res", f"mipmap-{name}")
    os.makedirs(d, exist_ok=True)
    icon(sz).save(os.path.join(d, "ic_launcher.png"))
print("ok")

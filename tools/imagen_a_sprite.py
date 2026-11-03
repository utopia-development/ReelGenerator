"""
Convierte una imagen pixel-art (o cualquier imagen) a un sprite de texto editable.

Uso:
    python tools/imagen_a_sprite.py foto.png sprites/personaje.txt --celda 16 --colores 16
    # --celda = tamaño en px de cada "pixel" de la imagen original (16 en tu imagen)
    # --sin-fondo quita el fondo gris conectado a los bordes (se vuelve transparente)
"""
import argparse
from collections import deque

import numpy as np
from PIL import Image

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def quitar_fondo(g, tol_paso=18, lum_min=70, lum_max=120, sat_max=32):
    """Flood-fill desde los bordes sobre pixeles grisaceos. Devuelve mascara True = fondo."""
    h, w, _ = g.shape
    lum = g.mean(axis=2)
    sat = g.max(axis=2) - g.min(axis=2)
    apto = (lum >= lum_min) & (lum <= lum_max) & (sat <= sat_max)
    fondo = np.zeros((h, w), bool)
    q = deque()
    for y in range(h):
        for x in (0, w - 1):
            if apto[y, x] and not fondo[y, x]:
                fondo[y, x] = True; q.append((y, x))
    for x in range(w):
        for y in (0, h - 1):
            if apto[y, x] and not fondo[y, x]:
                fondo[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not fondo[ny, nx] and apto[ny, nx]:
                if np.abs(g[ny, nx] - g[y, x]).max() <= tol_paso:
                    fondo[ny, nx] = True; q.append((ny, nx))
    return fondo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada"); ap.add_argument("salida")
    ap.add_argument("--celda", type=int, default=16)
    ap.add_argument("--colores", type=int, default=16)
    ap.add_argument("--sin-fondo", action="store_true")
    a = ap.parse_args()

    im = Image.open(a.entrada).convert("RGB")
    W, H = im.size
    gw, gh = W // a.celda, H // a.celda
    # muestrear el centro de cada celda
    arr = np.array(im)
    c = a.celda // 2
    g = arr[c::a.celda, c::a.celda][:gh, :gw].astype(float)

    fondo = quitar_fondo(g) if a.sin_fondo else np.zeros((gh, gw), bool)

    # cuantizar sobre COLORES UNICOS (asi los colores raros -dorado, camisa, piel- no se pierden
    # frente a los mil tonos de pelo casi negro) y luego asignar cada celda al mas cercano
    vis = g[~fondo]
    unicos = np.unique(vis.astype(np.uint8), axis=0)
    tira = Image.fromarray(unicos.reshape(-1, 1, 3)).quantize(
        colors=a.colores, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = np.array(tira.getpalette()[: a.colores * 3]).reshape(-1, 3)
    pal = np.unique(pal, axis=0)
    d = ((g[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
    idx = d.argmin(-1)
    usados = sorted(set(idx[~fondo].tolist()))
    mapa = {u: CHARS[i] for i, u in enumerate(usados)}

    lineas = ["# paleta: LETRA=R,G,B   ('.' = transparente)"]
    for u in usados:
        r, gg, b = pal[u]
        lineas.append(f"{mapa[u]}={r},{gg},{b}")
    lineas.append("# matriz")
    for y in range(gh):
        lineas.append("".join("." if fondo[y, x] else mapa[idx[y, x]] for x in range(gw)))
    open(a.salida, "w").write("\n".join(lineas) + "\n")
    print(f"OK: {gw}x{gh} celdas, {len(usados)} colores -> {a.salida}")


if __name__ == "__main__":
    main()

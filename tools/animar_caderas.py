"""
Genera frames de "baile de caderas" a partir de UN sprite de texto.

Los pies quedan plantados, las caderas se van de lado a lado y el torso/cabeza
hacen el contrapeso (al reves), como un baile de balanceo. Genera 3 frames:
    prefijo_1.txt  caderas a la IZQUIERDA
    prefijo_2.txt  centro (el sprite original, ya limpio)
    prefijo_3.txt  caderas a la DERECHA
y en config.toml los listas asi para ir y volver suave:
    frames = ["sprites/prefijo_1.txt", "sprites/prefijo_2.txt", "sprites/prefijo_3.txt", "sprites/prefijo_2.txt"]

Uso:
    python tools/animar_caderas.py original.txt sprites/fantasma --reducir 2
    # --reducir 2   : reduce el sprite a la mitad (64x96 -> 32x48)
    # --colores 9   : simplifica la paleta (0 = no tocar)
    # --fondo ABC   : letras de la paleta que son fondo y pasan a transparente
    # --balanceo 3  : cuantos pixeles se mueven las caderas (en el sprite final)
    # --cadera 0.46 : altura de la cadera (0 = cabeza, 1 = pies)
    # --pies 0.90   : por debajo de esta altura los pies no se mueven
    # --sin-limpiar : conservar las motas oscuras sueltas (por defecto se quitan)
"""
import argparse

import numpy as np

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def leer(ruta, fondo):
    pal, filas = {}, []
    for l in open(ruta).read().splitlines():
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s:
            k, v = s.split("=")
            pal[k.strip()] = tuple(int(x) for x in v.split(","))
        else:
            filas.append(s)
    h, w = len(filas), max(len(f) for f in filas)
    rgb = np.zeros((h, w, 3), np.float32)
    op = np.zeros((h, w), bool)
    for y, f in enumerate(filas):
        for x, c in enumerate(f):
            if c in pal and c not in fondo:
                rgb[y, x] = pal[c]
                op[y, x] = True
    return rgb, op


def cuantizar(rgb, op, k):
    """k colores, calculados sobre colores UNICOS (los tonos raros no se pierden)."""
    from scipy.cluster.vq import kmeans2
    px = rgb[op]
    unicos = np.unique(px, axis=0)
    if k <= 0 or len(unicos) <= k:
        pal = unicos
    else:
        pal, _ = kmeans2(unicos, k, minit="++", seed=3)
        pal = np.unique(np.round(pal), axis=0)
    d = ((rgb[:, :, None, :] - pal[None, None]) ** 2).sum(-1)
    idx = d.argmin(-1)
    lum = pal.mean(1)
    orden = np.argsort(lum)                      # paleta ordenada de oscuro a claro
    remap = np.empty(len(pal), int); remap[orden] = np.arange(len(pal))
    return remap[idx], pal[orden].astype(int)


def reducir(idx, op, pal, f):
    """Bloque f x f -> color mas frecuente (empate: el mas oscuro). Transparente si
    menos de la mitad del bloque es opaco."""
    h, w = (idx.shape[0] // f) * f, (idx.shape[1] // f) * f
    H2, W2 = h // f, w // f
    out = np.zeros((H2, W2), int)
    mask = np.zeros((H2, W2), bool)
    for y in range(H2):
        for x in range(W2):
            bo = op[y * f:(y + 1) * f, x * f:(x + 1) * f]
            if bo.sum() * 2 < f * f:
                continue
            vals = idx[y * f:(y + 1) * f, x * f:(x + 1) * f][bo]
            cuenta = np.bincount(vals, minlength=len(pal))
            out[y, x] = int(np.flatnonzero(cuenta == cuenta.max())[0])   # 0 = el mas oscuro
            mask[y, x] = True
    return out, mask


def limpiar(idx, mask, pal, cara=0.22, oscuro=90):
    """Quita motas oscuras SUELTAS (ruido) fuera de la cara; no toca contornos ni ojos."""
    h, w = idx.shape
    lum = pal.mean(1)
    osc = mask & (lum[idx] < oscuro)
    out = idx.copy()
    for y in range(int(h * cara), h):
        for x in range(w):
            if not osc[y, x]:
                continue
            vecinos4 = [(y + dy, x + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            if any(0 <= yy < h and 0 <= xx < w and osc[yy, xx] for yy, xx in vecinos4):
                continue                                    # parte de una linea: se queda
            cand = [idx[yy, xx] for yy in range(max(0, y - 1), min(h, y + 2))
                    for xx in range(max(0, x - 1), min(w, x + 2))
                    if mask[yy, xx] and not osc[yy, xx]]
            if cand:
                out[y, x] = np.bincount(cand).argmax()
    return out


def recortar(idx, mask, margen):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    idx, mask = idx[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    return (np.pad(idx, ((0, 0), (margen, margen))), np.pad(mask, ((0, 0), (margen, margen))))


def perfil(h, cadera, pies):
    """p(y): 0 en los pies, 1 en la cadera, y contrapeso negativo hacia la cabeza."""
    yc, yp = cadera * (h - 1), pies * (h - 1)
    y = np.arange(h, dtype=float)
    p = np.zeros(h)
    abajo = (y >= yc) & (y < yp)
    p[abajo] = (yp - y[abajo]) / (yp - yc)                       # cadera -> pies: 1 -> 0
    hombros = yc * 0.40
    arr1 = (y < yc) & (y >= hombros)
    p[arr1] = 1 - 1.5 * (yc - y[arr1]) / (yc - hombros)          # cadera -> hombros: 1 -> -0.5
    arr2 = y < hombros
    p[arr2] = -0.5 - 0.2 * (hombros - y[arr2]) / max(hombros, 1)  # hombros -> cabeza: -0.5 -> -0.7
    return p


def desplazar(idx, mask, desp):
    out_i, out_m = np.zeros_like(idx), np.zeros_like(mask)
    for y, d in enumerate(desp):
        d = int(d)
        out_i[y] = np.roll(idx[y], d)
        out_m[y] = np.roll(mask[y], d)
    return out_i, out_m


def escribir(ruta, idx, mask, pal, titulo):
    letras = CHARS[:len(pal)]
    lineas = [f"# {titulo}", "# paleta: LETRA=R,G,B   ('.' = transparente)"]
    usados = sorted(set(idx[mask].tolist()))
    for u in usados:
        r, g, b = pal[u]
        lineas.append(f"{letras[u]}={r},{g},{b}")
    lineas.append("# matriz")
    for y in range(idx.shape[0]):
        lineas.append("".join(letras[idx[y, x]] if mask[y, x] else "." for x in range(idx.shape[1])))
    open(ruta, "w").write("\n".join(lineas) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada"); ap.add_argument("prefijo")
    ap.add_argument("--reducir", type=int, default=1)
    ap.add_argument("--colores", type=int, default=9)
    ap.add_argument("--fondo", default="ABC")
    ap.add_argument("--balanceo", type=float, default=3.0)
    ap.add_argument("--cadera", type=float, default=0.46)
    ap.add_argument("--pies", type=float, default=0.90)
    ap.add_argument("--sin-limpiar", action="store_true", help="no quitar motas oscuras sueltas")
    a = ap.parse_args()

    rgb, op = leer(a.entrada, set(a.fondo))
    idx, pal = cuantizar(rgb, op, a.colores)
    if a.reducir > 1:
        idx, op = reducir(idx, op, pal, a.reducir)
    if not a.sin_limpiar:
        idx = limpiar(idx, op, pal)
    margen = int(np.ceil(a.balanceo)) + 1
    idx, op = recortar(idx, op, margen)
    h, w = idx.shape
    p = perfil(h, a.cadera, a.pies)
    for n, d in ((1, -1), (2, 0), (3, 1)):
        desp = np.round(a.balanceo * p * d).astype(int)
        i2, m2 = desplazar(idx, op, desp)
        escribir(f"{a.prefijo}_{n}.txt", i2, m2, pal, f"baile de caderas, frame {n}/3 ({['izquierda','centro','derecha'][n-1]})")
    print(f"OK: 3 frames de {w}x{h} px, {len(pal)} colores -> {a.prefijo}_1/2/3.txt")


if __name__ == "__main__":
    main()

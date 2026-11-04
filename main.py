#!/usr/bin/env python3
"""
pixelvoz  -  audio  ->  video pixel art que reacciona a la voz y a la musica.

  voz      -> controla la boca del personaje
  musica   -> controla viento, hojas, brillo del cielo, luna y "cabeceo" en el beat

Uso rapido:
    python main.py mi_audio.mp3                 # genera salida.mp4
    python main.py mi_audio.mp3 --demucs        # separa voz/musica con IA (mucho mejor)
    python main.py mi_audio.mp3 --preview 10    # solo los primeros 10 s (para probar rapido)
    python main.py mi_audio.mp3 --png 12.5      # guarda UN frame (t=12.5 s) como PNG, sin video
    python main.py mi_audio.mp3 -psyco          # ondas psicodelicas en el cielo al ritmo de medios/agudos
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
#  CONFIG  (todo lo "tocable" esta aqui arriba)
# --------------------------------------------------------------------------- #
W, H = 128, 72          # resolucion REAL del pixel art (baja a proposito)
ESCALA = 10             # salida = 1280x720  (W*ESCALA x H*ESCALA)
FPS = 24

# cielo: color arriba / color abajo (RGB)
CIELO_ARRIBA = (14, 16, 38)
CIELO_ABAJO = (74, 52, 92)
COLINA_LEJOS = (38, 38, 70)
COLINA_CERCA = (24, 40, 56)
SUELO = (16, 30, 30)
COLORES_HOJA = [(222, 126, 44), (236, 182, 56), (186, 74, 44), (150, 170, 64)]
N_HOJAS = 80
N_ESTRELLAS = 45

# modo -psyco: ondas de color translucidas sobre el cielo
PSYCO_ALFA_MAX = 0.40       # opacidad maxima del efecto (0 = invisible, 1 = tapa el cielo)
PSYCO_PASOS_COLOR = 12      # niveles de tono (posterizado -> look pixel art)
PSYCO_NIVELES_ALFA = 6      # niveles de transparencia con tramado (dithering)
PSYCO_MIN_ENTRE_ONDAS = 0.45  # segundos minimos entre una onda grande y la siguiente


# --------------------------------------------------------------------------- #
#  SPRITES  (matrices de texto)
# --------------------------------------------------------------------------- #
def _parse_paleta(linea):
    k, v = linea.split("=")
    return k.strip(), tuple(int(x) for x in v.split(","))


def cargar_sprite(ruta):
    """Lee un .txt con paleta (LETRA=R,G,B) y matriz de letras. '.' = transparente.
    Devuelve (rgb[h,w,3] uint8, mascara[h,w] bool)."""
    paleta, filas = {}, []
    for linea in Path(ruta).read_text().splitlines():
        s = linea.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s:
            k, c = _parse_paleta(s)
            paleta[k] = c
        else:
            filas.append(s)
    h, w = len(filas), max(len(f) for f in filas)
    rgb = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), bool)
    for y, f in enumerate(filas):
        for x, ch in enumerate(f):
            if ch in paleta:
                rgb[y, x] = paleta[ch]
                mask[y, x] = True
    return rgb, mask


def cargar_bocas(ruta):
    """boca.txt -> {estado: (rgb, mask, fila, col)}"""
    paleta, bocas, actual = {}, {}, None
    for linea in Path(ruta).read_text().splitlines():
        s = linea.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("@"):
            _, est, fila, col = s.split()
            actual = [int(est), int(fila), int(col), []]
            bocas[int(est)] = actual
        elif "=" in s:
            k, c = _parse_paleta(s)
            paleta[k] = c
        elif actual is not None:
            actual[3].append(s)
    out = {}
    for est, (_, fila, col, rows) in bocas.items():
        h, w = len(rows), max(len(r) for r in rows)
        rgb = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w), bool)
        for y, r in enumerate(rows):
            for x, ch in enumerate(r):
                if ch in paleta:
                    rgb[y, x] = paleta[ch]
                    mask[y, x] = True
        out[est] = (rgb, mask, fila, col)
    return out


def pegar(canvas, rgb, mask, x, y, mezcla=None, cantidad=0.0):
    """Pega un sprite en el canvas con recorte en los bordes.
    mezcla/cantidad: tine el sprite hacia un color (sirve para profundidad / neblina)."""
    h, w = mask.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    src = rgb[y0 - y:y1 - y, x0 - x:x1 - x]
    if mezcla is not None and cantidad > 0:
        src = (src * (1 - cantidad) + np.array(mezcla, np.float32) * cantidad).astype(np.uint8)
    sub = canvas[y0:y1, x0:x1]
    sub[m] = src[m]


def pegar_con_viento(canvas, rgb, mask, x, y, t, viento, fase, amp=3.0, mezcla=None, cantidad=0.0):
    """Mueve cada FILA del sprite en horizontal: la punta se mueve mas que la base."""
    h = mask.shape[0]
    for j in range(h):
        rel = ((h - 1 - j) / h) ** 1.4                      # 0 en la base, 1 en la punta
        desp = np.sin(t * 2.2 + fase + j * 0.30) * (0.25 + viento) * amp * rel
        pegar(canvas, rgb[j:j + 1], mask[j:j + 1], x + int(round(desp)), y + j, mezcla, cantidad)


def arcoiris(h):
    """Tono h (cualquier forma, ciclico 0-1) -> RGB [...,3] float 0-255, posterizado en pasos."""
    h = np.floor((h % 1.0) * PSYCO_PASOS_COLOR) / PSYCO_PASOS_COLOR
    fase = np.array([0.0, 1 / 3, 2 / 3], np.float32)
    return (0.5 + 0.5 * np.cos(2 * np.pi * (h[..., None] + fase))) * 255.0


def matriz_bayer(h, w):
    """Tramado ordenado 4x4 en [0,1): da el efecto de transparencia 'pixel art'."""
    b = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0
    return np.tile(b, (h // 4 + 1, w // 4 + 1))[:h, :w]


# --------------------------------------------------------------------------- #
#  AUDIO  ->  FEATURES POR FRAME
# --------------------------------------------------------------------------- #
def ffmpeg_a_wav(entrada, salida, sr=22050):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(entrada),
                    "-ac", "1", "-ar", str(sr), str(salida)], check=True)


def separar_demucs(audio, tmpdir):
    """Devuelve (ruta_voz, ruta_musica) usando demucs, o None si falla."""
    print("  Separando voz/musica con Demucs (la primera vez descarga el modelo)...")
    r = subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", str(tmpdir), str(audio)])
    if r.returncode != 0:
        return None
    voz = next(Path(tmpdir).rglob("vocals.wav"), None)
    mus = next(Path(tmpdir).rglob("no_vocals.wav"), None)
    return (voz, mus) if voz and mus else None


def voz_heuristica(y, sr):
    """Sin Demucs: parte armonica (HPSS) + pasa-banda de voz 300-3400 Hz.
    Funciona 'regular': guitarras/sintes medios se cuelan como voz."""
    import librosa
    from scipy.signal import butter, sosfilt
    arm, _ = librosa.effects.hpss(y)
    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    return sosfilt(sos, arm).astype(np.float32)


def a_frames(serie, tiempos, n, fps):
    """Interpola una serie (con sus tiempos) a exactamente n frames de video (sin deriva)."""
    return np.interp(np.arange(n) / fps, tiempos, serie).astype(np.float32)


def normalizar(x, pct=98):
    return np.clip(x / (np.percentile(x, pct) + 1e-9), 0, 1)


def suavizar(x, ataque=0.7, caida=0.25):
    """Sube rapido, baja lento (como un medidor de VU). Evita boca parpadeando."""
    out = np.zeros_like(x)
    v = 0.0
    for i, a in enumerate(x):
        v = v + (a - v) * (ataque if a > v else caida)
        out[i] = v
    return out


def features_psyco(S, fr, ts, sr, hop, n, fps):
    """Features extra para -psyco, SOLO de medios/agudos de la musica:
       medios   : energia 400-4000 Hz por frame
       pulso_ma : pulso (1 -> 0) en cada golpe de medios/agudos (hi-hats, snares, sintes...)
       ondas    : lista de ondas grandes (una por golpe fuerte), cada una recorre el cielo"""
    import librosa
    medios = suavizar(normalizar(a_frames(S[(fr >= 400) & (fr < 4000)].mean(0), ts, n, fps)), 0.6, 0.2)

    S_db = librosa.amplitude_to_db(S[fr >= 500], ref=np.max)
    env = librosa.onset.onset_strength(S=S_db, sr=sr, hop_length=hop)
    env = env / (np.percentile(env, 95) + 1e-9)
    wait = max(1, int(0.10 * sr / hop))
    idx = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, wait=wait, units="frames")
    t_on = librosa.frames_to_time(idx, sr=sr, hop_length=hop)
    fuerza = np.clip(env[idx] / 1.5, 0, 1) if len(idx) else np.array([])

    pulso = np.zeros(n, np.float32)
    for tt in t_on:
        f0 = int(tt * fps)
        f1 = min(n, f0 + int(fps))
        if f0 >= n:
            continue
        d = (np.arange(f0, f1) / fps) - tt
        pulso[f0:f1] = np.maximum(pulso[f0:f1], np.exp(-np.maximum(d, 0) / 0.14))

    rng = np.random.default_rng(11)
    ondas, ultimo = [], -9.0
    for tt, fz in zip(t_on, fuerza):
        if tt - ultimo < PSYCO_MIN_ENTRE_ONDAS or fz < 0.3:
            continue
        ultimo = tt
        ang = rng.uniform(-0.55, 0.55) + (np.pi if rng.random() < 0.5 else 0.0)   # sentido y diagonal
        ondas.append(dict(t0=float(tt), f=float(fz), ang=float(ang), dur=float(rng.uniform(1.8, 2.6)),
                          fases=rng.uniform(0, 6.28, 3).astype(np.float32), hue=float(rng.random())))
    print(f"  Psyco: {len(t_on)} golpes de medios/agudos -> {len(ondas)} ondas grandes")
    return dict(medios=medios, pulso_ma=pulso, ondas=ondas,
                ondas_t0=np.array([o["t0"] for o in ondas], np.float64))


def analizar(audio, fps, usar_demucs, tmpdir, psyco=False):
    import librosa
    wav = Path(tmpdir) / "mono.wav"
    ffmpeg_a_wav(audio, wav)
    y_mix, sr = librosa.load(wav, sr=22050, mono=True)
    dur = len(y_mix) / sr
    n = int(np.ceil(dur * fps))

    y_voz = y_mus = None
    if usar_demucs:
        res = separar_demucs(audio, Path(tmpdir) / "demucs")
        if res:
            y_voz, _ = librosa.load(res[0], sr=sr, mono=True)
            y_mus, _ = librosa.load(res[1], sr=sr, mono=True)
        else:
            print("  ! Demucs fallo, uso el metodo heuristico.")
    if y_voz is None:
        y_voz = voz_heuristica(y_mix, sr)
        y_mus = y_mix

    hop = 256
    def rms(y):
        r = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
        return r, librosa.times_like(r, sr=sr, hop_length=hop)

    r, t = rms(y_voz)
    voz = normalizar(a_frames(r, t, n, fps))
    voz = np.clip((voz - 0.06) / 0.94, 0, 1)              # compuerta de ruido
    voz = suavizar(voz, ataque=0.75, caida=0.30)

    r, t = rms(y_mus)
    musica = suavizar(normalizar(a_frames(r, t, n, fps)), ataque=0.4, caida=0.08)

    # bandas de la musica (graves / agudos)
    S = np.abs(librosa.stft(y_mus, n_fft=2048, hop_length=hop))
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    ts = librosa.times_like(S, sr=sr, hop_length=hop)
    graves = suavizar(normalizar(a_frames(S[fr < 150].mean(0), ts, n, fps)), 0.6, 0.15)
    agudos = suavizar(normalizar(a_frames(S[fr > 4000].mean(0), ts, n, fps)), 0.6, 0.2)

    # beats -> pulso que decae
    pulso = np.zeros(n, np.float32)
    try:
        _, beats = librosa.beat.beat_track(y=y_mus, sr=sr, units="time")
        tf = np.arange(n) / fps
        for b in np.atleast_1d(beats):
            d = tf - b
            pulso = np.maximum(pulso, np.where(d >= 0, np.exp(-d / 0.16), 0))
    except Exception as e:
        print("  ! No pude detectar beats:", e)

    viento = suavizar(musica, 0.05, 0.03)                  # viento lento
    print(f"  Audio: {dur:.1f}s -> {n} frames | frames con voz: {(voz > 0.1).mean() * 100:.0f}% "
          f"| beats: {int((np.diff((pulso > 0.9).astype(int)) == 1).sum())}")
    F = dict(n=n, voz=voz, musica=musica, graves=graves, agudos=agudos, pulso=pulso, viento=viento)
    if psyco:
        F.update(features_psyco(S, fr, ts, sr, hop, n, fps))
    return F


# --------------------------------------------------------------------------- #
#  ESCENA  (aqui vive todo el dibujo)
# --------------------------------------------------------------------------- #
class Escena:
    def __init__(self, ruta_personaje, seed=7, psyco=False, psyco_intensidad=1.0):
        self.rng = np.random.default_rng(seed)
        self.psyco = psyco
        self.psyco_k = float(psyco_intensidad)
        self.X, self.Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.bayer = matriz_bayer(H, W)
        self.per_rgb, self.per_mask = cargar_sprite(ruta_personaje)
        self.arb_rgb, self.arb_mask = cargar_sprite(AQUI / "sprites" / "arbol.txt")
        self.bocas = cargar_bocas(AQUI / "sprites" / "boca.txt")
        ph, pw = self.per_mask.shape
        self.px, self.py = (W - pw) // 2, H - ph + 1

        # --- cielo (gradiente) ---
        a, b = np.array(CIELO_ARRIBA, np.float32), np.array(CIELO_ABAJO, np.float32)
        k = np.linspace(0, 1, H)[:, None] ** 1.3
        self.cielo = (a * (1 - k) + b * k)[:, None, :] * np.ones((1, W, 1), np.float32)

        # --- estrellas ---
        self.est = np.stack([self.rng.integers(0, W, N_ESTRELLAS),
                             self.rng.integers(0, int(H * 0.5), N_ESTRELLAS)], 1)
        self.est_fase = self.rng.uniform(0, 6.28, N_ESTRELLAS)

        # --- luna (circulo con mordida = creciente) ---
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy, r = 104, 15, 7
        luna = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        mordida = (xx - (cx + 3)) ** 2 + (yy - (cy - 2)) ** 2 <= (r - 1) ** 2
        self.luna = luna & ~mordida

        # --- colinas y suelo (estaticos) ---
        x = np.arange(W)
        h1 = (H * 0.60 + 5 * np.sin(x * 0.045) + 3 * np.sin(x * 0.13 + 1)).astype(int)
        h2 = (H * 0.70 + 4 * np.sin(x * 0.07 + 2) + 2 * np.sin(x * 0.21)).astype(int)
        self.capa_est = np.zeros((H, W, 3), np.uint8)
        self.capa_mask = np.zeros((H, W), bool)
        for hh, col in ((h1, COLINA_LEJOS), (h2, COLINA_CERCA)):
            for xi in range(W):
                self.capa_est[hh[xi]:, xi] = col
                self.capa_mask[hh[xi]:, xi] = True
        self.suelo_y = H - 7
        self.capa_est[self.suelo_y:] = SUELO
        self.capa_mask[self.suelo_y:] = True
        for xi in range(0, W, 3):                           # briznas de pasto
            if self.rng.random() < 0.7:
                self.capa_est[self.suelo_y - 1, xi] = (30, 60, 48)
                self.capa_mask[self.suelo_y - 1, xi] = True

        # --- arboles: (x, y_base, fase, capa) ---
        th = self.arb_mask.shape[0]
        self.arboles_lejos = [(xi, int(h1[min(max(xi + 7, 0), W - 1)]) + 6 - th, self.rng.uniform(0, 6.28))
                              for xi in range(-4, W, 13)]
        self.arboles_cerca = [(xi, self.suelo_y + 3 - th, self.rng.uniform(0, 6.28))
                              for xi in (-3, 14, 98, 113)]

        # --- hojas ---
        n = N_HOJAS
        self.hx = self.rng.uniform(-10, W + 10, n)
        self.hy = self.rng.uniform(-5, H, n)
        self.hvy = self.rng.uniform(7, 16, n)
        self.hf = self.rng.uniform(0, 6.28, n)
        self.hvf = self.rng.uniform(1.5, 4.0, n)
        self.hcol = np.array(COLORES_HOJA, np.uint8)[self.rng.integers(0, len(COLORES_HOJA), n)]
        self.hfrente = self.rng.random(n) < 0.18
        self.horden = self.rng.permutation(n) / n            # cuales aparecen primero segun energia

    # -- hojas -----------------------------------------------------------
    def _mover_hojas(self, viento, pulso, musica):
        dt = 1.0 / FPS
        mult = 1.0 + 2.2 * pulso + 1.2 * musica
        self.hf += self.hvf * dt
        self.hx += (viento * 16 + 5 * np.sin(self.hf)) * dt * mult
        self.hy += self.hvy * dt * mult
        fuera = (self.hy > H) | (self.hx > W + 8)
        k = int(fuera.sum())
        if k:
            self.hy[fuera] = -2
            self.hx[fuera] = self.rng.uniform(-30, W, k)

    def _dibujar_hojas(self, canvas, musica, frente):
        visibles = self.horden < (0.15 + 0.6 * musica)
        sel = visibles & (self.hfrente == frente)
        ix, iy = self.hx[sel].astype(int), self.hy[sel].astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        canvas[iy[ok], ix[ok]] = self.hcol[sel][ok]
        # aleteo: algunas hojas ocupan 2 pixeles cuando sin(fase) > 0
        ale = ok & (np.sin(self.hf[sel]) > 0.3) & (ix + 1 < W)
        canvas[iy[ale], ix[ale] + 1] = self.hcol[sel][ale]

    # -- efecto psicodelico -----------------------------------------------
    def _psyco(self, canvas, t, n, F):
        """Mezcla translucida de color sobre el cielo:
           * remolino suave siempre presente (respira con medios/agudos)
           * cada golpe fuerte de medios/agudos lanza una ONDA IRREGULAR que cruza el cielo."""
        X, Y, k = self.X, self.Y, self.psyco_k
        medios, agud, pm = F["medios"][n], F["agudos"][n], F["pulso_ma"][n]

        campo = (np.sin(X * 0.07 + np.sin(Y * 0.11 + t * 0.6) * 2.5 + t * 0.8)
                 + np.sin(Y * 0.13 - t * 0.5 + np.sin(X * 0.05 + t * 0.4) * 2.0)) * 0.25 + 0.5
        a_amb = (0.03 + 0.08 * medios + 0.05 * agud + 0.10 * pm) * campo * k
        A = a_amb.astype(np.float32)
        C = A[..., None] * arcoiris(campo * 0.35 + t * 0.05 + pm * 0.10)

        t0s = F["ondas_t0"]
        i1 = int(np.searchsorted(t0s, t, side="right"))
        i0 = max(int(np.searchsorted(t0s, t - 3.6)), i1 - 4)          # maximo 4 ondas a la vez
        for o in F["ondas"][i0:i1]:
            edad, vida = t - o["t0"], o["dur"] + 1.0
            if edad >= vida:
                continue
            ca, sa = np.cos(o["ang"]), np.sin(o["ang"])
            proj = X * ca + Y * sa                                      # eje de avance
            perp = -X * sa + Y * ca                                     # eje transversal
            frente = proj.min() - 10 + (proj.max() - proj.min() + 20) * edad / o["dur"]
            f1, f2, f3 = o["fases"]
            irr = (5.0 * np.sin(perp * 0.23 + f1 + edad * 1.5)          # borde irregular
                   + 3.0 * np.sin(perp * 0.51 + f2 - edad * 2.1)
                   + 2.5 * np.sin(proj * 0.11 + f3))
            u = frente - proj + irr                                     # >0 = detras del frente
            env = np.where(u > 0, np.exp(-np.maximum(u, 0) / 26.0), np.exp(np.minimum(u, 0) / 2.5))
            ondulacion = 0.5 + 0.5 * np.cos(u * 0.55 - edad * 3.0)
            fade = np.clip((vida - edad) / 0.8, 0, 1)
            w = (o["f"] * 0.42 * k * fade) * env * (0.3 + 0.7 * ondulacion)
            A = A + w
            C = C + w[..., None] * arcoiris(o["hue"] + u * 0.012 + perp * 0.004 + edad * 0.12)

        a = np.clip(A, 0, PSYCO_ALFA_MAX)
        col = C / (A[..., None] + 1e-6)
        q = PSYCO_NIVELES_ALFA
        aq = (np.floor(a * q + self.bayer) / q)[..., None]              # transparencia con tramado
        return np.clip(canvas * (1 - aq) + col * aq, 0, 255).astype(np.uint8)

    # -- frame completo ----------------------------------------------------
    def frame(self, n, F):
        t = n / FPS
        voz, musica, graves = F["voz"][n], F["musica"][n], F["graves"][n]
        agudos, pulso, viento = F["agudos"][n], F["pulso"][n], F["viento"][n]

        # cielo: pulsa con graves y beat
        brillo = 1.0 + 0.30 * graves + 0.18 * pulso
        canvas = np.clip(self.cielo * brillo, 0, 255).astype(np.uint8)
        if self.psyco:                                       # ondas de color translucidas sobre el cielo
            canvas = self._psyco(canvas, t, n, F)

        # estrellas: titilan, mas con agudos
        for (sx, sy), fase in zip(self.est, self.est_fase):
            v = 0.45 + 0.55 * np.sin(t * (2 + 3 * agudos) + fase)
            if v > 0.2:
                canvas[sy, sx] = np.clip(np.array([200, 200, 230]) * (0.4 + 0.6 * v), 0, 255)

        # luna
        luz = np.clip(np.array([232, 226, 190]) * (0.85 + 0.25 * graves), 0, 255).astype(np.uint8)
        canvas[self.luna] = luz

        # colinas + suelo
        canvas[self.capa_mask] = self.capa_est[self.capa_mask]

        # arboles lejanos (con neblina) y cercanos
        for x, y, fase in self.arboles_lejos:
            pegar_con_viento(canvas, self.arb_rgb, self.arb_mask, x, y, t, viento, fase,
                             amp=2.0, mezcla=COLINA_LEJOS, cantidad=0.55)
        self._mover_hojas(viento, pulso, musica)
        self._dibujar_hojas(canvas, musica, frente=False)
        for x, y, fase in self.arboles_cerca:
            pegar_con_viento(canvas, self.arb_rgb, self.arb_mask, x, y, t, viento, fase,
                             amp=3.5, mezcla=CIELO_ABAJO, cantidad=0.0)

        # personaje (cabecea 1px en el beat)
        py = self.py - (1 if pulso > 0.55 else 0)
        pegar(canvas, self.per_rgb, self.per_mask, self.px, py)

        # boca segun nivel de voz
        est = 0 if voz < 0.12 else 1 if voz < 0.35 else 2 if voz < 0.65 else 3
        if est in self.bocas:
            rgb, mask, fila, col = self.bocas[est]
            pegar(canvas, rgb, mask, self.px + col, py + fila)

        # hojas en primer plano (delante del personaje)
        self._dibujar_hojas(canvas, musica, frente=True)
        return canvas


# --------------------------------------------------------------------------- #
#  VIDEO
# --------------------------------------------------------------------------- #
def escalar(canvas):
    return canvas.repeat(ESCALA, 0).repeat(ESCALA, 1)      # nearest-neighbour, rapido


def exportar(escena, F, audio, salida, n_frames, fps):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W * ESCALA}x{H * ESCALA}",
           "-r", str(fps), "-i", "-",
           "-i", str(audio),
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(salida)]
    # en --preview el video es mas corto que el audio y -shortest corta el audio solo
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for n in range(n_frames):
            proc.stdin.write(escalar(escena.frame(n, F)).tobytes())
            if n % (fps * 5) == 0:
                print(f"  frame {n}/{n_frames}  ({n / n_frames * 100:.0f}%)", end="\r")
    finally:
        proc.stdin.close()
        proc.wait()
    print(f"\nListo -> {salida}")


def main():
    global FPS
    ap = argparse.ArgumentParser(description="Audio -> video pixel art reactivo")
    ap.add_argument("audio")
    ap.add_argument("-o", "--salida", default="salida.mp4")
    ap.add_argument("--personaje", default=str(AQUI / "sprites" / "personaje.txt"))
    ap.add_argument("--demucs", action="store_true", help="separar voz/musica con IA")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--preview", type=float, default=0, help="renderizar solo N segundos")
    ap.add_argument("--png", type=float, default=None, help="guardar un frame en el segundo T como PNG")
    ap.add_argument("-psyco", "--psyco", action="store_true",
                    help="ondas psicodelicas translucidas en el cielo, al ritmo de medios/agudos")
    ap.add_argument("--psyco-intensidad", type=float, default=1.0,
                    help="fuerza del efecto psyco (0.3 sutil ... 1 normal ... 1.6 intenso)")
    a = ap.parse_args()
    FPS = a.fps

    with tempfile.TemporaryDirectory() as tmp:
        print("1/3 Analizando audio...")
        F = analizar(a.audio, FPS, a.demucs, tmp, psyco=a.psyco)
        print("2/3 Preparando escena...")
        escena = Escena(a.personaje, psyco=a.psyco, psyco_intensidad=a.psyco_intensidad)

        if a.png is not None:
            n = min(int(a.png * FPS), F["n"] - 1)
            for i in range(max(0, n - FPS), n):              # "calentar" las hojas
                escena.frame(i, F)
            Image.fromarray(escalar(escena.frame(n, F))).save("frame.png")
            print("Guardado frame.png")
            return

        total = F["n"] if not a.preview else min(F["n"], int(a.preview * FPS))
        print(f"3/3 Renderizando {total} frames...")
        exportar(escena, F, a.audio, a.salida, total, FPS)


if __name__ == "__main__":
    main()

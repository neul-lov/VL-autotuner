"""시험용 CV 음원 만들기: か · さ · は … 처럼 자음이 붙은 음절 45개 + 숨 (息 · 吸).
테토처럼 WORLDLINE-R로 렌더하면 ㅅ · ㅎ 같은 바람 소리 자음에도 음높이가 조금 실려서, 그걸 음표로 잘못 읽는지 시험할 수 있다.
자음은 잡음 · 짧은 터짐 · 콧소리를 만들어 붙인 것이고, 모음은 make_test_voicebank.py와 같은 사람 녹음이다.
숨은 테토처럼 노래만큼 크게 넣는다.

사용: python make_test_cv_voicebank.py <vocobox 폴더> <vocadito_14.wav> <만들 음원 폴더> [이름]
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

vocobox, vocadito, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
name = sys.argv[4] if len(sys.argv) > 4 else "TestCV"
out.mkdir(parents=True, exist_ok=True)
src = vocobox / "data/voices/martin/voyels/exports/mono"
rng = np.random.default_rng(1)

VOWELS = {"a": "_-a-c3.wav", "i": "_-i-c3.wav", "u": "_-ou-c3.wav", "e": "_-e-c3.wav", "o": "_-o-c3.wav"}
# 음절 → (자음 종류, 모음)
KANA = {"あ": ("", "a"), "い": ("", "i"), "う": ("", "u"), "え": ("", "e"), "お": ("", "o"),
        "か": ("k", "a"), "き": ("k", "i"), "く": ("k", "u"), "け": ("k", "e"), "こ": ("k", "o"),
        "さ": ("s", "a"), "し": ("sh", "i"), "す": ("s", "u"), "せ": ("s", "e"), "そ": ("s", "o"),
        "た": ("t", "a"), "ち": ("ch", "i"), "つ": ("ts", "u"), "て": ("t", "e"), "と": ("t", "o"),
        "な": ("n", "a"), "に": ("n", "i"), "ぬ": ("n", "u"), "ね": ("n", "e"), "の": ("n", "o"),
        "は": ("h", "a"), "ひ": ("hi", "i"), "ふ": ("f", "u"), "へ": ("h", "e"), "ほ": ("h", "o"),
        "ま": ("m", "a"), "み": ("m", "i"), "む": ("m", "u"), "め": ("m", "e"), "も": ("m", "o"),
        "ら": ("r", "a"), "り": ("r", "i"), "る": ("r", "u"), "れ": ("r", "e"), "ろ": ("r", "o"),
        "や": ("y", "a"), "ゆ": ("y", "u"), "よ": ("y", "o"), "わ": ("w", "a"), "ん": ("N", "u")}


def band(x, sr, lo, hi):
    return sosfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x)


def noise(sr, sec, lo, hi, level):
    x = band(rng.standard_normal(int(sr * sec)), sr, lo, hi)
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * level
    fade = int(sr * 0.01)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0.6, fade)
    return x


def vowel(v):
    y, sr = sf.read(src / VOWELS[v], dtype="float64", always_2d=True)
    y = y.mean(axis=1)
    hop = int(sr * 0.005)
    db = 20 * np.log10(np.array([np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop, hop)]))
    on = int(np.argmax(db > db.max() - 25)) * hop
    off = (len(db) - int(np.argmax(db[::-1] > db.max() - 25))) * hop
    return y[on:off], sr


def consonant(kind, vow, sr):
    """→ (자음 소리, 모음 앞을 얼마나 바꿀지 함수)."""
    lv = np.sqrt(np.mean(vow[: int(sr * 0.2)] ** 2))
    gap = np.zeros(int(sr * 0.02))
    if kind == "":
        return np.zeros(int(sr * 0.03))
    if kind in ("k", "t"):
        burst = noise(sr, 0.02, 1500 if kind == "k" else 3000, 7000, lv * 0.8)
        return np.concatenate([gap, gap, burst, noise(sr, 0.025, 1000, 6000, lv * 0.25)])
    if kind == "s":
        return noise(sr, 0.13, 4000, 10000, lv * 0.6)
    if kind == "sh":
        return noise(sr, 0.13, 2200, 7000, lv * 0.6)
    if kind in ("ch", "ts"):
        return np.concatenate([gap, noise(sr, 0.015, 2000, 8000, lv), noise(sr, 0.07, 2500 if kind == "ch" else 4000, 9000, lv * 0.5)])
    if kind in ("h", "hi", "f"):
        lo, hi = {"h": (800, 5000), "hi": (2500, 7000), "f": (1000, 6000)}[kind]
        return noise(sr, 0.11, lo, hi, lv * 0.35)
    if kind in ("n", "m", "N"):
        hum = sosfilt(butter(4, 350, btype="low", fs=sr, output="sos"), vow[: int(sr * (0.3 if kind == "N" else 0.07))])
        return hum / (np.sqrt(np.mean(hum ** 2)) + 1e-9) * lv * 0.3
    if kind == "r":
        return vow[: int(sr * 0.03)] * 0.15
    if kind in ("y", "w"):
        g, _ = vowel("i" if kind == "y" else "u")
        return g[: int(sr * 0.06)] * 0.7
    raise ValueError(kind)


lines = []
sr0 = None
for kana, (kind, v) in KANA.items():
    vow, sr = vowel(v)
    sr0 = sr
    cons = consonant(kind, vow, sr)
    lead = np.zeros(int(sr * 0.05))
    y = np.concatenate([lead, cons, vow])
    file = f"{kind or 'v'}_{v}.wav"
    sf.write(out / file, (y / max(1.0, np.abs(y).max() * 1.05)).astype(np.float32), sr, subtype="PCM_16")
    offset = 50 - 10  # ms
    pre = int(len(cons) / sr * 1000) + 10
    cons_ms = pre + 60
    cut = -(int(len(y) / sr * 1000) - offset - 40)
    # offset, consonant(고정), cutoff(음수 = offset부터 길이), preutter, overlap
    lines.append(f"{file}={kana},{offset},{cons_ms},{cut},{pre},{min(15, pre // 2)}")
y, sr = sf.read(vocadito, dtype="float64")
breath = y[int(3.42 * sr):int(3.62 * sr)]
fade = int(0.02 * sr)
breath[:fade] *= np.linspace(0, 1, fade)
breath[-fade:] *= np.linspace(1, 0, fade)
breath = np.concatenate([np.zeros(int(0.05 * sr)), breath, np.zeros(int(0.05 * sr))])
lv = np.sqrt(np.mean(vowel("a")[0][: int(sr0 * 0.2)] ** 2))
breath = breath / (np.sqrt(np.mean(breath[int(0.05 * sr):-int(0.05 * sr)] ** 2)) + 1e-9) * lv  # 테토처럼 노래만큼 크게
sf.write(out / "breath.wav", (breath / max(1.0, np.abs(breath).max() * 1.05)).astype(np.float32), sr, subtype="PCM_16")
for alias in ("息", "吸"):
    lines.append(f"breath.wav={alias},40,150,-200,30,10")
(out / "oto.ini").write_text("\n".join(lines) + "\n", encoding="utf-8")
(out / "character.txt").write_text(f"name={name}\n", encoding="utf-8")
(out / "character.yaml").write_text("text_file_encoding: utf-8\nportrait_opacity: 0.67\n"
                                    "default_phonemizer: OpenUtau.Plugin.Builtin.JapanesePresampPhonemizer\n", encoding="utf-8")
print("ok", out, len(KANA), "음절")

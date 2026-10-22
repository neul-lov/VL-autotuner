"""분리한 보컬처럼 만든 시험 보컬: 사람 노래(vocadito)에 잔향을 입히고 만든 반주와 섞은 뒤 Demucs로 다시 보컬만 뽑는다.
StemMixer로 분리한 보컬처럼 반주가 조금 새고, 반주가 큰 곳에서 목소리가 잠깐 줄어든다. 정답은 vocadito의 사람 표시 그대로.
소리는 재생하지 않는다. 결과 오디오는 시험에만 쓰고 올리지 않는다.

사용: python make_separated.py <vocadito 폴더> <결과 폴더> 3,10,14,22
      python make_separated.py --csd <CSD 폴더> <결과 폴더> 12   (CSD 한국 동요 12곡, accuracy.py와 같은 곡)
결과 폴더/sep_<번호 또는 이름>.wav (Demucs 보컬, 44.1kHz 모노). 다시 분리 · 잔향 제거 시험용으로 같은 크기로 맞춘
mix_<…>.wav (섞은 곡), wet_<…>.wav (잔향 입힌 보컬 = 분리 정답), dry_<…>.wav (원래 보컬 = 잔향 제거 정답),
bk_<…>.wav (Demucs 반주 = 섞은 곡 − Demucs 보컬, StemMixer가 주는 반주처럼)도 쓴다.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tetotune import audio  # noqa: E402

SR = 44100
BPM = 120.0


def _read_notes(p):
    rows = []
    for line in open(p):
        try:
            rows.append([float(v) for v in line.strip().split(",")])
        except ValueError:
            continue
    return np.array(rows).reshape(-1, 3)  # 시작, 음 높이(Hz), 길이


def _saw(f, t):
    return 2 * ((f * t) % 1.0) - 1


def _chords(notes, dur):
    """2초(한 마디)마다 그 안의 노래 음을 가장 많이 덮는 장 · 단3화음."""
    bar = 60 / BPM * 4
    out = []
    pcs = np.round(12 * np.log2(np.maximum(notes[:, 1], 1) / 440) + 69).astype(int) % 12 if len(notes) else []
    for k in range(int(np.ceil(dur / bar))):
        a, b = k * bar, (k + 1) * bar
        w = [(pc, min(b, s + d) - max(a, s)) for (s, _, d), pc in zip(notes, pcs) if s < b and s + d > a]
        best, score = (0, (0, 4, 7)), -1.0
        for root in range(12):
            for tri in ((0, 4, 7), (0, 3, 7)):
                sc = sum(x for pc, x in w if (pc - root) % 12 in tri) + 0.01 * (tri == (0, 4, 7))
                if sc > score:
                    best, score = (root, tri), sc
        out.append(best)
    return out, bar


def accompaniment(notes, dur, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    chords, bar = _chords(notes, dur)
    from scipy.signal import butter, sosfilt

    lp = butter(2, 2500, "low", fs=SR, output="sos")
    for k, (root, tri) in enumerate(chords):
        a, b = int(k * bar * SR), min(n, int((k + 1) * bar * SR))
        if b <= a:
            continue
        seg = np.zeros(b - a)
        tt = t[a:b]
        for iv in tri:  # 패드 (C3~C4 근처)
            f = 440 * 2 ** ((48 + root + iv - 69) / 12)
            for det in (-0.004, 0.0, 0.004):
                seg += 0.12 * _saw(f * (1 + det), tt)
        fb = 440 * 2 ** ((36 + root - 69) / 12)  # 베이스
        beat = 60 / BPM
        env = np.exp(-((tt - tt[0]) % beat) * 3)
        seg += 0.5 * (np.sin(2 * np.pi * fb * tt) + 0.3 * _saw(fb, tt)) * env
        fade = np.minimum(1, np.minimum(np.arange(b - a), np.arange(b - a)[::-1]) / (0.01 * SR))
        y[a:b] += sosfilt(lp, seg) * fade
    beat = 60 / BPM
    hp = butter(2, 6000, "high", fs=SR, output="sos")
    bp = butter(2, [1200, 5000], "band", fs=SR, output="sos")
    for i in range(int(dur / beat * 2)):
        s = int(i * beat / 2 * SR)
        if s >= n:
            break
        m = min(n - s, int(0.3 * SR))
        tt = np.arange(m) / SR
        if i % 2 == 0:  # 킥 (박마다)
            y[s:s + m] += 0.9 * np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
        if i % 4 == 2:  # 스네어 (2 · 4박)
            y[s:s + m] += 0.5 * sosfilt(bp, rng.standard_normal(m)) * np.exp(-tt * 18)
        y[s:s + m] += 0.12 * sosfilt(hp, rng.standard_normal(m)) * np.exp(-tt * 60)  # 하이햇 (8분)
    return y


def reverb(y, rt60=1.1, wet_db=-14.0, seed=1):
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(seed)
    m = int(rt60 * SR)
    tt = np.arange(m) / SR
    ir = rng.standard_normal(m) * np.exp(-6.9 * tt / rt60)
    ir[: int(0.02 * SR)] = 0  # 앞 20ms는 비운다 (선행 지연)
    wet = fftconvolve(y, ir)[: len(y)]
    wet *= 10 ** (wet_db / 20) * np.sqrt(np.mean(y ** 2)) / max(1e-9, np.sqrt(np.mean(wet ** 2)))
    return y + wet


def main():
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    argv = sys.argv[1:]
    csd = argv[0] == "--csd"
    if csd:
        argv = argv[1:]
    src, out = Path(argv[0]), Path(argv[1])
    if csd:
        sys.path.insert(0, str(ROOT / "devtools"))
        from accuracy import csd_songs

        items = [(name, wav, np.array([[a, 440 * 2 ** ((m - 69) / 12), b - a] for a, b, m, _ in rows]))
                 for name, wav, rows in csd_songs(src, int(argv[2]))]
    else:
        items = [(k, next(src.rglob(f"vocadito_{k}.wav")), _read_notes(next(src.rglob(f"vocadito_{k}_notesA1.csv"))))
                 for k in (int(x) for x in argv[2].split(","))]
    out.mkdir(parents=True, exist_ok=True)
    model = get_model("htdemucs")
    model.eval()
    for k, wav, notes in items:
        y, sr = audio.load(wav)
        if sr != SR:
            y = audio.resample(y, sr, SR)
        v = reverb(y)
        acc = accompaniment(notes, len(v) / SR, seed=k if isinstance(k, int) else sum(map(ord, k)))
        acc = np.pad(acc, (0, max(0, len(v) - len(acc))))[:len(v)]  # 초 → 샘플 반올림으로 한 샘플 모자랄 때가 있다
        acc *= np.sqrt(np.mean(v ** 2)) / max(1e-9, np.sqrt(np.mean(acc ** 2)))  # 보컬 : 반주 = 0dB
        mix = v + acc
        g = 1 / max(1.0, np.abs(mix).max() / 0.95)
        mix *= g
        x = torch.tensor(np.stack([mix, mix]), dtype=torch.float32)[None]
        with torch.no_grad():
            s = apply_model(model, x, device="cpu", split=True, overlap=0.25, progress=False)[0]
        voc = s[model.sources.index("vocals")].mean(0).numpy().astype(np.float64)
        audio.write(out / f"sep_{k}.wav", voc, SR)
        n = min(len(voc), len(mix))
        for name, x in (("mix", mix), ("wet", v * g), ("dry", y[:len(v)] * g), ("bk", mix[:n] - voc[:n])):
            audio.write(out / f"{name}_{k}.wav", x, SR)
        leak = 10 * np.log10(np.mean((voc[: len(v)] - v[: len(voc)]) ** 2) / np.mean(v ** 2))
        print(f"{k}: {len(v) / SR:.1f}s, 분리 보컬 오차 {leak:.1f}dB (원래 보컬 대비)", flush=True)


if __name__ == "__main__":
    main()

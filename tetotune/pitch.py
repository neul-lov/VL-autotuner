"""원래 보컬의 음높이를 5ms마다 읽는다.

- RMVPE (OpenUtau가 쓰는 노래 전용 모델): OpenUtau에 RMVPE를 설치해 두었으면 그 모델 파일을 그대로 쓴다.
  분리한 보컬에 반주가 조금 남아 있어도 잘 버틴다.
- WORLD Harvest: 모델이 없을 때. 깨끗한 보컬에서는 RMVPE만큼 정확하다.
읽은 뒤에는 옥타브 튐 · 짧은 튐을 고치고, 소리가 나는지(유성)는 크기와 주기성으로 다시 가린다.
"""
from pathlib import Path

import numpy as np

from . import audio

HOP = 0.005  # 초. OpenUtau 곡선 간격(5틱)과 비슷하게
FMIN, FMAX = 65.0, 1100.0


def hz_to_midi(f):
    f = np.asarray(f, dtype=np.float64)
    return np.where(f > 0, 69 + 12 * np.log2(np.maximum(f, 1e-6) / 440.0), np.nan)


def midi_to_hz(m):
    return 440.0 * 2 ** ((np.asarray(m) - 69) / 12)


# ---------------------------------------------------------------- RMVPE (있으면)
def rmvpe_model_path():
    """OpenUtau 의존성 폴더에 설치된 rmvpe.onnx (OpenUtau → 도구 → 의존성 설치)."""
    homes = [Path.home() / "Library" / "OpenUtau", Path.home() / ".local" / "share" / "OpenUtau"]
    for home in homes:
        for sub in ("rmvpe", "RMVPE"):
            p = home / "Dependencies" / sub / "rmvpe.onnx"
            if p.exists():
                return p
    return None


def _rmvpe(y16, n_frames):
    import onnxruntime as ort

    path = rmvpe_model_path()
    if path is None:
        return None
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    wave_in = next((i for i in inputs if "wav" in i.name.lower()), inputs[0])
    feeds = {wave_in.name: y16[None, :].astype(np.float32)}
    for i in inputs:
        if i is not wave_in:  # threshold (OpenUtau는 0.03)
            feeds[i.name] = np.array(0.03, dtype=np.float32).reshape([1] * len(i.shape) if i.shape else [])
    outs = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feeds))}
    f0 = next(v for k, v in outs.items() if "f0" in k.lower()).reshape(-1).astype(np.float64)
    uv = next((v for k, v in outs.items() if "uv" in k.lower()), None)
    if uv is not None:
        f0 = np.where(uv.reshape(-1) > 0.5, 0.0, f0)
    # RMVPE는 10ms 간격 → 5ms로
    t10 = np.arange(len(f0)) * 0.01
    t5 = np.arange(n_frames) * HOP
    voiced = np.interp(t5, t10, (f0 > 0).astype(float)) > 0.5
    lf = np.interp(t5, t10, np.log(np.maximum(f0, 1e-3)))
    return np.where(voiced, np.exp(lf), 0.0)


# ---------------------------------------------------------------- WORLD
def _harvest(y16):
    import pyworld

    x = y16.astype(np.float64)
    f0, t = pyworld.harvest(x, 16000, f0_floor=FMIN, f0_ceil=FMAX, frame_period=HOP * 1000)
    return pyworld.stonemask(x, f0, t, 16000)


def extract(y, sr, method="auto"):
    """→ (f0 Hz 배열, 0 = 소리 없음, 사용한 방법 이름). 프레임 i = i*HOP초."""
    y16 = audio.resample(y, sr, 16000)
    n = int(len(y16) / 16000 / HOP) + 1
    f0, used = None, "harvest"
    if method in ("auto", "rmvpe"):
        try:
            f0 = _rmvpe(y16, n)
            if f0 is not None:
                used = "rmvpe"
        except Exception:
            f0 = None
    if f0 is None:
        f0 = _harvest(y16)
    f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
    return f0, used


# ---------------------------------------------------------------- 정리
def frame_db(y, sr, n):
    """프레임마다 크기 (dB, 20ms 창)."""
    hop = int(sr * HOP)
    win = int(sr * 0.02)
    pad = np.pad(y, (win // 2, win + n * hop))
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    rms = np.sqrt((pad[idx] ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def _runs(mask):
    """True 구간들 [(시작, 끝)) 목록."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def clean(f0, db):
    """f0(Hz)와 크기(dB) → (midi 소수 배열, 유성 마스크).
    조용한 곳의 잡음, 30ms보다 짧은 유성 조각, 옥타브 튐, 한두 프레임 튐을 정리한다."""
    from scipy.ndimage import median_filter

    midi = hz_to_midi(f0)
    loud = db > max(-55.0, np.percentile(db, 95) - 38.0)
    voiced = np.isfinite(midi) & loud
    for a, b in _runs(voiced):
        if b - a < 6:
            voiced[a:b] = False
    out = np.where(voiced, midi, np.nan)
    for a, b in _runs(voiced):
        seg = out[a:b].copy()
        # 옥타브 튐: 주변 150ms 중앙값에서 9반음 넘게 벗어나면 옥타브를 접는다
        med = median_filter(seg, size=min(31, len(seg) | 1), mode="nearest")
        jump = seg - med
        seg = np.where(np.abs(jump) > 9, seg - 12 * np.round(jump / 12), seg)
        seg = median_filter(seg, size=3, mode="nearest")
        out[a:b] = seg
    # 소리가 시작 · 끝나는 순간(자음 · 성대 닫힘)에 음높이가 튀는 것: 이어지는 50ms와 2.5반음 넘게 다르면 버린다
    edge, look = int(0.04 / HOP), int(0.05 / HOP)
    for a, b in _runs(voiced):
        if b - a <= edge + look:
            continue
        ref = np.median(out[a + edge:a + edge + look])
        for i in range(a, a + edge):
            if abs(out[i] - ref) > 2.5:
                voiced[i] = False
        ref = np.median(out[b - edge - look:b - edge])
        for i in range(b - edge, b):
            if abs(out[i] - ref) > 2.5:
                voiced[i] = False
        # 가운데의 40ms보다 짧은 튐: 앞뒤 90ms 중앙값으로 바꾼다
        seg = out[a:b]
        med = median_filter(seg, size=min(19, (b - a) | 1), mode="nearest")
        spike = np.abs(seg - med) > 3
        for s0, s1 in _runs(spike):
            if s1 - s0 <= 8:
                seg[s0:s1] = med[s0:s1]
        out[a:b] = seg
    # 프레이즈가 짧으면 그 안의 중앙값부터 틀릴 수 있다: 앞뒤 3초의 노래 전체와 비교해서 한 번 더 접는다
    ctx = _context_median(out, voiced, int(1.5 / HOP))
    for a, b in _runs(voiced):
        d = np.nanmedian(out[a:b]) - np.nanmedian(ctx[a:b])
        if abs(d) > 9:
            out[a:b] -= 12 * np.round(d / 12)
    out[~voiced] = np.nan
    return out, voiced


def _context_median(x, mask, half):
    """프레임마다 앞뒤 half 프레임 안 유성 값의 중앙값 (40ms 간격으로 계산해서 보간)."""
    n = len(x)
    step = 8
    centers = np.arange(0, n, step)
    vals = np.full(len(centers), np.nan)
    for k, c in enumerate(centers):
        w = x[max(0, c - half):c + half][mask[max(0, c - half):c + half]]
        if len(w):
            vals[k] = np.median(w)
    ok = np.isfinite(vals)
    if not ok.any():
        return np.full(n, np.nan)
    return np.interp(np.arange(n), centers[ok], vals[ok])

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
    """OpenUtau 의존성 폴더에 설치된 rmvpe.onnx (OpenUtau → 도구 → 의존성 설치). onnxruntime이 없으면 못 쓰니 None."""
    import importlib.util

    if importlib.util.find_spec("onnxruntime") is None:
        return None
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
    feeds = {wave_in.name: np.clip(y16, -1, 1)[None, :].astype(np.float32)}
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
CHUNK = 10.0  # 초. Harvest는 긴 파일을 한 번에 넣으면 메모리를 아주 많이 쓴다 (4분 곡에 8GB 넘게): 나눠서 읽는다
PAD = 1.0     # 조각 앞뒤로 겹쳐 읽는 길이 (조각 끝의 음높이가 흔들리지 않게)


def _harvest_part(x):
    import pyworld

    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pyworld.harvest(x, 16000, f0_floor=FMIN, f0_ceil=FMAX, frame_period=HOP * 1000)
    return pyworld.stonemask(x, f0, t, 16000)


def _harvest(y16, workers=None):
    """10초 조각으로 나눠 (앞뒤 1초 겹침) 여러 프로세스에서 읽고 이어 붙인다. 프레임 i = i*HOP초."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    hop = int(16000 * HOP)
    total = int(len(y16) / 16000 / HOP) + 1
    step, pad = int(CHUNK / HOP), int(PAD / HOP)
    if total <= step + 2 * pad:
        return _harvest_part(y16)
    parts = []
    for f in range(0, total, step):
        a, b = max(0, f - pad), min(total, f + step + pad)
        parts.append((f, a, y16[a * hop:b * hop]))
    workers = workers or max(1, min(4, (os.cpu_count() or 2) // 2))
    results = None
    if workers > 1:
        # Mac은 새 프로세스를 spawn으로 띄워서, `if __name__ == "__main__"` 없이 부른 스크립트에서는 실패한다:
        # 그때는 한 프로세스에서 차례로 읽는다
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_harvest_part, [p[2] for p in parts]))
        except (RuntimeError, OSError, BrokenProcessPool):
            results = None
    if results is None:
        results = [_harvest_part(p[2]) for p in parts]
    out = np.zeros(total)
    for (f, a, _), r in zip(parts, results):
        k = min(step, total - f, len(r) - (f - a))
        out[f:f + k] = r[f - a:f - a + k]
    return out


AUTO_FUSION = True  # "auto"면 RMVPE(새 판) · SwiftF0 · Harvest를 합쳐 읽는다 (f0models.py, 모델이 있을 때)


def extract(y, sr, method="auto"):
    """→ (f0 Hz 배열, 0 = 소리 없음, 사용한 방법 이름). 프레임 i = i*HOP초.
    fusion: 로컬 AI 둘(RMVPE · SwiftF0)과 Harvest 중 둘 이상이 같은 음이라고 하는 곳만 (f0models.py)."""
    y16 = audio.resample(y, sr, 16000)
    n = int(len(y16) / 16000 / HOP) + 1
    f0, used = None, "harvest"
    if method == "fusion" or (method == "auto" and AUTO_FUSION):
        from . import f0models
        try:
            h = _harvest(y16)
            fz, names = f0models.extract(y16, n, h)
            if fz is not None:
                return np.pad(fz, (0, max(0, n - len(fz))))[:n], "fusion(" + "+".join(names) + ")"
            f0 = h  # 모델이 없으면 Harvest만
        except Exception:
            f0 = None
    if f0 is None and method in ("auto", "rmvpe"):
        try:
            f0 = _rmvpe(y16, n)
            if f0 is not None:
                used = "rmvpe"
        except Exception:
            f0 = None
    if f0 is None and method in ("rmvpe2", "swift"):  # 시험용: 모델 하나만
        from . import f0models
        fz, names = f0models.extract(y16, n, None, use=("rmvpe",) if method == "rmvpe2" else ("swift",))
        if fz is not None:
            return np.pad(fz, (0, max(0, n - len(fz))))[:n], method
    if f0 is None:
        f0 = _harvest(y16)
    f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
    return f0, used


# ---------------------------------------------------------------- 정리
def frame_db(y, sr, n):
    """프레임마다 크기 (dB, 20ms 창). 누적합으로 계산해서 긴 곡도 메모리를 적게 쓴다."""
    win = int(sr * 0.02)
    # 44.1kHz에서 5ms는 220.5샘플이라 정수 간격으로 자르면 곡 뒤로 갈수록 음높이 프레임보다 밀린다
    starts = np.round(np.arange(n) * HOP * sr).astype(np.int64)
    y = np.asarray(y, dtype=np.float64)
    right = max(0, int(starts[-1]) + win - len(y) - win // 2) if n else 0
    pad = np.pad(y, (win // 2, right))
    c = np.concatenate([[0.0], np.cumsum(pad * pad)])
    power = np.maximum(c[starts + win] - c[starts], 0.0) / win
    return 20 * np.log10(np.sqrt(power + 1e-12) + 1e-9)


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

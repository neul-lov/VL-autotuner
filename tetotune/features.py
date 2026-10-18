"""음높이 말고 사람 목소리가 바뀌는 것들: 크기, 숨 섞인 정도, 긴장도, 그리고 숨소리 위치."""
import numpy as np

from . import audio
from .pitch import HOP, _runs

SR16 = 16000
NFFT = 512


def _spectral(y16, n, chunk=4000):
    """프레임마다 대역 에너지 · 평탄도 · 켑스트럼 (n×512 행렬을 한 번에 만들지 않게 4000프레임씩)."""
    hop = int(SR16 * HOP)  # 80
    win = np.hanning(NFFT)
    pad = np.pad(np.asarray(y16, dtype=np.float64), (NFFT // 2, NFFT + n * hop))
    freqs = np.fft.rfftfreq(NFFT, 1 / SR16)
    bands = {name: (freqs >= lo) & (freqs < hi) for name, lo, hi in
             (("low", 80, 1500), ("mid", 1500, 2500), ("high", 2500, 7000))}
    out = {k: np.zeros(n) for k in ("low", "mid", "high", "flat")}
    ceps = np.zeros((n, 12))
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        idx = np.arange(a, b)[:, None] * hop + np.arange(NFFT)[None, :]
        spec = np.abs(np.fft.rfft(pad[idx] * win, axis=1)) ** 2
        for k, m in bands.items():
            out[k][a:b] = spec[:, m].sum(axis=1) + 1e-12
        out["flat"][a:b] = np.exp(np.log(spec[:, 4:] + 1e-12).mean(axis=1)) / (spec[:, 4:].mean(axis=1) + 1e-12)
        ceps[a:b] = np.fft.irfft(np.log(spec + 1e-10), axis=1)[:, 1:13]
    return out, ceps


def _d4c(x, f0, chunk=4000, pad=200):
    """WORLD D4C 비주기성 (1~5kHz 평균, dB). 긴 곡은 20초씩 (앞뒤 1초 겹침)."""
    import pyworld

    n = len(f0)
    hop = int(SR16 * HOP)
    out = np.zeros(n)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        pa, pb = max(0, a - pad), min(n, b + pad)
        seg = np.ascontiguousarray(x[pa * hop:pb * hop + NFFT])
        t = np.arange(pb - pa) * HOP
        ap = pyworld.d4c(seg, np.ascontiguousarray(f0[pa:pb]), t, SR16, fft_size=1024)
        apf = np.linspace(0, SR16 / 2, ap.shape[1])
        band = 10 * np.log10(ap[:, (apf >= 1000) & (apf <= 5000)].mean(axis=1) + 1e-6)
        out[a:b] = band[a - pa:b - pa]
    return out


def analyze(y, sr, f0_hz, voiced):
    """→ dict(novelty, breathiness, tension, high_ratio, flatness) 프레임 배열. 유성 아닌 곳은 nan (high_ratio · flatness · novelty는 모든 곳)."""
    n = len(f0_hz)
    y16 = audio.resample(y, sr, SR16)
    sp, ceps = _spectral(y16, n)
    low, mid, high = sp["low"], sp["mid"], sp["high"]
    high_ratio = (mid + high) / (low + mid + high)

    # 숨 섞인 정도: WORLD D4C의 비주기성. 높을수록 바람 소리가 많다.
    x = y16.astype(np.float64)
    f0 = np.where(voiced, f0_hz, 0.0).astype(np.float64)
    breath = _d4c(x, f0)
    # 긴장도: 높은 배음이 얼마나 센지 (스펙트럼 기울기). 힘을 주면 2.5kHz 위가 세진다.
    tension = 10 * np.log10(high / low)
    # 소리 빛깔(모음 · 자음)이 바뀌는 정도: 켑스트럼 낮은 차수(스펙트럼 윤곽, 음높이 · 비브라토와 무관)의 앞뒤 50ms 차이
    w = int(0.05 / HOP)
    cs = np.cumsum(np.vstack([np.zeros((1, ceps.shape[1])), ceps]), axis=0)
    novelty = np.zeros(n)
    i = np.arange(w, n - w)
    novelty[i] = np.linalg.norm((cs[i + w] - cs[i]) - (cs[i] - cs[i - w]), axis=1) / w
    return {
        "novelty": novelty,
        "breathiness": np.where(voiced, breath, np.nan),
        "tension": np.where(voiced, tension, np.nan),
        "high_ratio": high_ratio,
        "flatness": sp["flat"],
    }


def noisy_voiced(voiced, feats, min_len=0.05):
    """유성으로 잡혔지만 실제로는 잡음인 구간 (숨 · ㅅ · ㅎ 같은 바람 소리) → 마스크.
    합성한 숨 샘플(UTAU 息 · 吸), 크게 들이쉰 숨, WORLD로 다시 만든 자음은 Harvest가 음높이를 찾아 버려서
    따로 음표가 되고 가사가 한 칸씩 밀린다. 스펙트럼이 평탄하고(배음이 없고) 높은 대역이 대부분이며
    비주기성이 거의 최대인 구간이 50ms 넘게 이어질 때만 (사람 노래의 숨 섞인 모음은 배음이 남아 있어 걸리지 않는다)."""
    noisy = voiced & (feats["flatness"] > 0.015) & (feats["high_ratio"] > 0.6)
    for a, b in _runs(~noisy):  # 20ms 이하 틈은 메운다
        if 0 < a and b < len(noisy) and b - a <= 4 and voiced[a:b].all():
            noisy[a:b] = True
    out = np.zeros_like(voiced)
    for a, b in _runs(noisy):
        ap = feats["breathiness"][a:b]
        ap = ap[np.isfinite(ap)]
        if (b - a) * HOP >= min_len and len(ap) and np.median(ap) > -3:
            out[a:b] = True
    return out


def find_breaths(db, voiced, feats, min_len=0.12, max_len=1.4):
    """노래하지 않는 곳에서 숨소리를 찾는다 → [(시작초, 끝초, 크기dB, 들숨인지)].
    숨은 잡음처럼 넓게 퍼진 소리라서: 바닥 잡음보다 충분히 크고, 높은 대역 비율 · 평탄도가 높다.
    다음 소리 바로 앞 40ms는 자음(ㅅ · ㅎ 등)일 수 있어서 뺀다."""
    n = len(db)
    if not voiced.any():
        return []
    floor = np.percentile(db, 3)
    sing = np.median(db[voiced])
    # 크기 위쪽은 노래만큼까지: 마이크 가까이에서 들이쉰 숨이나 합성한 숨은 노래와 크기가 비슷하다
    cand = (~voiced) & (db > floor + 6) & (db < sing + 1) & (feats["high_ratio"] > 0.25) & (feats["flatness"] > 0.02)
    # 30ms 이하 틈은 메운다
    for a, b in _runs(~cand):
        if 0 < a and b < n and b - a <= 6:
            cand[a:b] = True
    onsets = [a for a, _ in _runs(voiced)]
    offsets = [b for _, b in _runs(voiced)]
    pad_on, pad_off = int(0.04 / HOP), int(0.03 / HOP)
    out = []
    for a, b in _runs(cand):
        nxt = next((o for o in onsets if o >= b), None)
        prv = max((o for o in offsets if o <= a), default=None)
        if nxt is not None and nxt - b < pad_on:
            b = nxt - pad_on
        if prv is not None and a - prv < pad_off:
            a = prv + pad_off
        dur = (b - a) * HOP
        if dur < min_len or np.mean(db[a:b]) < floor + 8:
            continue
        if dur > max_len:  # 긴 잡음은 숨이 아닐 수 있다: 다음 소리 쪽 끝만
            a = b - int(max_len / HOP)
        inhale = nxt is not None and (nxt - b) * HOP < 0.6
        out.append((a * HOP, b * HOP, float(np.percentile(db[a:b], 90)), inhale))
    return out

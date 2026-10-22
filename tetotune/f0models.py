"""로컬 AI 음 높이 읽기: RMVPE(RVC 2023-10-06 판, MIT)와 SwiftF0(MIT, pip로 설치)를 Harvest와 함께 쓴다.

분리한 보컬은 반주가 새고 잔향이 남아서 Harvest 혼자서는 다른 소리를 따라가거나 옥타브 · 다른 음으로 잠깐 튄다.
세 방법은 틀리는 곳이 서로 달라서, 둘 이상이 같은 음(80센트 안)이라고 하는 프레임만 쓰고 셋 다 다른 곳은 비워 둔다
(조교가 비운 곳을 앞뒤 · 음표 음으로 메운다). RMVPE 모델(362MB)은 처음 쓸 때 한 번 받는다 (판 고정 + SHA-256).
"""
import importlib.util
from pathlib import Path

import numpy as np

from .pitch import FMAX, FMIN, HOP, hz_to_midi

RMVPE_URL = ("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/"
             "0d7ebae452fb102c695b08f4e6f546be00603425/rmvpe.onnx")
RMVPE_SHA256 = "5370e71ac80af8b4b7c793d27efd51fd8bf962de3a7ede0766dac0befa3660fd"
RMVPE_MAX = 400 << 20
RMVPE_THRESHOLD = 0.03  # 이보다 확신이 낮으면 소리 없음 (RVC와 같은 값; 소리 있음은 다시 크기로 가린다)
CHUNK = 30.0  # 초. RMVPE는 긴 곡을 한 번에 넣으면 메모리를 많이 써서 조각으로 (앞뒤 PAD초 겹침)
PAD = 1.0
AGREE = 80.0  # 센트. 두 방법이 이만큼 안이면 같은 음
SOLO = 0.5  # 다른 방법과 맞지 않는 방법 하나는 무게가 이보다 클 때만 쓴다 (RMVPE 0.55만 넘는다). 1 = 둘 이상 맞을 때만


def _ort():
    return importlib.util.find_spec("onnxruntime") is not None


def rmvpe_path():
    from .game import models_home

    p = models_home() / "rmvpe" / "rmvpe.onnx"
    return p if p.exists() and _ort() else None


def ensure_rmvpe(progress=lambda s: None):
    """RMVPE 모델이 없으면 받는다 → 경로 또는 None (onnxruntime 없음)."""
    import tempfile

    from .game import _download, models_home

    if not _ort():
        return None
    found = rmvpe_path()
    if found:
        return found
    dest = models_home() / "rmvpe" / "rmvpe.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    progress("음 높이 읽기 모델(RMVPE) 받는 중 (처음 한 번)")
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        part = Path(tmp) / "rmvpe.onnx"
        _download(RMVPE_URL, part, RMVPE_SHA256, RMVPE_MAX, "음 높이 읽기 모델(RMVPE)")
        part.replace(dest)
    return dest


def swift_available():
    return _ort() and importlib.util.find_spec("swift_f0") is not None


# ---------------------------------------------------------------- RMVPE
def mel_basis(sr=16000, n_fft=1024, n_mels=128, fmin=30.0, fmax=8000.0):
    """librosa.filters.mel(htk=True, norm="slaney")와 같은 멜 필터 (RVC RMVPE가 쓰는 것)."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

    fft_f = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_f[None, :]
    w = np.zeros((n_mels, len(fft_f)))
    for i in range(n_mels):
        w[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w


def log_mel(y16, n_fft=1024, hop=160):
    """16kHz 소리 → 로그 멜 [128, 프레임] (torch.stft center=True · reflect · 주기 한 창, |X|, log(max(x, 1e-5)))."""
    y = np.asarray(y16, dtype=np.float64)
    pad = n_fft // 2
    if len(y) <= pad:
        y = np.pad(y, (0, pad + 1 - len(y)))
    x = np.pad(y, (pad, pad), mode="reflect")
    n = 1 + (len(x) - n_fft) // hop
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    basis = mel_basis(n_fft=n_fft)
    out = np.empty((basis.shape[0], n), dtype=np.float32)
    step = 2048
    for a in range(0, n, step):
        b = min(n, a + step)
        idx = (np.arange(a, b) * hop)[:, None] + np.arange(n_fft)[None, :]
        mag = np.abs(np.fft.rfft(x[idx] * win, axis=1))
        out[:, a:b] = np.log(np.maximum(basis @ mag.T, 1e-5))
    return out


CENTS = 20 * np.arange(360) + 1997.3794084376191  # 칸 k = 10Hz 기준 센트


def _decode(sal, thr=RMVPE_THRESHOLD):
    """[프레임, 360] → (Hz, 확신). 가장 센 칸 앞뒤 4칸의 가중 평균 (RVC to_local_average_cents)."""
    c = sal.argmax(axis=1)
    pad = np.pad(sal, ((0, 0), (4, 4)))
    cm = np.pad(CENTS, 4)
    idx = c[:, None] + np.arange(9)[None, :]
    w = np.take_along_axis(pad, idx, axis=1)
    cents = (w * cm[idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
    conf = sal.max(axis=1)
    hz = np.where(conf > thr, 10 * 2 ** (cents / 1200), 0.0)
    return hz, conf


class Rmvpe:
    def __init__(self, path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def salience(self, y16):
        """→ [10ms 프레임, 360]. 조각마다 앞뒤 PAD초를 겹쳐 넣고 가운데만 쓴다."""
        mel = log_mel(y16)
        n = mel.shape[1]
        step, pad = int(CHUNK * 100), int(PAD * 100)
        out = np.zeros((n, 360), dtype=np.float32)
        for a in range(0, n, step):
            lo, hi = max(0, a - pad), min(n, a + step + pad)
            m = mel[:, lo:hi]
            t = m.shape[1]
            m = np.pad(m, ((0, 0), (0, 32 * ((t - 1) // 32 + 1) - t)))
            h = self.sess.run(None, {self.inp: m[None].astype(np.float32)})[0][0, :t]
            k = min(step, n - a)
            out[a:a + k] = h[a - lo:a - lo + k]
        return out

    def f0(self, y16, n5):
        """→ (5ms 프레임 Hz, 확신). 10ms 둘을 5ms(80샘플) 엇갈려 읽어서 섞는다."""
        hz0, c0 = _decode(self.salience(y16))
        hz1, c1 = _decode(self.salience(np.concatenate([np.zeros(80), y16])))
        hz = np.zeros(n5)
        conf = np.zeros(n5)
        k0 = np.arange(len(hz0)) * 2
        k1 = np.arange(len(hz1)) * 2 - 1  # 80샘플 늦게 넣었으니 프레임 j는 (2j − 1) × 5ms
        for k, h, c in ((k0, hz0, c0), (k1, hz1, c1)):
            ok = (k >= 0) & (k < n5)
            hz[k[ok]], conf[k[ok]] = h[ok], c[ok]
        return hz, conf


# ---------------------------------------------------------------- SwiftF0
def swift_f0(y16, n5):
    """→ (5ms 프레임 Hz, 확신). SwiftF0는 16ms 간격이라 4번 엇갈려 (4ms 간격) 읽고 5ms로 옮긴다."""
    from swift_f0 import SwiftF0

    det = SwiftF0()
    ts, hz, cf = [], [], []
    for d in (0, 64, 128, 192):
        r = det.detect(np.concatenate([np.zeros(d), y16]).astype(np.float32), 16000, fmin=FMIN, fmax=FMAX)
        ts.append(r.timestamps - d / 16000)
        hz.append(r.pitch_hz)
        cf.append(r.confidence)
    t, hz, cf = (np.concatenate(v) for v in (ts, hz, cf))
    o = np.argsort(t)
    t, hz, cf = t[o], hz[o], cf[o]
    t5 = np.arange(n5) * HOP
    j = np.clip(np.searchsorted(t, t5), 1, len(t) - 1)
    j = np.where(np.abs(t[j - 1] - t5) < np.abs(t[j] - t5), j - 1, j)  # 가장 가까운 프레임 (소리 있음/없음이 섞이지 않게)
    conf = cf[j]
    return np.where(conf >= 0.5, hz[j], 0.0), conf


# ---------------------------------------------------------------- 합치기
def fuse(tracks, agree=AGREE, solo=None):
    """tracks: [(Hz, 무게)] 같은 길이 → Hz (0 = 믿을 수 없음 · 소리 없음).
    프레임마다 소리 있다는 방법들 중 서로 agree센트 안인 무리를 찾고, 무게 합이 가장 큰 무리가 둘 이상이면
    그 무리의 (무게) 평균. 하나만 소리 있다고 하면 그 방법의 무게가 0.5 넘을 때만 (RMVPE 혼자는 믿는다)."""
    m = np.stack([hz_to_midi(np.where(h > 0, h, np.nan)) for h, _ in tracks])  # [방법, 프레임]
    w = np.array([wt for _, wt in tracks], dtype=float)[:, None] * np.isfinite(m)
    k, n = m.shape
    best_w = np.zeros(n)
    best_v = np.full(n, np.nan)
    best_c = np.zeros(n, dtype=int)
    for i in range(k):  # 방법 i를 가운데로 한 무리
        near = np.isfinite(m) & np.isfinite(m[i]) & (np.abs(m - m[i]) * 100 <= agree)
        ww = np.where(near, w, 0.0)
        tot = ww.sum(0)
        val = np.where(tot > 0, (np.nan_to_num(m) * ww).sum(0) / np.maximum(tot, 1e-9), np.nan)
        cnt = near.sum(0)
        better = tot > best_w
        best_w, best_v, best_c = np.where(better, tot, best_w), np.where(better, val, best_v), np.where(better, cnt, best_c)
    ok = np.isfinite(best_v) & ((best_c >= 2) | (best_w > (SOLO if solo is None else solo)))
    return np.where(ok, 440.0 * 2 ** ((np.nan_to_num(best_v) - 69) / 12), 0.0)


WEIGHTS = {"rmvpe": 0.55, "swift": 0.3, "harvest": 0.3}


def extract(y16, n5, harvest_hz, use=("rmvpe", "swift", "harvest")):
    """→ (Hz, 쓴 방법 이름 목록). 모델이 없는 방법은 빠진다. 하나도 없으면 (None, [])."""
    tracks, names = [], []
    if "rmvpe" in use and rmvpe_path():
        hz, _ = Rmvpe(rmvpe_path()).f0(y16, n5)
        tracks.append((hz, WEIGHTS["rmvpe"]))
        names.append("rmvpe")
    if "swift" in use and swift_available():
        hz, _ = swift_f0(y16, n5)
        tracks.append((hz, WEIGHTS["swift"]))
        names.append("swift")
    if not tracks:
        return None, []
    if "harvest" in use and harvest_hz is not None:
        tracks.append((np.pad(harvest_hz, (0, max(0, n5 - len(harvest_hz))))[:n5], WEIGHTS["harvest"]))
        names.append("harvest")
    if len(tracks) == 1:
        return tracks[0][0], names
    return fuse(tracks), names

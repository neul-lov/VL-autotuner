"""분석용으로 보컬을 깨끗하게 (로컬 AI, onnxruntime): 반주가 샌 곳을 다시 분리하고 잔향을 걷어 낸다.

분리한 보컬에 남은 반주 · 잔향은 음 높이를 잘못 읽게 하고 (앞 음 잔향이 다음 음 첫머리에 남는다) 가짜 짧은 음표 ·
끊김을 만든다. 깨끗하게 한 소리는 음 높이 · 음표 · 가사 자리를 읽는 데만 쓰고, 크기 · 숨 · 미리 듣기는 원래 보컬로 한다.

- 다시 분리: Kimberley Jensen의 Mel-Band RoFormer (보컬, 가중치 MIT), audiojs의 ONNX 변환 (MIT).
  반주가 있으면 보컬 + 반주(= 원래 곡)에서 다시 뽑고, 없으면 보컬에서 한 번 더 뽑는다.
- 잔향 제거: anvuew의 dereverb_room (BS-RoFormer 모노, GPL-3.0), audiojs의 ONNX 변환. 앱에 넣지 않고 처음 쓸 때 받는다.
STFT는 모델 밖에서 한다 (torch.stft와 같게: Hann, 가운데 맞춤, 반사 채우기). 긴 곡은 조각으로 겹쳐 돌린다 (MSST demix처럼).
"""
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np

# "backing" = 반주 파일이 있을 때만 보컬 + 반주(= 원래 곡)에서 다시 뽑는다, "auto" = 반주가 없어도 보컬에서 한 번 더, "off" = 안 함.
# CI (vocadito 분리 보컬 14곡): 곡에서 다시 뽑으면 SDR 13.2 → 17.3dB, 음 높이 튐 0.39 → 0.20/분. 보컬에서 한 번 더는 그대로.
SEPARATE = "backing"
DEREVERB = "off"  # 잔향 제거는 SDR이 오히려 떨어지고 (14.0 → 10.3dB) 음 높이도 그대로라서 쓰지 않는다
# 시험용 (CI): 폴더를 주면 같은 보컬 · 반주 파일에서 다시 뽑은 보컬을 저장해 두고 다시 쓴다 (방식마다 다시 돌리지 않게)
CACHE = os.environ.get("TETOTUNE_CLEAN_CACHE", "")

MODELS = {
    "vocals": {
        "url": "https://huggingface.co/audiojs/mel-roformer/resolve/c1aba02c00a0bee852d6754fe6ede8164f90a4b3/"
               "mel-roformer.fp16.onnx",
        "sha256": "adcbbb826f12453fbb86652b503242d565a9c81cdef1bca2fe0e54752eba91ef",
        "max": 600 << 20, "what": "보컬 다시 분리 모델(Mel-RoFormer)",
        "n_fft": 2048, "hop": 441, "chunk": 352800, "channels": 2, "overlap": 2,
    },
    "dereverb": {
        "url": "https://huggingface.co/audiojs/dereverb-room/resolve/b4c2d14da9f2199099656195b06fd8253fd16e1d/"
               "dereverb-room.fp16.onnx",
        "sha256": "99797e086a71d1cb57d50f3b5be93dcb7c89c49d1537c60487a488a24aaee9ee",
        "max": 120 << 20, "what": "잔향 제거 모델(dereverb_room)",
        "n_fft": 2048, "hop": 512, "chunk": 384000, "channels": 1, "overlap": 2,
    },
}
# 모델 입력 [1, 채널×2, 주파수, 프레임]의 순서와 출력이 스펙트럼인지 마스크인지 (CI check_clean_models.py로 정한다)
LAYOUT = {"vocals": ("sc", "spec"), "dereverb": ("sc", "spec")}
SR = 44100


def _ort():
    return importlib.util.find_spec("onnxruntime") is not None


def wanted(name, backing=False):
    """이 조교에 name 모델을 쓸지 (설정과 반주 파일이 있는지로)."""
    mode = SEPARATE if name == "vocals" else DEREVERB
    return mode == "auto" or (mode == "backing" and bool(backing))


def model_path(name):
    from .game import models_home

    p = models_home() / "clean" / f"{name}.onnx"
    return p if _ort() and p.exists() else None


def ensure(name, progress=lambda s: None):
    import tempfile

    from .game import _download, models_home

    if not _ort():
        return None
    found = model_path(name)
    if found:
        return found
    m = MODELS[name]
    dest = models_home() / "clean" / f"{name}.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    progress(f"{m['what']} 받는 중 (처음 한 번)")
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        part = Path(tmp) / "m.onnx"
        _download(m["url"], part, m["sha256"], m["max"], m["what"])
        part.replace(dest)
    return dest


# ---------------------------------------------------------------- STFT (torch.stft / istft와 같게)
def _window(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)  # periodic Hann


def stft(x, n_fft, hop):
    """x: [채널, 샘플] → [채널, n_fft/2+1, 1 + 샘플/hop] 복소수 (center=True, reflect)."""
    w = _window(n_fft)
    pad = n_fft // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    frames = 1 + (xp.shape[1] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return np.fft.rfft(xp[:, idx] * w, axis=-1).transpose(0, 2, 1)


def istft(spec, n_fft, hop, length):
    """stft의 반대 (창 제곱으로 나눠 겹쳐 더하기) → [채널, length]."""
    w = _window(n_fft)
    frames = spec.shape[-1]
    y = np.fft.irfft(spec.transpose(0, 2, 1), n=n_fft, axis=-1) * w
    total = n_fft + hop * (frames - 1)
    out = np.zeros((spec.shape[0], total))
    norm = np.zeros(total)
    for t in range(frames):
        out[:, t * hop:t * hop + n_fft] += y[:, t]
        norm[t * hop:t * hop + n_fft] += w ** 2
    pad = n_fft // 2
    out = out[:, pad:pad + length]
    norm = norm[pad:pad + length]
    return out / np.where(norm > 1e-11, norm, 1.0)


def _pack(spec, layout):
    """[채널, F, T] 복소수 → [1, 채널×2, F, T]. "sc" = 채널마다 실수 · 허수, "cs" = 실수 모두 · 허수 모두."""
    re, im = spec.real, spec.imag
    if layout == "sc":
        parts = [p for c in range(spec.shape[0]) for p in (re[c], im[c])]
    else:
        parts = list(re) + list(im)
    return np.stack(parts)[None].astype(np.float32)


def _unpack(arr, channels, layout):
    a = arr[0].astype(np.float64)
    if layout == "sc":
        return np.stack([a[2 * c] + 1j * a[2 * c + 1] for c in range(channels)])
    return np.stack([a[c] + 1j * a[channels + c] for c in range(channels)])


class SpecModel:
    def __init__(self, name, path=None, layout=None):
        import onnxruntime as ort

        self.cfg = MODELS[name]
        self.layout, self.out = layout or LAYOUT[name]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path or model_path(name)), opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def _run(self, part):
        """part: [채널, chunk] → 같은 모양 (모델이 뽑은 소리)."""
        c = self.cfg
        spec = stft(part, c["n_fft"], c["hop"])
        y = self.sess.run(None, {self.inp: _pack(spec, self.layout)})[0]
        if y.ndim == 5:  # [1, 소리 수, 채널×2, F, T]: 첫 소리 (보컬)
            y = y[:, 0]
        est = _unpack(y, c["channels"], self.layout)
        if self.out == "mask":
            est = est * spec
        return istft(est, c["n_fft"], c["hop"], part.shape[1])

    def __call__(self, x, progress=None):
        """x: [채널, 샘플] (44.1kHz) → 같은 모양. MSST demix처럼 겹친 조각마다 돌려 가장자리를 서서히 섞는다."""
        c = self.cfg
        chunk, step = c["chunk"], c["chunk"] // c["overlap"]
        fade, border = chunk // 10, chunk - step
        n0 = x.shape[1]
        if n0 > 2 * border:
            x = np.pad(x, ((0, 0), (border, border)), mode="reflect")
        win = np.ones(chunk)
        win[:fade], win[-fade:] = np.linspace(0, 1, fade), np.linspace(1, 0, fade)
        out = np.zeros_like(x, dtype=np.float64)
        count = np.zeros(x.shape[1])
        starts = list(range(0, x.shape[1], step))
        for k, i in enumerate(starts):
            part = x[:, i:i + chunk]
            n = part.shape[1]
            part = np.pad(part, ((0, 0), (0, chunk - n)), mode="reflect" if n > chunk // 2 else "constant")
            y = self._run(part)
            w = win.copy()
            if k == 0:
                w[:fade] = 1
            if k == len(starts) - 1:
                w[-fade:] = 1
            out[:, i:i + n] += y[:, :n] * w[:n]
            count[i:i + n] += w[:n]
            if progress:
                progress((k + 1) / len(starts))
        out /= np.where(count > 0, count, 1.0)
        if n0 > 2 * border:
            out = out[:, border:-border]
        return out


def _to_channels(y, channels):
    y = np.atleast_2d(y)
    if channels == 1:
        return y.mean(0, keepdims=True)
    return np.repeat(y, 2, 0) if y.shape[0] == 1 else y[:2]


def vocals(y, sr, backing=None, progress=None):
    """보컬 y (+ 반주) → 다시 뽑은 보컬 (모노, sr). 모델이 없으면 None."""
    from . import audio

    path = model_path("vocals")
    if path is None:
        return None
    x = audio.resample(y, sr, SR) if sr != SR else np.asarray(y, dtype=np.float64)
    if backing is not None:
        b = audio.resample(backing, sr, SR) if sr != SR else np.asarray(backing, dtype=np.float64)
        b = np.atleast_2d(b).mean(0)
        b = np.pad(b, (0, max(0, len(x) - len(b))))[:len(x)]
        x = x + b
    peak = max(1e-9, np.abs(x).max())
    scale = min(1.0, 0.99 / peak)  # 반주를 더해서 넘치면 줄였다가 되돌린다
    out = SpecModel("vocals", path)(_to_channels(x * scale, 2), progress).mean(0) / scale
    return audio.resample(out, SR, sr) if sr != SR else out


def cache_file(vocal, backing, sr):
    """CACHE가 있으면 (보컬 파일, 반주 파일, sr, 모델)마다 다시 뽑은 보컬을 둘 곳. 없으면 None."""
    if not CACHE or not vocal:
        return None
    h = hashlib.sha256(f"{MODELS['vocals']['sha256']} {LAYOUT['vocals']} {sr}".encode())
    for p in (vocal, backing):
        h.update(Path(p).read_bytes() if p else b"-")
        h.update(b"|")
    return Path(CACHE) / f"vocals-{h.hexdigest()[:32]}.npy"


def vocals_cached(y, sr, vocal=None, backing_path=None, backing=None, progress=None):
    """vocals()와 같지만 CACHE에 같은 파일로 뽑은 것이 있으면 그것을 (float32로 저장)."""
    f = cache_file(vocal, backing_path if backing is not None else None, sr)
    if f is not None and f.exists():
        out = np.load(f)
        if abs(len(out) - len(y)) <= 0.01 * sr:  # 다시 표본을 맞추면 한두 샘플 다를 수 있다
            return out.astype(np.float64)
    out = vocals(y, sr, backing=backing, progress=progress)
    if f is not None and out is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part.npy")
        np.save(tmp, np.asarray(out, dtype=np.float32))
        tmp.replace(f)
    return out


def dereverb(y, sr, progress=None):
    from . import audio

    path = model_path("dereverb")
    if path is None:
        return None
    x = audio.resample(y, sr, SR) if sr != SR else np.asarray(y, dtype=np.float64)
    out = SpecModel("dereverb", path)(_to_channels(x, 1), progress)[0]
    return audio.resample(out, SR, sr) if sr != SR else out


def sdr(est, ref):
    """신호 대 왜곡 (dB, 길이는 짧은 쪽)."""
    n = min(len(est), len(ref))
    e, r = np.asarray(est[:n], float), np.asarray(ref[:n], float)
    return float(10 * np.log10(np.sum(r ** 2) / max(1e-12, np.sum((r - e) ** 2))))

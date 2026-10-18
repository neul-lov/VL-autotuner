"""오디오 읽기 · 쓰기. 분석은 모노 44.1kHz(특징) / 16kHz(음높이)로 한다."""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def load(path, sr=SR):
    """아무 오디오 파일 → (모노 float32, sr). wav · flac · aiff는 바로, mp3 · m4a 등은 ffmpeg으로."""
    path = Path(path)
    try:
        y, file_sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        exe = _ffmpeg()
        if not exe:
            raise ValueError(f"{path.name}: 이 형식은 ffmpeg이 있어야 읽을 수 있어요 (wav · flac은 바로 돼요)")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "in.wav"
            subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), str(out)],
                           check=True)
            y, file_sr = sf.read(out, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        y = resample(y, file_sr, sr)
    return y.astype(np.float32), sr


def resample(y, sr_from, sr_to):
    if sr_from == sr_to:
        return y
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(y, sr_to // g, sr_from // g).astype(np.float32)


def write(path, y, sr=SR):
    sf.write(str(path), np.asarray(y, dtype=np.float32), sr, subtype="PCM_16")

"""오디오 읽기 · 쓰기. 분석은 모노 44.1kHz(특징) / 16kHz(음높이)로 한다. 듣기만 하는 반주는 그대로 둔다."""
import re
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


def _decode(path, out, args, cause):
    """ffmpeg으로 path → out(wav). 못 읽으면 ffmpeg이 말한 까닭을 담아 ValueError."""
    exe = _ffmpeg()
    if not exe:
        raise ValueError(f"{path.name}: 이 형식은 ffmpeg이 있어야 읽을 수 있어요 (wav · flac은 바로 돼요)") from cause
    r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(path), "-vn", *args, str(out)],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        # 첫 줄이 까닭이다 ("[in#0 @ 0x…] Error opening input: Invalid data …", 예전 판은 "경로: Invalid data …")
        lines = [re.sub(r"^\[[^\]]* @ 0x[0-9a-f]+\]\s*", "", ln.strip()) for ln in r.stderr.splitlines() if ln.strip()]
        why = (lines[0] if lines else f"ffmpeg 오류 {r.returncode}").replace(str(path), path.name)
        raise ValueError(f"{path.name}: 오디오를 읽지 못했어요 ({why})") from cause


def load(path, sr=SR):
    """아무 오디오 파일 → (모노 float32, sr). wav · flac · aiff는 바로, mp3 · m4a 등은 ffmpeg으로."""
    path = Path(path)
    try:
        y, file_sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "in.wav"
            _decode(path, out, ["-ac", "1", "-ar", str(sr)], e)
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


# OpenUtau가 그대로 읽는 형식 (Wave.OpenFile: 머리가 RIFF · fLaC인 파일, 확장자가 .aiff인 파일 → NAudio가 푸는 PCM · float)
_PCM = {"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32"}
_AS_IS = {"WAV": (".wav", b"RIFF", _PCM | {"FLOAT", "DOUBLE"}), "WAVEX": (".wav", b"RIFF", _PCM | {"FLOAT", "DOUBLE"}),
          "FLAC": (".flac", b"fLaC", _PCM), "AIFF": (".aiff", b"FORM", {"PCM_16", "PCM_24", "PCM_32"})}


def save_copy(path, dest):
    """듣기만 하는 오디오(반주) → dest(확장자는 여기서 붙인다). 분석용이 아니라서 채널 · 음질을 그대로 둔다:
    wav · flac · aiff는 그대로 복사, 다른 형식은 채널 · 샘플레이트 그대로 24비트 wav로 푼다. → (쓴 파일, 길이 ms)."""
    path, dest = Path(path), Path(dest)
    try:
        info = sf.info(str(path))
        with open(path, "rb") as fp:
            head12 = fp.read(12)
        magic = head12[:4]
    except Exception:
        info = magic = head12 = None
    ext, head, subtypes = _AS_IS.get(info.format if info else None, (None, None, ()))
    # AIFF-C(sowt 등 압축 표시가 있는 것)와 8비트 AIFF는 OpenUtau가 못 읽을 수 있어서 wav로 푼다
    plain = info is not None and (info.format != "AIFF" or head12[8:12] == b"AIFF")
    if ext and magic == head and info.subtype in subtypes and plain:
        out = dest.with_suffix(ext)
        if not (out.exists() and out.samefile(path)):
            shutil.copyfile(path, out)
        return out, info.frames / info.samplerate * 1000
    out = dest.with_suffix(".wav")
    try:  # soundfile이 읽는 다른 형식 (ogg · mp3 · 압축 wav 등). load()와 같은 순서로 풀어야 보컬과 시간이 맞는다
        y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        sf.write(str(out), y, file_sr, subtype="PCM_24")
    except Exception as e:
        _decode(path, out, ["-c:a", "pcm_s24le"], e)
    info = sf.info(str(out))
    return out, info.frames / info.samplerate * 1000

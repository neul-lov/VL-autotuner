"""StemMixer에서 분리해 둔 곡 가져오기: 보컬 트랙, (있으면) 음표 · 가사, 보컬 뺀 반주.

StemMixer 곡 폴더: library/<곡 id>/meta.json, vocals.flac(예전 곡은 wav), user/vocals.*(내 녹음으로 바꾼 것),
midi/vocals.json ({"raw" 또는 "edited": [[시작, 끝, 음, 세기, 가사?], ...], "tempo"}).
"""
import json
import os
import plistlib
from pathlib import Path

import numpy as np


def library():
    """StemMixer 곡 폴더 찾기: 설치된 StemMixer.app이 가리키는 폴더 → 흔한 위치."""
    cands = []
    env = os.environ.get("STEMMIXER_LIBRARY")
    if env:
        cands.append(Path(env))
    for app in (Path("/Applications/StemMixer.app"), Path.home() / "Applications" / "StemMixer.app"):
        try:
            info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
            if info.get("StemMixerBackend"):
                cands.append(Path(info["StemMixerBackend"]) / "library")
        except Exception:
            pass
    for base in (Path.home(), Path.home() / "Documents", Path.home() / "Desktop", Path.home() / "Developer"):
        cands += [base / "stem-mixer" / "library", base / "StemMixer" / "library"]
    return next((c for c in cands if c.is_dir()), None)


def _track(folder, base):
    for ext in (".flac", ".wav"):
        p = folder / f"{base}{ext}"
        if p.exists():
            return p
    return None


def _vocal(folder, meta):
    user = (meta.get("replaced") or {}).get("vocals")
    if user and user.get("use"):
        p = _track(folder / "user", "vocals")
        if p:
            return p
    return _track(folder, "vocals")


def songs():
    lib = library()
    if lib is None:
        return []
    out = []
    for p in lib.glob("*/meta.json"):
        try:
            meta = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        folder = p.parent
        vocal = _vocal(folder, meta)
        if not vocal:
            continue
        notes = folder / "midi" / "vocals.json"
        has_notes, has_lyrics = notes.exists(), False
        if has_notes:
            try:
                d = json.loads(notes.read_text())
                has_lyrics = any(len(n) > 4 and n[4] for n in (d.get("edited") or d.get("raw") or []))
            except (OSError, ValueError):
                pass
        out.append({"id": folder.name, "title": meta.get("title") or folder.name, "duration": meta.get("duration"),
                    "vocal": str(vocal), "notes": has_notes, "lyrics": has_lyrics, "created": p.stat().st_mtime})
    out.sort(key=lambda s: -s["created"])
    return out


def _song_folder(song_id):
    """곡 id → 곡 폴더. 곡 폴더 바로 아래만 ("../" 같은 id로 다른 폴더를 읽지 못하게)."""
    lib = library()
    if lib is None or not song_id or "/" in song_id or "\\" in song_id:
        return None
    folder = (lib / song_id).resolve()
    if folder.parent != lib.resolve() or not (folder / "meta.json").is_file():
        return None
    return folder


def has_song(song_id):
    return _song_folder(song_id) is not None


def prepare(song_id, work):
    """→ (보컬 경로, 음표 파일 경로 또는 None, 반주 경로 또는 None, 제목)."""
    import soundfile as sf

    folder = _song_folder(song_id)
    if folder is None:
        raise ValueError("StemMixer 곡을 찾지 못했어요")
    meta = json.loads((folder / "meta.json").read_text())
    vocal = _vocal(folder, meta)
    notes = folder / "midi" / "vocals.json"
    backing = None
    others = [s for s in meta.get("stems") or [] if s != "vocals"]
    if others:
        mix = None
        for s in others:
            p = _track(folder, s)
            if p is None:
                continue
            y, sr = sf.read(p, dtype="float32", always_2d=True)
            mix = y if mix is None else mix[:len(y)] + y[:len(mix)]
        if mix is not None:
            work = Path(work)
            work.mkdir(parents=True, exist_ok=True)
            backing = work / "backing_from_stemmixer.wav"
            sf.write(backing, np.clip(mix, -1, 1), sr, subtype="PCM_24")  # 반주 음질은 그대로 (16비트로 줄이지 않게)
    return vocal, (notes if notes.exists() else None), backing, meta.get("title") or song_id

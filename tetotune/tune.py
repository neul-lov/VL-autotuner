"""자동 조교: 원래 보컬(+ 가사 · 음표) → 테토가 부르는 OpenUtau 프로젝트.

OpenUtau가 실제로 부르는 음높이 = 음표 높이(계단) + 음표 피치 점(포르타멘토) + 비브라토 + PITD 곡선.
여기서는 피치 점을 평평하게, 비브라토를 0으로 두고 "원래 보컬 음높이 − 음표 계단"을 PITD에 그대로 넣는다.
그래서 꺾기 · 미끄러짐 · 비브라토 · 음 끝 처짐이 사람이 부른 그대로 들어가고, 음표는 깔끔한 정답 음으로 남는다.
크기(dyn) · 숨 섞임(brec) · 긴장(tenc)도 원래 보컬에서 읽어 곡선으로 넣고, 숨 쉰 곳에는 숨 음표를 넣는다.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio, features, lyrics as lyr, notes as nt, pitch, voicebank
from .pitch import HOP

TICKS = 480
INTERVAL = 5  # OpenUtau 곡선 간격 (틱)


@dataclass
class Options:
    bpm: float = 0.0             # 0 = 음표 파일 빠르기, 없으면 120
    transpose: str = "auto"      # "auto" (남자 목소리면 한 옥타브 올림) 또는 반음 수 "-12" ~ "12"
    pitch_follow: float = 1.0    # 원래 음높이 따라가기 (0 = 음표 음만, 1 = 그대로)
    vibrato: float = 1.0         # 떨림(빠른 움직임) 세기
    correct: float = 0.3         # 음 가운데에서 음정 틀린 만큼 바로잡기 (0~1)
    dynamics: float = 0.7        # 크기 따라가기
    breathiness: float = 0.6     # 숨 섞인 소리 따라가기
    tension: float = 0.4         # 힘 준 소리 따라가기
    breaths: str = "auto"        # auto (음원 숨소리, 없으면 원래 숨) · voicebank · original · off
    breath_volume: float = 1.0
    pitch_method: str = "auto"   # auto (RMVPE 있으면) · harvest · rmvpe
    singer: str = ""             # 음원 폴더 경로 (비우면 테토를 찾는다)
    default_lyric: str = "ら"


@dataclass
class Result:
    folder: str
    ustx: str
    notes: int
    breaths: int
    transpose: int
    bpm: float
    singer: str | None
    phonemizer: str
    pitch_method: str
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------- 곡선 도우미
def _interp_gaps(x, mask):
    idx = np.arange(len(x))
    ok = mask & np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    return np.where(ok, x, np.interp(idx, idx[ok], x[ok]))


def _lowpass(x, hz):
    from scipy.signal import butter, filtfilt
    if len(x) < 30:
        return x.copy()
    b, a = butter(2, hz / (0.5 / HOP))
    return filtfilt(b, a, x)


def _smooth(x, sec):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, max(1, int(sec / HOP)))


def _simplify(xs, ys, tol):
    """선 위에 있는 점 지우기 (Ramer–Douglas–Peucker, 높이 차이 기준). 끝점은 남긴다."""
    n = len(xs)
    if n <= 2:
        return xs, ys
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t = (x[a + 1:b] - x[a]) / (x[b] - x[a])
        d = np.abs(y[a + 1:b] - (y[a] + t * (y[b] - y[a])))
        i = int(d.argmax())
        if d[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return [int(v) for v in x[keep]], [int(v) for v in y[keep]]


def _base_cents(notes, t):
    """OpenUtau가 음표만으로 만드는 계단: 그 시각 이후에 끝나는 첫 음표의 높이 (없으면 마지막 음표)."""
    ends = np.array([n.end for n in notes])
    tones = np.array([n.tone for n in notes]) * 100.0
    k = np.searchsorted(ends, t, side="right")
    return tones[np.minimum(k, len(notes) - 1)]


def _note_mask(notes, n, lead=0.0, tail=0.0):
    m = np.zeros(n, dtype=bool)
    for x in notes:
        m[max(0, int((x.start - lead) / HOP)):min(n, int((x.end + tail) / HOP) + 1)] = True
    return m


# ---------------------------------------------------------------- 본체
def tune(vocal, out_dir, lyrics_text="", notes_file=None, backing=None, options=None, progress=lambda p, s: None,
         title=None):
    o = options or Options()
    warnings = []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    progress(0.02, "원래 보컬 읽는 중")
    y, sr = audio.load(vocal)
    if len(y) < sr * 0.5:
        raise ValueError("보컬이 너무 짧아요 (0.5초 이상)")
    progress(0.08, "음높이 읽는 중")
    f0, used = pitch.extract(y, sr, o.pitch_method)
    n = len(f0)
    db = pitch.frame_db(y, sr, n)
    midi, voiced = pitch.clean(f0, db)
    if voiced.sum() * HOP < 0.3:
        raise ValueError("노래하는 소리를 찾지 못했어요 (보컬만 분리한 파일인지 확인해 주세요)")
    progress(0.35, "목소리 특징 읽는 중 (크기 · 숨 · 힘)")
    feats = features.analyze(y, sr, np.where(voiced, pitch.midi_to_hz(np.nan_to_num(midi, nan=60)), 0), voiced)
    # 음높이가 잡힌 숨소리는 노래가 아니다: 무성으로 돌려서 음표가 아니라 숨으로 찾게 한다
    noise = features.noisy_voiced(voiced, feats, db)
    if noise.any():
        voiced = voiced & ~noise
        midi = np.where(voiced, midi, np.nan)
        for k in ("breathiness", "tension"):
            feats[k] = np.where(voiced, feats[k], np.nan)

    # 조옮김: 테토 음역(대략 C4~C6)에 맞게
    med = float(np.nanmedian(midi[voiced]))
    if o.transpose == "auto":
        shift = 12 if med < 57 else 0  # A3보다 낮으면 남자 목소리로 보고 한 옥타브 올림
        if med + shift > 79:
            shift -= 12
    else:
        shift = int(round(float(o.transpose)))

    # 음표
    progress(0.5, "음표 나누는 중")
    lines, w = lyr.split(lyrics_text or "")
    warnings += w
    bpm = o.bpm
    if notes_file:
        notes, file_bpm = nt.read_file(notes_file)
        bpm = bpm or file_bpm
        if not notes:
            raise ValueError("음표 파일에 음표가 없어요")
    else:
        notes, _ = nt.segment(midi, voiced, db, feats["novelty"])
        if not notes:
            raise ValueError("음표를 만들지 못했어요")
    bpm = float(bpm or 120.0)
    if lines:
        phrases = len(nt._phrases(notes))
        if len(lines) != phrases and not notes_file:
            warnings.append(f"노래에서 숨 쉬는 곳(프레이즈)을 {phrases}개 찾았는데 가사는 {len(lines)}줄이에요. "
                            f"가사를 프레이즈마다 한 줄씩 나눠 주면 음절이 더 정확히 들어가요")
        notes = nt.align(notes, lines)
        n_syl = sum(1 for line in lines for s in line if s != "+")  # ー · + (늘이기)는 음절이 아니다
        got = sum(1 for x in notes if x.lyric != "+")
        if got != n_syl:
            warnings.append(f"가사 {n_syl}음절 중 {got}개만 음표에 들어갔어요 (음표가 모자라요). OpenUtau에서 확인해 주세요")
    # 가사 칸이 비면 음표 파일(StemMixer · ust · MIDI)의 가사를 쓰니까 말도 거기서 본다 (빈 가사를 채우기 전에)
    lang = lyr.language(lyrics_text) if lines else lyr.language("".join(x.lyric for x in notes))
    for i, x in enumerate(notes):
        if not x.lyric:
            # 가사가 없으면: 자음 · 쉼표 · 크기 꺼짐이 있는 곳만 새 음절, 음만 바뀌는 곳은 이어 부르기
            x.lyric = o.default_lyric if (i == 0 or x.strength >= 3 or notes_file) else "+"
        x.tone += shift

    # 가수
    singer_id, info = voicebank.find(o.singer or None)
    if singer_id is None:
        warnings.append("OpenUtau 가수 폴더에서 테토를 찾지 못했어요. OpenUtau에서 가수를 골라 주세요")
        info = {"phonemizer": voicebank.PRESAMP, "breath_in": None, "breath_out": None, "japanese": True, "name": None,
                "classic": True}
    phonemizer = info["phonemizer"]
    if lang == "ko" and info.get("japanese"):
        phonemizer = voicebank.KO_TO_JA

    # 숨
    progress(0.6, "숨소리 찾는 중")
    breaths = features.find_breaths(db, voiced, feats) if o.breaths != "off" else []
    alias_in, alias_out = info.get("breath_in"), info.get("breath_out")
    mode = o.breaths
    if mode == "auto":
        mode = "voicebank" if (alias_in or alias_out) else "original"
    if mode == "voicebank" and not (alias_in or alias_out):
        warnings.append("이 음원에 숨소리(息 · 吸 · br)가 없어서 원래 보컬의 숨을 따로 넣었어요")
        mode = "original"
    breath_notes = []
    if mode == "voicebank":
        sing_db = float(np.median(db[voiced]))
        for a, b, bdb, inhale in breaths:
            nxt = next((x for x in notes if x.start >= b - 0.01), None)
            prv = next((x for x in reversed(notes) if x.end <= a + 0.01), None)
            a2 = max(a, prv.end if prv else 0.0)
            b2 = min(b, nxt.start if nxt else b)
            if b2 - a2 < 0.08:
                continue
            tone = (nxt or prv or notes[0]).tone
            # 원래 보컬에서 숨이 노래보다 작은 만큼 = 음원 숨 샘플이 모음보다 작은 만큼 + vol
            gap = info.get("breath_gap_db")
            want = bdb - sing_db
            # (−3dB: 시험 렌더링에서 숨이 원래보다 3~6dB 크게 나와서 맞춘 값)
            gain = want - gap - 3 if gap is not None else want + 15
            vol = int(np.clip(100 * o.breath_volume * 10 ** (gain / 20), 5, 200))
            alias = (alias_in or alias_out) if inhale else (alias_out or alias_in)
            breath_notes.append(nt.Note(a2, b2, tone, alias))
            breath_notes[-1].strength = vol  # vol을 잠깐 여기에 둔다

    # 곡선
    progress(0.7, "곡선 만드는 중")
    t = np.arange(n) * HOP
    all_notes = sorted(notes + breath_notes, key=lambda x: x.start)
    target = (midi + shift) * 100.0
    sing_mask = voiced & _note_mask(notes, n, 0.05, 0.05)
    target = _interp_gaps(target, sing_mask)
    base = _base_cents(all_notes, t)
    dev = target - base
    lp = _lowpass(dev, 3.0)
    fine = dev - lp
    # 음 가운데(앞뒤 80ms 또는 25% 뺀 곳)에서만 음정 틀림을 바로잡는다 → 미끄러짐 · 꺾기는 그대로
    steady = np.zeros(n)
    for x in notes:
        edge = min(0.08, 0.25 * (x.end - x.start))
        steady[int((x.start + edge) / HOP):int((x.end - edge) / HOP)] = 1.0
    steady = _smooth(steady, 0.06)
    pitd = o.pitch_follow * (lp * (1 - o.correct * steady) + o.vibrato * fine)

    near = _smooth(_note_mask(notes, n, 0.1, 0.06).astype(float), 0.04)
    ref = float(np.median(db[voiced]))
    # 크기는 겉 윤곽만 (80ms 최댓값): 음절 사이 자음에서 잠깐 꺼지는 건 음원 자음이 이미 갖고 있다
    from scipy.ndimage import maximum_filter1d
    env = _smooth(maximum_filter1d(_interp_gaps(db, sing_mask), int(0.08 / HOP)), 0.08)
    rel = env - ref
    rel = np.where(rel < -6, -6 + (rel + 6) * 0.5, rel)  # 아주 작은 곳은 반만: 음원 소리가 묻히지 않게
    dyn = np.clip(rel * 10 * o.dynamics, -120, 60) * near

    def zcurve(raw, scale, lo, hi):
        v = raw[sing_mask & np.isfinite(raw)]
        if len(v) < 10:
            return np.zeros(n)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        z = (raw - q2) / max(1e-6, (q3 - q1) / 1.35)
        z = _interp_gaps(np.where(np.isfinite(z), z, np.nan), sing_mask)
        return np.clip(_smooth(z, 0.06) * scale, lo, hi) * near

    brec = zcurve(feats["breathiness"], 30 * o.breathiness, -50, 70)
    tenc = zcurve(feats["tension"], 25 * o.tension, -50, 50)

    tps = bpm / 60 * TICKS  # 초당 틱
    first = max(0.0, all_notes[0].start - 0.5)
    last = all_notes[-1].end + 0.3
    xs = np.arange(int(first * tps) // INTERVAL * INTERVAL, int(last * tps) + INTERVAL, INTERVAL)
    sec = xs / tps

    def curve(abbr, vals, lo, hi, tol):
        ys = np.clip(np.round(np.interp(sec, t, vals)), lo, hi).astype(int)
        cx, cy = _simplify(list(xs), list(ys), tol)
        return {"xs": cx, "ys": cy, "abbr": abbr}

    curves = [curve("pitd", pitd, -1200, 1200, 1.5), curve("dyn", dyn, -240, 120, 2)]
    if o.breathiness > 0:
        curves.append(curve("brec", brec, -100, 100, 2))
    if o.tension > 0:
        curves.append(curve("tenc", tenc, -100, 100, 2))

    # 파일
    progress(0.85, "OpenUtau 프로젝트 쓰는 중")
    guide = out / "guide_vocals.wav"
    audio.write(guide, y, sr)
    waves = [("원래 보컬 (참고용 · 음소거)", guide.name, len(y) / sr * 1000, True, 0)]
    if backing:  # 반주는 듣기만 하니까 모노로 줄이지 않고 그대로 (wav · flac · aiff는 복사, 다른 형식은 24비트 wav)
        file, ms = audio.save_copy(backing, out / "backing")
        waves.append(("반주", file.name, ms, False, -3))
    if mode == "original" and breaths:
        audio.write(out / "breaths.wav", _breath_track(y, sr, breaths, o.breath_volume), sr)
        waves.append(("숨소리 (원래 보컬)", "breaths.wav", len(y) / sr * 1000, False, -2))
    name = title or Path(vocal).stem
    _write_ustx(out / "project.ustx", name, bpm, all_notes, set(id(b) for b in breath_notes), curves, waves,
                singer_id, phonemizer, info.get("classic", True))

    preview = {
        "hop": HOP, "bpm": bpm, "shift": shift,
        "source": [None if not ok else round(float(v + shift), 2) for v, ok in zip(midi[::2], voiced[::2])],
        "pitch": [round(float((b + p) / 100), 2) for b, p in zip(base[::2], pitd[::2])],
        "dyn": [round(float(v), 1) for v in dyn[::4]],
        "notes": [[round(x.start, 3), round(x.end, 3), x.tone, x.lyric] for x in notes],
        "breaths": [[round(b.start, 3), round(b.end, 3)] for b in breath_notes] if mode == "voicebank"
        else [[round(a, 3), round(b, 3)] for a, b, *_ in breaths],
        "duration": len(y) / sr,
        "phrases": len(nt._phrases([x for x in notes])),
    }
    (out / "preview.json").write_text(json.dumps(preview, ensure_ascii=False), encoding="utf-8")
    res = Result(str(out), str(out / "project.ustx"), len(notes),
                 len(breath_notes) if mode == "voicebank" else (len(breaths) if mode == "original" else 0),
                 shift, bpm, info.get("name"), phonemizer, used, warnings)
    (out / "report.json").write_text(json.dumps({**asdict(res), "options": asdict(o)}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    progress(1.0, "끝")
    return res


def _breath_track(y, sr, breaths, gain):
    """원래 보컬에서 숨 쉰 곳만 남긴 트랙 (앞뒤 20ms 페이드)."""
    out = np.zeros_like(y)
    fade = int(0.02 * sr)
    for a, b, *_ in breaths:
        i, j = int(a * sr), min(len(y), int(b * sr))
        if j - i <= 2 * fade:
            continue
        env = np.ones(j - i)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        out[i:j] = y[i:j] * env * gain
    return out


def _write_ustx(path, name, bpm, notes, breath_ids, curves, waves, singer, phonemizer, classic=True):
    import yaml

    tps = bpm / 60 * TICKS
    vnotes = []
    for x in notes:
        a = int(round(x.start * tps))
        b = int(round(x.end * tps))
        if vnotes and a < vnotes[-1]["position"] + vnotes[-1]["duration"]:  # 겹치면 앞 음표를 줄인다
            a = max(a, vnotes[-1]["position"] + 15)
            vnotes[-1]["duration"] = a - vnotes[-1]["position"]
        if b - a < 15:
            continue
        exps = []
        if id(x) in breath_ids:
            exps.append({"index": 0, "abbr": "vol", "value": int(x.strength)})
        vnotes.append({
            "position": a, "duration": b - a, "tone": int(x.tone), "lyric": x.lyric,
            # 평평한 피치 점 + 비브라토 0: 음높이는 PITD 곡선이 다 맡는다
            "pitch": {"data": [{"x": 0, "y": 0, "shape": "l"}, {"x": 5, "y": 0, "shape": "l"}], "snap_first": True},
            "vibrato": {"length": 0, "period": 175, "depth": 25, "in": 10, "out": 10, "shift": 0, "drift": 0, "vol_link": 0},
            "phoneme_expressions": exps, "phoneme_overrides": [],
        })
    if not vnotes:
        raise ValueError("음표가 너무 짧아서 프로젝트를 만들지 못했어요 (빠르기를 확인해 주세요)")
    end = max(n["position"] + n["duration"] for n in vnotes)
    track = {"singer": singer or "", "phonemizer": phonemizer}
    if classic:  # UTAU 음원만 WORLDLINE-R. DiffSinger · ENUNU 등은 비워서 OpenUtau가 그 가수의 기본 렌더러를 고르게
        track["renderer_settings"] = {"renderer": "WORLDLINE-R"}
    tracks = [{**track, "track_name": "테토 (자동 조교)", "track_color": "Red", "mute": False, "solo": False,
               "volume": 0, "pan": 0}]
    wave_parts = []
    for i, (track_name, file, ms, mute, vol) in enumerate(waves, start=1):
        tracks.append({"singer": None, "phonemizer": None, "track_name": track_name, "track_color": "Blue",
                       "mute": mute, "solo": False, "volume": vol, "pan": 0})
        wave_parts.append({"name": track_name, "comment": "", "track_no": i, "position": 0,
                           "relative_path": file, "file_duration_ms": float(ms), "skip_ms": 0.0, "trim_ms": 0.0})
    project = {
        "name": name, "comment": "TetoTune 자동 조교. 1마디 첫 박 = 원래 보컬 0초",
        "output_dir": "Vocal", "cache_dir": "UCache", "ustx_version": "0.6", "key": 0,
        "time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}],
        "tempos": [{"position": 0, "bpm": float(round(bpm, 3))}],
        "tracks": tracks,
        "voice_parts": [{"name": name, "comment": "", "track_no": 0, "position": 0,
                         "duration": end + TICKS, "notes": vnotes, "curves": curves}],
        "wave_parts": wave_parts,
    }
    Path(path).write_text(yaml.safe_dump(project, allow_unicode=True, sort_keys=False), encoding="utf-8")

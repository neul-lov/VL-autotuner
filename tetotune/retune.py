"""구간 다시 조교하기 (앱에서): 고른 구간만 설정을 바꿔 다시 조교해서 project.ustx의 그 구간을 바꾼다.

구간은 소절(쉼으로 나뉜 덩어리) 경계로 넓힌다 (음표 중간에서 잘리지 않게). 원래 보컬에서 그 구간(+앞뒤 0.3초)만
잘라 조교하고, 곡 전체의 크기 · 숨 섞임 · 힘 기준(report.json "context")과 조옮김 · 빠르기는 그대로 쓴다.
- 음표는 그대로 (곡선만 새 설정으로): 지금 음표 · 가사를 그 구간의 음표 파일로 넣는다
- 음표도 다시 나누기: 가사(비우면 그 구간의 지금 음절)로 음표 나누기부터 다시
바꾸기 전 상태는 되돌리기 기록(.history)에 남는다. OpenUtau가 있으면 그 구간을 테토로 불러 보고 다듬는다.
"""
import json
import os
import shutil
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import yaml

from . import audio, edit, oupitch
from .notes import REST_GAP
from .tune import Options, _simplify, tune

PAD = 0.3     # 초. 잘라 낸 보컬 앞뒤 여유 (조교 · 크기 윤곽이 구간 끝에서 흔들리지 않게)
EDGE = 0.05   # 초. 구간 끝에서 이만큼 안쪽까지만 바꾼다


def _phrases(p):
    """부르는 음표를 소절로 묶어 [(시작초, 끝초, [부르는 음표 번호])]."""
    out = []
    sung = p.sung()
    for j, i in enumerate(sung):
        x = p.notes[i]
        s, e = x["position"] / p.tps, (x["position"] + x["duration"]) / p.tps
        if out and s - out[-1][1] <= REST_GAP:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2].append(j)
        else:
            out.append([s, e, [j]])
    return out


def snap(p, start, end):
    """고른 구간 → 걸치는 소절 전체 (a, b, 그 소절들)."""
    hit = [ph for ph in _phrases(p) if ph[0] < end and ph[1] > start]
    if not hit:
        return start, end, []
    return min(ph[0] for ph in hit), max(ph[1] for ph in hit), hit


def _excerpt_notes_ustx(p, hit, offset, path, transpose):
    """지금 음표(그 소절들)를 잘라 낸 보컬 시간으로 옮긴 음표 파일 (조교가 다시 조옮김하니까 조옮김 전 높이로)."""
    sung = p.sung()
    shift = int(round(offset * p.tps))
    notes = []
    for ph in hit:
        for j in ph[2]:
            x = dict(p.notes[sung[j]])
            x["position"] = x["position"] - shift
            x["tone"] = int(x["tone"]) - transpose
            if x["position"] >= 0:
                notes.append(x)
    proj = {"name": "구간", "ustx_version": "0.6", "tempos": [{"position": 0, "bpm": p.bpm}],
            "time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}],
            "tracks": [{"singer": "", "phonemizer": ""}],
            "voice_parts": [{"name": "구간", "track_no": 0, "position": 0, "notes": notes, "curves": []}]}
    Path(path).write_text(yaml.safe_dump(proj, allow_unicode=True, sort_keys=False), encoding="utf-8")


def same_renderer(p, ustx):
    """구간 프로젝트도 곡 프로젝트의 테토 트랙과 같은 렌더러로 (다듬기가 실제로 부를 소리로 재게).
    예전 판에서 만든 곡(WORLDLINE-R)이나 OpenUtau · EasyUtau에서 렌더러를 바꾼 곡도 그대로 따른다."""
    rs = (p.data.get("tracks") or [{}])[0].get("renderer_settings")
    proj = yaml.safe_load(Path(ustx).read_text(encoding="utf-8"))
    track = (proj.get("tracks") or [{}])[0]
    if not rs or not track.get("renderer_settings"):  # DiffSinger 등: 가수의 기본 렌더러 그대로
        return
    track["renderer_settings"] = dict(rs)
    Path(ustx).write_text(yaml.safe_dump(proj, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _lyrics_of(p, hit):
    sung = p.sung()
    lines = []
    for ph in hit:
        syl = [p.notes[sung[j]]["lyric"] for j in ph[2] if p.notes[sung[j]]["lyric"] != "+"]
        if syl:
            lines.append(" ".join(syl))
    return "\n".join(lines)


def _splice_curve(cur, new, a, b):
    """cur 곡선의 틱 a~b를 new 곡선 값으로 (5틱마다 다시 그려 선 위의 점은 지운다)."""
    grid = np.arange(a // 5 * 5, b + 5, 5)
    nx, ny = np.asarray(new["xs"], float), np.asarray(new["ys"], float)
    vals = np.round(np.interp(grid, nx, ny, left=0, right=0)).astype(int) if len(nx) else np.zeros(len(grid), int)
    keep = [(int(x), int(y)) for x, y in zip(cur.get("xs", []), cur.get("ys", [])) if x < grid[0] or x > grid[-1]]
    pts = sorted(keep + list(zip(grid.tolist(), vals.tolist())))
    cur["xs"], cur["ys"] = _simplify([q[0] for q in pts], [q[1] for q in pts], 1.5)


def _splice_array(old, new, a, b, step, offset):
    """미리 보기 곡선 배열(step초 간격)의 a~b초를 new(offset초부터 시작하는 같은 간격 배열)로."""
    out = list(old)
    for k in range(max(0, int(a / step)), min(len(out), int(b / step) + 1)):
        i = int(round((k * step - offset) / step))
        if 0 <= i < len(new):
            out[k] = new[i]
    return out


def _splice_wav(path, new_path, a, b, offset, out=None):
    """테토 미리 듣기 wav의 a~b초를 새 렌더(새 렌더의 0초 = 곡의 offset초)로 (끝은 20ms 겹쳐 잇기) → out (없으면 path)에 쓴다.
    a가 0보다 앞이거나 b가 wav 끝보다 뒤여도 (곡 첫 · 끝 음절) wav 안쪽만 바꾼다."""
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    z, sr2 = sf.read(str(new_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if z.ndim > 1:
        z = z.mean(axis=1)
    if sr2 != sr:
        z = audio.resample(z, sr2, sr)
    i, j = max(0, int(a * sr)), min(len(y), int(b * sr))
    if j <= i:
        sf.write(str(out or path), y, sr, subtype="PCM_16")
        return
    k0 = i - int(round(offset * sr))  # 새 렌더에서 i에 해당하는 곳
    seg = z[max(0, k0):max(0, k0 + (j - i))]
    if k0 < 0:  # 새 렌더가 i보다 늦게 시작하면 앞을 조용히
        seg = np.concatenate([np.zeros(min(-k0, j - i), np.float32), seg])
    seg = seg[:j - i]
    if len(seg) < j - i:
        seg = np.pad(seg, (0, j - i - len(seg)))
    fade = min(int(0.02 * sr), (j - i) // 4)
    w = np.ones(j - i, dtype=np.float32)
    if fade > 0:
        w[:fade] = np.linspace(0, 1, fade)
        w[-fade:] = np.linspace(1, 0, fade)
    y[i:j] = y[i:j] * (1 - w) + seg * w
    sf.write(str(out or path), y, sr, subtype="PCM_16")


def retune(folder, start, end, options=None, lyrics_text=None, renote=False, progress=lambda p, s: None):
    """→ 새 미리 보기 (+ "range": 실제로 바꾼 구간). 못 하면 ValueError (아무것도 바꾸지 않는다)."""
    folder = Path(folder)
    p = edit.Project(folder)
    report = p.report
    a, b, hit = snap(p, float(start), float(end))
    if b - a < 0.2:
        raise ValueError("구간이 너무 짧아요")
    y, sr = audio.load(folder / "guide_vocals.wav")
    off = max(0.0, a - PAD)
    clip = y[int(off * sr):int(min(len(y) / sr, b + PAD) * sr)]
    base = dict(report.get("options") or {})
    base.update(options or {})
    names = {f.name for f in fields(Options)}
    o = Options(**{k: v for k, v in base.items() if k in names})
    o.transpose = str(int(report.get("transpose", 0)))
    o.bpm = p.bpm
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav = tmp / "clip.wav"
        audio.write(wav, clip, sr)
        notes_file = None
        text = lyrics_text if lyrics_text is not None and lyrics_text.strip() else _lyrics_of(p, hit)
        if not renote and hit:
            notes_file = tmp / "notes.ustx"
            _excerpt_notes_ustx(p, hit, off, notes_file, int(report.get("transpose", 0)))
            text = ""  # 음표 파일의 가사를 그대로
        out = tmp / "out"
        progress(0.1, "구간 다시 조교하는 중")
        tune(wav, out, text, str(notes_file) if notes_file else None, None, o,
             progress=lambda v, s: progress(0.1 + 0.4 * v, s), title="구간", context=report.get("context"))
        same_renderer(p, out / "project.ustx")
        warning = None
        if o.refine != "off":  # 못 다듬어도 다시 조교한 곡선은 쓴다
            from . import refine
            warning = refine.for_result(out, lambda v, s: progress(0.5 + 0.4 * v, s))
        new = yaml.safe_load((out / "project.ustx").read_text(encoding="utf-8"))
        new_prev = json.loads((out / "preview.json").read_text(encoding="utf-8"))
        progress(0.92, "구간 바꾸는 중")
        shift = int(round(off * p.tps))
        lo, hi = int(round((a - EDGE) * p.tps)), int(round((b + EDGE) * p.tps))
        part = p.part
        grid, held = p.sung_pitch()  # 구간 밖은 지금 부르는 음 높이 그대로 (음표가 바뀌어도 PITD를 다시 써서 지킨다)
        new_notes = sorted(new["voice_parts"][0]["notes"], key=lambda x: x["position"])
        new_pitd = next((c for c in new["voice_parts"][0].get("curves", []) if c["abbr"] == "pitd"), None)
        inside = (grid >= lo) & (grid <= hi)
        if new_notes and inside.any():
            held = np.where(inside, oupitch.sung(new_notes, new_pitd, p.tps, grid - shift), held)
        keep = [x for x in part["notes"] if not lo <= x["position"] <= hi]
        add = []
        for x in new["voice_parts"][0]["notes"]:
            x = dict(x)
            x["position"] += shift
            if lo <= x["position"] <= hi:
                add.append(x)
        if not add:
            raise ValueError("이 구간에서 음표를 만들지 못했어요")
        merged = sorted(keep + add, key=lambda x: x["position"])
        for x, nxt in zip(merged, merged[1:]):  # 이웃 음표와 겹치면 앞 음표를 줄인다
            if x["position"] + x["duration"] > nxt["position"]:
                x["duration"] = max(15, nxt["position"] - x["position"])
        part["notes"] = merged
        cur = {c["abbr"]: c for c in part.get("curves", [])}
        for c in new["voice_parts"][0].get("curves", []):
            if c["abbr"] == "pitd":
                continue  # 아래 keep_pitch가 다시 쓴다
            c = {"abbr": c["abbr"], "xs": [x + shift for x in c["xs"]], "ys": list(c["ys"])}
            if c["abbr"] not in cur:
                cur[c["abbr"]] = {"abbr": c["abbr"], "xs": [], "ys": []}
                part.setdefault("curves", []).append(cur[c["abbr"]])
            _splice_curve(cur[c["abbr"]], c, lo, hi)
        p.keep_pitch(grid, held)
        # 바꿀 것을 모두 결과 폴더 안의 임시 파일로 먼저 만들고, 되돌리기 기록(소리 포함)을 남긴 뒤 한 번에 바꿔 넣는다.
        # 중간에 실패하면 기록으로 되돌려서 소리 · 음표 · 미리 보기가 서로 어긋나지 않게
        prev_path = folder / "preview.json"
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        step_p, step_d = prev["hop"] * 2, prev["hop"] * 4
        prev["pitch"] = _splice_array(prev["pitch"], new_prev["pitch"], a - EDGE, b + EDGE, step_p, off)
        prev["dyn"] = _splice_array(prev["dyn"], new_prev["dyn"], a - EDGE, b + EDGE, step_d, off)
        moved = [[round(f[0] + off, 3), round(f[1] + off, 3)] + f[2:] for f in new_prev.get("flags", [])]
        prev["flags"] = sorted([f for f in prev.get("flags", []) if not (f[0] < b and f[1] > a)] +
                               [f for f in moved if f[0] < b + EDGE and f[1] > a - EDGE], key=lambda f: f[0])
        spliced = (out / "teto_preview.wav").exists() and (folder / "teto_preview.wav").exists()
        staged = {}
        try:
            if spliced:
                staged["teto_preview.wav"] = folder / ".retune_teto_preview.wav"
                _splice_wav(folder / "teto_preview.wav", out / "teto_preview.wav", a - EDGE, b + EDGE, off,
                            out=staged["teto_preview.wav"])
            staged["project.ustx"] = folder / ".retune_project.ustx"
            staged["project.ustx"].write_text(p.dump(), encoding="utf-8")
            staged["preview.json"] = folder / ".retune_preview.json"
            staged["preview.json"].write_text(json.dumps(prev, ensure_ascii=False), encoding="utf-8")
            edit.snapshot(folder, wav=spliced)
            try:
                for name in ("teto_preview.wav", "preview.json", "project.ustx"):
                    if name in staged:
                        os.replace(staged[name], folder / name)
            except BaseException:
                edit.undo(folder)
                raise
        finally:
            for f in staged.values():
                f.unlink(missing_ok=True)
    prev = edit.refresh_preview(folder, stale=not spliced)
    prev["range"] = [round(a, 3), round(b, 3)]
    if warning:
        prev["warning"] = warning
    return prev

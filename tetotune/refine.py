"""불러 보고 고치기: 조교한 프로젝트를 OpenUtau로 실제로 렌더해서(소리는 재생하지 않는다) 원래 보컬과 비교하고,
다른 만큼 음 높이(PITD) · 크기(DYN) 곡선을 고쳐 다시 렌더한다. 가장 닮은 회차를 남긴다.

테토 음원은 음절마다 크기 · 음 높이 버릇이 있어서(콧소리 · 자음 앞에서 작아짐, 음 바뀌는 곳에서 처짐) 곡선을
원래 보컬대로만 넣으면 그 버릇이 겹쳐 뚝뚝 끊기거나 음이 떨어져 들린다. 실제 렌더를 재서 그만큼 빼 준다.

render(ustx 경로) → 테토 트랙 wav 경로 (openutau.py의 Renderer, CI에서는 render.sh)
"""
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from . import audio, confidence, pitch
from .pitch import HOP

ROUNDS = 3          # 고치는 횟수 (렌더는 ROUNDS + 1번). 2번보다 3번이 끊김 · 크기 차이가 조금 더 줄었다
PITCH_GAIN = 0.8    # 한 번에 고치는 비율 (렌더 음높이 읽기 오차에 흔들리지 않게 조금 덜)
DYN_GAIN = 0.7
PITCH_STEP = 200.0  # 센트. 한 번에 고치는 최대
PITCH_MIN = 30.0    # 센트. 이보다 작은 음 높이 차이는 고치지 않는다 (렌더 음높이 읽기 잡음)
DYN_STEP = 9.0      # dB
# 음 높이 목표: "fixed" = 처음 조교한 음 높이 (렌더를 잘못 읽은 만큼이 회차마다 쌓이지 않게),
# "off" = 음 높이는 고치지 않음 (WORLDLINE-R은 준 음 높이를 그대로 부른다), "old" = 예전 (지난 회차 곡선이 목표)
PITCH_FIX = "fixed"
PITCH_TOTAL = 100.0  # 센트. "fixed"일 때 처음 곡선에서 고치는 최대
DYN_TOTAL = 6.0      # dB. 처음 크기 곡선에서 고치는 최대 (0 = 제한 없음)
DYN_VOICED = True    # 크기는 원래 보컬 · 렌더가 둘 다 소리 나는 곳에서만 (테토 자음을 키우거나 모음을 줄이지 않게)


def _runs(mask):
    return pitch._runs(mask)


def _lowpass(x, hz):
    from scipy.signal import butter, filtfilt
    if len(x) < 30:
        return x
    b, a = butter(2, hz / (0.5 / HOP))
    return filtfilt(b, a, x)


def _fill_short(x, ok, max_gap):
    """ok가 아닌 곳: 양쪽이 ok이고 max_gap 프레임 안이면 이어 긋고, 아니면 0."""
    out = np.where(ok, x, 0.0)
    idx = np.arange(len(x))
    if not ok.any():
        return out
    for a, b in _runs(~ok):
        if a > 0 and b < len(x) and b - a <= max_gap:
            out[a:b] = np.interp(idx[a:b], [a - 1, b], [x[a - 1], x[b]])
    return out


def _envelope(db, mask, win):
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    idx = np.arange(len(db))
    filled = np.interp(idx, idx[mask], db[mask]) if mask.any() else db
    return uniform_filter1d(maximum_filter1d(filled, max(1, int(win / HOP))), max(1, int(win / HOP)))


def measure(wav, n, loaded=None):
    """렌더 wav → (음높이 센트 · 유성 · 크기 dB), 조교할 때와 같은 방법으로 (Harvest). loaded: 이미 읽은 (y, sr)."""
    y, sr = loaded or audio.load(wav)
    f0, _ = pitch.extract(y, sr, "harvest")
    f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
    db = pitch.frame_db(y, sr, n)
    m, v = pitch.clean(f0, db)
    return np.where(v, m * 100, np.nan), v, db


def _goal(z):
    """음 높이 목표 (센트): 처음 조교한 곡선 (PITCH_FIX가 "old"면 지금 곡선)."""
    if PITCH_FIX != "old" and "pitd0" in z:
        return z["base"] + z["pitd0"]
    return z["base"] + z["pitd"]


def score(z, cents, voiced, db):
    """닮은 정도 (작을수록): 음 높이 오차 90% (센트) / 40 + 크기 오차 평균 (dB) / 3."""
    want = _goal(z)
    both = z["sing"] & z["voiced"] & voiced
    if both.sum() < 20:
        return 99.0, {}
    err = np.abs(cents[both] - want[both])
    rel = _rel(z, db)
    m = z["sing"]
    derr = np.abs(rel[m] - z["rel"][m])
    s = float(np.percentile(err, 90) / 40 + np.mean(derr) / 3)
    return s, {"pitch_p90": round(float(np.percentile(err, 90)), 1), "pitch_median": round(float(np.median(err)), 1),
               "db_mean": round(float(np.mean(derr)), 2), "voiced": round(float(both.sum() / max(1, (z["sing"] & z["voiced"]).sum())), 3)}


def _rel(z, db):
    m = z["sing"]
    env = _envelope(db, m, 0.02)
    return env - float(np.median(db[m & z["voiced"]] if (m & z["voiced"]).any() else db[m]))


def step(z, cents, voiced, db):
    """한 번 고치기: z의 pitd · dyn을 바꾼다."""
    n = len(cents)
    want = _goal(z)
    # 음 높이: 원래 보컬도 렌더도 소리 나는 곳, 소리 시작 · 끝 30ms는 빼고 (자음 · 숨에서 음높이 읽기가 튄다)
    ok = z["sing"] & z["voiced"] & voiced
    edge = int(0.03 / HOP)
    for a, b in _runs(voiced):
        ok[a:min(b, a + edge)] = False
        ok[max(a, b - edge):b] = False
    res = np.where(ok, want - cents, 0.0)
    res = np.where(np.abs(res) > 600, 0.0, res)  # 옥타브 틀림(읽기 오류)은 고치지 않는다
    from scipy.ndimage import median_filter
    res = median_filter(_fill_short(res, ok, int(0.08 / HOP)), size=int(0.06 / HOP) | 1, mode="nearest")
    # 잠깐 튀는 읽기 잡음은 두고, PITCH_MIN센트 넘게 40ms 넘게 이어지는 차이(음이 처지거나 뜨는 곳)만 고친다
    keep = np.zeros(n, dtype=bool)
    for a, b in _runs((np.abs(res) > PITCH_MIN) & ok):
        if b - a >= int(0.04 / HOP):
            keep[max(0, a - 4):b + 4] = True
    res = np.clip(_lowpass(np.where(keep, res, 0.0), 6.0), -PITCH_STEP, PITCH_STEP)
    if PITCH_FIX != "off":
        z["pitd"] = z["pitd"] + PITCH_GAIN * res
        if PITCH_FIX == "fixed" and "pitd0" in z:
            z["pitd"] = z["pitd0"] + np.clip(z["pitd"] - z["pitd0"], -PITCH_TOTAL, PITCH_TOTAL)
    # 크기: 원래 보컬이 노래하는 곳에서 렌더 크기 윤곽이 목표(조교할 때 원래 보컬 크기)와 같아지게
    m = z["sing"] & z["voiced"] & voiced if DYN_VOICED else z["sing"]
    d = np.where(m, z["rel"] - _rel(z, db), 0.0)
    d = np.clip(_fill_short(d, m, int(0.1 / HOP)), -DYN_STEP, DYN_STEP)
    from scipy.ndimage import uniform_filter1d
    d = uniform_filter1d(d, int(0.04 / HOP))
    dyn = z["dyn"] + DYN_GAIN * 10 * d * z["near"]
    if DYN_TOTAL > 0 and "dyn0" in z:
        dyn = z["dyn0"] + np.clip(dyn - z["dyn0"], -10 * DYN_TOTAL, 10 * DYN_TOTAL)
    z["dyn"] = np.clip(dyn, -200, 120)  # −240은 OpenUtau에서 소리 끔
    return z


def write_curves(ustx, z):
    from .tune import TICKS, make_curves, pitd_curve

    proj = yaml.safe_load(Path(ustx).read_text(encoding="utf-8"))
    part = proj["voice_parts"][0]
    t = np.arange(len(z["pitd"])) * HOP
    if int(z.get("exact", 0)):  # PITD는 OpenUtau가 음표로 만드는 음 높이 기준으로 틱마다 (tune.pitd_curve)
        tps = float(z["bpm"]) / 60 * TICKS
        notes = sorted(part["notes"], key=lambda x: x["position"])
        curves = [pitd_curve(t, z["base"] + z["pitd"], notes, [x["pitch"] for x in notes], tps, z["span"])] + \
            make_curves(t, {"dyn": z["dyn"]}, float(z["bpm"]), z["span"])
    else:
        curves = make_curves(t, {"pitd": z["pitd"], "dyn": z["dyn"]}, float(z["bpm"]), z["span"])
    new = {c["abbr"]: c for c in curves}
    part["curves"] = [new.get(c["abbr"], c) for c in part["curves"]] + \
        [c for k, c in new.items() if k not in {c["abbr"] for c in part["curves"]}]
    Path(ustx).write_text(yaml.safe_dump(proj, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _render_checked(render, ustx, tries=2):
    """render를 부르고 소리가 있는 wav면 (경로, 읽은 (y, sr)) (재는 데 다시 읽지 않게), 두 번 다 비었으면 (None, None).
    못 한 까닭(렌더 오류)은 last_render_error에 남긴다 (앱 경고에 보여 주려고)."""
    global last_render_error
    last_render_error = None
    for _ in range(tries):
        try:
            wav = Path(render(ustx))
            y, sr = audio.load(wav)
        except Exception as e:
            last_render_error = str(e)
            print(f"테토로 불러 보기 실패: {e}", flush=True)
            continue
        if len(y) > sr * 0.5 and float(np.abs(y).max()) > 1e-3:
            return wav, (y, sr)
        last_render_error = "렌더가 비었어요"
    return None, None


last_render_error = None


def refine(folder, render, rounds=None, progress=lambda p, s: None):
    """→ {"rounds": [회차마다 점수], "best": 고른 회차, "preview": 남긴 렌더 wav 이름 또는 None}.
    고른 회차의 곡선을 project.ustx에 남기고 그 렌더를 teto_preview.wav로 둔다 (앱 미리 듣기)."""
    folder = Path(folder)
    rounds = ROUNDS if rounds is None else rounds
    z = {k: v for k, v in np.load(folder / "curves.npz").items()}
    n = len(z["pitd"])
    ustx = folder / "project.ustx"
    work = folder / ".refine"
    work.mkdir(exist_ok=True)
    history, best, measured = [], None, None
    try:
        for r in range(rounds + 1):
            progress(r / (rounds + 1), f"테토로 불러 보는 중 ({r + 1}/{rounds + 1})")
            wav, loaded = _render_checked(render, ustx)
            if wav is None:  # 렌더가 비었으면 (OpenUtau가 가끔 소리 없는 파일을 낸다) 여기까지 고른 것으로
                history.append({"round": r, "error": last_render_error or "렌더가 비었어요"})
                break
            cents, voiced, db = measure(wav, n, loaded)
            s, info = score(z, cents, voiced, db)
            history.append({"round": r, "score": round(s, 3), **info})
            if best is None or s < best[0]:
                best = (s, r)
                shutil.copy(ustx, work / "best.ustx")
                shutil.copy(wav, work / "best.wav")
                np.savez_compressed(work / "best.npz", pitd=z["pitd"], dyn=z["dyn"])
                measured = (cents, voiced, db)
            if r == rounds:
                break
            z = step(z, cents, voiced, db)
            write_curves(ustx, z)
        if best is None:
            raise RuntimeError(f"테토로 불러 보지 못했어요 ({last_render_error or '렌더가 비었어요'})")
        shutil.copy(work / "best.ustx", ustx)
        b = np.load(work / "best.npz")
        np.savez_compressed(folder / "curves.npz", **{**z, "pitd": b["pitd"], "dyn": b["dyn"]})
        shutil.move(str(work / "best.wav"), folder / "teto_preview.wav")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    prev_path = folder / "preview.json"
    if prev_path.exists():  # 앱 미리 보기의 음높이 · 크기 곡선도 고른 회차로
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        prev["pitch"] = [round(float(v / 100), 2) for v in (z["base"] + b["pitd"])[::2]]
        prev["dyn"] = [round(float(v), 1) for v in b["dyn"][::4]]
        cents, voiced, db = measured
        rf = confidence.render_flags(z["base"] + b["pitd"], z["sing"], z["voiced"], cents, voiced, z["rel"], _rel(z, db))
        prev["flags"] = confidence.merge([f for f in prev.get("flags", []) if f[3] != "render"] + rf)
        prev_path.write_text(json.dumps(prev, ensure_ascii=False), encoding="utf-8")
    out = {"rounds": history, "best": best[1] if best else None, "preview": "teto_preview.wav"}
    (folder / "refine.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def for_result(folder, progress=lambda p, s: None):
    """앱 · 명령줄에서: 조교 결과 폴더를 사용자의 OpenUtau로 불러 보고 다듬는다. report.json에 결과를 적는다.
    → 경고 문장 또는 None. OpenUtau · 렌더 도구가 없으면 아무것도 안 한다."""
    from . import openutau

    folder = Path(folder)
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not openutau.available() or not report.get("singer"):
        return None
    work = folder / ".render"
    warning = None
    try:
        out = refine(folder, lambda u: openutau.render(u, work), progress=progress)
        report["refine"] = out
        report["teto_preview"] = out["preview"]
    except Exception as e:  # 처음 조교한 결과는 그대로 남는다
        warning = f"테토로 불러 보고 다듬지 못했어요 (조교 결과는 그대로예요): {e}"
        report.setdefault("warnings", []).append(warning)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return warning

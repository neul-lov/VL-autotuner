"""조교 결과 고치기 (앱에서): 음절을 음표 사이에서 옮기기 · 늘여 부르기 · 음표 나누기 · 합치기 · 가사 고치기, 되돌리기.

project.ustx를 바로 고친다 (OpenUtau에서 고친 것도 그대로 두고 그 위에). 음표 높이가 바뀌는 곳(합치기)은
PITD 곡선을 그만큼 옮겨서 테토가 실제로 부르는 음 높이는 그대로 둔다.
고치기 전 상태는 결과 폴더의 .history에 남겨서 되돌릴 수 있다.

음표 번호 = 미리 보기(preview.json "notes")의 번호 = 숨 음표를 뺀 부르는 음표 순서.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from .notes import REST_GAP

TICKS = 480
INTERVAL = 5
MIN_TICKS = 30  # 나눈 음표 하나의 가장 짧은 길이
KEEP = 30  # 되돌리기 기록 수


class EditError(ValueError):
    pass


class Project:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.path = self.folder / "project.ustx"
        self.data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        report = self.folder / "report.json"
        self.report = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        self.breath = set(self.report.get("breath_aliases") or [])
        self.part = self.data["voice_parts"][0]
        self.bpm = float(self.data["tempos"][0]["bpm"])
        self.tps = self.bpm / 60 * TICKS

    @property
    def notes(self):
        return self.part["notes"]

    def sung(self):
        """부르는 음표들의 (ustx 안 번호) 목록."""
        return [i for i, x in enumerate(self.notes) if x["lyric"] not in self.breath]

    def phrase(self, j):
        """부르는 음표 j가 든 소절 (쉼으로 나뉜 덩어리)의 부르는 음표 번호 목록."""
        sung = self.sung()
        gap = REST_GAP * self.tps
        a = j
        while a > 0 and self._gap(sung[a - 1], sung[a]) <= gap:
            a -= 1
        b = j
        while b + 1 < len(sung) and self._gap(sung[b], sung[b + 1]) <= gap:
            b += 1
        return list(range(a, b + 1))

    def _gap(self, i, k):
        x, y = self.notes[i], self.notes[k]
        return y["position"] - (x["position"] + x["duration"])

    def save(self):
        self.path.write_text(yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # ---------------------------------------------------------------- PITD
    def _curve(self, abbr):
        return next((c for c in self.part.get("curves", []) if c["abbr"] == abbr), None)

    def base_at(self, ticks):
        """OpenUtau가 음표만으로 만드는 계단 (센트): 그 틱 뒤에 끝나는 첫 음표의 높이."""
        ns = sorted(self.notes, key=lambda x: x["position"])
        ends = np.array([x["position"] + x["duration"] for x in ns])
        tones = np.array([x["tone"] for x in ns]) * 100.0
        k = np.searchsorted(ends, ticks, side="right")
        return tones[np.minimum(k, len(ns) - 1)]

    def shift_pitd(self, a, b, delta):
        """틱 a~b의 PITD에 delta(틱 배열 → 센트)를 더한다."""
        c = self._curve("pitd")
        if c is None:
            c = {"xs": [], "ys": [], "abbr": "pitd"}
            self.part.setdefault("curves", []).append(c)
        xs = np.asarray(c["xs"], dtype=float)
        ys = np.asarray(c["ys"], dtype=float)
        grid = np.arange(a // INTERVAL * INTERVAL, b + INTERVAL, INTERVAL)
        cur = np.interp(grid, xs, ys) if len(xs) else np.zeros(len(grid))
        new = np.clip(np.round(cur + delta(grid)), -1200, 1200).astype(int)
        keep = [(int(x), int(y)) for x, y in zip(c["xs"], c["ys"]) if x < grid[0] or x > grid[-1]]
        pts = sorted(keep + list(zip(grid.tolist(), new.tolist())))
        from .tune import _simplify
        sx, sy = _simplify([p[0] for p in pts], [p[1] for p in pts], 1.5)
        c["xs"], c["ys"] = sx, sy


# ---------------------------------------------------------------- 고치기
def _lyrics(p, idx):
    sung = p.sung()
    return [p.notes[sung[j]]["lyric"] for j in idx]


def _assign(p, idx, lyrics):
    sung = p.sung()
    for j, ly in zip(idx, lyrics):
        p.notes[sung[j]]["lyric"] = ly


def _split(p, j, tick=None):
    sung = p.sung()
    i = sung[j]
    x = p.notes[i]
    a, d = x["position"], x["duration"]
    t = int(round((a + d / 2 if tick is None else tick) / INTERVAL) * INTERVAL)
    t = min(max(t, a + MIN_TICKS), a + d - MIN_TICKS)
    if d < 2 * MIN_TICKS or not (a < t < a + d):
        raise EditError("음표가 너무 짧아서 나눌 수 없어요")
    y = json.loads(json.dumps(x))  # 피치 점 · 비브라토 · 표현 그대로 복사
    x["duration"] = t - a
    y["position"], y["duration"], y["lyric"] = t, a + d - t, "+"
    y["phoneme_expressions"] = []
    y["phoneme_overrides"] = []
    p.notes.insert(i + 1, y)


def op_lyric(p, j, text):
    text = (text or "").strip()
    if not text:
        text = "+"
    if " " in text or len(text) > 16:
        raise EditError("음표 하나에는 음절 하나만 써 주세요")
    _assign(p, [j], [text])


def op_move_left(p, j):
    """음절 시작을 한 음표 앞으로: 앞 음표가 늘이기(+)일 때 그 음표부터 이 음절을 부른다."""
    ph = p.phrase(j)
    if j == ph[0] or _lyrics(p, [j])[0] == "+" or _lyrics(p, [j - 1])[0] != "+":
        raise EditError("앞 음표가 늘여 부르는(+) 음표일 때만 음절을 앞으로 옮길 수 있어요")
    _assign(p, [j - 1, j], [_lyrics(p, [j])[0], "+"])


def op_move_right(p, j):
    """음절 시작을 한 음표 뒤로: 뒤 음표가 늘이기(+)일 때 이 음표는 앞 음절을 늘여 부른다."""
    ph = p.phrase(j)
    if j == ph[-1] or _lyrics(p, [j])[0] == "+" or _lyrics(p, [j + 1])[0] != "+":
        raise EditError("뒤 음표가 늘여 부르는(+) 음표일 때만 음절을 뒤로 옮길 수 있어요")
    if j == ph[0]:
        raise EditError("소절 첫 음표는 늘여 부를 앞 음절이 없어요")
    _assign(p, [j, j + 1], ["+", _lyrics(p, [j])[0]])


def op_extend(p, j):
    """이 음표는 앞 음절을 늘여 부르고(+), 이 음표부터의 음절은 한 음표씩 뒤로 민다.
    소절 끝에 자리가 없으면 마지막 음표를 반으로 나눠 자리를 만든다."""
    ph = p.phrase(j)
    if j == ph[0]:
        raise EditError("소절 첫 음표는 늘여 부를 앞 음절이 없어요")
    if _lyrics(p, [j])[0] == "+":
        raise EditError("이미 늘여 부르는 음표예요")
    idx = [k for k in ph if k >= j]
    ly = _lyrics(p, idx)
    if ly[-1] != "+":
        _split(p, idx[-1])
        idx.append(idx[-1] + 1)
        ly.append("+")
    _assign(p, idx, ["+"] + ly[:-1])


def op_unextend(p, j):
    """늘여 부르던(+) 이 음표에서 다음 음절을 부른다: 뒤 음절들을 한 음표씩 앞으로 당긴다."""
    ph = p.phrase(j)
    if _lyrics(p, [j])[0] != "+":
        raise EditError("늘여 부르는(+) 음표가 아니에요")
    idx = [k for k in ph if k >= j]
    ly = _lyrics(p, idx)
    if all(x == "+" for x in ly):
        raise EditError("이 소절 뒤에 당겨 올 음절이 없어요")
    _assign(p, idx, ly[1:] + ["+"])


def op_split(p, j, time=None):
    _split(p, j, None if time is None else time * p.tps)


def op_merge(p, j):
    """이 음표와 다음 음표를 하나로 (다음 음표의 음절은 한 음표씩 뒤로 민다). 테토가 부르는 음 높이는 그대로."""
    ph = p.phrase(j)
    if j == ph[-1]:
        raise EditError("소절 마지막 음표는 다음 음표와 합칠 수 없어요")
    if _lyrics(p, [j + 1])[0] != "+":  # 다음 음표의 음절은 뒤로 민다 (자리가 없으면 마지막 음표를 나눈다)
        op_extend(p, j + 1)
    sung = p.sung()
    i, k = sung[j], sung[j + 1]
    x, y = p.notes[i], p.notes[k]
    a, end = x["position"], y["position"] + y["duration"]
    old = p.base_at(np.arange(a // INTERVAL * INTERVAL, end + INTERVAL, INTERVAL))
    tone = x["tone"] if x["duration"] >= y["duration"] else y["tone"]
    x["duration"], x["tone"] = end - a, tone
    del p.notes[k]
    grid0 = np.arange(a // INTERVAL * INTERVAL, end + INTERVAL, INTERVAL)
    new = p.base_at(grid0)
    p.shift_pitd(grid0[0], grid0[-1], lambda g: np.interp(g, grid0, old - new))


OPS = {"lyric": op_lyric, "move_left": op_move_left, "move_right": op_move_right, "extend": op_extend,
       "unextend": op_unextend, "split": op_split, "merge": op_merge}


# ---------------------------------------------------------------- 기록 · 미리 보기
def _history(folder):
    return Path(folder) / ".history"


def snapshot(folder):
    h = _history(folder)
    h.mkdir(exist_ok=True)
    n = max([int(f.stem) for f in h.glob("*.ustx") if f.stem.isdigit()] or [0]) + 1
    shutil.copy(Path(folder) / "project.ustx", h / f"{n:04d}.ustx")
    if (Path(folder) / "preview.json").exists():
        shutil.copy(Path(folder) / "preview.json", h / f"{n:04d}.json")
    for f in sorted(h.glob("*.ustx"))[:-KEEP]:
        f.unlink()
        (h / f"{f.stem}.json").unlink(missing_ok=True)
    return n


def history_count(folder):
    h = _history(folder)
    return len(list(h.glob("*.ustx"))) if h.exists() else 0


def undo(folder):
    h = _history(folder)
    files = sorted(h.glob("*.ustx")) if h.exists() else []
    if not files:
        raise EditError("되돌릴 것이 없어요")
    last = files[-1]
    shutil.move(str(last), Path(folder) / "project.ustx")
    if (h / f"{last.stem}.json").exists():
        shutil.move(str(h / f"{last.stem}.json"), Path(folder) / "preview.json")
    return refresh_preview(folder)


def refresh_preview(folder, p=None, touched=None, stale=True):
    """project.ustx의 음표로 미리 보기 음표를 다시 쓴다. touched(초 구간)에 걸친 가사 표시는 지운다 (사용자가 확인했으니).
    stale: 테토 미리 듣기(teto_preview.wav)가 지금 음표와 달라졌다 (앱이 "다시 불러 보기"를 보여 준다)."""
    folder = Path(folder)
    p = p or Project(folder)
    prev_path = folder / "preview.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else {}
    prev["notes"] = [[round(p.notes[i]["position"] / p.tps, 3), round((p.notes[i]["position"] + p.notes[i]["duration"]) / p.tps, 3),
                      int(p.notes[i]["tone"]), p.notes[i]["lyric"]] for i in p.sung()]
    if touched:
        a, b = touched
        prev["flags"] = [f for f in prev.get("flags", []) if not (f[3] == "lyric" and f[0] < b and f[1] > a)]
    if stale and (folder / "teto_preview.wav").exists():
        prev["teto_stale"] = True
    prev_path.write_text(json.dumps(prev, ensure_ascii=False), encoding="utf-8")
    prev["history"] = history_count(folder)
    return prev


def apply(folder, op, index, **kw):
    """→ 새 미리 보기. 못 하면 EditError (아무것도 바꾸지 않는다)."""
    if op not in OPS:
        raise EditError("모르는 고치기예요")
    p = Project(folder)
    sung = p.sung()
    if not 0 <= index < len(sung):
        raise EditError("음표를 찾지 못했어요")
    x = p.notes[sung[index]]
    span = (x["position"] / p.tps, (x["position"] + x["duration"]) / p.tps)
    OPS[op](p, index, **kw)
    snapshot(folder)
    p.save()
    return refresh_preview(folder, p, touched=span)

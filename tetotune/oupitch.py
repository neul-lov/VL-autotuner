"""OpenUtau가 음표만으로 만드는 음 높이(계단 + 음표 피치 점)를 OpenUtau와 똑같이 계산한다 (RenderPhrase.cs).

OpenUtau가 실제로 부르는 음 높이 = 이 값 + PITD 곡선. PITD는 "원하는 음 높이 − 이 값"으로 넣으니까, 여기가
OpenUtau 계산과 조금이라도 다르면 음표가 바뀌는 곳마다 그만큼 음이 튄다. 예전에는 경계에서 계단(snap_first,
5ms 직선)이 PITD 격자(5틱)와 어긋나서 음표가 바뀔 때마다 수백 센트가 짧게 튀었다.

그래서 붙어 있는 두 음표의 높이가 다르면 경계 앞뒤 PORTA_MS/2씩 부드러운 곡선(io)으로 넘어가게 피치 점을 쓰고
(어디서 재도 OpenUtau와 같은 값), 음표 사이에 짧은 틈이 있으면 뒤 음표의 첫 피치 점을 틈 시작(앞 음표 끝)에 앞 음표 높이로 둔다:
틈이 선행발성보다 짧으면 OpenUtau는 선행발성을 틈 길이로 줄여서 뒤 프레이즈가 앞 음표 끝 틱부터 재는데, 그 틱의 계단이
앞 프레이즈(앞 음표 높이)와 뒤 프레이즈(뒤 음표 높이)에서 달라 PITD 하나로는 둘 다 맞출 수 없었다 (자음 첫머리가 튐).
"""
import math

import numpy as np

PORTA_MS = 30.0  # 붙어 있는 두 음표 높이가 다를 때 넘어가는 길이 (ms, 짧은 음표는 그 1/4까지). 0 = 예전 방식
GAP_RAMP_MS = 400.0  # 이보다 짧은 틈 뒤 음표는 틈 시작부터 앞 음표 높이에서 넘어온다 (선행발성이 이보다 긴 음원은 거의 없다)


def pitch_data(notes, tps):
    """notes: [{"position", "duration", "tone"}] (틱, 시작 순서) → 음표마다 ustx "pitch" 항목."""
    out = []
    for k, n in enumerate(notes):
        prev = notes[k - 1] if k else None
        if PORTA_MS <= 0:  # 예전: OpenUtau 기본 (snap_first가 앞 음표 높이에서 5ms 직선으로)
            out.append({"data": [{"x": 0, "y": 0, "shape": "l"}, {"x": 5, "y": 0, "shape": "l"}], "snap_first": True})
            continue
        touch = prev is not None and prev["position"] + prev["duration"] == n["position"]
        gap_ms = (n["position"] - prev["position"] - prev["duration"]) / tps * 1000 if prev is not None else 0
        if not touch and prev is not None and prev["tone"] != n["tone"] and 0 < gap_ms <= GAP_RAMP_MS:
            d = (prev["tone"] - n["tone"]) * 10
            out.append({"data": [{"x": round(-gap_ms, 3), "y": d, "shape": "io"}, {"x": 0, "y": 0, "shape": "io"}],
                        "snap_first": False})
        elif touch and prev["tone"] != n["tone"]:
            w = min(PORTA_MS, 0.5 * min(prev["duration"], n["duration"]) / tps * 1000)
            d = (prev["tone"] - n["tone"]) * 10
            # 첫 점 앞 5ms는 평평하게: OpenUtau는 첫 점보다 한 칸(5틱) 앞에서부터 계산해서, 곡선이 바로 시작하면 그 칸이 조금 튄다
            out.append({"data": [{"x": round(-w / 2 - 5, 3), "y": d, "shape": "io"}, {"x": round(-w / 2, 3), "y": d, "shape": "io"},
                                 {"x": round(w / 2, 3), "y": 0, "shape": "io"}], "snap_first": False})
        else:
            out.append({"data": [{"x": 0, "y": 0, "shape": "io"}, {"x": 5, "y": 0, "shape": "io"}], "snap_first": False})
    return out


def _ease(x0, x1, y0, y1, x, shape):
    """MusicMath.InterpolateShape (io · sp: 사인 in-out, i · o: 반쪽, 그 밖 직선)."""
    if x1 - x0 < 1e-6:
        return np.full_like(x, y1, dtype=float)
    u = (x - x0) / (x1 - x0)
    if shape in ("io", "sp"):
        return y0 + (y1 - y0) * (1 - np.cos(u * math.pi)) / 2
    if shape == "i":
        return y0 + (y1 - y0) * (1 - np.cos(u * math.pi / 2))
    if shape == "o":
        return y0 + (y1 - y0) * np.sin(u * math.pi / 2)
    return y0 + (y1 - y0) * u


def _points(n, data, tps):
    """음표 하나의 피치 점 → [(틱, 센트, 모양)] (OpenUtau가 앞뒤에 채우는 점까지). snap_first는 앞 음표를 보고 계산."""
    pos, end, tone = n["position"], n["position"] + n["duration"], n["tone"]
    pts = [(int(round(pos + p["x"] * tps / 1000)), p["y"] * 10 + tone * 100, p.get("shape", "io")) for p in data["data"]]
    if not pts:
        pts = [(pos, tone * 100, "io"), (end, tone * 100, "io")]
    if pts[0][0] > pos:  # (프레이즈 첫 음표면 프레이즈 시작부터: 값이 같아서 더해지는 게 없다)
        pts.insert(0, (pos, pts[0][1], "io"))
    if pts[-1][0] < end:
        pts.append((end, pts[-1][1], "io"))
    return pts


def snap(notes, data):
    """UNote.Validate의 snap_first: 앞 음표가 딱 붙어 있으면 첫 점을 앞 음표 높이로, 아니면 0."""
    out = []
    for k, (n, d) in enumerate(zip(notes, data)):
        d = {"data": [dict(p) for p in d["data"]], "snap_first": d.get("snap_first", True)}
        if d["snap_first"] and d["data"]:
            prev = notes[k - 1] if k else None
            touch = prev is not None and prev["position"] + prev["duration"] == n["position"]
            d["data"][0]["y"] = (prev["tone"] - n["tone"]) * 10 if touch else 0
        out.append(d)
    return out


def base(notes, data, tps, xs):
    """틱 xs에서 OpenUtau가 음표 · 피치 점만으로 만드는 음 높이 (센트). notes · data는 ustx에 쓰는 그대로 (숨 음표 포함)."""
    xs = np.asarray(xs, dtype=float)
    if not notes:
        return np.zeros_like(xs)
    data = snap(notes, data)
    ends = np.array([n["position"] + n["duration"] for n in notes], dtype=float)
    tones = np.array([n["tone"] for n in notes], dtype=float) * 100
    # 계단: 그 틱 뒤에 끝나는 첫 음표의 높이 (마지막 음표 뒤는 마지막 음표)
    out = tones[np.minimum(np.searchsorted(ends, xs, side="right"), len(notes) - 1)].copy()
    for k, (n, d) in enumerate(zip(notes, data)):
        pts = _points(n, d, tps)
        lo, hi = pts[0][0], pts[-1][0]
        i0, i1 = np.searchsorted(xs, lo, side="left"), np.searchsorted(xs, hi, side="left")
        if i1 <= i0:
            continue
        x = xs[i0:i1]
        prev_end = ends[k - 1] if k else -np.inf
        basep = np.where(x < prev_end, tones[k - 1] if k else tones[k], tones[k])
        y = np.zeros_like(x)
        for (xa, ya, sh), (xb, yb, _) in zip(pts, pts[1:]):
            m = (x >= xa) & (x < xb)
            y[m] = _ease(xa, xb, ya, yb, x[m], sh)
        out[i0:i1] += y - basep
    return out


def _ramped(n, m, tps):
    """n 뒤 틈 다음의 m이 틈 시작부터 넘어오는 피치 점을 받는지 (pitch_data와 같은 조건)."""
    gap_ms = (m["position"] - n["position"] - n["duration"]) / tps * 1000 if tps else 0
    return PORTA_MS > 0 and n["tone"] != m["tone"] and 0 < gap_ms <= GAP_RAMP_MS


def curve_xs(notes, span_ticks, tps=None):
    """PITD 점을 둘 틱과 그 점에서 쓸 계단 쪽 → (xs, at). 틱마다 두고 선 위의 점은 나중에 지운다: OpenUtau는 프레이즈마다
    다른 곳에서 5틱 간격으로 재서, 5틱마다만 두면 빠르게 넘어가는 곳(큰 도약 · 넘어가는 곡선)에서 그 사이 직선만큼 어긋난다.
    긴 틈 앞에서 끝나는 음표의 끝(End)은 앞 음표 쪽(at = End − 0.5)으로 잰다: OpenUtau 계단은 End에서 다음 음표 높이로
    바뀌는데 앞 음표의 프레이즈는 End까지 앞 음표 높이다 (짧은 틈은 피치 점으로 양쪽을 맞춰서 필요 없다)."""
    a, b = span_ticks
    xs = {x: float(x) for x in range(a - 1, b + 2)}
    for n, m in zip(notes, notes[1:]):
        end = n["position"] + n["duration"]
        if m["position"] > end and not _ramped(n, m, tps) and end in xs:
            xs[end] = end - 0.5
    keys = sorted(xs)
    return np.array(keys), np.array([xs[x] for x in keys])


def sample(curve, xs):
    """UCurve.Sample: 점 사이는 직선, 반올림. 첫 점 앞 · 마지막 점 뒤는 0 (PITD 기본값)."""
    xs = np.asarray(xs, dtype=float)
    if not curve or not curve.get("xs"):
        return np.zeros_like(xs)
    cx, cy = np.asarray(curve["xs"], dtype=float), np.asarray(curve["ys"], dtype=float)
    v = np.round(np.interp(xs, cx, cy))
    return np.where((xs < cx[0]) | (xs > cx[-1]), 0.0, v)


def sung(notes, curve, tps, xs):
    """ustx 음표(피치 점 포함) + PITD 곡선 → 틱 xs에서 OpenUtau가 부르는 음 높이 (센트, 비브라토 0)."""
    data = [n.get("pitch") or {"data": [], "snap_first": True} for n in notes]
    return base(notes, data, tps, xs) + sample(curve, xs)


def rewrite(part, tps, grid, intended, span=None):
    """음표를 바꾼 뒤: 모든 음표의 피치 점을 다시 쓰고, 부르는 음 높이가 intended(틱 grid에서, 센트)가 되게 PITD를 다시 쓴다."""
    from .tune import CURVE_RANGE, _simplify

    notes = sorted(part["notes"], key=lambda x: x["position"])
    part["notes"] = notes
    data = pitch_data(notes, tps)
    for n, d in zip(notes, data):
        n["pitch"] = d
    span = span or (int(grid[0]), int(grid[-1]))
    xs, at = curve_xs(notes, span, tps=tps)
    lo, hi, tol = CURVE_RANGE["pitd"]
    ys = np.clip(np.round(np.interp(at, grid, intended) - base(notes, data, tps, at)), lo, hi).astype(int)
    cx, cy = _simplify(list(xs), list(ys), tol)
    curves = part.setdefault("curves", [])
    cur = next((c for c in curves if c["abbr"] == "pitd"), None)
    if cur is None:
        cur = {"xs": [], "ys": [], "abbr": "pitd"}
        curves.append(cur)
    cur["xs"], cur["ys"] = cx, cy

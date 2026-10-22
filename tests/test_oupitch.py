"""oupitch가 OpenUtau와 같은 음 높이를 내는지: RenderPhrase.cs의 음 높이 계산을 줄마다 그대로 옮긴 것과 비교한다.
OpenUtau는 프레이즈마다 (첫 음소 위치 − 선행발성)부터 5틱 간격으로 재서, 격자가 어디서 시작하든 맞아야 한다."""
import math

import numpy as np

from tetotune import oupitch, tune

TPS = 960.0  # 120bpm, 480틱 = 0.5초


def _ease(x0, x1, y0, y1, x, shape):
    if x1 - x0 < 1e-6:
        return y1
    if shape in ("io", "sp"):
        return y0 + (y1 - y0) * (1 - math.cos((x - x0) / (x1 - x0) * math.pi)) / 2
    if shape == "i":
        return y0 + (y1 - y0) * (1 - math.cos((x - x0) / (x1 - x0) * math.pi / 2))
    if shape == "o":
        return y0 + (y1 - y0) * math.sin((x - x0) / (x1 - x0) * math.pi / 2)
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _sample(curve, x):
    """UCurve.Sample."""
    xs, ys = curve["xs"], curve["ys"]
    if x in xs:
        return ys[xs.index(x)]
    i = int(np.searchsorted(xs, x))
    if 0 < i < len(xs):
        return round(ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]))
    return 0


def _render_phrase(notes, curve, i0, i1, leading):
    """RenderPhrase 생성자의 음 높이 부분 (비브라토 · 음원 주파수 보정 없음) → (틱, 센트)."""
    end_of = [n["position"] + n["duration"] for n in notes]
    u = list(range(i0, i1))
    if i0 > 0 and end_of[i0 - 1] == notes[i0]["position"]:
        u.insert(0, i0 - 1)
    if i1 < len(notes) and end_of[i1 - 1] == notes[i1]["position"]:
        u.append(i1)
    position, end = notes[i0]["position"], end_of[i1 - 1]
    pitch_start = position - leading
    pitches = [0.0] * ((end - pitch_start) // 5 + 1)
    index = 0
    for k in u:
        while pitch_start + index * 5 < end_of[k] and index < len(pitches):
            pitches[index] = notes[k]["tone"] * 100
            index += 1
    index = max(1, index)
    while index < len(pitches):
        pitches[index] = pitches[index - 1]
        index += 1
    for k in u:
        n = notes[k]
        pos_ms = n["position"] / TPS * 1000
        pts = [[round((pos_ms + p["x"]) * TPS / 1000), p["y"] * 10 + n["tone"] * 100, p["shape"]] for p in n["pitch"]["data"]]
        if not pts:
            pts = [[n["position"], n["tone"] * 100, "io"], [end_of[k], n["tone"] * 100, "io"]]
        if k == u[0] and pts[0][0] > pitch_start:
            pts.insert(0, [pitch_start, pts[0][1], "io"])
        elif pts[0][0] > n["position"]:
            pts.insert(0, [n["position"], pts[0][1], "io"])
        if pts[-1][0] < end_of[k]:
            pts.append([end_of[k], pts[-1][1], "io"])
        index = max(0, int((pts[0][0] - pitch_start) / 5))
        for p0, p1 in zip(pts, pts[1:]):
            x = pitch_start + index * 5
            while x < p1[0] and index < len(pitches):
                base = notes[k - 1]["tone"] * 100 if k > 0 and x < end_of[k - 1] else n["tone"] * 100
                pitches[index] += _ease(p0[0], p1[0], p0[1], p1[1], x, p0[2]) - base
                index += 1
                x += 5
    xs = [pitch_start + i * 5 for i in range(len(pitches))]
    return np.array(xs), np.array([p + _sample(curve, x) for p, x in zip(pitches, xs)])


def _song(seed):
    """붙은 음표 · 틈 있는 음표 · 큰 도약 · 짧은 음표가 섞인 노래와, 원하는 음 높이 (비브라토 + 넘어가는 곡선)."""
    rng = np.random.default_rng(seed)
    notes, pos, tone = [], 480, 60
    for _ in range(40):
        dur = int(rng.choice([30, 60, 120, 240, 480, 720]))
        tone = int(np.clip(tone + rng.choice([-12, -7, -5, -2, -1, 0, 1, 2, 3, 5, 9, 12]), 48, 76))
        notes.append({"position": pos, "duration": dur, "tone": tone, "lyric": "a"})
        pos += dur + int(rng.choice([0, 0, 0, 0, 3, 17, 120, 480]))
    t = np.arange(0, (pos + 960) / TPS, 0.005)
    steps = np.full(len(t), np.nan)
    for n in notes:
        steps[(t * TPS >= n["position"]) & (t * TPS < n["position"] + n["duration"])] = n["tone"] * 100
    idx = np.arange(len(t))
    ok = ~np.isnan(steps)
    steps = np.interp(idx, idx[ok], steps[ok])
    k = np.hanning(17)  # 80ms로 넘어가기 (사람 노래쯤)
    smooth = np.convolve(np.pad(steps, 8, mode="edge"), k / k.sum(), mode="valid")
    return notes, t, smooth + 30 * np.sin(2 * np.pi * 5.5 * t)


def test_sung_pitch_matches_openutau_on_every_grid():
    for seed in range(3):
        notes, t, intended = _song(seed)
        pdata = oupitch.pitch_data(notes, TPS)
        for n, d in zip(notes, pdata):
            n["pitch"] = d
        curve = tune.pitd_curve(t, intended, notes, pdata, TPS, (0, t[-1]))
        # 프레이즈 = 붙어 있는 음표 묶음 (틈이 있으면 OpenUtau도 나눈다)
        phrases, a = [], 0
        for i in range(1, len(notes) + 1):
            if i == len(notes) or notes[i]["position"] != notes[i - 1]["position"] + notes[i - 1]["duration"]:
                phrases.append((a, i))
                a = i
        worst = 0.0
        for leading in (0, 1, 2, 3, 4, 13, 58):
            for i0, i1 in phrases:
                gap = notes[i0]["position"] - (notes[i0 - 1]["position"] + notes[i0 - 1]["duration"]) if i0 else 999
                xs, got = _render_phrase(notes, curve, i0, i1, min(leading, gap))
                want = np.interp(xs / TPS, t, intended)
                worst = max(worst, float(np.abs(got - want).max()))
                # oupitch.sung도 같은 값 (편집 · 재조교가 쓰는 계산). 틈 앞 음표의 끝 틱은 두 프레이즈가 다르게 재서 뺀다
                inner = xs < xs[-1]
                assert np.abs(oupitch.sung(notes, curve, TPS, xs[inner]) - got[inner]).max() < 1.01
        assert worst < 3.0, (seed, worst)


def test_old_writer_spiked_at_note_changes():
    """예전 방식(PORTA_MS = 0: snap_first + 5ms 직선)이면 같은 시험에서 음표가 바뀔 때 크게 튄다: 시험이 문제를 본다."""
    old = oupitch.PORTA_MS
    try:
        oupitch.PORTA_MS = 0.0
        notes, t, intended = _song(0)
        pdata = oupitch.pitch_data(notes, TPS)
        for n, d in zip(notes, pdata):
            n["pitch"] = d
        # 예전에는 PITD를 계단(_base_cents)에 맞춰 5틱마다 썼다
        xs = np.arange(0, int(t[-1] * TPS), 5)
        base = tune._base_cents([type("N", (), {"start": n["position"] / TPS, "end": (n["position"] + n["duration"]) / TPS,
                                                "tone": n["tone"]})() for n in notes], xs / TPS)
        ys = np.round(np.interp(xs / TPS, t, intended) - base).astype(int)
        curve = {"xs": [int(x) for x in xs], "ys": [int(y) for y in ys]}
        worst = 0.0
        for leading in (1, 3):
            xs2, got = _render_phrase(notes, curve, 0, 1, 0)
            for i0 in range(len(notes) - 1):
                i1 = i0 + 1
                while i1 < len(notes) and notes[i1]["position"] == notes[i1 - 1]["position"] + notes[i1 - 1]["duration"]:
                    i1 += 1
                xs2, got = _render_phrase(notes, curve, i0, i1, leading)
                worst = max(worst, float(np.abs(got - np.interp(xs2 / TPS, t, intended)).max()))
        assert worst > 100
    finally:
        oupitch.PORTA_MS = old

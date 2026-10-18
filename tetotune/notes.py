"""음표 만들기: 원래 보컬의 음높이에서 음표를 나누거나, 이미 있는 MIDI · ustx · ust를 읽고, 가사를 음표에 맞춘다.

음표 시작 = 모음이 시작하는 곳(소리가 나기 시작하는 곳). OpenUtau는 음표 시작에 모음을 두고
자음은 그 앞에(선행발성만큼) 끼워 넣기 때문에, 사람이 부른 박자와 같아진다.
"""
from pathlib import Path

import numpy as np

from .pitch import HOP, _runs

TICKS = 480
REST_GAP = 0.15  # 이보다 긴 무성 구간은 쉼표 (숨 · 프레이즈 사이)
CONSONANT_GAP = 0.22  # 바람 소리(자음)로 찬 틈은 이만큼까지 프레이즈 안 (features.CONSONANT_MAX와 같게)
LINE_MATCH = True  # 가사 줄 수 ≠ 프레이즈 수일 때 줄 경계를 지키며 짝짓기 (끄면 음절 수만 나눈다)


class Note:
    __slots__ = ("start", "end", "tone", "lyric", "strength", "dips")

    def __init__(self, start, end, tone, lyric="", strength=10.0):
        self.start, self.end, self.tone, self.lyric = float(start), float(end), int(tone), lyric
        self.strength = strength  # 앞 음표와의 경계가 얼마나 "새 음절" 같은지
        self.dips = []  # [(초, 점수)] 같은 음 안에서 새 음절이 시작했을 만한 곳 (같은 음으로 다른 음절을 부를 때)

    def __repr__(self):
        return f"Note({self.start:.3f}-{self.end:.3f} {self.tone} {self.lyric!r})"


def tuning_offset(midi, voiced):
    """사람마다 기준음이 조금씩 다르다 (예: 전체가 20센트 높음). 음표 높이를 정할 때 그만큼 빼고 반올림한다."""
    m = midi[voiced & np.isfinite(midi)]
    if len(m) == 0:
        return 0.0
    ang = 2 * np.pi * (m - np.round(m))
    return float(np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi))


def _viterbi(seg, weight, offset, switch=6.0):
    lo, hi = int(np.floor(np.nanmin(seg))) - 1, int(np.ceil(np.nanmax(seg))) + 2
    states = np.arange(lo, hi)
    cost = np.minimum((seg[:, None] - offset - states[None, :]) ** 2, 4.0) * weight[:, None]
    n, k = cost.shape
    acc = cost[0].copy()
    back = np.zeros((n, k), dtype=np.int32)
    for i in range(1, n):
        best = acc.argmin()
        stay = acc
        move = acc[best] + switch
        choose_move = move < stay
        back[i] = np.where(choose_move, best, np.arange(k))
        acc = np.where(choose_move, move, stay) + cost[i]
    path = np.empty(n, dtype=np.int32)
    path[-1] = acc.argmin()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return states[path]


def _cuts(db_s, nov_z, a, b, edge_sec=0.1):
    """a..b 프레임(한 음표) 안에서 새 음절이 시작했을 만한 곳 → [(프레임, 점수, 크기 꺼짐 dB)].
    점수 = 크기가 꺼진 깊이/2 + 소리 빛깔이 바뀐 정도. 같은 음으로 "あ え"처럼 부른 곳을 찾는다."""
    w = int(0.15 / HOP)
    edge = int(edge_sec / HOP)
    score = np.zeros(b - a)
    depth = np.zeros(b - a)
    for i in range(a + edge, b - edge):
        left = db_s[max(a, i - w):i].max()
        right = db_s[i + 1:min(b, i + w)].max()
        depth[i - a] = max(0.0, min(left, right) - db_s[i])
        score[i - a] = depth[i - a] / 2 + 0.5 * max(0.0, nov_z[i] - 2.0)
    out = []
    sup = int(0.08 / HOP)
    order = np.argsort(-score)
    taken = np.zeros(b - a, dtype=bool)
    for k in order:
        if score[k] < 1.5:
            break
        if taken[max(0, k - sup):k + sup].any():
            continue
        taken[k] = True
        out.append((a + int(k), float(score[k]), float(depth[k])))
    return sorted(out)


def segment(midi, voiced, db, novelty=None, auto_split_db=6.0, noise=None):
    """음높이 → 음표 목록. 프레이즈(쉼표로 나뉜 덩어리)마다 음 높이가 머무는 곳을 찾는다 (비터비).
    noise: 음높이가 잡혔지만 바람 소리라서 무성으로 돌린 곳 (features.noisy_voiced). 그런 틈이나 노래만큼 큰 틈은
    ㅅ · ㅎ 같은 자음이라 REST_GAP보다 조금 길어도 프레이즈를 끊지 않는다."""
    from scipy.ndimage import uniform_filter1d

    offset = tuning_offset(midi, voiced)
    db_s = uniform_filter1d(db, 5)
    n = len(midi)
    if novelty is None:
        novelty = np.zeros(n)
    ref = np.median(novelty[voiced]) if voiced.any() else 1.0
    spread = np.median(np.abs(novelty[voiced] - ref)) * 1.4826 + 1e-9 if voiced.any() else 1.0
    nov_z = (novelty - ref) / spread
    # 짧은 무성 구간(자음)은 메워서 한 프레이즈로, 긴 구간은 쉼표
    gaps = []
    bridged = voiced.copy()
    runs = _runs(voiced)
    sing = np.median(db[voiced]) if voiced.any() else 0.0
    for (a0, b0), (a1, b1) in zip(runs, runs[1:]):
        consonant = (a1 - b0) * HOP <= CONSONANT_GAP and (
            (noise is not None and noise[b0:a1].mean() >= 0.5) or np.median(db[b0:a1]) > sing - 20)
        if (a1 - b0) * HOP <= REST_GAP or consonant:
            bridged[b0:a1] = True
            gaps.append((b0, a1))
    filled = midi.copy()
    idx = np.arange(n)
    ok = voiced & np.isfinite(midi)
    if ok.any():
        filled = np.where(ok, midi, np.interp(idx, idx[ok], midi[ok]))
    notes = []
    for a, b in _runs(bridged):
        seg = filled[a:b]
        weight = np.where(voiced[a:b], 1.0, 0.25)
        path = _viterbi(seg, weight, offset)
        starts = [0] + [i for i in range(1, len(path)) if path[i] != path[i - 1]]
        # 무성 틈(자음) 끝 = 새 음절 시작
        for g0, g1 in gaps:
            if a < g1 < b:
                starts.append(g1 - a)
        starts = sorted(set(starts))
        # 60ms보다 짧은 음은 앞 음에 붙인다 (꺾기 · 미끄러짐)
        merged = []
        for s in starts:
            if merged and (s - merged[-1]) * HOP < 0.06 and not any(g1 - a == s for _, g1 in gaps):
                continue
            merged.append(s)
        bounds = merged + [b - a]
        phrase = []
        for s, e in zip(bounds, bounds[1:]):
            if e <= s:
                continue
            core = seg[s + (e - s) // 5:e - (e - s) // 5 or e]
            core = core[np.isfinite(core)] if len(core) else seg[s:e]
            tone = int(np.round(np.median(core) - offset))
            is_gap = any(g1 - a == s for _, g1 in gaps)
            if not phrase:
                strength = 10.0
            elif is_gap:
                strength = 5.0
            else:  # 음만 바뀜: 모음도 바뀌었으면(소리 빛깔) 새 음절일 가능성이 크다
                k = a + s
                bonus = max(0.0, float(nov_z[max(0, k - 6):k + 7].max()) - 1.0)
                strength = 1.0 + min(2.0, 0.5 * abs(tone - phrase[-1].tone)) + min(4.0, bonus)
            phrase.append(Note((a + s) * HOP, (a + e) * HOP, tone, strength=strength))
        # 같은 음 안에서 크기가 확 꺼지면 다른 음절 (예: "ら ら")
        out = []
        for note in phrase:
            fa, fb = int(round(note.start / HOP)), int(round(note.end / HOP))
            cuts = _cuts(db_s, nov_z, fa, fb)
            strong = [i for i, sc, d in cuts if d >= auto_split_db]
            note.dips = [(i * HOP, sc) for i, sc, d in cuts if d < auto_split_db]
            if not strong:
                out.append(note)
                continue
            score = {i: sc for i, sc, d in cuts}
            edges = [fa] + strong + [fb]
            for j, (s, e) in enumerate(zip(edges, edges[1:])):
                part = Note(s * HOP, e * HOP, note.tone, strength=note.strength if j == 0 else 3.0 + score[s] / 2)
                part.dips = [(t, d) for t, d in note.dips if part.start < t < part.end]
                out.append(part)
        notes.extend(_merge_short(_snap(out)))
    return notes, offset


def _snap(notes, near=0.15, min_score=4.0):
    """음이 바뀐 곳(비터비 경계)이 실제 음절 시작(크기 꺼짐 · 소리 빛깔 바뀜)보다 조금 어긋났으면 그쪽으로 옮긴다."""
    for i in range(1, len(notes)):
        a, b = notes[i - 1], notes[i]
        if abs(a.end - b.start) > 1e-6 or b.strength >= 5:
            continue
        cands = [(sc, t) for t, sc in a.dips if a.end - t < near] + [(sc, t) for t, sc in b.dips if t - b.start < near]
        if not cands:
            continue
        sc, t = max(cands)
        if sc < min_score or t - a.start < 0.06 or b.end - t < 0.06:
            continue
        a.dips = [(x, d) for x, d in a.dips if x < t - near]
        b.dips = [(x, d) for x, d in b.dips if x > t + near]
        a.end = b.start = t
        b.strength = max(b.strength, 3.0 + sc / 2)
    return notes


def _merge_short(notes, min_len=0.09):
    """90ms보다 짧고 앞뒤와 붙어 있는 음표(음이 넘어가는 중간)는 높이가 가까운 이웃에 합친다."""
    notes = list(notes)
    i = 0
    while i < len(notes) and len(notes) > 1:
        x = notes[i]
        if x.end - x.start >= min_len:
            i += 1
            continue
        prv = notes[i - 1] if i > 0 and x.strength < 10 else None  # 쉼표 너머로는 붙이지 않는다
        nxt = notes[i + 1] if i + 1 < len(notes) and notes[i + 1].strength < 10 else None
        if nxt and (not prv or abs(nxt.tone - x.tone) <= abs(prv.tone - x.tone)):
            nxt.start, nxt.strength = x.start, max(nxt.strength, x.strength)
            nxt.dips = x.dips + nxt.dips
        elif prv:
            prv.end = x.end
            prv.dips += x.dips
        elif i + 1 < len(notes):  # 앞뒤가 쉼표인 짧은 조각 (숨 끝 · 잡음): 버리고 다음 음이 프레이즈 시작
            notes[i + 1].strength = max(notes[i + 1].strength, x.strength)
        notes.pop(i)
    return notes


# ---------------------------------------------------------------- 가사 맞추기
def _split_note(notes, k, t, strength):
    a = notes[k]
    b = Note(t, a.end, a.tone, strength=strength)
    b.dips = [(x, d) for x, d in a.dips if x > t]
    a.end = t
    a.dips = [(x, d) for x, d in a.dips if x < t]
    notes.insert(k + 1, b)


NEUTRAL = 2.5  # 이보다 약한 경계는 새 음절보다 "이어 부르기"일 가능성이 크다


def _evidence(notes):
    """한 프레이즈 안에서 새 음절이 시작될 수 있는 곳들 (센 순서): 음표 경계 + 음표 안 후보."""
    ev = [(n.strength, "b", i, n.start) for i, n in enumerate(notes) if i > 0]
    ev += [(d, "c", i, t) for i, n in enumerate(notes) for t, d in n.dips]
    return sorted(ev, key=lambda e: -e[0])


def _utilities(ev, n):
    """음절 k개(0..n)를 이 프레이즈에 넣을 때의 그럴듯함 (센 경계 k−1개를 고른 합, 경계가 모자라면 하나에 −6).
    누적합도 sum()으로 구한다 (sum()은 소수 오차를 보정하며 더해서, 하나씩 더하면 값이 조금 달라진다)."""
    d = [e[0] - NEUTRAL for e in ev]
    top = np.array([sum(d[:j]) for j in range(len(d) + 1)], dtype=float)
    k = np.arange(n + 1)
    u = top[np.clip(k - 1, 0, len(d))] - 6.0 * np.maximum(0, k - 1 - len(d))
    u[0] = -np.inf  # 프레이즈마다 1개 이상
    return u


def _align_group(notes, syls):
    """프레이즈 하나에 음절을 넣는다. 음표 경계와 음표 안 후보를 함께 놓고 센 순서로 음절 시작을 고른다:
    고르지 않은 음표 경계는 "+"(한 음절을 여러 음에 걸쳐 부르기), 고른 음표 안 후보에서는 음표를 나눈다."""
    if not notes or not syls:
        return notes
    ev = _evidence(notes)
    chosen = ev[:len(syls) - 1]
    starts = {notes[0].start} | {e[3] for e in chosen if e[1] == "b"}
    for _, kind, i, t in sorted((e for e in chosen if e[1] == "c"), key=lambda e: -e[3]):
        k = next(j for j, n in enumerate(notes) if n.start < t < n.end)
        _split_note(notes, k, t, 3.0)
        starts.add(t)
    missing = len(syls) - len(starts)
    while missing > 0:  # 후보가 모자라면 긴 음표를 반으로
        k = max(range(len(notes)), key=lambda i: notes[i].end - notes[i].start)
        n = notes[k]
        if n.end - n.start < 0.16:
            break
        t = (n.start + n.end) / 2
        _split_note(notes, k, t, 2.0)
        starts.add(t)
        missing -= 1
    s = 0
    for n in notes:
        if n.start in starts and s < len(syls):
            n.lyric = syls[s]
            s += 1
        else:
            n.lyric = "+"
    return notes


def _phrases(notes):
    groups, cur = [], []
    for n in notes:
        if cur and n.strength >= 10:
            groups.append(cur)
            cur = []
        cur.append(n)
    if cur:
        groups.append(cur)
    return groups


def align(notes, lines):
    """가사 줄 수 = 프레이즈 수면 줄마다 그대로. 가사가 한 줄이면 음절을 프레이즈에 나누는 방법 중
    경계가 가장 그럴듯한 것을 고른다 (동적 계획법). 줄이 여럿인데 프레이즈 수와 다르면 줄 경계를 지키며
    줄과 프레이즈를 짝짓는다 (_match_lines): 숨 아닌 곳에서 프레이즈가 끊겨도 가사가 다음 줄로 밀리지 않게."""
    syls = [s for line in lines for s in line]
    if not syls:
        return notes
    groups = _phrases(notes)
    if len(groups) == len(lines) or len(lines) == 1 or not LINE_MATCH:
        counts = [len(line) for line in lines] if len(groups) == len(lines) else _distribute(groups, len(syls))
        return _fill(groups, counts, syls)
    out = []
    for gs, ls in _match_lines(groups, lines):
        line = [s for l in ls for s in l]
        out.extend(_fill(gs, [len(line)] if len(gs) == 1 else _distribute(gs, len(line)), line))
    return out


def _fill(groups, counts, syls):
    out, s = [], 0
    for g, k in zip(groups, counts):
        out.extend(_align_group(g, syls[s:s + k]) if k else _mark_rest(g))
        s += k
    return out


def _match_lines(groups, lines, most=4):
    """프레이즈 묶음과 가사 줄 묶음을 순서대로 짝짓는다 → [(프레이즈들, 줄들)]. 한쪽은 늘 하나:
    (프레이즈 여럿, 줄 하나) 또는 (프레이즈 하나, 줄 여럿). 음절이 경계에 가장 잘 맞는 짝을 고른다 (동적 계획법)."""
    G, L = len(groups), len(lines)
    NEG = -1e18
    best = [[NEG] * (L + 1) for _ in range(G + 1)]
    back = [[None] * (L + 1) for _ in range(G + 1)]
    best[0][0] = 0.0
    evs = [_evidence(g) for g in groups]

    def score(g0, g1, l0, l1):
        k = sum(len(line) for line in lines[l0:l1])
        if g1 - g0 == 1:
            v = float(_utilities(evs[g0], k)[k])
        elif k < g1 - g0:
            return NEG
        else:
            counts = _distribute(groups[g0:g1], k)
            v = sum(float(_utilities(evs[g], c)[c]) for g, c in zip(range(g0, g1), counts))
        # 합칠수록 조금씩 덜 그럴듯하다 (보통은 한 줄 = 한 프레이즈)
        return v - 2.0 * ((g1 - g0 - 1) + (l1 - l0 - 1))

    for g in range(G + 1):
        for l in range(L + 1):
            if best[g][l] == NEG:
                continue
            moves = [(g + m, l + 1) for m in range(1, most + 1)] + [(g + 1, l + m) for m in range(2, most + 1)]
            for g2, l2 in moves:
                if g2 > G or l2 > L:
                    continue
                v = score(g, g2, l, l2)
                if v > NEG / 2 and best[g][l] + v > best[g2][l2]:
                    best[g2][l2] = best[g][l] + v
                    back[g2][l2] = (g, l)
    if back[G][L] is None:  # 너무 어긋나면 모두 한 덩어리로 (음절 수만 나눈다)
        return [(groups, lines)]
    pairs, g, l = [], G, L
    while (g, l) != (0, 0):
        g0, l0 = back[g][l]
        pairs.append((groups[g0:g], lines[l0:l]))
        g, l = g0, l0
    return pairs[::-1]




def _mark_rest(g):
    for n in g:
        n.lyric = "+"
    return g


def _distribute(groups, total):
    """음절 total개를 프레이즈마다 몇 개씩 넣을지. 프레이즈마다 1개 이상.
    best[j] = 지금 프레이즈까지 (프레이즈 수 + j)개를 썼을 때 가장 좋은 점수. 프레이즈마다 k를 하나씩 늘리며
    j 전체를 numpy로 한꺼번에 넓힌다. 점수가 같으면 큰 k (= 앞 프레이즈들에 적게)."""
    G = len(groups)
    if G == 0 or total < G:  # 음절이 프레이즈보다 적다: 앞에서부터 하나씩
        return [1 if i < total else 0 for i in range(G)]
    n = total - G + 1  # 한 프레이즈에 넣을 수 있는 최대 개수 (남은 프레이즈에 1개씩 남긴다)
    best = np.full(n, -np.inf)
    best[0] = 0.0
    backs = []
    for g in groups:
        u = _utilities(_evidence(g), n)
        nxt, back = np.full(n, -np.inf), np.zeros(n, dtype=np.int64)
        for k in range(1, n + 1):  # k개를 넣으면 j → j + k − 1
            v = best[:n - k + 1] + u[k]
            cur = nxt[k - 1:]
            back[k - 1:][v >= cur] = k
            np.maximum(cur, v, out=cur)
        best = nxt
        backs.append(back)
    counts, j = [], n - 1
    for back in reversed(backs):
        counts.append(int(back[j]))
        j -= counts[-1] - 1
    return counts[::-1]


# ---------------------------------------------------------------- 파일에서 읽기
def _tick_to_sec(tempos):
    tempos = sorted(tempos) or [(0, 120.0)]
    marks, sec = [], 0.0
    for i, (tick, bpm) in enumerate(tempos):
        if i:
            sec += (tick - tempos[i - 1][0]) / TICKS * 60 / tempos[i - 1][1]
        marks.append((tick, sec, bpm))

    def conv(tick):
        t0, s0, bpm = next((m for m in reversed(marks) if m[0] <= tick), marks[0])
        return s0 + (tick - t0) / TICKS * 60 / bpm
    return conv


def read_file(path):
    """MIDI · OpenUtau(.ustx) · UTAU(.ust) → (음표 목록, 첫 빠르기). 파일의 0초 = 오디오 0초."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ustx":
        import yaml
        proj = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        tempos = [(int(t.get("position", 0)), float(t["bpm"])) for t in proj.get("tempos") or []] \
            or [(0, float(proj.get("bpm", 120)))]
        conv = _tick_to_sec(tempos)
        parts = proj.get("voice_parts") or []
        track = min((int(p.get("track_no", 0)) for p in parts), default=0)
        raw = []
        for part in parts:
            if int(part.get("track_no", 0)) != track:
                continue
            off = int(part.get("position", 0))
            for n in part.get("notes") or []:
                a = off + int(n["position"])
                raw.append((conv(a), conv(a + int(n["duration"])), int(n["tone"]), str(n.get("lyric") or "")))
        bpm = tempos[0][1]
    elif suffix == ".ust":
        raw, bpm = _read_ust(path)
    elif suffix in (".mid", ".midi"):
        raw, bpm = _read_midi(path)
    elif suffix == ".json":  # StemMixer 음표 (midi/vocals.json): 초 단위
        import json
        d = json.loads(path.read_text())
        raw = [(float(n[0]), float(n[1]), int(n[2]), str(n[4]) if len(n) > 4 and n[4] else "")
               for n in (d.get("edited") or d.get("raw") or [])]
        bpm = float(d.get("tempo") or 120)
    else:
        raise ValueError("음표 파일은 MIDI(.mid) · OpenUtau(.ustx) · UTAU(.ust) · StemMixer 음표만 돼요")
    # 같은 때 시작하는 음(화음 · 겹친 음)은 하나만 남긴다: 높은 음이 뒤에 오게 놓아서 그것이 남는다 (보통 맨 위가 멜로디)
    raw.sort(key=lambda r: (r[0], r[2], r[1]))
    notes = []
    for s, e, tone, lyric in raw:
        if e - s < 0.03:
            continue
        if notes and s < notes[-1].end:  # 겹치면 앞 음표를 자른다
            notes[-1].end = s
            if s - notes[-1].start < 0.03:  # 거의 남지 않으면 버린다 (가사는 이 음표가 받는다)
                lyric = lyric or notes.pop().lyric
        gap = s - notes[-1].end if notes else 99
        notes.append(Note(s, e, tone, lyric, strength=10.0 if gap > REST_GAP else 5.0))
    return notes, bpm


def _read_ust(path):
    data = path.read_bytes()
    text = data.decode("utf-8") if b"Charset=UTF-8" in data[:400] else data.decode("cp932", errors="replace")
    sections, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[#"):
            cur = {"_": line}
            sections.append(cur)
        elif cur is not None and "=" in line:
            k, v = line.split("=", 1)
            cur[k] = v
    setting = next((s for s in sections if s["_"] == "[#SETTING]"), {})
    bpm = float(setting.get("Tempo", "120").replace(",", ".") or 120)
    first, sec, raw = bpm, 0.0, []
    for s in sections:
        if not s["_"][2:-1].isdigit():
            continue
        if s.get("Tempo"):
            bpm = float(s["Tempo"].replace(",", "."))
        dur = int(float(s.get("Length", 0))) / TICKS * 60 / bpm
        lyric = s.get("Lyric", "R").strip()
        if lyric not in ("R", "r", "") and s.get("NoteNum"):
            raw.append((sec, sec + dur, int(s["NoteNum"]), lyric))
        sec += dur
    return raw, first


VOCAL_TRACK = ("vocal", "vox", "melody", "voice", "歌", "ボーカル", "보컬")


def _midi_text(t):
    """mido는 글자를 latin-1로 읽는다 → UTF-8, 아니면 cp932 (일본 MIDI)."""
    for enc in ("utf-8", "cp932"):
        try:
            return t.encode("latin-1").decode(enc)
        except Exception:
            continue
    return t


def _read_midi(path):
    """MIDI → 음표. (트랙, 채널)마다 모아서 노래 파트를 고른다: 가사가 붙은 음표가 많은 것 → 이름이 보컬 · 멜로디인 것 →
    음표가 많은 것 (채널 10 = 드럼은 뺀다). 빠르기가 중간에 바뀌어도 초를 맞게 센다."""
    import mido

    mid = mido.MidiFile(str(path))
    tempos, names, parts = [], {}, {}  # parts: (트랙, 채널) → [(시작 틱, 끝 틱, 음, 가사)]
    for ti, tr in enumerate(mid.tracks):
        tick, lyric, name = 0, "", ""
        on = {}  # (채널, 음) → [(시작 틱, 가사)] 먼저 누른 것부터
        for msg in tr:
            tick += msg.time
            if msg.type == "set_tempo":
                tempos.append((tick, msg.tempo))
            elif msg.type in ("track_name", "instrument_name"):
                name += " " + _midi_text(msg.name)
            elif msg.type == "lyrics":
                lyric = _midi_text(msg.text).strip()
            elif msg.type not in ("note_on", "note_off") or msg.channel == 9:
                continue
            elif msg.type == "note_on" and msg.velocity > 0:
                on.setdefault((msg.channel, msg.note), []).append((tick, lyric))
                lyric = ""
            elif on.get((msg.channel, msg.note)):
                # 먼저 누른 음부터 닫는다: 같은 음을 이어 부르면 다음 note_on이 앞 note_off보다 먼저 올 수 있다 (같은 틱 · 겹침).
                # 이 틱에 막 누른 음뿐이면 길이 0인 음이라 버린다 (pretty_midi와 같게). 겹친 곳은 read_file이 자른다
                t0, ly = on[(msg.channel, msg.note)].pop(0)
                if t0 < tick:
                    parts.setdefault((ti, msg.channel), []).append((t0, tick, msg.note, ly))
        names[ti] = name.lower()
    tempos = sorted(tempos) or [(0, 500000)]
    if tempos[0][0] > 0:
        tempos.insert(0, (0, tempos[0][1]))
    conv = _tick_to_sec([(t * TICKS / mid.ticks_per_beat, 60_000_000 / us) for t, us in tempos])
    raw = []
    if parts:
        def rank(key):
            notes = parts[key]
            return (sum(1 for n in notes if n[3]), any(w in names[key[0]] for w in VOCAL_TRACK), len(notes))
        for t0, t1, p, ly in parts[max(parts, key=rank)]:
            raw.append((conv(t0 * TICKS / mid.ticks_per_beat), conv(t1 * TICKS / mid.ticks_per_beat), p, ly))
    return raw, 60_000_000 / tempos[0][1]

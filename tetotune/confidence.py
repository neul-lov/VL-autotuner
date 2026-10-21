"""틀렸을 수 있는 곳 찾기: 음 높이를 잘 못 읽었거나, 음표 높이가 애매하거나, 가사가 그 음표에 맞는지 확실하지 않거나,
테토로 불러 보니 원래 보컬과 다르게 나온 곳. 앱 타임라인에 표시해서 그 구간만 들어 보고 고칠 수 있게 한다.

표시 하나 = [시작초, 끝초, 수준(1 = 확인해 보기, 2 = 틀렸을 가능성이 큼), 종류("pitch" · "lyric" · "render"), 이유]
"""
import numpy as np

from .pitch import HOP, _runs

MERGE_GAP = 0.15  # 초. 이보다 가까운 표시는 하나로


def note_flags(notes, midi, voiced, shift):
    """조교한 음표마다 (원래 보컬 음높이 midi는 조옮김 전, 음표 tone은 조옮김 뒤)."""
    from scipy.ndimage import median_filter

    out = []
    n = len(midi)
    for x in notes:
        a, b = int(x.start / HOP), min(n, int(x.end / HOP))
        if b - a < 8:
            continue
        reasons, level, kind = [], 0, "pitch"
        vf = float(voiced[a:b].mean())
        if vf < 0.5:
            reasons.append("음 높이를 거의 못 읽었어요 (작거나 숨 섞인 소리 · 반주가 섞였을 수 있어요)")
            level = max(level, 2 if vf < 0.25 else 1)
        c0, c1 = a + (b - a) // 5, b - (b - a) // 5
        core = midi[c0:c1][voiced[c0:c1]]
        if len(core) >= 6:
            if x.lyric != "+" and abs(float(np.median(core)) + shift - x.tone) > 0.6:
                reasons.append("음표 높이가 두 음 사이예요 (어느 쪽인지 들어 보세요)")
                level = max(level, 1)
            if len(core) >= 12:
                wobble = float(np.std(core - median_filter(core, size=11, mode="nearest")))
                if wobble > 0.35:
                    reasons.append("음 높이 읽기가 흔들려요")
                    level = max(level, 1)
            if float(np.ptp(core)) > 7:
                reasons.append("음 높이가 크게 튀어요 (옥타브를 잘못 읽었을 수 있어요)")
                level = max(level, 2)
        if reasons:
            out.append([round(x.start, 3), round(x.end, 3), level, kind, " · ".join(reasons)])
        if getattr(x, "weak", False) and x.lyric != "+":
            out.append([round(x.start, 3), round(x.end, 3), 1, "lyric", f"「{x.lyric}」이(가) 이 음표에서 시작하는지 확실하지 않아요"])
    return out


def render_flags(want_cents, sing, src_voiced, cents, voiced, rel_want, rel_got, min_len=0.08):
    """테토로 불러 본 결과가 목표와 다른 곳: 음 높이 80센트 넘게, 크기 6dB 넘게, 원래는 소리 나는데 테토는 조용한 곳."""
    out = []
    both = sing & src_voiced & voiced
    err = np.where(both, np.abs(cents - want_cents), 0.0)
    k = int(min_len / HOP)
    for a, b in _runs(err > 80):
        if b - a >= k:
            c = int(np.median(err[a:b]))
            out.append([round(a * HOP, 3), round(b * HOP, 3), 2 if c > 200 else 1, "render",
                        f"테토가 {c}센트쯤 다른 음으로 불러요"])
    d = np.where(sing & src_voiced, rel_got - rel_want, 0.0)
    for a, b in _runs(d < -6):
        if b - a >= k:
            out.append([round(a * HOP, 3), round(b * HOP, 3), 2 if np.median(d[a:b]) < -12 else 1, "render",
                        "테토 소리가 원래보다 작아요 (끊겨 들릴 수 있어요)"])
    for a, b in _runs(d > 6):
        if b - a >= k:
            out.append([round(a * HOP, 3), round(b * HOP, 3), 1, "render", "테토 소리가 원래보다 커요"])
    return out


def merge(flags, gap=MERGE_GAP):
    """같은 종류의 가까운 표시를 하나로 (이유는 겹치지 않게 모으고 수준은 높은 쪽)."""
    out = []
    for f in sorted(flags, key=lambda f: (f[3], f[0])):
        if out and out[-1][3] == f[3] and f[0] - out[-1][1] <= gap:
            last = out[-1]
            last[1] = max(last[1], f[1])
            last[2] = max(last[2], f[2])
            if f[4] not in last[4]:
                last[4] = last[4] + " · " + f[4]
        else:
            out.append(list(f))
    return sorted(out, key=lambda f: f[0])

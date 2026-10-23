"""어디서 틀리는지 나눠 보기: CSD 사람 노래(깨끗한 것 · 분리한 것)를 새 기본으로 조교하면서 단계마다 음절 자리 ·
음 높이를 정답과 비교한다 (소리는 재생하지 않고 숫자만. CSD는 CC BY-NC-SA라 숫자만 찍는다).

단계: GAME 음표 → 가사 넣기(notes.align) → 가사 맞추기 모델이 찾은 모음 시작(raw) → 마지막 음표.
- 마지막에 50ms 넘게 틀린 음절이 어느 단계 탓인지 (모델이 맞게 찾았는데 음표에 못 넣음 / 모델이 틀림 / 맞추기를 안 함)
- 모델 시간에 일정하게 더하거나 음절 시작(자음)과 섞으면 얼마나 맞는지
- 음 높이가 틀린 음절이 원래 소리(가수가 부른 음)와는 맞는지 (정답 표시와 가수가 다른지, 우리가 음을 잘못 고르는지)

사용: python diagnose.py --csd csd --csd-n 12 [--sep out/csdsep] [--json out/diag.json]
"""
import argparse
import difflib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "devtools"))
import accuracy as acc  # noqa: E402
from tetotune import align_fa, game  # noqa: E402
from tetotune import notes as nt  # noqa: E402

CAP = {}
_align, _transcribe = nt.align, game.transcribe


def _cap_align(notes, lines):
    CAP["phrases"] = len(nt._phrases(notes))
    CAP["lines"] = len(lines)
    out = _align(notes, lines)
    CAP["aligned"] = [(x.start, x.end, x.tone, x.lyric) for x in out]
    return out


def _cap_game(*a, **k):
    out = _transcribe(*a, **k)
    CAP["game"] = list(out or [])
    return out


nt.align = _cap_align
game.transcribe = _cap_game


def _jamo(s):
    c = ord(s) - 0xAC00
    return c // 588, (c % 588) // 28, c % 28


GLIDE = {2, 3, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19}  # ㅑ ㅒ ㅕ ㅖ ㅘ ㅙ ㅚ ㅛ ㅝ ㅞ ㅟ ㅠ ㅢ


def kind(syl, prev):
    """음절 종류: 모음 시작(ㅇ) · 그중 앞 음절이 받침 없이 끝남(모음 → 모음) · 반모음 · 울리는 자음 · 그 밖."""
    cho, jung, _ = _jamo(syl)
    vv = prev is not None and _jamo(prev)[2] == 0
    if cho == 11:
        return "모음→모음" if vv else "모음 시작"
    if jung in GLIDE:
        return "반모음"
    return "울리는 자음" if nt._voiced_onset(syl) else "그 밖 자음"


def _map(truth_syl, est_syl):
    sm = difflib.SequenceMatcher(None, truth_syl, est_syl, autojunk=False)
    return {i + k: j + k for i, j, n in sm.get_matching_blocks() for k in range(n)}


def _rate(err, n, tol):
    a = np.abs(np.array([e for e in err if e is not None and np.isfinite(e)]))
    return float(np.sum(a <= tol) / max(1, n))


def song(name, wav, rows):
    CAP.clear()
    with acc.patched(acc.FUSION), tempfile.TemporaryDirectory() as tmp:
        prev = acc.run_tune(wav, acc.csd_lyrics(rows), Path(tmp) / "o")
    log = list(align_fa.LAST)
    truth = [r for r in rows if r[3] != "+"]
    tsyl = [r[3] for r in truth]
    shift = prev["shift"]
    final = [x for x in prev["notes"] if x[3] != "+"]
    aligned = [x for x in CAP.get("aligned", []) if x[3] != "+"]
    m_final, m_al = _map(tsyl, [x[3] for x in final]), _map(tsyl, [x[3] for x in aligned])
    # 가사 맞추기 기록: 프레이즈마다 음절 · 모델 모음 시작 · 음절 시작 · 맞춤 여부 · 안 된 까닭
    fa_syl, fa = [], []
    for r in log:
        syls = r.get("syls") or []
        raw, raws = r.get("raw"), r.get("raw_start")
        why = r.get("why") or r.get("skip") or ("unknown" if "unknown" in r else ("error" if "error" in r else ""))
        for k, s in enumerate(syls):
            fa_syl.append(s)
            fa.append({"v": raw[k] if raw else None, "s": raws[k] if raws else None, "placed": bool(r.get("placed")),
                       "why": "" if r.get("placed") else (why or "?"), "first": k == 0, "last": k == len(syls) - 1,
                       "conf": r.get("conf")})
    m_fa = _map(tsyl, fa_syl)
    hop = prev["hop"]
    src = np.array([np.nan if v is None else v - shift for v in prev["source"]], dtype=float)  # 2프레임마다
    out = []
    for i, t in enumerate(truth):
        d = {"t": t[0], "dur": t[1] - t[0], "midi": t[2], "kind": kind(t[3], truth[i - 1][3] if i else None)}
        if i in m_final:
            x = final[m_final[i]]
            d["f"] = (x[0] - t[0]) * 1000
            d["tone"] = x[2] - shift - t[2]
        if i in m_al:
            d["a"] = (aligned[m_al[i]][0] - t[0]) * 1000
            d["atone"] = aligned[m_al[i]][2] - t[2]
        if i in m_fa:
            f = fa[m_fa[i]]
            d.update({"placed": f["placed"], "why": f["why"], "first": f["first"], "last": f["last"], "conf": f["conf"]})
            if f["v"] is not None:
                d["v"] = (f["v"] - t[0]) * 1000
                d["s"] = (f["s"] - t[0]) * 1000
        a, b = t[0] + 0.25 * (t[1] - t[0]), t[1] - 0.25 * (t[1] - t[0])  # 음절 가운데 절반에서 가수가 부른 음
        seg = src[int(a / (2 * hop)):max(int(a / (2 * hop)) + 1, int(b / (2 * hop)))]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            d["sung"] = float(np.median(seg)) - t[2]
        out.append(d)
    g = CAP.get("game", [])
    gm = acc.match_onsets([t[0] for t in truth], [x[0] for x in g])
    info = {"name": name, "n": len(truth), "lines": CAP.get("lines"), "phrases": CAP.get("phrases"),
            "game_notes": len(g), "game_hit50": len(gm) / max(1, len(truth)),
            "tuning_cents": float(np.nanmedian([d["sung"] for d in out if "sung" in d]) * 100) if out else None}
    print(f"  {name}: 음절 {len(truth)}개, 가사 {info['lines']}줄 / 찾은 프레이즈 {info['phrases']}개, "
          f"GAME 음표 {len(g)}개 (정답 시작 50ms 안 {info['game_hit50']:.3f}), 가수 음정 치우침 {info['tuning_cents']:+.0f}센트, "
          f"마지막 50ms 안 {_rate([d.get('f') for d in out], len(out), 50):.3f}", flush=True)
    return info, out


def report(label, songs, syl):
    n = len(syl)
    print(f"\n## {label}: {len(songs)}곡, 음절 {n}개", flush=True)
    print("| 단계 | 들어간 음절 | 50ms 안 | 100ms 안 | 치우침 중앙 ms | 절댓값 중앙 ms |", flush=True)
    print("|---|---|---|---|---|---|", flush=True)
    for key, title in (("a", "가사 넣기 (GAME 음표에)"), ("s", "모델 음절 시작 (자음)"), ("v", "모델 모음 시작"), ("f", "마지막 음표")):
        e = [d[key] for d in syl if key in d]
        if not e:
            continue
        print(f"| {title} | {len(e) / n:.3f} | {_rate(e, n, 50):.3f} | {_rate(e, n, 100):.3f} | {np.median(e):+.0f} | "
              f"{np.median(np.abs(e)):.0f} |", flush=True)
    gh = np.mean([s["game_hit50"] for s in songs])
    print(f"GAME 음표 시작이 정답 음절 시작 50ms 안 (가사 없이 가까운 것끼리): {gh:.3f}", flush=True)

    cov = [d for d in syl if "v" in d]
    cells = []
    for off in range(-40, 50, 10):
        cells.append(f"{off:+d}ms {_rate([d['v'] + off for d in cov], n, 50):.3f}")
    print("모델 모음 시작에 더하면 (50ms 안, 모든 음절 대비): " + ", ".join(cells), flush=True)
    cells = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        cells.append(f"{w:.2f} {_rate([d['s'] + w * (d['v'] - d['s']) for d in cov], n, 50):.3f}")
    print("음절 시작 → 모음 시작 사이 어디가 정답에 가까운지 (0 = 음절 시작, 1 = 모음 시작): " + ", ".join(cells), flush=True)
    cells = []
    for off in range(-40, 50, 10):
        cells.append(f"{off:+d}ms {_rate([d['f'] + off for d in syl if 'f' in d], n, 50):.3f}")
    print("마지막 음표 시작에 더하면: " + ", ".join(cells), flush=True)

    # 마지막에 50ms 넘게 틀린 음절은 어느 단계 탓인지
    bad = [d for d in syl if "f" in d and abs(d["f"]) > 50]
    cause = Counter()
    for d in bad:
        if not d.get("placed"):
            w = d.get("why") or "?"
            cause["맞추기 안 함: " + ("음표 나누기 실패" if w.startswith("split") else "틈" if w.startswith(("gap", "end"))
                                   else w)] += 1
        elif "v" in d and abs(d["v"]) <= 50:
            cause["모델은 맞음, 음표가 다른 자리"] += 1
        else:
            cause["모델이 틀림"] += 1
    print(f"마지막에 50ms 넘게 틀린 음절 {len(bad)}개 ({len(bad) / n:.3f}): "
          + ", ".join(f"{k} {v}" for k, v in cause.most_common()), flush=True)
    lost = [d for d in bad if d.get("placed") and "v" in d and abs(d["v"]) <= 50]
    if lost:
        print("  모델은 맞았는데 음표가 다른 자리: 마지막 오차 중앙 "
              f"{np.median([d['f'] for d in lost]):+.0f}ms, 처음 음절 {sum(d.get('first', False) for d in lost)}개, "
              f"가사 넣기 그대로인 것 {sum('a' in d and abs(d['a'] - d['f']) < 1 for d in lost)}개", flush=True)
    print("| 음절 종류 | 수 | 마지막 50ms 안 | 치우침 중앙 | 모델 모음 시작 50ms 안 | 모델 치우침 중앙 |", flush=True)
    print("|---|---|---|---|---|---|", flush=True)
    for k in ("모음 시작", "모음→모음", "반모음", "울리는 자음", "그 밖 자음"):
        g = [d for d in syl if d["kind"] == k]
        if not g:
            continue
        f = [d["f"] for d in g if "f" in d]
        v = [d["v"] for d in g if "v" in d]
        print(f"| {k} | {len(g)} | {_rate(f, len(g), 50):.3f} | {np.median(f) if f else float('nan'):+.0f} | "
              f"{_rate(v, len(g), 50):.3f} | {np.median(v) if v else float('nan'):+.0f} |", flush=True)
    for k, sel in (("프레이즈 첫 음절", lambda d: d.get("first")), ("프레이즈 끝 음절", lambda d: d.get("last")),
                   ("짧은 음절 (<0.2초)", lambda d: d["dur"] < 0.2), ("확신 < 0.5인 프레이즈", lambda d: (d.get("conf") or 1) < 0.5)):
        g = [d for d in syl if sel(d)]
        if g:
            print(f"{k}: {len(g)}개, 마지막 50ms 안 {_rate([d.get('f') for d in g], len(g), 50):.3f}, "
                  f"모델 50ms 안 {_rate([d.get('v') for d in g], len(g), 50):.3f}", flush=True)

    # 음 높이
    tone = Counter()
    for d in syl:
        if "tone" not in d:
            continue
        x = d["tone"]
        sung = d.get("sung")
        sang_same = sung is not None and abs(sung) < 0.5
        if x == 0:
            tone["맞음"] += 1
        elif abs(x) == 12:
            tone["옥타브"] += 1
        elif sung is not None and not sang_same and round(sung) == x:
            tone[f"가수가 그 음으로 부름 (±{abs(x)})"] += 1
        else:
            tone[f"우리가 잘못 고름 (±{min(abs(x), 3)}{'+' if abs(x) >= 3 else ''})"] += 1
    print("음절 음 높이: " + ", ".join(f"{k} {v}" for k, v in tone.most_common()), flush=True)
    sung = [d["sung"] for d in syl if "sung" in d]
    if sung:
        a = np.abs(np.array(sung))
        print(f"가수가 부른 음 (음절 가운데 절반 중앙) − 정답 음: 25센트 안 {np.mean(a < 0.25):.3f}, 50센트 안 {np.mean(a < 0.5):.3f}, "
              f"곡마다 치우침 {', '.join('%+.0f' % s['tuning_cents'] for s in songs)}센트", flush=True)
    sh = [d for d in syl if "tone" in d and d["dur"] < 0.2]
    lg = [d for d in syl if "tone" in d and d["dur"] >= 0.2]
    print(f"음 높이 맞음: 짧은 음절 {np.mean([d['tone'] == 0 for d in sh]) if sh else float('nan'):.3f} ({len(sh)}개), "
          f"긴 음절 {np.mean([d['tone'] == 0 for d in lg]) if lg else float('nan'):.3f} ({len(lg)}개), "
          f"가사 넣기 단계 {np.mean([d['atone'] == 0 for d in syl if 'atone' in d]):.3f}", flush=True)
    mism = [s for s in songs if s["lines"] != s["phrases"]]
    print(f"가사 줄 수와 찾은 프레이즈 수가 다른 곡 {len(mism)}개: "
          + ", ".join(f"{s['name']} {s['lines']}/{s['phrases']}" for s in mism), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csd", required=True)
    ap.add_argument("--csd-n", type=int, default=12)
    ap.add_argument("--sep", help="make_separated.py --csd 결과 폴더: 있으면 분리한 보컬로")
    ap.add_argument("--json")
    args = ap.parse_args()
    items = acc.csd_songs(args.csd, args.csd_n)
    label = "CSD 깨끗한 보컬"
    if args.sep:
        items = [(n, Path(args.sep) / f"sep_{n}.wav", r) for n, _, r in items if (Path(args.sep) / f"sep_{n}.wav").exists()]
        label = "CSD 분리한 보컬"
    songs, syl = [], []
    for name, wav, rows in items:
        info, out = song(name, wav, rows)
        songs.append(info)
        syl += out
    report(label, songs, syl)
    if args.json:
        Path(args.json).write_text(json.dumps({"songs": songs, "syl": syl}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

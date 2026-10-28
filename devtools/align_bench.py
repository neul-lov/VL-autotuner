"""가사 맞추기 엔진 비교: CSD 한국 동요(사람이 부른 노래, 음절마다 시작 정답)에서 엔진만 떼어 견준다.

정답 음절을 쉼(0.25초 넘는 틈)마다 프레이즈로 나누고, 이웃 프레이즈를 --joint초까지 묶어 (앱의 align_fa.JOINT처럼)
같은 소리 구간 · 같은 음절을 엔진마다 준다. 엔진이 찾은 음절 시작 · 모음 시작을 정답 시작과 비교한다.
음표 나누기 · 가사 넣기와 상관없이 엔진 자체가 얼마나 맞는지 보려는 것 (끝까지 조교한 결과는 accuracy.py).
소리는 재생하지 않고 숫자만 찍는다 (CSD는 CC BY-NC-SA라 숫자만).

엔진 (--engines, 쉼표로):
  hfa   지금 쓰는 HubertFA (일본어 노래 모델, 한글은 일본어 음소로), 모음 시작 + align_fa.OFFSET
  mfa   Montreal Forced Aligner 한국어 모델 (말소리로 배운 HMM, korean_mfa). PATH에 mfa가 있어야 한다
  mms   torchaudio MMS_FA (여러 말 wav2vec2 CTC, 한글은 uroman으로 로마자)
합치기: 엔진이 셋 이상이면 음절마다 가운데 값, 둘이면 평균도 같이 잰다.

사용: python align_bench.py --csd csd [--csd-n 0] [--sep out/csdsep] --engines hfa,mfa,mms [--json out/align.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "devtools"))
import accuracy as acc  # noqa: E402
from tetotune import align_fa, audio  # noqa: E402

GAP = 0.25  # 초. 정답에서 이보다 긴 틈은 프레이즈 사이 (accuracy.csd_lyrics와 같게)


# ---------------------------------------------------------------- 시험 조각
def items_for(name, wav, rows, joint):
    """노래 하나 → 맞출 조각 [{song, a, b, syls, truth}]: 음절은 "+"를 뺀 정답 음절 순서대로."""
    phrases, cur = [], []
    for r in rows:
        if cur and r[0] - cur[-1][1] > GAP:
            phrases.append(cur)
            cur = []
        cur.append(r)
    if cur:
        phrases.append(cur)
    phrases = [p for p in phrases if any(r[3] != "+" for r in p)]

    def window(i, j):
        a = max(0.0, phrases[i][0][0] - align_fa.WIN_BEFORE, phrases[i - 1][-1][1] if i else 0.0)
        b = phrases[j][-1][1] + align_fa.WIN_AFTER
        if j + 1 < len(phrases):
            b = min(b, phrases[j + 1][0][0])
        return a, b

    out, i = [], 0
    while i < len(phrases):
        j = i
        while joint > 0 and j + 1 < len(phrases) and window(i, j + 1)[1] - window(i, j + 1)[0] <= joint:
            j += 1
        a, b = window(i, j)
        syl = [r for p in phrases[i:j + 1] for r in p if r[3] != "+"]
        out.append({"song": name, "wav": str(wav), "a": a, "b": b, "syls": [r[3] for r in syl], "truth": [r[0] for r in syl]})
        i = j + 1
    return out


_AUDIO = {}


def clip(item, sr=None):
    """조각 소리 (모노, sr). 같은 노래는 한 번만 읽는다."""
    key = (item["wav"], sr)
    if key not in _AUDIO:
        _AUDIO.clear()
        y, s = audio.load(item["wav"])
        if sr and sr != s:
            y, s = audio.resample(y, s, sr), sr
        _AUDIO[key] = (y, s)
    y, s = _AUDIO[key]
    return y[int(item["a"] * s):int(item["b"] * s)], s


# ---------------------------------------------------------------- 엔진
class HFA:
    """지금 앱이 쓰는 가사 맞추기 (align_fa.Aligner)."""
    name = "hfa"

    def __init__(self):
        folder = align_fa.model_dir() or align_fa.ensure_model(print)
        if folder is None:
            raise RuntimeError("HubertFA 모델 없음")
        self.al = align_fa.Aligner(folder)

    def prepare(self, items):
        pass

    def align(self, item):
        y, sr = clip(item)
        times, conf = self.al.align(y, sr, item["syls"])
        if times is None:
            return None
        return [(item["a"] + s, item["a"] + v + align_fa.OFFSET) for s, v, _ in times]


ENGINES = {"hfa": HFA}


def engine(name):
    if name not in ENGINES:
        # 다른 엔진은 따로 파일로 (무거운 라이브러리를 엔진을 쓸 때만 불러온다)
        import align_engines
        align_engines.register(sys.modules[__name__])
    return ENGINES[name]()


# ---------------------------------------------------------------- 채점
GLIDE = {2, 3, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19}  # ㅑ ㅒ ㅕ ㅖ ㅘ ㅙ ㅚ ㅛ ㅝ ㅞ ㅟ ㅠ ㅢ


def kind(syl, prev):
    """음절 종류 (diagnose.kind와 같게): 모음 시작(ㅇ) · 모음 → 모음 · 반모음 · 울리는 자음 · 그 밖."""
    from tetotune import notes as nt

    c = ord(syl) - 0xAC00
    cho, jung = c // 588, (c % 588) // 28
    if cho == 11:
        return "모음→모음" if prev is not None and (ord(prev) - 0xAC00) % 28 == 0 else "모음 시작"
    if jung in GLIDE:
        return "반모음"
    return "울리는 자음" if nt._voiced_onset(syl) else "그 밖 자음"


def score(items, preds, label):
    """preds[k]: 조각 k의 음절마다 (음절 시작, 모음 시작) 또는 None → 표 한 줄 값."""
    n, ev, es, kinds = 0, [], [], {}
    for it, p in zip(items, preds):
        n += len(it["syls"])
        if p is None:
            continue
        for k, ((s, v), t) in enumerate(zip(p, it["truth"])):
            ev.append((v - t) * 1000)
            es.append((s - t) * 1000)
            kd = kind(it["syls"][k], it["syls"][k - 1] if k else None)
            kinds.setdefault(kd, []).append((v - t) * 1000)
    a = np.abs(np.array(ev)) if ev else np.array([np.nan])

    def rate(x, tol, total):
        return float(np.sum(np.abs(np.array(x)) <= tol) / max(1, total)) if len(x) else 0.0
    return {"label": label, "n": n, "covered": len(ev) / max(1, n), "hit20": rate(ev, 20, n), "hit50": rate(ev, 50, n),
            "hit100": rate(ev, 100, n), "med": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "bias": float(np.median(ev)) if ev else float("nan"), "start_hit50": rate(es, 50, n),
            "mix75_hit50": rate([s + 0.75 * (v - s) for s, v in zip(es, ev)], 50, n),
            "kinds": {k: [len(v), rate(v, 50, len(v)), float(np.median(v))] for k, v in kinds.items()}}


def fuse(preds_by_engine, how):
    """엔진별 예측 → 음절마다 가운데 값 (how="median") 또는 평균 ("mean"). 한 엔진이라도 못 맞춘 조각은 나머지로."""
    names = list(preds_by_engine)
    out = []
    for k in range(len(preds_by_engine[names[0]])):
        got = [np.array(preds_by_engine[e][k], dtype=float) for e in names if preds_by_engine[e][k] is not None]
        if not got:
            out.append(None)
            continue
        st = np.stack(got)
        val = np.median(st, axis=0) if how == "median" else np.mean(st, axis=0)
        out.append([tuple(x) for x in val])
    return out


def print_table(rows):
    print("| 엔진 | 맞춘 음절 | 모음 시작 20 / 50 / 100ms 안 | 오차 중앙 / 90% ms | 치우침 ms | 음절 시작 50ms 안 | "
          "음절→모음 3/4 50ms 안 |", flush=True)
    print("|---|---|---|---|---|---|---|", flush=True)
    for r in rows:
        print(f"| {r['label']} | {r['covered']:.3f} | {r['hit20']:.3f} / {r['hit50']:.3f} / {r['hit100']:.3f} | "
              f"{r['med']:.0f} / {r['p90']:.0f} | {r['bias']:+.0f} | {r['start_hit50']:.3f} | {r['mix75_hit50']:.3f} |",
              flush=True)
    kinds = sorted({k for r in rows for k in r["kinds"]})
    print("\n| 엔진 | " + " | ".join(f"{k} 50ms 안 (치우침)" for k in kinds) + " |", flush=True)
    print("|---|" + "---|" * len(kinds), flush=True)
    for r in rows:
        cells = [f"{r['kinds'][k][1]:.3f} ({r['kinds'][k][2]:+.0f})" if k in r["kinds"] else "" for k in kinds]
        print(f"| {r['label']} | " + " | ".join(cells) + " |", flush=True)


def report(items, preds):
    """엔진마다 + 합치기 + 가장 가까운 것 고르기 (상한) → 표 줄들."""
    names = list(preds)
    rows = [score(items, preds[e], e) for e in names]
    if len(names) >= 3:
        rows.append(score(items, fuse(preds, "median"), "가운데 값 (" + "+".join(names) + ")"))
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair = {k: preds[k] for k in (names[i], names[j])}
                rows.append(score(items, fuse(pair, "mean"), f"평균 ({names[i]}+{names[j]})"))
        # 엔진 중 하나라도 맞으면 (어느 것을 고를지 완벽히 안다면 얼마까지 되는지)
        best = []
        for k, it in enumerate(items):
            got = [preds[e][k] for e in names if preds[e][k] is not None]
            if not got:
                best.append(None)
                continue
            best.append([min((g[s] for g in got), key=lambda p: abs(p[1] - t)) for s, t in enumerate(it["truth"])])
        rows.append(score(items, best, "가장 가까운 것 고르기 (상한)"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csd", required=True)
    ap.add_argument("--csd-n", type=int, default=0, help="CSD 몇 곡 (0 = 모두)")
    ap.add_argument("--sep", help="make_separated.py --csd 결과 폴더: 있으면 분리한 보컬로")
    ap.add_argument("--engines", default="hfa")
    ap.add_argument("--joint", type=float, default=align_fa.JOINT, help="초. 이웃 프레이즈를 이 길이까지 묶는다 (0 = 프레이즈마다)")
    ap.add_argument("--json")
    args = ap.parse_args()
    songs = acc.csd_songs(args.csd, args.csd_n)
    if args.sep:
        songs = [(n, Path(args.sep) / f"sep_{n}.wav", r) for n, _, r in songs if (Path(args.sep) / f"sep_{n}.wav").exists()]
    items = [it for name, wav, rows in songs for it in items_for(name, wav, rows, args.joint)]
    print(f"CSD {len(songs)}곡, 조각 {len(items)}개, 음절 {sum(len(it['syls']) for it in items)}개"
          f"{' (분리한 보컬)' if args.sep else ''}, 묶는 길이 {args.joint}초", flush=True)
    preds = {}
    for name in args.engines.split(","):
        try:
            eng = engine(name)
            eng.prepare(items)
            out = []
            for it in items:
                try:
                    out.append(eng.align(it))
                except Exception as e:  # 한 조각만 못 하면 그 조각만 빼고
                    print(f"  {name} {it['song']} {it['a']:.1f}s: {str(e)[:200]}", flush=True)
                    out.append(None)
        except Exception as e:
            print(f"{name}: 쓰지 못함 ({str(e)[:300]})", flush=True)
            continue
        preds[name] = out
        r = score(items, out, name)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "kinds"},
                         ensure_ascii=False), flush=True)
    tables = {"모든 곡": list(range(len(items)))}
    if not args.csd_n and not args.sep:  # 정확도 비교(accuracy.py)에 쓰는 12곡과 나머지 (맞추며 고른 값이 다른 곡에도 맞는지)
        tuned = {n for n, _, _ in acc.csd_songs(args.csd, 12)}
        tables["정확도 비교 12곡"] = [k for k, it in enumerate(items) if it["song"] in tuned]
        tables["나머지 곡"] = [k for k, it in enumerate(items) if it["song"] not in tuned]
    rows = []
    for title, idx in tables.items():
        if not idx:
            continue
        sub = [items[k] for k in idx]
        part = report(sub, {e: [p[k] for k in idx] for e, p in preds.items()})
        print(f"\n### {title} (조각 {len(sub)}개, 음절 {sum(len(it['syls']) for it in sub)}개)\n", flush=True)
        print_table(part)
        rows += [dict(r, table=title) for r in part]
    if args.json:  # 표 + 음절마다 오차만 (정답 시각 · 가사는 CSD 것이라 싣지 않는다). 합치기는 오차만으로 다시 잴 수 있다
        err = {e: [None if q is None else [[round((a - t) * 1000, 1), round((v - t) * 1000, 1)] for (a, v), t in zip(q, it["truth"])]
                   for q, it in zip(p, items)] for e, p in preds.items()}
        meta = [{"song": it["song"], "kinds": [kind(x, it["syls"][k - 1] if k else None) for k, x in enumerate(it["syls"])],
                 "gaps": [round(b - a, 3) for a, b in zip(it["truth"], it["truth"][1:])]} for it in items]
        Path(args.json).write_text(json.dumps({"rows": rows, "items": meta, "err": err}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

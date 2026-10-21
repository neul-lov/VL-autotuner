"""분리한 보컬로 조교했을 때 테토에게 부르라고 넣는 음 높이(렌더 전)가 깨끗한 원래 보컬과 어디서 크게 다른지 찾고,
넣는 음 높이를 만드는 방법(VARIANTS)마다 비교한다.

렌더 비교(compare_render.py --truth)의 "떨어짐 · 튐"과 같은 기준으로 센다: 깨끗한 보컬도 소리 나는 음표 안에서
넣은 음 높이가 깨끗한 보컬(Harvest)보다 150센트 넘게 낮은 · 높은 곳이 30ms 넘게. 렌더 없이 조교만 하니까 빠르다.
같은 곳을 깨끗한 보컬로 조교했을 때와도 비교한다 (분리 때문인지, 조교 방법 때문인지).
- 음표는 맞는데 벗어남: 그 자리 음표가 깨끗한 보컬과 150센트 안인데 넣은 음 높이만 벗어난 곳 = 테토가 맞는 음에서 확
  떨어지거나 튀는 곳. 음표도 깨끗한 보컬과 다른 곳은 대부분 사람이 음절 사이 · 음 끝에서 잠깐 꺾은 것을 분리한 보컬(잔향)이
  덮어서 테토가 음표 음을 그대로 부르는 곳이다.
- 왜: 정답 의심(깨끗한 보컬 Harvest가 사람이 표시한 vocadito f0와 150센트 넘게 다름) · 빈 곳 잇기(분리한 보컬이 거기서
  소리 없음이라 앞뒤를 이어 그음) · 읽기 틀림(분리한 보컬에서 읽은 음 높이부터 다름)

사용: python target_check.py <vocadito 폴더> <분리한 보컬 폴더 (sep_<번호>.wav)> [번호,번호,...]
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "devtools"))
from f0_robust import truth  # noqa: E402
from tetotune import audio, features, pitch  # noqa: E402
from tetotune.pitch import HOP  # noqa: E402
from tetotune.tune import Options, tune  # noqa: E402

OPTS = {"transpose": "0", "pitch_method": "harvest"}


def analyse(y, sr, n):
    """compare_render.py와 같은 방법: Harvest → clean → 음높이가 잡힌 잡음 빼기."""
    f0, _ = pitch.extract(y, sr, "harvest")
    f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
    db = pitch.frame_db(y, sr, n)
    m, v = pitch.clean(f0, db)
    feats = features.analyze(y, sr, np.where(v, pitch.midi_to_hz(np.nan_to_num(m, nan=60)), 0), v)
    v = v & ~features.noisy_voiced(v, feats, db)
    return np.where(v, m, np.nan), v


def where(nl, t):
    k = next((i for i, x in enumerate(nl) if x[0] <= t < x[1]), None)
    if k is None:
        return "음표 밖", None
    x = nl[k]
    w = "음표 시작" if t - x[0] < 0.08 else ("음표 끝" if x[1] - t < 0.08 else "음표 가운데")
    if k and abs(nl[k - 1][1] - x[0]) < 1e-3 and t - x[0] < 0.08:
        w = "이어진 음표 경계"
    if k + 1 < len(nl) and abs(nl[k + 1][0] - x[1]) < 1e-3 and x[1] - t < 0.08:
        w = "이어진 음표 경계"
    return w, k


VARIANTS = {  # 이름 → (옥타브 접기, 긴 틈 기준 초)
    "지금": (False, 0.0),
    "옥타브 접기": (True, 0.0),
    "긴 틈은 음표 음 50ms": (False, 0.05),
    "옥타브 접기 + 긴 틈 50ms": (True, 0.05),
}


def check(wav, ms, vs, annot, n):
    """wav로 한 번 조교하고(음 높이 읽기 · 음표는 같으니까), 넣는 음 높이만 방법마다 다시 만들어 깨끗한 보컬과 비교."""
    import copy

    import tetotune.tune as tm

    got = {}
    real = tm._fill_gaps

    def grab(target, mask, notes, base, min_gap, **kw):
        got.update(target=target.copy(), mask=mask.copy(), notes=copy.deepcopy(notes), base=base.copy())
        return real(target, mask, notes, base, min_gap, **kw)

    tm._fill_gaps = grab
    octave, tm.OCTAVE_FIX = tm.OCTAVE_FIX, False  # 접기 전 음 높이를 받는다
    try:
        with tempfile.TemporaryDirectory() as tmp:
            res = tune(wav, tmp, "", options=Options(**OPTS), title="t")
    finally:
        tm._fill_gaps = real
        tm.OCTAVE_FIX = octave
    out = {}
    for name, (octave, gap) in VARIANTS.items():
        t = got["target"]
        if octave:
            t = tm._fold_octaves(t, got["mask"], got["notes"])
        t = tm._fill_gaps(t, got["mask"], got["notes"], got["base"], gap)
        want = (got["base"] + tm._pitd(t, got["base"], got["notes"], Options(**OPTS))) / 100
        out[name] = compare(want, got["mask"] & np.isfinite(got["target"]), got["notes"], ms, vs, annot, n)
        out[name]["method"] = res.pitch_method
    return out


def compare(want, sv, notes, ms, vs, annot, n):
    m = min(n, len(want))
    ms, vs, annot, want, sv = ms[:m], vs[:m], annot[:m], want[:m], sv[:m]
    nl = [[x.start, x.end, x.tone, x.lyric] for x in notes]
    inside = np.zeros(m, bool)
    step = np.full(m, np.nan)
    for s, e, tone, _ in nl:
        inside[int(s / HOP):int(e / HOP)] = True
        step[int(s / HOP):int(e / HOP)] = tone
    use = vs & inside
    diff = np.where(use, (want - np.nan_to_num(ms)) * 100, 0.0)
    events = []
    for kind, mask in (("떨어짐", diff < -150), ("튐", diff > 150)):
        for a, b in pitch._runs(mask):
            if b - a < 6:
                continue
            sl = slice(a, b)
            t = (a + b) / 2 * HOP
            w, k = where(nl, t)
            c = float(np.median(diff[sl]))
            aerr = (ms[sl] - annot[sl]) * 100
            aerr = float(np.nanmedian(aerr)) if np.isfinite(aerr).any() else float("nan")
            nerr = float(np.nanmedian((step[sl] - ms[sl]) * 100))
            voiced = float(sv[sl].mean())
            if np.isfinite(aerr) and abs(aerr) > 150:
                why = "정답 의심"
            elif voiced < 0.5:
                why = "빈 곳 잇기"
            else:
                why = "읽기 틀림"
            events.append({"t": round(t, 2), "ms": int((b - a) * HOP * 1000), "kind": kind, "where": w, "why": why,
                           "cents": int(c), "note": int(nerr) if np.isfinite(nerr) else None,
                           "note_ok": bool(np.isfinite(nerr) and abs(nerr) < 150),
                           "truth_vs_annot": int(aerr) if np.isfinite(aerr) else None, "sep_voiced": round(voiced, 2)})
    err = np.abs(diff[use])
    return {"minutes": m * HOP / 60, "events": events, "rpa": float((err <= 50).mean()) if use.any() else 0.0,
            "notes": len(nl)}


def main():
    root, sep = Path(sys.argv[1]), Path(sys.argv[2])
    want = {int(x) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 else None
    out = {"분리한 보컬": [], "깨끗한 보컬": []}
    for wav in sorted(sep.glob("sep_*.wav"), key=lambda p: int(p.stem.split("_")[1])):
        k = int(wav.stem.split("_")[1])
        if want and k not in want:
            continue
        clean_wav = next(root.rglob(f"vocadito_{k}.wav"))
        yc, sr = audio.load(clean_wav)
        n = int(len(yc) / sr / HOP) + 1
        ms, vs = analyse(yc, sr, n)
        annot, _ = truth(k, root, n)
        for label, w in (("분리한 보컬", wav), ("깨끗한 보컬", clean_wav)):
            r = check(w, ms, vs, annot, n)
            out[label].append(r)
            line = " · ".join(f"{name} {len(v['events'])}({sum(e['note_ok'] for e in v['events'])})" for name, v in r.items())
            print(f"vocadito {k} {label}: {line}", flush=True)
            if label == "분리한 보컬":
                base = r["지금"]["events"]
                for name, v in r.items():
                    if name == "지금":
                        continue
                    old = {(e["t"], e["kind"]) for e in base}
                    new = {(e["t"], e["kind"]) for e in v["events"]}
                    for e in base:
                        if e["note_ok"] and (e["t"], e["kind"]) not in new:
                            print(f"    {name} 없앰", json.dumps(e, ensure_ascii=False), flush=True)
                    for e in v["events"]:
                        if (e["t"], e["kind"]) not in old:
                            print(f"    {name} 새로", json.dumps(e, ensure_ascii=False), flush=True)
                for e in base:
                    if e["note_ok"]:
                        print("    음표 맞는데 벗어남", json.dumps(e, ensure_ascii=False), flush=True)
    print()
    for label, rs in out.items():
        print(f"\n### {label}\n")
        print("| 방법 | 벗어남 /분 | 음표는 맞는데 벗어남 /분 | 600센트 넘게 /분 | 50센트 안 |")
        print("|---|---|---|---|---|")
        for name in VARIANTS:
            minutes = sum(r[name]["minutes"] for r in rs)
            evs = [e for r in rs for e in r[name]["events"]]
            ok = sum(e["note_ok"] for e in evs)
            big = sum(abs(e["cents"]) > 600 for e in evs)
            rpa = np.mean([r[name]["rpa"] for r in rs])
            print(f"| {name} | {len(evs) / minutes:.2f} | {ok / minutes:.2f} | {big / minutes:.2f} | {rpa:.4f} |", flush=True)
        evs = [e for r in rs for e in r["지금"]["events"]]
        cnt = {}
        for e in evs:
            kk = f"{'음표 맞음' if e['note_ok'] else '음표도 다름'} · {e['why']} · {e['where']}"
            cnt[kk] = cnt.get(kk, 0) + 1
        print("\n지금 방법 벗어남 종류: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])), flush=True)


if __name__ == "__main__":
    main()

"""분리한 보컬로 조교했을 때 테토에게 부르라고 넣는 음 높이(렌더 전)가 깨끗한 원래 보컬과 어디서 크게 다른지 찾는다.

렌더 비교(compare_render.py --truth)의 "떨어짐 · 튐"과 같은 기준으로 센다: 깨끗한 보컬도 소리 나는 음표 안에서
넣은 음 높이가 깨끗한 보컬(Harvest)보다 150센트 넘게 낮은 · 높은 곳이 30ms 넘게. 렌더 없이 조교만 하니까 빠르다.
같은 곳을 깨끗한 보컬로 조교했을 때와도 비교한다 (분리 때문인지, 조교 방법 때문인지).

곳마다 무엇 때문인지:
- 정답 의심: 깨끗한 보컬 Harvest가 사람이 표시한 음 높이(vocadito f0)와 150센트 넘게 다르다 (비교 기준이 틀림)
- 빈 곳 잇기: 분리한 보컬이 거기서 소리 없음(무성 · 숨으로 뺌)이라 앞뒤를 이어 그었다
- 읽기 틀림: 분리한 보컬에서 읽은 음 높이부터 다르다
- 바로잡기: 읽은 음 높이는 맞는데 음 가운데 바로잡기(correct)가 음표 쪽으로 당겼다 (음표가 틀림)

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


def check(wav, ms, vs, annot, n):
    """wav로 조교 → 넣은 음 높이를 깨끗한 보컬 음 높이(ms, vs)와 비교."""
    with tempfile.TemporaryDirectory() as tmp:
        res = tune(wav, tmp, "", options=Options(**OPTS), title="t")
        prev = json.loads((Path(tmp) / "preview.json").read_text())
        z = {k: v for k, v in np.load(Path(tmp) / "curves.npz").items()}
    m = min(n, len(z["base"]))
    ms, vs, annot = ms[:m], vs[:m], annot[:m]
    want = (z["base"][:m] + z["pitd"][:m]) / 100
    target = z["target"][:m] / 100
    sv = z["voiced"][:m]
    nl = prev["notes"]
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
            terr = float(np.nanmedian((target[sl] - ms[sl]) * 100))
            aerr = (ms[sl] - annot[sl]) * 100
            aerr = float(np.nanmedian(aerr)) if np.isfinite(aerr).any() else float("nan")
            nerr = float(np.nanmedian((step[sl] - ms[sl]) * 100))
            voiced = float(sv[sl].mean())
            if np.isfinite(aerr) and abs(aerr) > 150:
                why = "정답 의심"
            elif voiced < 0.5:
                why = "빈 곳 잇기"
            elif abs(terr) > 150:
                why = "읽기 틀림"
            else:
                why = "바로잡기"
            events.append({"t": round(t, 2), "ms": int((b - a) * HOP * 1000), "kind": kind, "where": w, "why": why,
                           "cents": int(c), "target": int(terr), "note": int(nerr) if np.isfinite(nerr) else None,
                           "truth_vs_annot": int(aerr) if np.isfinite(aerr) else None, "sep_voiced": round(voiced, 2)})
    err = np.abs(diff[use])
    return {"minutes": m * HOP / 60, "events": events, "rpa": float((err <= 50).mean()) if use.any() else 0.0,
            "notes": len(nl), "method": res.pitch_method}


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
            r["song"] = k
            out[label].append(r)
            print(f"vocadito {k} {label}: {len(r['events'])}곳 / {r['minutes']:.2f}분 · 50센트 안 {r['rpa']:.3f} · "
                  f"음표 {r['notes']} ({r['method']})", flush=True)
            if label == "분리한 보컬":
                for e in r["events"]:
                    print("   ", json.dumps(e, ensure_ascii=False), flush=True)
    print()
    for label, rs in out.items():
        minutes = sum(r["minutes"] for r in rs)
        evs = [e for r in rs for e in r["events"]]
        print(f"\n### {label}: {len(evs) / max(1e-9, minutes):.2f}곳/분 (떨어짐 {sum(e['kind'] == '떨어짐' for e in evs) / minutes:.2f} · "
              f"튐 {sum(e['kind'] == '튐' for e in evs) / minutes:.2f}), 50센트 안 {np.mean([r['rpa'] for r in rs]):.3f}")
        for key in ("why", "where"):
            cnt = {}
            for e in evs:
                cnt[e[key]] = cnt.get(e[key], 0) + 1
            print("| " + " | ".join(f"{k} {v / minutes:.2f}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])) + " |")
        both = {}
        for e in evs:
            kk = f"{e['why']} · {e['where']}"
            both[kk] = both.get(kk, 0) + 1
        print("| " + " | ".join(f"{k} {v}" for k, v in sorted(both.items(), key=lambda kv: -kv[1])[:10]) + " |", flush=True)


if __name__ == "__main__":
    main()

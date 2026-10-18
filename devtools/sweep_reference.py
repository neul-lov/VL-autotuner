"""정답을 아는 시험 곡 여러 개를 설정 몇 가지로 조교해서 점수를 표로 비교한다 (렌더 없이, 음표 · 가사 · 숨만).
사용: python sweep_reference.py <시험 곡 폴더>...   (폴더마다 ref_ref.wav · truth.json · lyrics.txt)
CI에서 실제 테토로 렌더한 시험 곡에 써서, 기준값을 바꿀 때 어느 쪽이 나은지 한 번에 본다."""
import functools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tetotune import features, notes  # noqa: E402
from tetotune.tune import Options, tune  # noqa: E402


def score(truth, prev):
    """eval_reference.py와 같은 채점 → (가사 맞음, 음 높이 맞음, 음표 수, 시작 오차 중앙 ms, 숨 찾음, 엉뚱한 숨)."""
    est = prev["notes"]
    syll = [n for n in est if n[3] != "+"] or est
    tn = truth["notes"]
    on, tone_ok, lyric_ok = [], 0, 0
    for s, e, tone, ly in tn:
        j = int(np.argmin([abs(x[0] - s) for x in syll]))
        on.append(abs(syll[j][0] - s) * 1000)
        tone_ok += syll[j][2] - prev["shift"] == tone
        lyric_ok += syll[j][3] == ly
    hit = sum(any(min(b, y) - max(a, x) > 0.3 * (b - a) for x, y in prev["breaths"]) for a, b in truth["breaths"])
    extra = sum(1 for x, y in prev["breaths"] if not any(min(b, y) - max(a, x) > 0 for a, b in truth["breaths"]))
    return lyric_ok, tone_ok, len(est), float(np.median(on)), hit, extra


def variants():
    """이름 → 바꿀 것들 [(모듈, 이름, 값)]. 첫 번째가 지금 기본값."""
    nv = features.noisy_voiced
    out = {"기본": []}
    for ml in (0.08, 0.12):
        out[f"잡음 최소 {ml}s"] = [(features, "noisy_voiced", functools.partial(nv, min_len=ml))]
    out["자음 틈 안 메움"] = [(notes, "CONSONANT_GAP", 0.0)]
    if hasattr(notes, "LINE_MATCH"):
        out["줄 맞추기 끔"] = [(notes, "LINE_MATCH", False)]
    if hasattr(notes, "ONSET_MAX"):
        out["자음 조각 끔"] = [(notes, "ONSET_MAX", 0.0)]
        out["짧은 음표 모두"] = [(notes, "ONSET_ANY", True)]
        for v in (0.12, 0.18):
            out[f"자음 조각 {v}s"] = [(notes, "ONSET_MAX", v)]
    return out


def main():
    refs = [Path(p) for p in sys.argv[1:] if (Path(p) / "truth.json").exists()]
    rows = []
    for name, patches in variants().items():
        saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
        for m, a, v in patches:
            setattr(m, a, v)
        try:
            res = []
            for d in refs:
                with tempfile.TemporaryDirectory() as tmp:
                    tune(d / "ref_ref.wav", Path(tmp) / "out", (d / "lyrics.txt").read_text(encoding="utf-8"),
                         options=Options(transpose="0"), title="ref")
                    prev = json.loads((Path(tmp) / "out" / "preview.json").read_text())
                res.append(score(json.load(open(d / "truth.json")), prev))
        finally:
            for m, a, v in saved:
                setattr(m, a, v)
        rows.append((name, res))
    total = sum(len(json.load(open(d / "truth.json"))["notes"]) for d in refs)
    print(f"시험 곡 {len(refs)}개 ({', '.join(d.name for d in refs)}), 음절 {total}개")
    print(f"{'설정':<14} {'가사':>9} {'음 높이':>9} {'음표 수':>8} {'시작 오차':>9} {'숨':>7}  곡마다 가사")
    for name, res in rows:
        ly = sum(r[0] for r in res)
        to = sum(r[1] for r in res)
        nn = sum(r[2] for r in res)
        on = np.median([r[3] for r in res])
        hit = sum(r[4] for r in res)
        ex = sum(r[5] for r in res)
        print(f"{name:<14} {ly:>4}/{total:<4} {to:>4}/{total:<4} {nn:>8} {on:>7.0f}ms {hit:>3}+{ex:<3}  "
              + " ".join(str(r[0]) for r in res))


if __name__ == "__main__":
    main()

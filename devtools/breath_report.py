"""정답을 아는 시험 곡에서 숨 자리 · 노래 자리의 특징을 숫자로 보여 준다 (숨 찾기 기준을 맞출 때).
사용: python breath_report.py <정답 json> <보컬 wav> [결과 폴더]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tetotune import audio, features, pitch  # noqa: E402
from tetotune.pitch import HOP  # noqa: E402


def stats(name, a, b, db, voiced, feats, sing):
    sl = slice(int(a / HOP), int(b / HOP))
    br = feats["breathiness"][sl]
    br = br[np.isfinite(br)]
    print(f"{name:>10} {a:6.2f}-{b:5.2f}s  유성 {voiced[sl].mean():4.0%}  크기 {np.percentile(db[sl], 90) - sing:6.1f}dB"
          f"  높은대역 {np.median(feats['high_ratio'][sl]):.2f}  평탄 {np.median(feats['flatness'][sl]):.3f}"
          f"  비주기 {np.median(br) if len(br) else float('nan'):.2f}")


def main():
    truth = json.load(open(sys.argv[1]))
    y, sr = audio.load(sys.argv[2])
    f0, _ = pitch.extract(y, sr, "harvest")
    n = len(f0)
    db = pitch.frame_db(y, sr, n)
    midi, voiced = pitch.clean(f0, db)
    feats = features.analyze(y, sr, np.where(voiced, pitch.midi_to_hz(np.nan_to_num(midi, nan=60)), 0), voiced)
    sing = float(np.median(db[voiced]))
    print(f"바닥 {np.percentile(db, 3) - sing:.1f}dB (노래 중앙값 기준)")
    for a, b in truth["breaths"]:
        stats("숨", a, b, db, voiced, feats, sing)
    for s, e, _, ly in truth["notes"][::5]:
        stats(f"노래 {ly}", s, min(e, s + 0.3), db, voiced, feats, sing)
    found = features.find_breaths(db, voiced, feats)
    print("찾은 숨:", [(round(a, 2), round(b, 2)) for a, b, *_ in found])
    if len(sys.argv) > 3:
        prev = json.load(open(Path(sys.argv[3]) / "preview.json"))
        tn = truth["notes"]
        print("정답 → 만든 음표 (시작초 가사):")
        for s, e, tone, ly in prev["notes"]:
            near = min(tn, key=lambda t: abs(t[0] - s))
            mark = "" if abs(near[0] - s) < 0.05 and near[3] == ly else "  <-"
            print(f"  {s:6.2f}-{e:5.2f} {ly}  (정답 {near[0]:6.2f} {near[3]}){mark}")


if __name__ == "__main__":
    main()

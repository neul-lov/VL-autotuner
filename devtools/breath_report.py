"""정답을 아는 시험 곡에서 숨 자리 · 노래 자리의 특징을 숫자로 보여 준다 (숨 찾기 기준을 맞출 때).
사용: python breath_report.py <정답 json> <보컬 wav> [결과 폴더]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tetotune import audio, features, pitch  # noqa: E402
from tetotune import notes as nt  # noqa: E402
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
    noise = features.noisy_voiced(voiced, feats, db)
    print("음높이가 잡힌 잡음(숨):", [(round(a * HOP, 2), round(b * HOP, 2)) for a, b in pitch._runs(noise)])
    sing = float(np.median(db[voiced]))
    print(f"바닥 {np.percentile(db, 3) - sing:.1f}dB (노래 중앙값 기준)")
    for a, b in truth["breaths"]:
        stats("숨", a, b, db, voiced, feats, sing)
    for s, e, _, ly in truth["notes"][::5]:
        stats(f"노래 {ly}", s, min(e, s + 0.3), db, voiced, feats, sing)
    voiced = voiced & ~noise
    found = features.find_breaths(db, voiced, feats)
    # 자음 조각 합치기 전의 짧은 음표 (음절 앞 자음이 따로 음표가 되는지)
    keep, nt.ONSET_MAX = getattr(nt, "ONSET_MAX", 0.0), 0.0
    segs, _ = nt.segment(np.where(voiced, midi, np.nan), voiced, db, feats["novelty"], noise=noise,
                        hiband=feats["high_ratio"])
    nt.ONSET_MAX = keep
    for i, x in enumerate(segs):
        if x.end - x.start < 0.16:
            prv = segs[i - 1] if i and x.strength < 10 else None
            nxt = segs[i + 1] if i + 1 < len(segs) and segs[i + 1].strength < 10 else None
            sl = slice(int(x.start / HOP), int(x.end / HOP))
            print(f"짧은 음표 {x.start:.2f}-{x.end:.2f} 음 {x.tone} 세기 {x.strength:.1f}"
                  f" | 앞 {prv.tone if prv else '-'} | 뒤 {nxt.tone if nxt else '-'} 세기 {nxt.strength if nxt else 0:.1f}"
                  f" | 잡음 {noise[max(0, sl.start - 10):sl.stop].mean():.2f}"
                  f" | 자음 같음 {nt._consonant_like(x, voiced, feats['high_ratio'])}")
            stats("", x.start, x.end, db, voiced, feats, sing)
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

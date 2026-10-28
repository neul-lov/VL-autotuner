"""음 높이 읽기 방법 비교 (렌더 없이): Harvest · RMVPE(새 판) · SwiftF0 · CREPE · 셋 합치기 · 넷 합치기를
사람이 표시한 정답(vocadito f0)과 비교.
조교처럼 읽은 뒤 정리(pitch.clean)하고 음표 안 빈 곳은 앞뒤를 이어 메운 다음 잰다 (f0_robust.py와 같은 기준).
- 튐 /분: 정답과 150센트 넘게 다른 곳이 30ms 넘게 이어진 횟수
- RPA: 정답 소리 나는 곳 중 50센트 안 · RPA10: 10센트 안 (정밀함)
분리한 보컬과 깨끗한 보컬 둘 다. 소리는 재생하지 않는다.

사용: python f0_models.py <vocadito 폴더> <분리한 보컬 폴더 (sep_<번호>.wav)>
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "devtools"))
from f0_robust import score, truth  # noqa: E402
from tetotune import audio, f0models, pitch  # noqa: E402

METHODS = ("harvest", "rmvpe2", "swift", "crepe", "fusion", "fusion4")  # fusion4 = 합치기에 CREPE도


def analyze(wav, k, root):
    y, sr = audio.load(wav)
    out = {}
    for m in METHODS:
        if m == "fusion4":
            if not f0models.crepe_path():
                continue
            saved, f0models.USE = f0models.USE, ("rmvpe", "crepe", "swift", "harvest")
            try:
                f0, used = pitch.extract(y, sr, "fusion")
            finally:
                f0models.USE = saved
        else:
            f0, used = pitch.extract(y, sr, m)
        if m != "harvest" and used == "harvest":
            continue
        n = len(f0)
        db = pitch.frame_db(y, sr, n)
        midi, voiced = pitch.clean(f0, db)
        tmidi, notes = truth(k, root, n)
        r = score(midi, voiced, tmidi, notes)
        if r is None:
            continue
        ok = notes & np.isfinite(tmidi)
        r["rpa10_n"] = int((np.abs(r["target"][ok] - tmidi[ok]) * 100 <= 10).sum())
        r["used"] = used
        out[m] = r
    return out


def table(results, label):
    print(f"\n| 방법 | {label} 튐 /분 | RPA (50센트) | RPA10 | 남긴 프레임 |")
    print("|---|---|---|---|---|")
    for m in METHODS:
        rs = [r[m] for r in results if m in r]
        if not rs:
            continue
        g = sum(r["glitch"] for r in rs) / max(1e-9, sum(r["minutes"] for r in rs))
        n = max(1, sum(r["ok_n"] for r in rs))
        print(f"| {m} ({rs[0]['used']}) | {g:.2f} | {sum(r['rpa_n'] for r in rs) / n:.3f} | "
              f"{sum(r['rpa10_n'] for r in rs) / n:.3f} | {np.mean([r['kept'] for r in rs]):.2f} |", flush=True)


def main():
    root, sep = Path(sys.argv[1]), Path(sys.argv[2])
    sep_res, clean_res = [], []
    for wav in sorted(sep.glob("sep_*.wav"), key=lambda p: int(p.stem.split("_")[1])):
        k = int(wav.stem.split("_")[1])
        sep_res.append(analyze(wav, k, root))
        clean_res.append(analyze(next(root.rglob(f"vocadito_{k}.wav")), k, root))
        print(f"vocadito {k}: " + ", ".join(f"{m} 분리 튐 {r['glitch']}" for m, r in sep_res[-1].items()), flush=True)
    table(sep_res, "분리한 보컬")
    table(clean_res, "깨끗한 보컬")


if __name__ == "__main__":
    main()

"""보컬 다시 분리 · 잔향 제거 모델(tetotune/clean.py)이 맞게 돌아가는지, 얼마나 도움이 되는지 (소리는 재생하지 않는다).

make_separated.py 결과 폴더(sep_ · mix_ · wet_ · dry_ · bk_<번호>.wav)를 쓴다.
1) 모델 입력 · 출력 모양을 찍고, 입력 순서(sc · cs) × 출력(스펙트럼 · 마스크) 네 가지를 두 곡에 돌려서 SDR이 가장 높은 것을 고른다.
   틀린 순서는 SDR이 0dB 근처나 그 아래로 나온다.
2) 고른 것으로 모든 곡: Demucs 보컬 대비 SDR, 그리고 음 높이 읽기 (사람이 표시한 vocadito 정답과 비교, f0_models.py와 같은 기준).

사용: python check_clean_models.py <분리한 보컬 폴더> <vocadito 폴더>
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "devtools"))
from f0_robust import score, truth  # noqa: E402
from tetotune import audio, clean, pitch  # noqa: E402


def load(folder, name, k):
    y, sr = audio.load(folder / f"{name}_{k}.wav")
    assert sr == clean.SR
    return y


def io_info(path):
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for x in s.get_inputs():
        print(f"  입력 {x.name} {x.shape} {x.type}", flush=True)
    for x in s.get_outputs():
        print(f"  출력 {x.name} {x.shape} {x.type}", flush=True)


def run(name, x, layout):
    m = clean.SpecModel(name, clean.model_path(name), layout)
    t = time.time()
    y = m(x)
    return y, time.time() - t


def pitch_row(y, k, root, method):
    f0, used = pitch.extract(y, clean.SR, method)
    n = len(f0)
    midi, voiced = pitch.clean(f0, pitch.frame_db(y, clean.SR, n))
    tmidi, notes = truth(k, root, n)
    r = score(midi, voiced, tmidi, notes)
    return r, used


def main():
    sep, root = Path(sys.argv[1]), Path(sys.argv[2])
    ks = sorted(int(p.stem.split("_")[1]) for p in sep.glob("mix_*.wav") if p.stem.split("_")[1].isdigit())
    if not ks:
        print("mix_<번호>.wav가 없어요 (make_separated.py를 새로 돌려 주세요)")
        return
    for name in clean.MODELS:
        p = clean.ensure(name, print)
        print(f"{name}: {p} ({p.stat().st_size >> 20}MB)", flush=True)
        io_info(p)
    best = {}
    for name in clean.MODELS:
        scores = {}
        for layout in (("sc", "spec"), ("cs", "spec"), ("sc", "mask"), ("cs", "mask")):
            if name == "dereverb" and layout[0] == "cs":
                continue  # 모노는 실수 · 허수 둘뿐이라 순서가 같다
            vals = []
            for k in ks[:2]:
                if name == "vocals":
                    est, sec = run(name, np.stack([load(sep, "mix", k)] * 2), layout)
                    vals.append(clean.sdr(est.mean(0), load(sep, "wet", k)))
                else:
                    est, sec = run(name, load(sep, "wet", k)[None], layout)
                    vals.append(clean.sdr(est[0], load(sep, "dry", k)))
            scores[layout] = float(np.mean(vals))
            print(f"{name} {layout}: SDR {scores[layout]:.2f}dB (곡마다 {[round(v, 2) for v in vals]}, 마지막 곡 {sec:.0f}초)",
                  flush=True)
        best[name] = max(scores, key=scores.get)
        print(f"==> {name} 고른 것: {best[name]}", flush=True)
    clean.LAYOUT.update(best)

    print("\n| 곡 | Demucs 보컬 SDR | 곡에서 다시 분리 | Demucs 보컬 한 번 더 | 잔향 (원래 보컬 대비) | 잔향 제거 | "
          "다시 분리 + 잔향 제거 |")
    print("|---|---|---|---|---|---|---|")
    sums = []
    rows = {"Demucs 보컬": [], "곡에서 다시 분리": [], "Demucs 보컬 한 번 더": [], "Demucs 보컬 → 잔향 제거": [],
            "다시 분리 → 잔향 제거": []}
    for k in ks:
        mix, wet, dry, sepv, bk = (load(sep, n, k) for n in ("mix", "wet", "dry", "sep", "bk"))
        n = min(len(mix), len(sepv), len(bk))
        re = clean.vocals(sepv[:n], clean.SR, backing=bk[:n])
        twice = clean.vocals(sepv, clean.SR)
        dr_sep = clean.dereverb(sepv, clean.SR)
        dr_re = clean.dereverb(re, clean.SR)
        row = [clean.sdr(sepv, wet), clean.sdr(re, wet), clean.sdr(twice, wet), clean.sdr(wet, dry),
               clean.sdr(dr_sep, dry), clean.sdr(dr_re, dry)]
        sums.append(row)
        print(f"| {k} | " + " | ".join(f"{v:.2f}" for v in row) + " |", flush=True)
        for label, y in zip(rows, (sepv, re, twice, dr_sep, dr_re)):
            rows[label].append((k, y))
    print("| 평균 | " + " | ".join(f"{v:.2f}" for v in np.mean(sums, 0)) + " |")

    for method in ("harvest", "fusion"):
        print(f"\n| 음 높이 ({method}) | 튐 /분 | RPA (50센트) | 남긴 프레임 |")
        print("|---|---|---|---|")
        for label, items in rows.items():
            rs = []
            for k, y in items:
                r, used = pitch_row(y, k, root, method)
                if r is not None:
                    rs.append(r)
            if not rs:
                continue
            g = sum(r["glitch"] for r in rs) / max(1e-9, sum(r["minutes"] for r in rs))
            nn = max(1, sum(r["ok_n"] for r in rs))
            print(f"| {label} ({used}) | {g:.2f} | {sum(r['rpa_n'] for r in rs) / nn:.3f} | "
                  f"{np.mean([r['kept'] for r in rs]):.2f} |", flush=True)


if __name__ == "__main__":
    main()

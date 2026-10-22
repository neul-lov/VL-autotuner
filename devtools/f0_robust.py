"""분리한 보컬에서 음높이를 짧게 잘못 읽는 곳(확 떨어졌다 돌아오거나 튀는 곳)을 줄이는 방법 비교. 렌더 없이 빠르게.

테토가 부르는 음높이 = 원래 보컬에서 읽은 음높이 (믿을 수 없는 프레임은 빼고 앞뒤를 이어서). 그래서 읽은 음높이를
사람이 표시한 정답(vocadito f0)과 바로 비교한다. 조교(tune.py)처럼 음표 안의 빈 곳은 앞뒤를 이어 메운 뒤 잰다.
- 튐 /분: 정답과 150센트 넘게 다른 곳이 30ms 넘게 이어진 횟수 (렌더 비교의 "떨어짐 · 튐"과 같은 기준)
- RPA: 정답 소리 나는 곳 중 50센트 안
분리한 보컬과 깨끗한 보컬(나빠지지 않는지) 둘 다 잰다.

사용: python f0_robust.py <vocadito 폴더> <분리한 보컬 폴더 (sep_<번호>.wav)>
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tetotune import audio, pitch  # noqa: E402
from tetotune.pitch import HOP  # noqa: E402


def _csv(p):
    rows = []
    for row in csv.reader(open(p)):
        try:
            rows.append([float(v) for v in row])
        except ValueError:
            continue
    return np.array(rows)


def truth(k, root, n):
    f0 = _csv(next(root.rglob(f"vocadito_{k}_f0.csv")))
    nts = _csv(next(root.rglob(f"vocadito_{k}_notesA1.csv")))
    t = np.arange(n) * HOP
    idx = np.clip(np.searchsorted(f0[:, 0], t), 0, len(f0) - 1)
    hz = f0[idx, 1]
    midi = np.where(hz > 0, pitch.hz_to_midi(np.maximum(hz, 1)), np.nan)
    notes = np.zeros(n, bool)
    for s, _, d in nts:
        notes[int((s - 0.05) / HOP):int((s + d + 0.05) / HOP) + 1] = True
    return midi, notes[:n]


def interp_gaps(x, mask):
    out = x.copy()
    idx = np.arange(len(x))
    if mask.sum() >= 2:
        out[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return out


def score(est, voiced, tmidi, notes, base=None):
    base = voiced if base is None else base
    use = voiced & notes & np.isfinite(est)
    if use.sum() < 2:
        return None
    target = interp_gaps(np.where(use, est, 0.0), use)
    ok = notes & np.isfinite(tmidi)
    err = np.where(ok, np.abs(target - np.nan_to_num(tmidi)) * 100, 0.0)
    runs = [r for r in pitch._runs(err > 150) if r[1] - r[0] >= 6]
    return {"glitch": len(runs), "minutes": len(est) * HOP / 60, "rpa_n": int((err[ok] <= 50).sum()), "ok_n": int(ok.sum()),
            "kept": float(use.sum() / max(1, (base & notes).sum())), "runs": runs, "target": target}


# ---------------------------------------------------------------- 믿을 수 없는 프레임 찾기 (후보)
def _local_max(db, sec):
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(db, int(sec / HOP) | 1)


def quiet(db, voiced, drop):
    """둘레 0.3초에서 가장 큰 소리보다 drop dB 넘게 작은 곳 (반주가 새는 소리가 목소리보다 클 수 있다)."""
    return voiced & (db < _local_max(np.where(voiced, db, -120), 0.3) - drop)


def edge_quiet(db, voiced, drop, edge=0.15):
    """소리 덩어리(유성 구간) 앞뒤 edge초 안에서만 quiet."""
    q = quiet(db, voiced, drop)
    m = np.zeros(len(db), bool)
    k = int(edge / HOP)
    for a, b in pitch._runs(voiced):
        m[a:min(b, a + k)] = True
        m[max(a, b - k):b] = True
    return q & m


def harmonicity(y16, f0, win=0.03):
    """프레임마다 f0 주기만큼 밀어 본 자기 상관 (1 = 깨끗한 주기 소리, 낮을수록 다른 소리가 섞임)."""
    n = len(f0)
    hop, w = int(16000 * HOP), int(16000 * win)
    out = np.zeros(n)
    x = np.pad(np.asarray(y16, float), (w, 2 * w + 400))
    for i in range(n):
        if f0[i] <= 0:
            continue
        T = int(round(16000 / f0[i]))
        s = i * hop + w - w // 2
        a, b = x[s:s + w], x[s + T:s + T + w]
        d = np.sqrt((a @ a) * (b @ b))
        out[i] = (a @ b) / d if d > 0 else 0
    return out


def dio_track(y16):
    import pyworld
    x = np.ascontiguousarray(y16, dtype=np.float64)
    f0, t = pyworld.dio(x, 16000, f0_floor=pitch.FMIN, f0_ceil=pitch.FMAX, frame_period=HOP * 1000)
    return pyworld.stonemask(x, f0, t, 16000)


def candidates(y16, f0, db, midi, voiced):
    """이름 → 믿을 수 있는 유성 마스크."""
    n = len(midi)
    out = {"지금": voiced}
    for d in (12, 18, 24):
        out[f"작은 소리 빼기 {d}dB"] = voiced & ~quiet(db, voiced, d)
        out[f"끝 작은 소리 빼기 {d}dB"] = voiced & ~edge_quiet(db, voiced, d)
    h = harmonicity(y16, np.where(voiced, f0[:n], 0))
    for th in (0.5, 0.65, 0.8):
        out[f"주기성 {th}"] = voiced & (h >= th)
    dio = dio_track(y16)
    dio = np.pad(dio, (0, max(0, n - len(dio))))[:n]
    dm = np.where(dio > 0, pitch.hz_to_midi(np.maximum(dio, 1)), np.nan)
    for c in (80, 150):
        agree = np.isfinite(dm) & (np.abs(dm - midi) * 100 <= c)
        out[f"DIO와 같을 때만 {c}센트"] = voiced & agree
    out["주기성 0.65 + 끝 작은 소리 18dB"] = out["주기성 0.65"] & out["끝 작은 소리 빼기 18dB"]
    out["주기성 0.65 + DIO 150센트"] = out["주기성 0.65"] & out["DIO와 같을 때만 150센트"]
    return out


def analyze(wav, k, root):
    y, sr = audio.load(wav)
    f0, _ = pitch.extract(y, sr, "harvest")
    n = len(f0)
    db = pitch.frame_db(y, sr, n)
    midi, voiced = pitch.clean(f0, db)
    tmidi, notes = truth(k, root, n)
    y16 = audio.resample(y, sr, 16000)
    cands = candidates(y16, f0, db, midi, voiced)
    res = {name: score(midi, m, tmidi, notes, voiced) for name, m in cands.items()}
    if describe and res["지금"]:
        explain(res["지금"], midi, voiced, tmidi, db, y16, f0, k)
    return res


describe = False


def explain(r, midi, voiced, tmidi, db, y16, f0, k):
    """지금 방법의 튐마다: 몇 ms · 몇 센트, 그 프레임이 유성인지, 둘레보다 몇 dB 작은지, 주기성, DIO와 차이, 유성 덩어리 끝에서 얼마나."""
    h = harmonicity(y16, np.where(voiced, f0[:len(midi)], 0))
    dio = dio_track(y16)
    dio = np.pad(dio, (0, max(0, len(midi) - len(dio))))[:len(midi)]
    dm = np.where(dio > 0, pitch.hz_to_midi(np.maximum(dio, 1)), np.nan)
    lm = _local_max(np.where(voiced, db, -120), 0.3)
    starts = [a for a, b in pitch._runs(voiced)]
    ends = [b for a, b in pitch._runs(voiced)]
    for a, b in r["runs"]:
        sl = slice(a, b)
        err = (r["target"][sl] - tmidi[sl]) * 100
        v = voiced[sl].mean()
        rel = float(np.median((db - lm)[sl]))
        hh = float(np.median(h[sl][voiced[sl]])) if voiced[sl].any() else float("nan")
        dd = np.abs(dm[sl] - midi[sl]) * 100
        dd = float(np.nanmedian(dd)) if np.isfinite(dd).any() else float("nan")
        edge = min([abs(a - x) for x in starts + ends] or [999]) * HOP * 1000
        print(f"  {k} t={a * HOP:.2f} {int((b - a) * HOP * 1000)}ms {int(np.nanmedian(err)):+d}c 유성 {v:.2f} "
              f"둘레보다 {rel:.1f}dB 주기성 {hh:.2f} DIO차 {dd:.0f}c 끝에서 {edge:.0f}ms", flush=True)


def table(results, label):
    names = list(results[0].keys())
    print(f"\n| 방법 | {label} 튐 /분 | RPA | 남긴 프레임 |")
    print("|---|---|---|---|")
    for name in names:
        rs = [r[name] for r in results if r[name]]
        g = sum(r["glitch"] for r in rs) / max(1e-9, sum(r["minutes"] for r in rs))
        rpa = sum(r["rpa_n"] for r in rs) / max(1, sum(r["ok_n"] for r in rs))
        kept = np.mean([r["kept"] for r in rs])
        print(f"| {name} | {g:.2f} | {rpa:.3f} | {kept:.2f} |", flush=True)


def main():
    root, sep = Path(sys.argv[1]), Path(sys.argv[2])
    sep_res, clean_res = [], []
    global describe
    for wav in sorted(sep.glob("sep_*.wav"), key=lambda p: int(p.stem.split("_")[1])):
        k = int(wav.stem.split("_")[1])
        describe = True
        sep_res.append(analyze(wav, k, root))
        describe = False
        clean_res.append(analyze(next(root.rglob(f"vocadito_{k}.wav")), k, root))
        print(f"vocadito {k}: 분리 튐 {sep_res[-1]['지금']['glitch']} · 깨끗 튐 {clean_res[-1]['지금']['glitch']}", flush=True)
    table(sep_res, "분리한 보컬")
    table(clean_res, "깨끗한 보컬")


if __name__ == "__main__":
    main()

"""OpenUtau가 렌더링한 wav를 원래 보컬과 비교한다 (재생 없음, 숫자만).
사용: python compare_render.py <결과 폴더> <렌더 wav> [그림.png] [--truth 정답 보컬.wav] [--f0-truth 음 높이.csv]
--truth: 조교에 넣은 보컬(guide_vocals.wav) 대신 이 보컬과 비교 (분리한 보컬로 조교했을 때 깨끗한 원래 보컬)
--f0-truth: 원래 보컬의 음 높이를 Harvest로 읽지 않고 사람이 표시한 음 높이(시각, Hz; 0 = 소리 없음)를 쓴다.
  Harvest로 읽으면 Harvest로 곡선을 만든 조교가 유리해서, 사람이 표시한 정답이 있으면 둘 다 잰다"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tetotune import audio, features, pitch  # noqa: E402
from tetotune.pitch import HOP  # noqa: E402


def _runs(mask, min_len=1):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return [(a, b) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if b - a >= min_len]


def _wobble(m, mask):
    """50ms 중앙값에서 벗어나는 정도 (센트 RMS): 비브라토보다 빠른 떨림 · 튐."""
    from scipy.ndimage import median_filter

    dev = []
    for a, b in _runs(mask, 12):
        seg = m[a:b]
        dev.append((seg - median_filter(seg, size=11, mode="nearest")) * 100)
    return float(np.sqrt(np.mean(np.concatenate(dev) ** 2))) if dev else float("nan")


def smoothness(prev, ms, vs, dbs, mr, vr, dbr, notes):
    """끊김 · 갑자기 떨어짐 · 흔들림을 센다 (1분마다 몇 번). 일어난 곳마다 무엇 때문인지도 적는다.
    - 끊김: 원래 보컬은 음표 안에서 소리 나는데 렌더는 원래보다 15dB 넘게 더 작아진 곳이 30ms 넘게 (노래 크기 기준)
    - 떨어짐 · 튐: 둘 다 소리 나는데 렌더 음 높이가 원래보다 150센트 넘게 낮은 · 높은 곳이 30ms 넘게
    - 흔들림: 렌더가 원래보다 얼마나 더 떨리는지 (1 = 같음)"""
    n = len(ms)
    minutes = max(1e-6, n * HOP / 60)
    sing_r = float(np.median(dbr[vr])) if vr.any() else -60.0
    both = vs & vr
    sing_s = float(np.median(dbs[vs])) if vs.any() else -60.0
    # 원래 보컬도 작아진 곳(콧소리 · 여린 소리)은 끊김이 아니다: 원래보다 15dB 넘게 더 작아진 곳만
    drop = _runs(vs & notes & (dbr < sing_r - 12) & ((dbr - sing_r) < (dbs - sing_s) - 15), 6)
    diff = np.where(both, (mr - ms) * 100, 0.0)
    low = _runs(diff < -150, 6)
    high = _runs(diff > 150, 6)
    intended = np.array([np.nan if v is None else v for v in prev["pitch"]], dtype=float)
    nl = prev["notes"]
    step = np.full(n, np.nan)
    for s, e, tone, _ in nl:
        step[int(s / HOP):int(e / HOP)] = tone

    def why(a, b, kind):
        t = (a + b) / 2 * HOP
        k = next((i for i, x in enumerate(nl) if x[0] <= t < x[1]), None)
        if k is None:
            nxt = next((x for x in nl if x[0] > t), None)
            prv = next((x for x in reversed(nl) if x[1] <= t), None)
            gap = (nxt[0] - prv[1]) if nxt and prv else 99
            where = f"음표 사이 틈 {gap * 1000:.0f}ms" if gap < 0.5 else "쉼"
        else:
            x = nl[k]
            where = "음표 시작" if t - x[0] < 0.08 else ("음표 끝" if x[1] - t < 0.08 else "음표 가운데")
            if k and abs(nl[k - 1][1] - x[0]) < 1e-3 and t - x[0] < 0.08:
                where = "이어진 음표 경계"
        i = min(len(intended) - 1, int(t / HOP / 2))
        src = ms[int(t / HOP)]
        curve = abs(intended[i] - src) * 100 > 100 if np.isfinite(src) and np.isfinite(intended[i]) else False
        cons = not vs[max(0, a - 6):b + 6].all()
        # 음표가 원래 보컬과 맞는 곳(150센트 안)에서 벗어남 = 맞는 음에서 확 떨어지거나 튀는 것. 음표도 다르면 대부분
        # 사람이 음절 사이 · 음 끝에서 잠깐 꺾은 것을 테토가 음표 음 그대로 부른 곳
        ne = (step[a:b] - ms[a:b]) * 100
        note_ok = bool(np.isfinite(ne).any() and abs(np.nanmedian(ne)) < 150)
        return {"t": round(t, 2), "ms": int((b - a) * HOP * 1000), "kind": kind, "where": where,
                "lyric": nl[k][3] if k is not None else "", "curve": bool(curve), "near_unvoiced": bool(cons),
                "note_ok": note_ok,
                "cents": int(np.nanmedian(diff[a:b])) if kind != "끊김" else None,
                "db": round(float(np.median(dbr[a:b]) - sing_r), 1)}
    events = [why(a, b, "끊김") for a, b in drop] + [why(a, b, "떨어짐") for a, b in low] + \
        [why(a, b, "튐") for a, b in high]
    kinds = {}
    for e in events:
        key = f"{e['kind']} · {e['where'].split(' ')[0] if e['where'].startswith('음표 사이') else e['where']}" \
            f"{' · 곡선' if e['curve'] else ''}{' · 자음 옆' if e['near_unvoiced'] else ''}"
        kinds[key] = kinds.get(key, 0) + 1
    wr, ws = _wobble(mr, both), _wobble(ms, both)
    return {"dropouts_per_min": round(len(drop) / minutes, 2), "pitch_drops_per_min": round(len(low) / minutes, 2),
            "pitch_jumps_per_min": round(len(high) / minutes, 2),
            "off_note_ok_per_min": round(sum(e["note_ok"] for e in events if e["kind"] != "끊김") / minutes, 2),
            "wobble_ratio": round(wr / ws, 3) if ws > 0 else None, "wobble_cents": round(wr, 1),
            "event_kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])), "events": events[:60]}


def f0_frames(csv_path, n):
    """사람이 표시한 음 높이 csv (시각, Hz) → HOP 프레임마다 (midi, 소리 있음). 가장 가까운 표시가 10ms 넘게 멀면 소리 없음."""
    rows = []
    for line in Path(csv_path).read_text().splitlines():
        try:
            rows.append([float(v) for v in line.split(",")[:2]])
        except ValueError:
            continue  # 머리글
    ref = np.array(rows, dtype=float).reshape(-1, 2)
    if len(ref) < 2:
        return np.full(n, np.nan), np.zeros(n, dtype=bool)
    t = np.arange(n) * HOP
    i = np.clip(np.searchsorted(ref[:, 0], t), 1, len(ref) - 1)
    i = np.where(np.abs(ref[i - 1, 0] - t) < np.abs(ref[i, 0] - t), i - 1, i)
    m = pitch.hz_to_midi(np.where(np.abs(ref[i, 0] - t) <= 0.01, ref[i, 1], 0.0))
    return m, np.isfinite(m)


def main():
    argv = sys.argv[1:]
    opts = {}
    for flag in ("--truth", "--f0-truth"):
        if flag in argv:
            k = argv.index(flag)
            opts[flag] = argv[k + 1]
            argv = argv[:k] + argv[k + 2:]
    truth, f0_truth = opts.get("--truth"), opts.get("--f0-truth")
    folder, render = Path(argv[0]), argv[1]
    png = argv[2] if len(argv) > 2 else None
    prev = json.loads((folder / "preview.json").read_text())
    ys, sr = audio.load(truth or folder / "guide_vocals.wav")
    yr, _ = audio.load(render)
    n = int(len(ys) / sr / HOP) + 1
    yr = np.pad(yr, (0, max(0, len(ys) - len(yr))))[:len(ys)]

    def analyse(y):
        f0, _ = pitch.extract(y, sr, "harvest")
        f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
        db = pitch.frame_db(y, sr, n)
        m, v = pitch.clean(f0, db)
        # 음높이가 잡힌 잡음(숨 · ㅅ · ㅎ)은 음높이가 아니라서 비교에서 뺀다 (조교할 때와 같은 기준)
        feats = features.analyze(y, sr, np.where(v, pitch.midi_to_hz(np.nan_to_num(m, nan=60)), 0), v)
        v = v & ~features.noisy_voiced(v, feats, db)
        return np.where(v, m, np.nan), v, db

    ms, vs, dbs = analyse(ys)
    if f0_truth:
        ms, vs = f0_frames(f0_truth, n)
    mr, vr, dbr = analyse(yr)
    ms = ms + prev["shift"]
    both = vs & vr
    if not both.any():
        sys.exit(f"렌더({render})에 원래 보컬과 같이 소리 나는 곳이 없어요: 렌더 유성 {int(vr.sum())}프레임, 원래 {int(vs.sum())}프레임")
    err = np.abs(mr[both] - ms[both]) * 100
    notes = np.zeros(n, dtype=bool)
    for s, e, *_ in prev["notes"]:
        notes[int(s / HOP):int(e / HOP)] = True
    # 음표 계단만 (조교 안 한 상태)과 원래 보컬의 차이: 비교 기준
    step = np.full(n, np.nan)
    for s, e, tone, _ in prev["notes"]:
        step[int(s / HOP):int(e / HOP)] = tone
    okb = vs & np.isfinite(step)
    err_step = np.abs(step[okb] - ms[okb]) * 100
    corr = np.corrcoef(dbs[both], dbr[both])[0, 1]
    # 크기 윤곽 차이 (dB): 원래 보컬이 음표 안에서 소리 나는 곳, 각자 노래 크기(중앙값) 기준
    m = vs & notes
    rel_s, rel_r = dbs - np.median(dbs[m]), dbr - np.median(dbr[m])
    loud_mae = float(np.mean(np.abs(np.clip(rel_r[m], -40, 20) - np.clip(rel_s[m], -40, 20))))
    out = {
        "pitch_err_cents_median": round(float(np.median(err)), 1),
        "pitch_err_cents_p90": round(float(np.percentile(err, 90)), 1),
        "within_50_cents": round(float(np.mean(err < 50)), 3),
        "notes_only_err_cents_median": round(float(np.median(err_step)), 1),
        "notes_only_err_cents_p90": round(float(np.percentile(err_step, 90)), 1),
        "voiced_overlap": round(float(both.sum() / max(1, vs.sum())), 3),
        "loudness_corr": round(float(corr), 3),
        "loudness_mae_db": round(loud_mae, 2),
    }
    sing = np.median(dbr[vr])
    br = []
    for s, e in prev["breaths"]:
        a, b = int(s / HOP), int(e / HOP)
        br.append(round(float(np.percentile(dbr[a:b], 90) - sing), 1))
    out["breath_db_vs_singing"] = br
    sing_s = np.median(dbs[vs])
    out["source_breath_db_vs_singing"] = [round(float(np.percentile(dbs[int(s / HOP):int(e / HOP)], 90) - sing_s), 1)
                                          for s, e in prev["breaths"]]
    out.update(smoothness(prev, ms, vs, dbs, mr, vr, dbr, notes))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(n) * HOP
        fig, ax = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
        ax[0].plot(t, ms, "k", lw=1.2, label="source (+transpose)")
        ax[0].plot(t, mr, "r", lw=0.8, label="OpenUtau render")
        ax[0].plot(t, step, "c", lw=0.8, alpha=0.6, label="notes")
        for s, e in prev["breaths"]:
            ax[0].axvspan(s, e, color="g", alpha=0.2)
        ax[0].legend()
        ax[1].plot(t, dbs, "k", lw=0.8)
        ax[1].plot(t, dbr, "r", lw=0.8)
        plt.tight_layout()
        plt.savefig(png, dpi=75)


# Mac은 Harvest 조각을 새 프로세스(spawn)로 읽어서, 이 파일을 다시 불러도 비교가 또 돌지 않게 막아 둔다
if __name__ == "__main__":
    main()

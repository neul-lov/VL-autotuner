"""OpenUtau가 렌더링한 wav를 원래 보컬과 비교한다 (재생 없음, 숫자만).
사용: python compare_render.py <결과 폴더> <렌더 wav> [그림.png]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tetotune import audio, features, pitch  # noqa: E402
from tetotune.pitch import HOP  # noqa: E402


def main():
    folder, render = Path(sys.argv[1]), sys.argv[2]
    prev = json.loads((folder / "preview.json").read_text())
    ys, sr = audio.load(folder / "guide_vocals.wav")
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
    out = {
        "pitch_err_cents_median": round(float(np.median(err)), 1),
        "pitch_err_cents_p90": round(float(np.percentile(err, 90)), 1),
        "within_50_cents": round(float(np.mean(err < 50)), 3),
        "notes_only_err_cents_median": round(float(np.median(err_step)), 1),
        "notes_only_err_cents_p90": round(float(np.percentile(err_step, 90)), 1),
        "voiced_overlap": round(float(both.sum() / max(1, vs.sum())), 3),
        "loudness_corr": round(float(corr), 3),
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
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 3:
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
        plt.savefig(sys.argv[3], dpi=75)


# Mac은 Harvest 조각을 새 프로세스(spawn)로 읽어서, 이 파일을 다시 불러도 비교가 또 돌지 않게 막아 둔다
if __name__ == "__main__":
    main()

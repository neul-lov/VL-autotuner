"""같은 조교 결과(project.ustx)를 보코더만 바꿔 렌더해 견준다 (vocoder.yml. 소리는 재생하지 않고 숫자만, 렌더는 이 기계에만).

accuracy.py를 ACC_KEEP_PROJECTS=<폴더>로 돌리면 곡마다 조교 · 고치기까지 끝난 폴더가 <폴더>/<방식>/<모음>/<곡>/에 남는다
(project.ustx · preview.json · guide_vocals.wav · teto_preview.wav, meta.json에 견줄 정답 보컬).
run은 그 project.ustx를 지금 깔린 보코더로 다시 렌더해서 accuracy.py와 같은 잣대(compare_render.py)로 잰다.
OpenUtau는 렌더마다 새로 떠서 보코더를 다시 읽으니, 보코더 파일을 바꾸고 렌더 캐시를 비운 뒤 다시 돌리면 된다.
렌더가 고치기 마지막 렌더(teto_preview.wav)와 얼마나 같은지(SNR dB)도 적는다: 같은 보코더면 거의 같고, 바꿨으면 달라야 한다.

사용: python vocoder_ab.py run <남긴 폴더> --render <OpenUtau.app/Contents/MacOS> <작업 폴더> --label "원래 보코더" \
          --json a.json [--keep keep_ab]     (렌더를 keep_ab/<이름>/<모음>/<곡>.wav로, 원래 보컬은 keep_ab/원래 보컬/에)
      python vocoder_ab.py table a.json b.json [--mos mos.json] [--base "원래 보코더"]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "devtools"))

KEYS = ("pitch_err_cents_median", "within_50_cents", "loudness_corr", "loudness_mae_db", "dropouts_per_min",
        "pitch_drops_per_min", "pitch_jumps_per_min", "wobble_ratio")


def snr_db(ref, y):
    """ref에 견준 y의 SNR (같으면 99)."""
    n = min(len(ref), len(y))
    if n == 0:
        return float("nan")
    d = float(np.sum((ref[:n] - y[:n]) ** 2))
    return 99.0 if d == 0 else round(min(99.0, 10 * np.log10(float(np.sum(ref[:n] ** 2)) / d)), 1)


def run(args):
    import accuracy as acc
    from tetotune import audio

    ou, work = Path(args.render[0]).resolve(), Path(args.render[1]).resolve()
    res, t0 = {}, time.time()
    for ustx in sorted(Path(args.projects).glob("*/*/*/project.ustx")):
        folder = ustx.parent
        key, song = folder.parent.name, folder.name
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        t = time.time()
        try:
            wav = acc._render_only(ou, work, ustx)
            c = acc.compare_wav(folder, wav, meta["truth"])
            if meta.get("f0_truth"):
                by = acc.compare_wav(folder, wav, meta["truth"], meta["f0_truth"])
                c["by_f0"] = {k: by.get(k) for k in KEYS}
        except Exception as e:  # 한 곡이 안 되면 그 곡만 빼고 (어느 쪽에서 빠졌는지 표에 수로 보인다)
            print(f"{args.label} {key} {song} 렌더 못 함: {type(e).__name__}: {str(e)[:300]}", flush=True)
            shutil.rmtree(folder / "_r", ignore_errors=True)
            continue
        y, _ = audio.load(wav)
        prev, _ = audio.load(folder / "teto_preview.wav")
        c["snr_vs_preview_db"] = snr_db(prev, y)
        base = Path(args.keep or "", args.base, key, song).with_suffix(".wav")
        if args.keep and args.label != args.base and base.exists():  # 원래 보코더로 다시 렌더한 것과 (바꿔 끼운 게 먹혔는지)
            c["snr_vs_base_db"] = snr_db(audio.load(base)[0], y)
        if args.keep:
            dst = Path(args.keep, args.label, key, song).with_suffix(".wav")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(wav, dst)
            orig = Path(args.keep, "원래 보컬", key, song).with_suffix(".wav")
            if not orig.exists():
                orig.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(meta["truth"] or folder / "guide_vocals.wav", orig)
        shutil.rmtree(folder / "_r", ignore_errors=True)
        row = {"song": song, **{k: c.get(k) for k in KEYS}, "snr_vs_preview_db": c["snr_vs_preview_db"]}
        if "snr_vs_base_db" in c:
            row["snr_vs_base_db"] = c["snr_vs_base_db"]
        if "by_f0" in c:
            row["by_f0"] = c["by_f0"]
        res.setdefault(key, []).append(row)
        print(f"{args.label} {key} {song} ({time.time() - t:.0f}초)", json.dumps({k: row[k] for k in KEYS + ("snr_vs_preview_db",)},
                                                                         ensure_ascii=False), flush=True)
    n = sum(len(v) for v in res.values())
    print(f"{args.label}: {n}곡 렌더 ({time.time() - t0:.0f}초)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({args.label: res}, ensure_ascii=False), encoding="utf-8")
    if not n:
        raise SystemExit("렌더한 곡이 없어요")


def _mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
    return f"{np.mean(v):.3f}" if v else ""


def table(args):
    res = {}
    for p in args.jsons:
        res.update(json.loads(Path(p).read_text(encoding="utf-8")))
    sets = sorted({s for v in res.values() for s in v})
    print("\n### 같은 조교 결과를 보코더만 바꿔 렌더 (곡 평균; 소리 크기 · 음 높이 잣대는 accuracy.py와 같음)\n")
    print("| 보코더 | 모음 (곡) | " + " | ".join(KEYS) + " | 고치기 렌더와 같은 정도 (SNR dB) |")
    print("|---|---|" + "---|" * (len(KEYS) + 1))
    for label, by_set in res.items():
        for s in sets:
            rows = by_set.get(s, [])
            if rows:
                print(f"| {label} | {s} ({len(rows)}) | " + " | ".join(_mean(rows, k) for k in KEYS)
                      + f" | {_mean(rows, 'snr_vs_preview_db')} |")
    if not args.mos:
        return
    mos = json.loads(Path(args.mos).read_text(encoding="utf-8"))
    base = mos.get(args.base, {})
    names = sorted({n for v in mos.values() for s in v.values() for sc in s.values() for n in sc})
    for label in [v for v in mos if v not in (args.base, "원래 보컬")]:
        print(f"\n### 자연스러움 AI 채점: {label} − {args.base} (같은 곡끼리 뺀 값의 평균, 더 높은 곡 수 / 곡 수)\n")
        print("| 채점기 | " + " | ".join(sets) + " | 모두 |")
        print("|---|" + "---|" * (len(sets) + 1))
        for name in names:
            cells, allv = [], []
            for s in sets:
                d = [sc[name] - base[s][song][name] for song, sc in mos[label].get(s, {}).items()
                     if name in sc and name in base.get(s, {}).get(song, {})]
                allv += d
                cells.append(f"{np.mean(d):+.3f} ({sum(x > 0 for x in d)}/{len(d)})" if d else "")
            total = f"{np.mean(allv):+.3f} ({sum(x > 0 for x in allv)}/{len(allv)})" if allv else ""
            print(f"| {name} | " + " | ".join(cells) + f" | {total} |", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table"])
    ap.add_argument("projects", nargs="?")
    ap.add_argument("jsons", nargs="*")
    ap.add_argument("--render", nargs=2)
    ap.add_argument("--label")
    ap.add_argument("--json")
    ap.add_argument("--keep")
    ap.add_argument("--mos")
    ap.add_argument("--base", default="원래 보코더")
    args = ap.parse_args()
    if args.cmd == "table":
        args.jsons = ([args.projects] if args.projects else []) + args.jsons
        table(args)
    else:
        if not (args.projects and args.render and args.label):
            ap.error("run은 <남긴 폴더> --render <OpenUtau 폴더> <작업 폴더> --label 이름")
        run(args)


if __name__ == "__main__":
    main()

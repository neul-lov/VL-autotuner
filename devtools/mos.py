"""노래 자연스러움 AI 채점: 사람이 매길 MOS(1~5점)를 흉내 내는 모델로 렌더한 테토와 원래 보컬을 같은 잣대로 잰다.

- singmos_pro: SingMOS-Pro (노래 합성 · 변환 결과 7,981개에 사람이 매긴 점수로 배움, wav2vec2 large). 주 잣대
- singmos_v1: SingMOS v1 (VoiceMOS 2024 노래 부문 기준 모델, wav2vec2 base)
- utmos: UTMOS22 strong (말소리 자연스러움으로 배움, 참고만)
곡을 8초 조각으로 나눠 소리가 충분한 조각만 같은 크기로 맞춰 매기고 평균한다. 소리는 재생하지 않고 숫자만 찍는다.

사용: python mos.py <폴더> --json out.json   (폴더/<방식>/<모음>/<곡>.wav, accuracy.py ACC_KEEP_RENDERS)
      python mos.py --merge a.json b.json ...  (잡마다 나온 점수를 표 하나로)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEG = 8.0  # 초
SR = 16000
PREDICTORS = {  # 이름 → (torch.hub 저장소:판, 모델, 두 번째 인자)
    "singmos_pro": ("South-Twilight/SingMOS:v1.1.2", "singmos_pro", "length"),
    "singmos_v1": ("South-Twilight/SingMOS:v1.1.2", "singmos_v1", "length"),
    "utmos": ("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", "sr"),
}


def segments(y):
    """16kHz 모노 → 8초 조각들 (50ms 틀 중 곡 최대보다 30dB 넘게 작은 것이 절반 넘으면 뺀다), 조각마다 RMS −20dBFS로."""
    n, hop = int(SEG * SR), int(0.05 * SR)
    frames = np.sqrt(np.mean(y[:len(y) // hop * hop].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    top = np.percentile(20 * np.log10(frames), 99) if len(frames) else -100
    out = []
    for a in range(0, max(1, len(y) - n // 2), n):
        seg = y[a:a + n]
        if len(seg) < 2 * SR:
            continue
        fr = frames[a // hop:(a + len(seg)) // hop]
        if np.mean(20 * np.log10(fr) > top - 30) < 0.5:
            continue
        seg = seg * (10 ** (-20 / 20) / (np.sqrt(np.mean(seg ** 2)) + 1e-9))
        out.append(np.clip(seg, -1, 1).astype(np.float32))
    return out


def score_files(files, names):
    """wav 목록 → {파일: {채점기: 곡 점수}} (채점기를 못 불러오면 그 채점기만 뺀다)."""
    import torch

    from tetotune import audio

    clips = {}
    for f in files:
        y, sr = audio.load(f)
        if sr != SR:
            y = audio.resample(y, sr, SR)
        clips[f] = segments(y)
    out = {f: {} for f in files}
    for name in names:
        repo, model, arg = PREDICTORS[name]
        try:
            # skip_validation: 저장소 확인에 GitHub API를 부르지 않는다 (CI Mac에서 이름 없는 API 한도에 걸려 403)
            pred = torch.hub.load(repo, model, trust_repo=True, skip_validation=True)
            pred.eval()
        except Exception as e:
            print(f"{name}: 불러오지 못함 ({str(e)[:300]})", flush=True)
            continue
        print(f"{name}: {repo} {model}", flush=True)
        for f, segs in clips.items():
            if not segs:
                continue
            vals = []
            with torch.no_grad():
                for seg in segs:
                    w = torch.from_numpy(seg).unsqueeze(0)
                    second = torch.tensor([w.shape[1]], dtype=torch.long) if arg == "length" else SR
                    vals.append(float(pred(w, second).reshape(-1)[0]))
            out[f][name] = float(np.mean(vals))
        del pred
    return out


def collect(root):
    """폴더/<방식>/<모음>/<곡>.wav → {방식: {모음: {곡: 파일}}}."""
    res = {}
    for f in sorted(Path(root).glob("*/*/*.wav")):
        res.setdefault(f.parent.parent.name, {}).setdefault(f.parent.name, {})[f.stem] = f
    return res


def print_tables(res):
    """res: {방식: {모음: {곡: {채점기: 점수}}}} → 채점기마다 표 (방식 × 모음, 곡 평균)."""
    sets = sorted({k for v in res.values() for k in v})
    names = [n for n in PREDICTORS if any(n in sc for v in res.values() for s in v.values() for sc in s.values())]
    for name in names:
        print(f"\n### 자연스러움 AI 채점 · {name} (1~5점, 곡 평균, 높을수록 자연스러움)\n", flush=True)
        print("| 방식 | " + " | ".join(sets) + " | 모두 |")
        print("|---|" + "---|" * (len(sets) + 1))
        for v in sorted(res, key=lambda x: (x != "원래 보컬", x)):
            cells, allv = [], []
            for s in sets:
                vals = [sc[name] for sc in res[v].get(s, {}).values() if name in sc]
                allv += vals
                cells.append(f"{np.mean(vals):.3f} ({len(vals)})" if vals else "")
            total = f"{np.mean(allv):.3f}" if allv else ""
            print(f"| {v} | " + " | ".join(cells) + f" | {total} |", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--merge", nargs="*")
    ap.add_argument("--predictors", default="singmos_pro,singmos_v1,utmos")
    args = ap.parse_args()
    if args.merge is not None:
        res = {}
        for p in args.merge:
            for v, sets in json.loads(Path(p).read_text(encoding="utf-8")).items():
                for s, songs in sets.items():
                    res.setdefault(v, {}).setdefault(s, {}).update(songs)
        print_tables(res)
        return
    files = collect(args.root)
    flat = [f for v in files.values() for s in v.values() for f in s.values()]
    print(f"채점할 곡 {len(flat)}개 (방식 {len(files)}가지)", flush=True)
    scores = score_files(flat, args.predictors.split(","))
    res = {v: {s: {song: scores[f] for song, f in songs.items()} for s, songs in sets.items()} for v, sets in files.items()}
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print_tables(res)


if __name__ == "__main__":
    main()

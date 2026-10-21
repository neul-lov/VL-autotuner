"""조교 방식 몇 가지를 같은 시험 곡들로 돌려서 정확도를 표로 비교한다 (소리는 재생하지 않는다).

- 테토 정답 노래 (make_reference_song.py로 만들어 실제 테토로 렌더한 것, 폴더마다 ref_ref.wav · truth.json · lyrics.txt):
  가사 넣고 조교 → 정답 음표와 하나씩 짝지어(50ms 안) 음 시작 F1 · 가사 · 음 높이 · 시작 오차 (울리는 자음 · 그 밖 따로)
- vocadito 사람 노래 (사람이 표시한 음 높이 · 음표가 있다): 가사 없이 조교 → 음 높이 정확도(RPA, 50센트)와
  음표 정확도(mir_eval COnP · COnPOff · COn F1)
- --render: 몇 가지 방식만 vocadito 몇 곡을 실제 테토로 렌더해서 원래 보컬과 비교 (compare_render.py)

사용: python accuracy.py --refs out/ref* --vocadito vocadito폴더 [--render OpenUtau폴더 작업폴더 --render-songs 10,14]
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tetotune import features, notes, pitch  # noqa: E402
from tetotune import tune as tune_mod  # noqa: E402
from tetotune.tune import Options, tune  # noqa: E402


def variants():
    """이름 → [(모듈, 이름, 값)]. 첫 번째가 지금 기본값."""
    return {
        "기본": [],
        "모음 시작": [(notes, "VOWEL_ONSET", True)],
        "구간 음정 10초": [(notes, "LOCAL_OFFSET", 10.0)],
        "구간 음정 20초": [(notes, "LOCAL_OFFSET", 20.0)],
        "꺾기 남기기 6dB": [(pitch, "EDGE_KEEP_DB", 6.0)],
        "꺾기 남기기 12dB": [(pitch, "EDGE_KEEP_DB", 12.0)],
        "크기 창 30ms": [(tune_mod, "DYN_WIN", 0.03)],
        "RMVPE": [(None, "pitch_method", "rmvpe")],
    }


RENDER_VARIANTS = ("기본", "꺾기 남기기 6dB", "크기 창 30ms", "RMVPE")
OPTS = {"transpose": "0", "pitch_method": "harvest"}  # 방식마다 (None, 옵션 이름, 값)으로 바꾼다


# ---------------------------------------------------------------- 같은 입력이면 다시 계산하지 않기 (방식마다 음 높이 읽기는 같다)
def _key(*args):
    h = hashlib.sha1()
    for a in args:
        h.update(np.ascontiguousarray(a).tobytes() if isinstance(a, np.ndarray) else repr(a).encode())
    return h.hexdigest()


def _memo(fn):
    cache = {}

    def wrapped(*args, **kw):
        k = _key(*args, *sorted(kw.items()))
        if k not in cache:
            cache[k] = fn(*args, **kw)
        v = cache[k]
        return {a: (b.copy() if isinstance(b, np.ndarray) else b) for a, b in v.items()} if isinstance(v, dict) else \
            tuple(x.copy() if isinstance(x, np.ndarray) else x for x in v)
    return wrapped


pitch.extract = _memo(pitch.extract)
features.analyze = _memo(features.analyze)


def patched(patches):
    class Ctx:
        def __enter__(self):
            self.saved = [(m, a, getattr(m, a)) for m, a, _ in patches if m is not None]
            self.opts = dict(OPTS)
            for m, a, v in patches:
                if m is None:
                    OPTS[a] = v
                else:
                    setattr(m, a, v)

        def __exit__(self, *exc):
            for m, a, v in self.saved:
                setattr(m, a, v)
            OPTS.clear()
            OPTS.update(self.opts)
    return Ctx()


def run_tune(wav, lyrics_text, out):
    res = tune(wav, out, lyrics_text, options=Options(**OPTS), title="t")
    if res.pitch_method != OPTS["pitch_method"]:
        raise RuntimeError(f"{OPTS['pitch_method']}를 쓰지 못했어요 ({res.pitch_method})")
    return json.loads((Path(out) / "preview.json").read_text())


# ---------------------------------------------------------------- 테토 정답 노래
def match_onsets(truth_on, est_on, tol=0.05):
    """정답 · 추정 음 시작을 하나씩 짝짓는다 (가까운 순서로, 한 번 쓴 것은 다시 안 씀) → [(정답 번호, 추정 번호)]."""
    pairs = sorted((abs(t - e), i, j) for i, t in enumerate(truth_on) for j, e in enumerate(est_on) if abs(t - e) <= tol)
    used_i, used_j, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
    return out


def score_ref(truth, prev):
    syl = [x for x in prev["notes"] if x[3] != "+"]
    tn = truth["notes"]
    pairs = match_onsets([x[0] for x in tn], [x[0] for x in syl])
    tp = len(pairs)
    p, r = tp / max(1, len(syl)), tp / max(1, len(tn))
    lyric = sum(syl[j][3] == tn[i][3] for i, j in pairs)
    tone = sum(syl[j][2] - prev["shift"] == tn[i][2] for i, j in pairs)
    err = {"v": [], "o": []}
    for i, t in enumerate(tn):  # 시작 오차: 가장 가까운 추정 음절 (짝이 없어도)
        e = min(abs(x[0] - t[0]) for x in syl) * 1000 if syl else 999
        err["v" if notes._voiced_onset(t[3]) else "o"].append(e)
    return {"n": len(tn), "tp": tp, "p": p, "r": r, "lyric": lyric, "tone": tone, "err_v": err["v"], "err_o": err["o"]}


def eval_refs(refs, patches):
    res = []
    for d in refs:
        with patched(patches), tempfile.TemporaryDirectory() as tmp:
            prev = run_tune(d / "ref_ref.wav", (d / "lyrics.txt").read_text(encoding="utf-8"), Path(tmp) / "o")
        res.append(score_ref(json.load(open(d / "truth.json")), prev))
    n = sum(r["n"] for r in res)
    tp = sum(r["tp"] for r in res)
    est = sum(r["tp"] / r["p"] if r["p"] else 0 for r in res)
    f1 = 2 * tp / max(1, n + est)
    ev = [e for r in res for e in r["err_v"]]
    eo = [e for r in res for e in r["err_o"]]
    return {"ref_f1": f1, "ref_lyric": sum(r["lyric"] for r in res), "ref_tone": sum(r["tone"] for r in res), "ref_n": n,
            "ref_err_v": float(np.median(ev)) if ev else float("nan"), "ref_err_o": float(np.median(eo)) if eo else float("nan"),
            "ref_err_v90": float(np.percentile(ev, 90)) if ev else float("nan")}


# ---------------------------------------------------------------- vocadito
def _read_csv(p):
    rows = []
    for row in csv.reader(open(p)):
        try:
            rows.append([float(v) for v in row])
        except ValueError:
            continue  # 머리글
    return np.array(rows)


def vocadito_songs(root):
    """→ [(번호, 오디오, f0 csv, 음표 csv)] (음표는 A1 표시)."""
    root = Path(root)
    out = []
    for audio_path in sorted(root.rglob("vocadito_*.wav"), key=lambda p: int(p.stem.split("_")[1])):
        k = audio_path.stem.split("_")[1]
        f0 = next(root.rglob(f"vocadito_{k}_f0.csv"), None)
        nts = next(root.rglob(f"vocadito_{k}_notesA1.csv"), None)
        if f0 and nts:
            out.append((int(k), audio_path, f0, nts))
    return out


def score_vocadito(prev, f0_csv, notes_csv):
    import mir_eval

    f0 = _read_csv(f0_csv)
    src = np.array([np.nan if v is None else v for v in prev["source"]], dtype=float)
    t_est = np.arange(len(src)) * prev["hop"] * 2
    hz = np.where(np.isfinite(src), pitch.midi_to_hz(np.nan_to_num(src, nan=60)), 0.0)
    mel = mir_eval.melody.evaluate(f0[:, 0], f0[:, 1], t_est, hz)
    nn = _read_csv(notes_csv)  # 시작, 음 높이(Hz), 길이
    ref_iv = np.stack([nn[:, 0], nn[:, 0] + nn[:, 2]], axis=1)
    est = [x for x in prev["notes"] if x[1] > x[0]]
    # 가사 없이 조교하면 같은 음을 이어 부르는 "+" 음표가 생긴다: 사람 표시는 음표 하나라서 이어 붙인다
    merged = []
    for x in est:
        if merged and x[3] == "+" and abs(merged[-1][1] - x[0]) < 1e-3 and merged[-1][2] == x[2]:
            merged[-1][1] = x[1]
        else:
            merged.append(list(x))
    est_iv = np.array([[x[0], x[1]] for x in merged]).reshape(-1, 2)
    est_hz = pitch.midi_to_hz(np.array([x[2] - prev["shift"] + prev.get("offset", 0.0) for x in merged]))
    tr = mir_eval.transcription
    conp = tr.precision_recall_f1_overlap(ref_iv, nn[:, 1], est_iv, est_hz, offset_ratio=None)[2]
    conpoff = tr.precision_recall_f1_overlap(ref_iv, nn[:, 1], est_iv, est_hz)[2]
    con = tr.onset_precision_recall_f1(ref_iv, est_iv)[2]
    return {"rpa": mel["Raw Pitch Accuracy"], "vfa": mel["Voicing False Alarm"], "vr": mel["Voicing Recall"],
            "conp": conp, "conpoff": conpoff, "con": con}


def eval_vocadito(songs, patches, keep=None):
    res = []
    for k, wav, f0, nts in songs:
        with patched(patches), tempfile.TemporaryDirectory() as tmp:
            out = Path(keep) / str(k) if keep else Path(tmp) / "o"
            prev = run_tune(wav, "", out)
        res.append(score_vocadito(prev, f0, nts))
    return {f"v_{m}": float(np.mean([r[m] for r in res])) for m in res[0]} if res else {}


# ---------------------------------------------------------------- 렌더
def render_compare(ou, work, folder):
    render = folder / "render.wav"
    subprocess.run([str(ROOT / "devtools/openutau_harness/render.sh"), str(ou), str(work), str(folder / "project.ustx"),
                    str(render)], check=True, stdout=subprocess.DEVNULL)
    wav = sorted(folder.glob("render_*.wav"))[0]
    out = subprocess.run([sys.executable, str(ROOT / "devtools/compare_render.py"), str(folder), str(wav)],
                         check=True, capture_output=True, text=True).stdout
    for f in folder.glob("render*.wav"):
        f.unlink()
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--vocadito")
    ap.add_argument("--render", nargs=2, metavar=("OPENUTAU", "WORK"))
    ap.add_argument("--render-songs", default="10,14")
    ap.add_argument("--only", help="쉼표로 나눈 방식 이름만")
    args = ap.parse_args()
    refs = [Path(p) for p in args.refs if (Path(p) / "truth.json").exists()]
    songs = vocadito_songs(args.vocadito) if args.vocadito else []
    vs = variants()
    if args.only:
        vs = {k: v for k, v in vs.items() if k in args.only.split(",")}
    print(f"테토 정답 노래 {len(refs)}곡 · vocadito {len(songs)}곡", flush=True)
    rows = {}
    for name, patches in vs.items():
        row = {}
        try:
            if refs:
                row.update(eval_refs(refs, patches))
            if songs:
                row.update(eval_vocadito(songs, patches))
        except RuntimeError as e:
            print(name, "건너뜀:", e, flush=True)
            continue
        rows[name] = row
        print(name, json.dumps({k: round(v, 4) for k, v in row.items()}, ensure_ascii=False), flush=True)
    if refs or songs:
        print()
        print("| 방식 | 정답 노래 음 시작 F1 | 가사 | 음 높이 | 시작 오차 울리는 자음 (중앙/90%) · 그 밖 | "
              "사람 노래 음 높이 RPA | 유성 오경보 | 음표 COnP | COnPOff | COn |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for name, r in rows.items():
            ref = (f"{r['ref_f1']:.3f} | {r['ref_lyric']}/{r['ref_n']} | {r['ref_tone']}/{r['ref_n']} | "
                   f"{r['ref_err_v']:.0f}/{r['ref_err_v90']:.0f} · {r['ref_err_o']:.0f}ms") if "ref_f1" in r else " | | | "
            hum = (f"{r['v_rpa']:.3f} | {r['v_vfa']:.3f} | {r['v_conp']:.3f} | {r['v_conpoff']:.3f} | {r['v_con']:.3f}"
                   ) if "v_rpa" in r else " | | | | "
            print(f"| {name} | {ref} | {hum} |")
    if args.render and songs:
        want = {int(x) for x in args.render_songs.split(",")}
        print()
        print("| 방식 | 곡 | 렌더 음 높이 오차 중앙/90% (센트) | 50센트 안 | 크기 상관 | 숨 크기 (테토 / 원래) |")
        print("|---|---|---|---|---|---|")
        for name in RENDER_VARIANTS:
            if name not in vs:
                continue
            if name not in rows:
                continue
            for k, wav, f0, nts in songs:
                if k not in want:
                    continue
                with patched(vs[name]), tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp) / "o"
                    run_tune(wav, "", out)
                    c = render_compare(Path(args.render[0]), Path(args.render[1]), out)
                print(f"| {name} | {k} | {c['pitch_err_cents_median']}/{c['pitch_err_cents_p90']} | {c['within_50_cents']} | "
                      f"{c['loudness_corr']} | {c['breath_db_vs_singing']} / {c['source_breath_db_vs_singing']} |", flush=True)


if __name__ == "__main__":
    main()

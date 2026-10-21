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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tetotune import features, game, notes, pitch  # noqa: E402
from tetotune import refine as refine_mod  # noqa: E402
from tetotune import tune as tune_mod  # noqa: E402
from tetotune.tune import Options, tune  # noqa: E402


class Cfg:
    refine = refine_mod.ROUNDS  # 렌더해서 고치기 횟수 (앱 기본처럼). 0 = 안 함


def variants():
    """이름 → [(모듈, 이름, 값)]. 첫 번째가 지금 기본값. (None, 옵션 이름, 값)은 조교 설정을 바꾼다."""
    pr4 = [(tune_mod, "OCTAVE_FIX", False), (tune_mod, "GAP_NOTE", 0.0)]  # main (PR #4)
    return {
        "새 기본": [],
        "PR4 기본": pr4,
        "긴 틈만": [(tune_mod, "OCTAVE_FIX", False), (tune_mod, "GAP_NOTE", 0.05)],
        "옥타브 접기만": [(tune_mod, "OCTAVE_FIX", True), (tune_mod, "GAP_NOTE", 0.0)],
    }


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
        if isinstance(v, dict):
            return {a: (b.copy() if isinstance(b, np.ndarray) else b) for a, b in v.items()}
        if isinstance(v, tuple):
            return tuple(x.copy() if isinstance(x, np.ndarray) else x for x in v)
        return list(v) if isinstance(v, list) else v
    return wrapped


pitch.extract = _memo(pitch.extract)
_transcribe_once = game.transcribe
game.transcribe = _memo(game.transcribe)
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
    if not res.pitch_method.startswith(OPTS["pitch_method"]) or (notes.USE_GAME and "+game" not in res.pitch_method):
        raise RuntimeError(f"고른 방법을 쓰지 못했어요 ({res.pitch_method}, {res.warnings})")
    if not notes.USE_GAME and "+game" in res.pitch_method:
        raise RuntimeError("GAME을 껐는데 썼어요")
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
def render_compare(ou, work, folder, truth=None):
    render = folder / "render.wav"
    subprocess.run([str(ROOT / "devtools/openutau_harness/render.sh"), str(ou), str(work), str(folder / "project.ustx"),
                    str(render)], check=True, stdout=subprocess.DEVNULL)
    wav = sorted(folder.glob("render_*.wav"))[0]
    cmd = [sys.executable, str(ROOT / "devtools/compare_render.py"), str(folder), str(wav)]
    if truth:
        cmd += ["--truth", str(truth)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    for f in folder.glob("render*.wav"):
        f.unlink()
    return json.loads(out)


def _render_only(ou, work, ustx):
    """refine.py가 부르는 렌더: ustx 옆 _r 폴더에 렌더해서 테토 트랙 wav 경로."""
    d = Path(ustx).parent / "_r"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    subprocess.run([str(ROOT / "devtools/openutau_harness/render.sh"), str(ou), str(work), str(ustx), str(d / "render.wav")],
                   check=True, stdout=subprocess.DEVNULL)
    return sorted(d.glob("render_*.wav"))[0]


def compare_wav(folder, wav, truth=None):
    cmd = [sys.executable, str(ROOT / "devtools/compare_render.py"), str(folder), str(wav)]
    if truth:
        cmd += ["--truth", str(truth)]
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


RENDER_KEYS = ("pitch_err_cents_median", "pitch_err_cents_p90", "within_50_cents", "loudness_corr", "loudness_mae_db",
               "dropouts_per_min",
               "pitch_drops_per_min", "pitch_jumps_per_min", "off_note_ok_per_min", "wobble_ratio")


def _mean(cs, k):
    v = [c[k] for c in cs if c.get(k) is not None]
    return float(np.mean(v)) if v else float("nan")


def render_set(name, items, patches, args):
    """items: [(이름, 보컬, 가사, 비교할 정답 보컬 또는 None)] → 곡마다 비교 결과."""
    out = []
    for label, wav, text, truth in items:
        with patched(patches), tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "o"
            try:
                run_tune(wav, text, folder)
                if Cfg.refine:
                    rr = refine_mod.refine(folder, lambda u: _render_only(Path(args.render[0]), Path(args.render[1]), u),
                                           rounds=Cfg.refine)
                    print(f"{name} {label} 고치기", json.dumps(rr["rounds"], ensure_ascii=False), flush=True)
                    c = compare_wav(folder, folder / "teto_preview.wav", truth)
                else:
                    c = render_compare(Path(args.render[0]), Path(args.render[1]), folder, truth)
            except (RuntimeError, subprocess.CalledProcessError) as e:
                print(f"{name} {label} 렌더 못 함: {e}", flush=True)
                continue
        c["song"] = label
        events = c.pop("events", [])
        print(f"{name} {label}", json.dumps({k: c.get(k) for k in RENDER_KEYS + ("event_kinds",)}, ensure_ascii=False),
              flush=True)
        for e in events[:12]:
            print("   ", json.dumps(e, ensure_ascii=False), flush=True)
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--refs-ko", nargs="*", default=[], help="한글 가사 테토 정답 노래 (KO to JA)")
    ap.add_argument("--vocadito")
    ap.add_argument("--separated", help="make_separated.py 결과 폴더 (sep_<번호>.wav)")
    ap.add_argument("--render", nargs=2, metavar=("OPENUTAU", "WORK"))
    ap.add_argument("--render-songs", default="10,14")
    ap.add_argument("--render-sep-songs", help="분리한 보컬은 이 곡들을 렌더 (없으면 --render-songs)")
    ap.add_argument("--render-refs", type=int, default=0, help="테토 정답 노래 몇 곡을 조교해서 다시 렌더해 비교할지 (한글도 같은 수)")
    ap.add_argument("--only", help="쉼표로 나눈 방식 이름만")
    ap.add_argument("--json", help="결과를 이 파일에 (여러 잡 결과를 합칠 때)")
    args = ap.parse_args()
    refs = [Path(p) for p in args.refs if (Path(p) / "truth.json").exists()]
    krefs = [Path(p) for p in args.refs_ko if (Path(p) / "truth.json").exists()]
    songs = vocadito_songs(args.vocadito) if args.vocadito else []
    sep = []
    if args.separated and songs:
        by_k = {k: (f0, nts, wav) for k, wav, f0, nts in songs}
        for f in sorted(Path(args.separated).glob("sep_*.wav")):
            k = int(f.stem.split("_")[1])
            if k in by_k:
                sep.append((k, f, by_k[k][0], by_k[k][1], by_k[k][2]))
    vs = variants()
    if args.only:
        vs = {k: v for k, v in vs.items() if k in args.only.split(",")}
    print(f"테토 정답 노래 {len(refs)}곡 · 한글 {len(krefs)}곡 · vocadito {len(songs)}곡 · 분리한 보컬 {len(sep)}곡", flush=True)
    if songs and game.model_dir():  # GAME이 같은 노래에 늘 같은 음표를 내는지 (잡마다 비교가 흔들리지 않게)
        from tetotune import audio
        y, sr = audio.load(songs[0][1])
        f0, _ = pitch.extract(y, sr, "harvest")
        _, v = pitch.clean(f0, pitch.frame_db(y, sr, len(f0)))
        a, b = _transcribe_once(y, sr, v, pitch.HOP), _transcribe_once(y, sr, v, pitch.HOP)
        print(f"GAME 두 번 같은 결과: {a == b} ({len(a)}음표)", flush=True)
    results = {}
    for name, patches in vs.items():
        row = {}
        try:
            if refs:
                row.update(eval_refs(refs, patches))
            if krefs:
                row.update({"k" + k: v for k, v in eval_refs(krefs, patches).items()})
            if songs:
                row.update(eval_vocadito(songs, patches))
            if sep:
                row.update({"s" + k: v for k, v in eval_vocadito([(k, w, f0, nts) for k, w, f0, nts, _ in sep], patches).items()})
        except RuntimeError as e:
            print(name, "건너뜀:", e, flush=True)
            continue
        print(name, json.dumps({k: round(v, 4) for k, v in row.items()}, ensure_ascii=False), flush=True)
        rend = {}
        if args.render:
            want = {int(x) for x in args.render_songs.split(",")}
            n = args.render_refs
            sets = {
                "ref": [(d.name, d / "ref_ref.wav", (d / "lyrics.txt").read_text(encoding="utf-8"), None) for d in refs[:n]],
                "kref": [(d.name, d / "ref_ref.wav", (d / "lyrics.txt").read_text(encoding="utf-8"), None) for d in krefs[:n]],
                "vocadito": [(str(k), wav, "", None) for k, wav, _, _ in songs if k in want],
                "sep": [(str(k), f, "", clean) for k, f, _, _, clean in sep
                        if k in ({int(x) for x in args.render_sep_songs.split(",")} if args.render_sep_songs else want)],
            }
            for key, items in sets.items():
                if items:
                    rend[key] = render_set(name, items, patches, args)
        results[name] = {"row": row, "render": rend}
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print_tables(results)


def print_tables(results):
    """여러 방식 결과(main의 results, 잡 여러 개를 합친 것도) → 마크다운 표."""
    rows = {k: v["row"] for k, v in results.items()}

    def ref_cell(r, p=""):
        if p + "ref_f1" not in r:
            return " | | "
        return f"{r[p + 'ref_f1']:.3f} | {r[p + 'ref_lyric']}/{r[p + 'ref_n']} | {r[p + 'ref_tone']}/{r[p + 'ref_n']}"

    def hum_cell(r, p=""):
        if p + "v_rpa" not in r:
            return " | | "
        return f"{r[p + 'v_rpa']:.3f} | {r[p + 'v_conp']:.3f} | {r[p + 'v_con']:.3f}"
    print()
    print("| 방식 | 테토 정답 F1 | 가사 | 음 높이 | 한글 정답 F1 | 가사 | 음 높이 | 사람 RPA | COnP | COn | "
          "분리 보컬 RPA | COnP | COn |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {ref_cell(r)} | {ref_cell(r, 'k')} | {hum_cell(r)} | {hum_cell(r, 's')} |")
    labels = {"ref": "테토 정답", "kref": "한글 정답", "vocadito": "사람 노래", "sep": "분리한 보컬"}
    for key, label in labels.items():
        have = [(name, v["render"][key]) for name, v in results.items() if v["render"].get(key)]
        if not have:
            continue
        print()
        print(f"| 방식 | {label} → 조교 → 테토 렌더 | 음 높이 오차 중앙/90% | 50센트 안 | 크기 상관 | 크기 차이 dB | "
              f"끊김 /분 | 떨어짐 /분 | 튐 /분 | 음표는 맞는데 벗어남 /분 | 흔들림 (1=원래) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, cs in have:
            print(f"| {name} | {len(cs)}곡 평균 | {_mean(cs, 'pitch_err_cents_median'):.1f}/{_mean(cs, 'pitch_err_cents_p90'):.1f}"
                  f" | {_mean(cs, 'within_50_cents'):.3f} | {_mean(cs, 'loudness_corr'):.3f} | "
                  f"{_mean(cs, 'loudness_mae_db'):.2f} | "
                  f"{_mean(cs, 'dropouts_per_min'):.2f} | {_mean(cs, 'pitch_drops_per_min'):.2f} | "
                  f"{_mean(cs, 'pitch_jumps_per_min'):.2f} | {_mean(cs, 'off_note_ok_per_min'):.2f} | "
                  f"{_mean(cs, 'wobble_ratio'):.2f} |")
        kinds = {}
        for name, cs in have:
            for c in cs:
                for k, v in c.get("event_kinds", {}).items():
                    kinds.setdefault(name, {}).setdefault(k, 0)
                    kinds[name][k] += v
        for name, kd in kinds.items():
            top = sorted(kd.items(), key=lambda kv: -kv[1])[:8]
            print(f"\n{label} · {name} 문제 종류: " + ", ".join(f"{k} {v}" for k, v in top))


def merge_main():
    """python accuracy.py --merge 결과*.json: 잡마다 나온 결과를 합쳐 표로."""
    results = {}
    for f in sys.argv[2:]:
        results.update(json.loads(Path(f).read_text(encoding="utf-8")))
    order = list(variants())
    print_tables({k: results[k] for k in sorted(results, key=lambda k: order.index(k) if k in order else 99)})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_main()
    else:
        main()

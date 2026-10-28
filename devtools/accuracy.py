"""조교 방식 몇 가지를 같은 시험 곡들로 돌려서 정확도를 표로 비교한다 (소리는 재생하지 않는다).

- 테토 정답 노래 (make_reference_song.py로 만들어 실제 테토로 렌더한 것, 폴더마다 ref_ref.wav · truth.json · lyrics.txt):
  가사 넣고 조교 → 정답 음표와 하나씩 짝지어(50ms 안) 음 시작 F1 · 가사 · 음 높이 · 시작 오차 (울리는 자음 · 그 밖 따로)
- vocadito 사람 노래 (사람이 표시한 음 높이 · 음표가 있다): 가사 없이 조교 → 음 높이 정확도(RPA, 50센트)와
  음표 정확도(mir_eval COnP · COnPOff · COn F1)
- CSD 한국 동요 (사람이 부른 노래, 음절마다 시작 · 끝 · 음 높이 정답, CC BY-NC-SA 4.0): 가사를 숨 쉬는 곳마다 한 줄로
  넣고 조교 → 음절이 맞는 때에 들어갔는지(50 · 100ms 안, 치우침), 그 음절의 음 높이, 음표 정확도(COnP).
  지금까지는 사람 노래에서 가사 위치를 잰 적이 없었다 (테토 정답 노래는 음절마다 음표 하나라 쉽다)
- --render: 몇 가지 방식만 vocadito 몇 곡을 실제 테토로 렌더해서 원래 보컬과 비교 (compare_render.py)

사용: python accuracy.py --refs out/ref* --vocadito vocadito폴더 [--render OpenUtau폴더 작업폴더 --render-songs 10,14]
"""
import argparse
import csv
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tetotune import align_fa, clean, f0models, features, game, notes, oupitch, pitch  # noqa: E402
from tetotune import refine as refine_mod  # noqa: E402
from tetotune import tune as tune_mod  # noqa: E402
from tetotune.tune import Options, tune  # noqa: E402


class Cfg:
    refine = refine_mod.ROUNDS  # 렌더해서 고치기 횟수 (앱 기본처럼). 0 = 안 함


# 이번에 고친 것마다 예전 값. 하나씩 끈 판과 비교하면 무엇이 얼마나 도왔는지 (또는 해쳤는지) 보인다
OFF = {
    "OpenUtau 그대로 PITD": [(tune_mod, "OU_EXACT", False), (oupitch, "PORTA_MS", 0.0), (oupitch, "GAP_RAMP_MS", 0.0)],
    "가사 맞추기 모델": [(align_fa, "USE_FA", False)],
    "GAME 모음 시작 그대로": [(notes, "VOWEL_ONSET_GAME", True)],
    "っ · + 고침": [(notes, "CLOSURE", False), (notes, "PLUS_FIX", False)],
    "GAME 음 높이 고침": [(notes, "TONE_FROM_F0", False)],
    "잔떨림 · 바로잡기 문턱": [(tune_mod, "JITTER_HZ", 0.0), (tune_mod, "CORRECT_GATE", 0.0)],
    "다듬기 음 높이 한도": [(refine_mod, "PITCH_FIX", "old")],
    "숨 자리": [(tune_mod, "BREATH_ROOM", False)],
    "보컬 다시 분리": [(clean, "SEPARATE", "off")],
    # PR10 뒤: 나눠 보기(diagnose.py)에서 50ms 넘게 틀린 음절의 78%는 모델이 찾은 모음 시작 자체가 틀린 것(17ms쯤 늦음),
    # 18%는 음표를 나누지 못해 프레이즈째 버린 것이었고, 가사 줄 수와 찾은 프레이즈 수가 다른 곡에서 많이 틀렸다
    "이웃 프레이즈 함께 맞추기": [(align_fa, "JOINT", 0.0)],
    "가사 맞추기 시간 보정 · 덜 버리기": [(align_fa, "OFFSET", 0.0), (align_fa, "SOFT_SPLIT", False),
                                (align_fa, "GAP_EXTEND", False)],
}
PR10 = OFF["이웃 프레이즈 함께 맞추기"] + OFF["가사 맞추기 시간 보정 · 덜 버리기"]
OLD_RENDER = [(tune_mod, "RENDERER", "WORLDLINE-R"), (tune_mod, "RESAMPLER", None)]  # PR12까지 (main) 렌더러
FUSION = [(None, "pitch_method", "fusion")]
# 기본값을 바꾸면 어떨지 (새 기본에 더해서)
ALT = {
    "+ GAME 경계 증거": [(notes, "GAME_EVIDENCE", True)],
    "+ 다듬기 크기 한도": [(refine_mod, "DYN_TOTAL", 6.0), (refine_mod, "DYN_VOICED", True)],
    # 사람 노래 음절 시작은 더 맞지만 (CSD 0.764 → 0.777, 분리 0.743 → 0.769) 테토 렌더가 나빠진다 (README)
    "가사 맞추기 −15ms": [(align_fa, "OFFSET", -0.015)],
    # 렌더 엔진 (OpenUtau 0.1.576 안에 든 것): 새 기본은 hifisampler (CLASSIC + 신경망 보코더 pc-nsf-hifigan).
    # WORLDLINE-R1.1은 hnsep(배음 · 잡음 나누기 AI) 특징 + 5ms 프레임, WORLDLINE-R2는 WORLD 특징 → 같은 보코더 (메뉴에는 없다)
    "렌더 WORLDLINE-R1.1": [(tune_mod, "RENDERER", "WORLDLINE-R1.1"), (tune_mod, "RESAMPLER", None)],
    "렌더 WORLDLINE-R2": [(tune_mod, "RENDERER", "WORLDLINE-R2"), (tune_mod, "RESAMPLER", None)],
    "렌더 CLASSIC worldline": [(tune_mod, "RENDERER", "CLASSIC"), (tune_mod, "RESAMPLER", "worldline")],
    # hifisampler는 숨 섞임(BREC) · 힘(TENC) 곡선이 있으면 hnsep으로 배음 · 잡음을 나눠 섞는다: 곡선 없이 부르면 어떤지
    "hifisampler − 숨 · 힘 곡선": [(None, "breathiness", 0.0), (None, "tension", 0.0)],
    # 음 높이 읽기에 CREPE full도 (넷 합치기), 음표 나누기 GAME 큰 · 작은 모델 (같은 1.0.3, 받아 둔 폴더가 있을 때)
    "+ CREPE 음 높이": [(f0models, "USE", ("rmvpe", "crepe", "swift", "harvest"))],
    "GAME 큰 모델": [(game, "FOLDER", str(game.models_home() / "game-large"))],
    "GAME 작은 모델": [(game, "FOLDER", str(game.models_home() / "game-small"))],
    # 빨리 견주기(quick, CSD 12곡 음절 시작 50ms 안, PR10 기본 0.706)에서 본 것 — 손으로 다시 보려면 "새 기본 & align_fa.X=값":
    # 함께 맞추기 15 · 30 · 45초 0.758 · 0.777 · 0.768, 시간 보정 −5 · −15 · −25ms (15초 함께) 0.743 · 0.758 · 0.739
    # (테토 정답 노래는 −5ms가 조금 낫다), 여러 번 돌려 가운데(TTA 3 · 5) 0.748 · 0.743 (보정 · 덜 버리기만 0.750),
    # 첫 음절 0.4초 · 앞뒤 여유 0.6 · 0.5초 · 받침을 영어 음소로(KO_FINALS) · ㅓ를 중국어 음소로(KO_EO): ±0.005 안
}


def variants():
    """이름 → [(모듈, 이름, 값)]. 첫 번째가 지금 기본값. (None, 옵션 이름, 값)은 조교 설정을 바꾼다."""
    out = {"새 기본": FUSION, "예전 기본 (WORLDLINE-R)": FUSION + OLD_RENDER, "PR10 기본": FUSION + PR10 + OLD_RENDER,
           "PR6 기본": [p for v in OFF.values() for p in v] + OLD_RENDER, "− 음 높이 합치기 (Harvest)": []}
    for k, v in OFF.items():
        out[f"− {k}"] = FUSION + v
    for k, v in ALT.items():
        out[k] = FUSION + v
    out["다듬기 없음"] = FUSION + [(Cfg, "refine", 0)]
    return out


MODS = {"align_fa": align_fa, "clean": clean, "f0models": f0models, "game": game, "notes": notes, "oupitch": oupitch,
        "refine": refine_mod, "tune": tune_mod}


def with_flags(vs, name):
    """"방식 이름 & align_fa.FIRST_MAX=0.4 & ..." 꼴이면 그 방식에 스위치를 더 바꾼 판 (손으로 돌리는 빠른 비교에서 이어 붙여 보기)."""
    import ast

    base, *extra = name.split(" & ")
    out = list(vs[base])
    for e in extra:
        key, val = e.split("=", 1)
        mod, flag = key.strip().split(".")
        if not hasattr(MODS[mod], flag):
            raise SystemExit(f"없는 스위치: {key}")
        try:
            v = ast.literal_eval(val.strip())
        except (ValueError, SyntaxError):
            v = val.strip()
        out.append((MODS[mod], flag, v))
    return out


OPTS = {"transpose": "0", "pitch_method": "harvest"}  # 방식마다 (None, 옵션 이름, 값)으로 바꾼다


# ---------------------------------------------------------------- 같은 입력이면 다시 계산하지 않기 (방식마다 음 높이 읽기는 같다)
def _key(*args):
    h = hashlib.sha1()
    for a in args:
        h.update(np.ascontiguousarray(a).tobytes() if isinstance(a, np.ndarray) else repr(a).encode())
    return h.hexdigest()


def _memo(fn, setting=lambda: None):
    """같은 입력(+ setting()이 돌려주는 스위치 값)이면 다시 계산하지 않는다."""
    cache = {}

    def wrapped(*args, **kw):
        k = _key(setting(), *args, *sorted(kw.items()))
        if k not in cache:
            cache[k] = fn(*args, **kw)
        v = cache[k]
        if isinstance(v, dict):
            return {a: (b.copy() if isinstance(b, np.ndarray) else b) for a, b in v.items()}
        if isinstance(v, tuple):
            return tuple(x.copy() if isinstance(x, np.ndarray) else x for x in v)
        return list(v) if isinstance(v, list) else v
    return wrapped


pitch.extract = _memo(pitch.extract, lambda: (f0models.USE, f0models.WEIGHTS.get("crepe")))
_transcribe_once = game.transcribe
game.transcribe = _memo(game.transcribe, lambda: str(game.model_dir()))
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


FA_LOG = []  # 가사 맞추기 모델이 프레이즈마다 남긴 기록 (확신 · 맞춤 여부 · 옮긴 거리), 세트마다 요약해서 찍는다
FA_GAIN = []  # 맞춘 프레이즈마다 (확신, 가장 멀리 옮긴 ms, 맞추기 전 · 뒤 정답 시작 50ms 안 음절 수)
USED = set()


def run_tune(wav, lyrics_text, out):
    wav = Path(wav)  # 분리한 보컬(sep_…)은 옆의 반주(bk_…)를 같이 준다 (StemMixer처럼)
    bk = wav.with_name("bk_" + wav.name[4:]) if wav.name.startswith("sep_") else None
    align_fa.LAST.clear()  # 가사 맞추기를 끈 방식에서 앞 곡 기록이 남지 않게
    res = tune(wav, out, lyrics_text, backing=str(bk) if bk and bk.exists() else None, options=Options(**OPTS),
               title="t")
    FA_LOG.extend(align_fa.LAST)
    if bk and clean.wanted("vocals", True) and not clean.model_path("vocals") and "noclean" not in USED:
        USED.add("noclean")
        print("  보컬 다시 분리 모델이 없어서 분리한 보컬을 그대로 읽었어요", flush=True)
    if res.pitch_method not in USED:
        USED.add(res.pitch_method)
        print(f"  쓴 방법: {res.pitch_method}", flush=True)
    if not res.pitch_method.startswith(OPTS["pitch_method"]) or (notes.USE_GAME and "+game" not in res.pitch_method):
        raise RuntimeError(f"고른 방법을 쓰지 못했어요 ({res.pitch_method}, {res.warnings})")
    if not notes.USE_GAME and "+game" in res.pitch_method:
        raise RuntimeError("GAME을 껐는데 썼어요")
    return json.loads((Path(out) / "preview.json").read_text())


def fa_gain(truth_on, log=None):
    """마지막 조교의 가사 맞추기 기록 → 맞춘 프레이즈마다 정답 음절 시작(50ms 안)이 맞추기 전보다 몇 개 늘었는지를 FA_GAIN에."""
    for r in log if log is not None else align_fa.LAST:
        if not r.get("before") or r.get("after") is None:
            continue
        a, b = min(r["before"] + r["after"]) - 0.15, max(r["before"] + r["after"]) + 0.15
        t = [x for x in truth_on if a <= x <= b]
        FA_GAIN.append((r["conf"], r.get("moved_max_ms") or 0.0, len(match_onsets(t, r["before"])),
                        len(match_onsets(t, r["after"]))))


def fa_summary(label):
    """가사 맞추기 모델 기록 요약을 찍고 비운다 (확신 문턱 MIN_CONF을 정할 때 본다)."""
    rows = [r for r in FA_LOG if "conf" in r]
    errs = [r["error"] for r in FA_LOG if "error" in r]
    unknown = [r["unknown"] for r in FA_LOG if "unknown" in r]
    FA_LOG.clear()
    if FA_GAIN:  # 확신 · 옮긴 거리마다 맞추기가 정답 음절 시작을 얼마나 늘리고 줄였는지
        g = np.array(FA_GAIN, dtype=float)
        FA_GAIN.clear()
        cells = []
        for lo, hi in ((0, .55), (.55, .6), (.6, .65), (.65, .7), (.7, 1.01)):
            m = (g[:, 0] >= lo) & (g[:, 0] < hi)
            cells.append(f"{lo:.2f}~ {int(m.sum())}개 {int(g[m, 3].sum() - g[m, 2].sum()):+d}")
        moves = []
        for lo, hi in ((0, 60), (60, 150), (150, 300), (300, 1e9)):
            m = (g[:, 1] >= lo) & (g[:, 1] < hi)
            moves.append(f"{lo:.0f}ms~ {int(m.sum())}개 {int(g[m, 3].sum() - g[m, 2].sum()):+d}")
        print(f"  가사 맞추기 {label} 맞추기 전 → 뒤 정답 시작 50ms 안 음절: {int(g[:, 2].sum())} → {int(g[:, 3].sum())}; "
              f"확신별 {', '.join(cells)}; 가장 멀리 옮긴 거리별 {', '.join(moves)}", flush=True)
    if not rows and not errs and not unknown:
        return
    conf = np.array([r["conf"] for r in rows]) if rows else np.array([np.nan])
    moved = [r["moved_ms"] for r in rows if r.get("moved_ms") is not None]
    print(f"  가사 맞추기 {label}: 프레이즈 {len(rows)}개 중 {sum(r['placed'] for r in rows)}개 맞춤, "
          f"확신 10/50/90% {np.nanpercentile(conf, 10):.3f}/{np.nanpercentile(conf, 50):.3f}/{np.nanpercentile(conf, 90):.3f}, "
          f"옮긴 거리 중앙 {np.median(moved) if moved else float('nan'):.0f}ms, 오류 {len(errs)} {errs[:2]}, "
          f"모르는 음절이라 건너뜀 {len(unknown)} {unknown[:3]}", flush=True)


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
    hit = {i for i, _ in pairs}
    miss = []  # 짝을 못 찾은 정답 음절: (시각, 정답 가사, 가장 가까운 추정 음절까지 ms, 그 가사)
    for i, t in enumerate(tn):
        if i not in hit and syl:
            j = min(range(len(syl)), key=lambda j: abs(syl[j][0] - t[0]))
            miss.append((round(t[0], 2), t[3], int((syl[j][0] - t[0]) * 1000), syl[j][3]))
    return {"n": len(tn), "tp": tp, "p": p, "r": r, "lyric": lyric, "tone": tone, "err_v": err["v"], "err_o": err["o"],
            "miss": miss, "extra": len(syl) - tp}


def eval_refs(refs, patches, label=""):
    res = []
    for d in refs:
        with patched(patches), tempfile.TemporaryDirectory() as tmp:
            prev = run_tune(d / "ref_ref.wav", (d / "lyrics.txt").read_text(encoding="utf-8"), Path(tmp) / "o")
        truth = json.load(open(d / "truth.json"))
        fa_gain([t[0] for t in truth["notes"]])
        res.append(score_ref(truth, prev))
        r = res[-1]
        if r["miss"] or r["extra"]:  # 어느 음절을 놓치는지 (방식끼리 비교해서 무엇이 가사 · 음표를 떨어뜨리는지 찾게)
            print(f"  {label} {d.name}: 놓침 {len(r['miss'])} · 남는 음절 {r['extra']}",
                  " ".join(f"{t}s {a}→{dt:+d}ms {b}" for t, a, dt, b in r["miss"][:8]), flush=True)
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


# ---------------------------------------------------------------- CSD (사람이 부른 한국 동요)
HANGUL_SYL = ("가", "힣")


def _hangul(s):
    return len(s) == 1 and HANGUL_SYL[0] <= s <= HANGUL_SYL[1]


# CSD csv · txt의 음절은 발음 기호 (n_a, s_a_n, w_a: 음소를 _로 이음). 가사(lyric)는 한글 그대로.
CSD_VOWELS = {"a", "ae", "eo", "e", "o", "u", "eu", "ui", "i", "oe"}
_CSD_JUNG = [("", "a"), ("", "ae"), ("y", "a"), ("y", "ae"), ("", "eo"), ("", "e"), ("y", "eo"), ("y", "e"), ("", "o"),
             ("w", "a"), ("w", "ae"), ("", "oe"), ("y", "o"), ("", "u"), ("w", "eo"), ("w", "e"), ("w", "i"), ("y", "u"),
             ("", "eu"), ("", "ui"), ("", "i")]
_CSD_CHO = [{"g", "k", "kk"}, {"kk", "g", "k"}, {"n"}, {"d", "t", "tt"}, {"tt", "d", "t"}, {"r", "l", "n"}, {"m"},
            {"b", "p", "pp"}, {"pp", "b", "p"}, {"s", "ss"}, {"ss", "s"}, set(), {"j", "jj", "ch"}, {"jj", "j", "ch"},
            {"ch", "j"}, {"k", "g"}, {"t", "d"}, {"p", "b"}, {"h"}]
_CSD_SAME = [{"ae", "e"}, {"oe", "e", "ae"}, {"ui", "i", "e"}]


def _csd_vowel_eq(a, b):
    return a == b or any(a in g and b in g for g in _CSD_SAME)


def _csd_token(tok):
    """n_a → (첫 자음, 반모음, 모음). 모음이 없으면 모음은 None."""
    ph = [x for x in tok.split("_") if x]
    vi = next((i for i, x in enumerate(ph) if x in CSD_VOWELS), None)
    if vi is None:
        return (ph[0] if ph else "", "", None)
    head = ph[:vi]
    glide = head[-1] if head and head[-1] in ("y", "w") else ""
    onset = head[0] if head and head[0] not in ("y", "w") else ""
    return onset, glide, ph[vi]


def csd_align(text, tokens):
    """한글 가사 음절 text와 CSD 발음 음절 tokens(음표마다, 같은 수이거나 더 많음)를 맞춘다 → 음표마다 음절 또는 "+",
    모음이 맞은 비율. 한 음절을 여러 음으로 부른 곳은 뒤 음표가 "+" (보통 모음만 적혀 있다)."""
    n, m = len(text), len(tokens)
    if n == 0 or n > m:
        return None, 0.0
    lyr = []
    for c in text:
        code = ord(c) - 0xAC00
        cho, jung = code // 588, (code % 588) // 28
        lyr.append((cho, *_CSD_JUNG[jung]))
    tok = [_csd_token(t) for t in tokens]

    def assign(i, j):
        cho, glide, vowel = lyr[i]
        on, gl, v = tok[j]
        sc = 3.0 if v is not None and _csd_vowel_eq(vowel, v) else -2.0
        if glide == gl:
            sc += 0.5
        if (on in _CSD_CHO[cho]) or (cho == 11 and on == "") or (cho == 18 and on == ""):
            sc += 1.0
        return sc

    def cont(i, j):
        on, gl, v = tok[j]
        return 1.0 if not on and v is not None and _csd_vowel_eq(lyr[i][2], v) else -1.5

    neg = -1e9
    dp = np.full((n + 1, m + 1), neg)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 = 음절 시작, 2 = 앞 음절 이어 부르기
    dp[0, 0] = 0.0
    for j in range(1, m + 1):
        for i in range(1, min(n, j) + 1):
            a = dp[i - 1, j - 1] + assign(i - 1, j - 1)
            c = dp[i, j - 1] + cont(i - 1, j - 1) if j - 1 >= i else neg
            if a >= c:
                dp[i, j], back[i, j] = a, 1
            else:
                dp[i, j], back[i, j] = c, 2
    out, i, j, ok = ["+"] * m, n, m, 0
    while j > 0:
        if back[i, j] == 1:
            out[j - 1] = text[i - 1]
            ok += tok[j - 1][2] is not None and _csd_vowel_eq(lyr[i - 1][2], tok[j - 1][2])
            i -= 1
        j -= 1
    return out, ok / n


def csd_songs(root, count=0):
    """→ [(이름, 오디오, [(시작, 끝, midi, 음절 또는 "+")])] (korean, 같은 노래를 두 조로 부른 것 중 a만).
    csv 음절은 발음 기호라서 가사 파일(lyric, 한글)과 모음 · 자음으로 맞춘다. 한 음절을 여러 음으로 부른 곳은 음표가
    나뉘어 있어서, 남는 음표는 "+"로 둔다. 잘 맞지 않는 노래(모음이 맞은 비율 90% 아래)는 뺀다."""
    import unicodedata

    root = Path(root)
    out = []
    for wav in sorted(root.rglob("korean/wav/*a.wav")):
        csvp = wav.parent.parent / "csv" / (wav.stem + ".csv")
        if not csvp.exists():
            continue
        rows = []
        for row in csv.reader(open(csvp, encoding="utf-8-sig")):
            try:
                rows.append([float(row[0]), float(row[1]), int(round(float(row[2]))),
                             unicodedata.normalize("NFC", row[3].strip())])
            except (ValueError, IndexError):
                continue  # 머리글
        if not rows:
            continue
        lyr = wav.parent.parent / "lyric" / (wav.stem + ".txt")
        raw = unicodedata.normalize("NFC", lyr.read_text(encoding="utf-8-sig")) if lyr.exists() else ""
        text = [c for c in raw if _hangul(c)]
        syl = [r[3] for r in rows]
        if sum(map(_hangul, syl)) < len(syl) / 2:  # 발음 기호
            lab, rate = csd_align(text, syl)
            if lab is None or rate < 0.9:
                print(f"CSD {wav.stem}: 가사 {len(text)}음절 · 음표 {len(rows)}개를 맞추지 못해서 뺌 (모음 {rate:.2f}), "
                      f"처음 {syl[:6]} {''.join(text[:12])}", flush=True)
                continue
            for r, x in zip(rows, lab):
                r[3] = x
        elif text and len(text) < len(syl):
            keep = set()
            for a, b, k in difflib.SequenceMatcher(None, syl, text, autojunk=False).get_matching_blocks():
                keep.update(range(a, a + k))
            for i, r in enumerate(rows):
                if i not in keep:
                    r[3] = "+"
        for r in rows:
            if not _hangul(r[3]):
                r[3] = "+"
        out.append((wav.stem, wav, [tuple(r) for r in rows]))
    if count and len(out) > count:  # 고르게 (앞 · 뒤 노래 고루)
        out = [out[round(i * (len(out) - 1) / (count - 1))] for i in range(count)] if count > 1 else out[:1]
    return out


def csd_lyrics(rows, gap=0.25):
    """정답 음절 → 가사 글 (노래하는 사람이 쓰듯 숨 쉬는 틈마다 한 줄)."""
    lines, cur, end = [], "", None
    for s, e, _, syl in rows:
        if end is not None and s - end > gap and cur:
            lines.append(cur)
            cur = ""
        if syl != "+":
            cur += syl
        end = e
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def score_csd(prev, rows):
    import mir_eval

    truth = [r for r in rows if r[3] != "+"]
    est = [x for x in prev["notes"] if x[3] != "+"]
    sm = difflib.SequenceMatcher(None, [r[3] for r in truth], [x[3] for x in est], autojunk=False)
    pairs = [(i + k, j + k) for i, j, n in sm.get_matching_blocks() for k in range(n)]
    err = {"v": [], "o": [], "a": []}
    tone = 0
    for i, j in pairs:
        e = (est[j][0] - truth[i][0]) * 1000
        err["v" if notes._voiced_onset(truth[i][3]) else "o"].append(e)
        if (ord(truth[i][3]) - 0xAC00) // 588 == 11:  # 첫소리 ㅇ: 정답 시작이 곧 모음 시작
            err["a"].append(e)
        tone += est[j][2] - prev["shift"] == truth[i][2]
    ref_iv = np.array([[r[0], r[1]] for r in rows]).reshape(-1, 2)
    ref_hz = pitch.midi_to_hz(np.array([r[2] for r in rows], dtype=float))
    merged = []
    for x in prev["notes"]:
        if x[1] <= x[0]:
            continue
        if merged and x[3] == "+" and abs(merged[-1][1] - x[0]) < 1e-3 and merged[-1][2] == x[2]:
            merged[-1][1] = x[1]
        else:
            merged.append(list(x))
    est_iv = np.array([[x[0], x[1]] for x in merged]).reshape(-1, 2)
    est_hz = pitch.midi_to_hz(np.array([x[2] - prev["shift"] for x in merged], dtype=float))
    conp = mir_eval.transcription.precision_recall_f1_overlap(ref_iv, ref_hz, est_iv, est_hz, offset_ratio=None)[2]
    n_a = sum((ord(r[3]) - 0xAC00) // 588 == 11 for r in truth)
    return {"n": len(truth), "placed": len(pairs), "tone": tone, "err_v": err["v"], "err_o": err["o"], "err_a": err["a"],
            "n_a": n_a, "conp": conp}


def eval_csd(songs, patches, keep=None):
    res = []
    for name, wav, rows in songs:
        with patched(patches), tempfile.TemporaryDirectory() as tmp:
            out = Path(keep) / name if keep else Path(tmp) / "o"
            prev = run_tune(wav, csd_lyrics(rows), out)
        fa_gain([r[0] for r in rows if r[3] != "+"])
        res.append(score_csd(prev, rows))
    if not res:
        return {}
    n = sum(r["n"] for r in res)
    errs = [e for r in res for e in r["err_v"] + r["err_o"]]
    ev = [e for r in res for e in r["err_v"]]
    eo = [e for r in res for e in r["err_o"]]
    a = np.abs(np.array(errs)) if errs else np.array([np.nan])
    ea = np.array([e for r in res for e in r["err_a"]])
    return {"c_n": n, "c_placed": sum(r["placed"] for r in res) / max(1, n),
            "c_hit50": float(np.sum(a <= 50) / max(1, n)), "c_hit100": float(np.sum(a <= 100) / max(1, n)),
            "c_bias_v": float(np.median(ev)) if ev else float("nan"), "c_bias_o": float(np.median(eo)) if eo else float("nan"),
            "c_abs90": float(np.percentile(a, 90)), "c_tone": sum(r["tone"] for r in res) / max(1, n),
            "c_conp": float(np.mean([r["conp"] for r in res])),
            "c_vow50": float(np.sum(np.abs(ea) <= 50) / max(1, sum(r["n_a"] for r in res))),
            "c_vow_med": float(np.median(np.abs(ea))) if len(ea) else float("nan")}


def fa_check(csd, lines=3):
    """가사 맞추기 모델만 CSD 첫 노래 몇 줄에 돌려서 음절마다 정답 시작 · 모델 음절 시작 · 모음 시작을 찍는다 (ms)."""
    from tetotune import audio

    folder = align_fa.model_dir()
    if not csd or folder is None:
        print("가사 맞추기 모델 없음", flush=True)
        return
    al = align_fa.Aligner(folder)
    print(f"가사 맞추기 모델: {folder}, 음소 {len(al.ids)}개, 사전 {len(al.dict)}개, 앞붙임 {al.prefix!r}, "
          f"{al.sr}Hz / {al.hop}", flush=True)
    langs = sorted({k.split("/")[0] for k in al.ids if "/" in k})
    print(f"  언어: {langs}, 한글용 다른 언어 음소: {[p for p in ('en/n', 'en/m', 'en/ng', 'en/l', 'zh/e') if p in al.ids]}",
          flush=True)
    name, wav, rows = csd[0]
    y, sr = audio.load(wav)
    groups, cur = [], []
    for r in rows:
        if cur and r[0] - cur[-1][1] > 0.25:
            groups.append(cur)
            cur = []
        cur.append(r)
    groups.append(cur)
    errs = []
    for g in groups[:lines]:
        syl = [r for r in g if r[3] != "+"]
        a, b = max(0.0, g[0][0] - 0.4), g[-1][1] + 0.3
        times, conf = al.align(y[int(a * sr):int(b * sr)], sr, [r[3] for r in syl])
        if times is None:
            print(f"  {name} {''.join(r[3] for r in syl)}: 맞추지 못함 (확신 {conf:.3f})", flush=True)
            continue
        cells = []
        for r, (ts, tv, _) in zip(syl, times):
            cells.append(f"{r[3]} {r[0] * 1000:.0f}/{(a + ts) * 1000:.0f}/{(a + tv) * 1000:.0f}")
            errs.append((a + ts - r[0]) * 1000)
        print(f"  {name} 확신 {conf:.3f}: " + ", ".join(cells), flush=True)
    if errs:
        print(f"  음절 시작 오차 (모델 − 정답) 중앙 {np.median(errs):+.0f}ms, 절댓값 중앙 {np.median(np.abs(errs)):.0f}ms", flush=True)


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


def compare_wav(folder, wav, truth=None, f0_truth=None):
    cmd = [sys.executable, str(ROOT / "devtools/compare_render.py"), str(folder), str(wav)]
    if truth:
        cmd += ["--truth", str(truth)]
    if f0_truth:
        cmd += ["--f0-truth", str(f0_truth)]
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


RENDER_KEYS = ("pitch_err_cents_median", "pitch_err_cents_p90", "within_50_cents", "loudness_corr", "loudness_mae_db",
               "dropouts_per_min",
               "pitch_drops_per_min", "pitch_jumps_per_min", "off_note_ok_per_min", "wobble_ratio")


def _mean(cs, k):
    v = [c[k] for c in cs if c.get(k) is not None]
    return float(np.mean(v)) if v else float("nan")


KEEP = os.environ.get("ACC_KEEP_RENDERS")  # 있으면 렌더 · 원래 보컬을 <여기>/<방식>/<모음>/<곡>.wav로 남긴다 (mos.py, 올리지 않음)


def _keep(src, *parts):
    dst = Path(KEEP, *parts).with_suffix(".wav")
    if Path(src).exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def render_set(name, items, patches, args, key=""):
    """items: [(이름, 보컬, 가사, 비교할 정답 보컬 또는 None[, 사람이 표시한 음 높이 csv])] → 곡마다 비교 결과.
    음 높이 csv가 있으면 그것을 기준으로 잰 결과도 c["by_f0"]에 (Harvest 기준은 Harvest 곡선이 유리하다)."""
    out = []
    for label, wav, text, truth, *f0_truth in items:
        with patched(patches), tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "o"
            try:
                run_tune(wav, text, folder)
                if Cfg.refine:
                    rr = refine_mod.refine(folder, lambda u: _render_only(Path(args.render[0]), Path(args.render[1]), u),
                                           rounds=Cfg.refine)
                    print(f"{name} {label} 고치기", json.dumps(rr["rounds"], ensure_ascii=False), flush=True)
                    c = compare_wav(folder, folder / "teto_preview.wav", truth)
                    if f0_truth:
                        c["by_f0"] = compare_wav(folder, folder / "teto_preview.wav", truth, f0_truth[0])
                    if KEEP:
                        _keep(folder / "teto_preview.wav", name, key, label)
                        _keep(wav, "원래 보컬", key, label)
                else:
                    c = render_compare(Path(args.render[0]), Path(args.render[1]), folder, truth)
            except (RuntimeError, subprocess.CalledProcessError) as e:
                print(f"{name} {label} 렌더 못 함: {e}", flush=True)
                continue
        c["song"] = label
        if "by_f0" in c:
            c["by_f0"].pop("events", None)
            c["by_f0"]["song"] = label
            print(f"{name} {label} (사람이 표시한 음 높이 기준)",
                  json.dumps({k: c["by_f0"].get(k) for k in RENDER_KEYS + ("event_kinds",)}, ensure_ascii=False), flush=True)
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
    ap.add_argument("--csd", help="CSD 폴더 (korean/wav · csv · lyric)")
    ap.add_argument("--csd-n", type=int, default=12, help="CSD 노래 몇 곡 (고르게 고른다)")
    ap.add_argument("--csd-sep", help="make_separated.py --csd 결과 폴더 (sep_<이름>.wav): 분리한 보컬로도 CSD를 잰다")
    ap.add_argument("--render-csd", type=int, default=0, help="CSD 몇 곡을 가사 넣고 조교해서 테토로 렌더해 비교할지")
    ap.add_argument("--only", help="쉼표로 나눈 방식 이름만")
    ap.add_argument("--json", help="결과를 이 파일에 (여러 잡 결과를 합칠 때)")
    ap.add_argument("--tag", default="", help="결과 방식 이름 뒤에 붙일 말 (다른 기계에서 돈 같은 방식과 나눠 볼 때)")
    ap.add_argument("--render-only", action="store_true", help="음표 · 가사 · 음 높이 표는 건너뛰고 렌더 비교만 (렌더 엔진 견주기)")
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
    csd = csd_songs(args.csd, args.csd_n) if args.csd else []
    csd_sep = []
    if args.csd_sep and csd:
        csd_sep = [(name, Path(args.csd_sep) / f"sep_{name}.wav", rows, wav) for name, wav, rows in csd
                   if (Path(args.csd_sep) / f"sep_{name}.wav").exists()]
    for name, wav, rows in csd[:2]:
        print(f"CSD {name}: 음표 {len(rows)}개, 음절 {sum(r[3] != '+' for r in rows)}개, 가사 줄 {len(csd_lyrics(rows).splitlines())}줄, "
              f"처음: {rows[:4]}", flush=True)
    fa_check(csd)
    vs = variants()
    if args.only:
        vs = {k: with_flags(vs, k) for k in args.only.split(",")}
    print(f"테토 정답 노래 {len(refs)}곡 · 한글 {len(krefs)}곡 · vocadito {len(songs)}곡 · 분리한 보컬 {len(sep)}곡 · "
          f"CSD {len(csd)}곡 (분리 {len(csd_sep)}곡)", flush=True)
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
            FA_LOG.clear()
            if not args.render_only:
                if refs:
                    row.update(eval_refs(refs, patches, name))
                    fa_summary(f"{name} 테토 정답")
                if krefs:
                    row.update({"k" + k: v for k, v in eval_refs(krefs, patches, name).items()})
                    fa_summary(f"{name} 한글 정답")
                if songs:
                    row.update(eval_vocadito(songs, patches))
                if sep:
                    row.update({"s" + k: v for k, v in eval_vocadito([(k, w, f0, nts) for k, w, f0, nts, _ in sep], patches).items()})
                if csd:
                    row.update(eval_csd(csd, patches))
                    fa_summary(f"{name} CSD")
                if csd_sep:
                    row.update({"s" + k: v for k, v in eval_csd([(n, w, r) for n, w, r, _ in csd_sep], patches).items()})
                    fa_summary(f"{name} 분리 CSD")
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
                "vocadito": [(str(k), wav, "", None, f0) for k, wav, f0, _ in songs if k in want],
                "sep": [(str(k), f, "", clean, f0) for k, f, f0, _, clean in sep
                        if k in ({int(x) for x in args.render_sep_songs.split(",")} if args.render_sep_songs else want)],
                "csd": [(name, wav, csd_lyrics(rows), None) for name, wav, rows in csd[:args.render_csd]],
                "csdsep": [(name, w, csd_lyrics(rows), clean) for name, w, rows, clean in csd_sep[:args.render_csd]],
            }
            for key, items in sets.items():
                if items:
                    rend[key] = render_set(name, items, patches, args, key)
                    by_f0 = [c.pop("by_f0") for c in rend[key] if "by_f0" in c]
                    if by_f0:
                        rend[key + "_f0"] = by_f0
        results[name + args.tag] = {"row": row, "render": rend}
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
    def csd_cell(r, p=""):
        if p + "c_hit50" not in r:
            return " | | | | | | "
        return (f"{r[p + 'c_placed']:.3f} | {r[p + 'c_hit50']:.3f} / {r[p + 'c_hit100']:.3f} | "
                f"{r.get(p + 'c_vow50', float('nan')):.3f} / {r.get(p + 'c_vow_med', float('nan')):.0f} | "
                f"{r[p + 'c_bias_v']:+.0f} / {r[p + 'c_bias_o']:+.0f} | "
                f"{r[p + 'c_abs90']:.0f} | {r[p + 'c_tone']:.3f} | {r[p + 'c_conp']:.3f}")
    if any("c_hit50" in r for r in rows.values()):
        print()
        print("| 방식 | CSD 음절 들어감 | 음절 시작 50/100ms 안 | 모음 첫 음절 50ms 안 / 오차 중앙 ms | "
              "치우침 ms (울리는 자음 / 그 밖) | 시작 오차 90% ms | 음절 음 높이 | COnP | "
              "분리 CSD 들어감 | 50/100ms 안 | 모음 첫 음절 | 치우침 | 90% | 음 높이 | COnP |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, r in rows.items():
            print(f"| {name} | {csd_cell(r)} | {csd_cell(r, 's')} |")
    labels = {"ref": "테토 정답", "kref": "한글 정답", "vocadito": "사람 노래", "vocadito_f0": "사람 노래 (사람이 표시한 음 높이 기준)",
              "sep": "분리한 보컬", "sep_f0": "분리한 보컬 (사람이 표시한 음 높이 기준)", "csd": "CSD (가사)",
              "csdsep": "분리 CSD (가사)"}
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

    def rank(k):  # 기계마다 (꼬리말 없는 Mac 먼저), 그 안에서는 variants 순서
        base = next((v for v in order if k == v or k.startswith(v + " (")), None)
        return (k not in order, order.index(base) if base else 99, k)
    print_tables({k: results[k] for k in sorted(results, key=rank)})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_main()
    else:
        main()

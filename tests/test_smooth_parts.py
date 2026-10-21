"""끊김 · 음 떨어짐 보완(틈 메우기 · 렌더해서 고치기)의 작은 부분 시험. 렌더 없이 돈다."""
import numpy as np
import yaml

from tetotune import notes as nt, refine, tune as T
from tetotune.pitch import HOP


def _db(n, quiet=()):
    db = np.full(n, -20.0)
    for a, b in quiet:
        db[int(a / HOP):int(b / HOP)] = -70.0
    return db


def test_legato_fills_consonant_gaps_only():
    n = int(3.0 / HOP)
    db = _db(n, quiet=[(1.2, 1.45)])  # 1.2~1.45초는 진짜 쉼 (아주 조용)
    xs = [nt.Note(0.0, 0.5, 60), nt.Note(0.58, 1.0, 62),  # 80ms 틈: 자음 → 메운다
          nt.Note(1.0, 1.2, 62), nt.Note(1.45, 1.8, 64),  # 쉼 → 그대로
          nt.Note(1.9, 2.3, 65), nt.Note(2.6, 2.9, 67)]   # 0.3초 넘는 틈 → 그대로
    breaths = [(1.82, 1.88, -30, True)]  # 1.8~1.9초 틈에서 숨을 쉼 → 그대로
    T._legato(xs, breaths, db, -20.0)
    assert xs[0].end == xs[1].start == 0.58
    assert xs[2].end == 1.2
    assert xs[3].end == 1.8
    assert xs[4].end == 2.3


def test_fill_short_bridges_only_short_holes():
    x = np.array([1.0, 1, 0, 0, 3, 3, 0, 0, 0, 0, 0, 5])
    ok = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], bool)
    out = refine._fill_short(x, ok, 2)
    assert np.allclose(out[2:4], [5 / 3, 7 / 3])  # 1 → 3 사이를 잇는다
    assert np.all(out[6:11] == 0)  # 긴 구멍은 0


def _z(n, render_offset_db=0.0):
    base = np.full(n, 6000.0)
    sing = np.zeros(n, bool)
    sing[100:n - 100] = True
    return {"base": base, "pitd": np.zeros(n), "dyn": np.zeros(n), "sing": sing, "voiced": sing.copy(),
            "rel": np.zeros(n), "near": sing.astype(float), "bpm": np.array(120.0), "span": np.array([0.0, n * HOP])}


def test_step_corrects_sustained_flat_and_quiet_spot():
    n = int(2.0 / HOP)
    z = _z(n)
    cents = np.full(n, 6000.0)
    a, b = int(0.8 / HOP), int(1.0 / HOP)
    cents[a:b] = 5900.0  # 렌더가 200ms 동안 100센트 처짐 → 올려 준다
    cents[int(1.4 / HOP):int(1.41 / HOP)] = 5800.0  # 10ms 튐은 읽기 잡음 → 그대로
    voiced = z["sing"].copy()
    db = np.full(n, -20.0)
    db[int(0.55 / HOP):int(0.75 / HOP)] = -32.0  # 렌더가 여기서 12dB 작다 → 크게
    z2 = refine.step(dict(z), cents, voiced, db)
    mid = (a + b) // 2
    assert 50 < z2["pitd"][mid] < 110
    assert abs(z2["pitd"][int(1.405 / HOP)]) < 5
    assert z2["dyn"][int(0.65 / HOP)] > 40  # dB × 10
    assert abs(z2["dyn"][int(1.2 / HOP)]) < 10


def test_write_curves_keeps_other_curves(tmp_path):
    n = int(1.0 / HOP)
    ustx = tmp_path / "project.ustx"
    proj = {"voice_parts": [{"curves": [{"abbr": "pitd", "xs": [0], "ys": [0]}, {"abbr": "brec", "xs": [0, 5], "ys": [3, 3]}]}]}
    ustx.write_text(yaml.safe_dump(proj, allow_unicode=True))
    z = _z(n)
    z["pitd"] = np.full(n, 50.0)
    refine.write_curves(ustx, z)
    out = yaml.safe_load(ustx.read_text())["voice_parts"][0]["curves"]
    by = {c["abbr"]: c for c in out}
    assert set(by) == {"pitd", "brec", "dyn"} and by["brec"]["ys"] == [3, 3]
    assert max(by["pitd"]["ys"]) == 50


def test_refine_keeps_best_round(tmp_path):
    """렌더가 고칠수록 나빠지면 첫 회차를 남긴다."""
    import soundfile as sf

    sr = 44100
    n = int(1.5 / HOP)
    z = _z(n)
    np.savez_compressed(tmp_path / "curves.npz", **z)
    proj = {"voice_parts": [{"curves": []}]}
    (tmp_path / "project.ustx").write_text(yaml.safe_dump(proj))
    calls = []

    def render(ustx):
        k = len(calls)
        calls.append(k)
        t = np.arange(int(1.5 * sr)) / sr
        f = 261.63 * 2 ** (0.3 * k / 12)  # 회차마다 30센트씩 더 틀린다
        y = 0.3 * np.sin(2 * np.pi * f * t) * (t > 0.5) * (t < 1.0)
        p = tmp_path / f"r{k}.wav"
        sf.write(p, y, sr)
        return p
    r = refine.refine(tmp_path, render, rounds=2)
    assert len(calls) == 3 and r["best"] == 0
    assert (tmp_path / "teto_preview.wav").exists() and (tmp_path / "refine.json").exists()


def test_deglitch_removes_short_dips_keeps_steps_and_long_bends():
    from tetotune import tune
    from tetotune.pitch import HOP
    n = int(2.0 / HOP)
    t = np.arange(n) * HOP
    x = np.where(t < 1.0, 6000.0, 6200.0)  # 음이 바뀌는 계단
    x[(t > 0.5) & (t < 0.55)] -= 700  # 50ms 확 떨어짐 (잘못 읽음)
    x[(t > 1.4) & (t < 1.7)] += 300  # 300ms 꺾기 (진짜)
    y = tune._deglitch(x, np.ones(n, bool), 150)
    assert np.max(np.abs(y[(t > 0.48) & (t < 0.57)] - 6000)) < 40
    assert abs(y[int(0.99 / HOP)] - 6000) < 1 and abs(y[int(1.01 / HOP)] - 6200) < 1
    assert abs(y[int(1.55 / HOP)] - 6500) < 1

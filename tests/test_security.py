"""보안 · 데이터 안정성: 폴더 밖 파일을 읽지 않는지, 작업이 끝없이 쌓이지 않는지, 받은 모델을 확인하는지,
구간 다시 조교가 곡 처음에서도 되는지 · 실패해도 결과가 어긋나지 않는지."""
import json
import os
import zipfile

import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi.testclient import TestClient

from tetotune import edit, game, retune, stemmixer, voicebank


def _wav(path, seconds=1.0, value=0.1, sr=44100):
    sf.write(str(path), np.full(int(seconds * sr), value, np.float32), sr)


# ---------------------------------------------------------------- StemMixer meta.json
def test_stemmixer_stems_stay_inside_song_folder(tmp_path, monkeypatch):
    lib = tmp_path / "sm"
    song = lib / "song1"
    song.mkdir(parents=True)
    secret = tmp_path / "secret.wav"
    _wav(secret, value=0.5)
    _wav(song / "vocals.wav")
    _wav(song / "drums.wav", value=0.01)
    (song / "linked.wav").symlink_to(secret)
    stems = ["vocals", "drums", "../../secret", str(tmp_path / "secret"), "linked", ".hidden", 3, None]
    (song / "meta.json").write_text(json.dumps({"title": "곡", "stems": stems, "replaced": "이상한 값"}))
    monkeypatch.setenv("STEMMIXER_LIBRARY", str(lib))
    vocal, _, backing, _ = stemmixer.prepare("song1", tmp_path / "work")
    assert vocal == song / "vocals.wav"
    y, _ = sf.read(str(backing))
    assert np.allclose(y, 0.01, atol=1e-4)  # 드럼만 (곡 폴더 밖 secret은 섞이지 않는다)
    assert stemmixer._track(song, "../../secret") is None
    assert stemmixer._track(song, "linked") is None


def test_stemmixer_bad_meta_is_skipped(tmp_path, monkeypatch):
    lib = tmp_path / "sm"
    (lib / "a").mkdir(parents=True)
    (lib / "a" / "meta.json").write_text("[1, 2]")
    _wav(lib / "a" / "vocals.wav")
    monkeypatch.setenv("STEMMIXER_LIBRARY", str(lib))
    assert stemmixer.songs() == []


# ---------------------------------------------------------------- oto.ini
def test_oto_samples_stay_inside_voicebank(tmp_path):
    vb = tmp_path / "vb"
    (vb / "sub").mkdir(parents=True)
    outside = tmp_path / "outside.wav"
    _wav(outside)
    _wav(vb / "sub" / "a.wav")
    (vb / "sub" / "oto.ini").write_text("\n".join([
        "a.wav=a,0,0,0,0,0",
        "../../outside.wav=up,0,0,0,0,0",
        f"{outside}=abs,0,0,0,0,0",
    ]), encoding="utf-8")
    otos = voicebank._otos(vb)
    assert otos["a"][0] == (vb / "sub" / "a.wav").resolve()
    assert otos["up"][0] is None and otos["abs"][0] is None
    assert voicebank._level(otos["up"]) is None


# ---------------------------------------------------------------- 작업 대기열
@pytest.fixture()
def server_mod(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "OUTPUT", tmp_path / "out")
    monkeypatch.setattr(server, "jobs", {})
    return server


def test_too_many_jobs_are_refused(server_mod):
    for _ in range(server_mod.MAX_ACTIVE):
        server_mod._new_job()
    with pytest.raises(Exception) as e:
        server_mod._new_job()
    assert getattr(e.value, "status_code", None) == 429


def test_finished_jobs_are_pruned(server_mod, monkeypatch):
    old = server_mod._new_job()
    server_mod._finish(server_mod.jobs[old], status="done")
    server_mod.jobs[old]["_ended"] -= server_mod.JOB_TTL + 1
    recent = server_mod._new_job()
    server_mod._finish(server_mod.jobs[recent], status="done")
    for _ in range(server_mod.MAX_JOBS + 5):
        server_mod._finish(server_mod.jobs[server_mod._new_job()], status="error")
    server_mod._new_job()
    done = [k for k, j in server_mod.jobs.items() if j["status"] != "running"]
    assert old not in server_mod.jobs and len(done) <= server_mod.MAX_JOBS
    client = TestClient(server_mod.app, base_url="http://127.0.0.1:8766")
    k = done[-1]
    assert "_ended" not in client.get(f"/api/jobs/{k}").json()


def test_failed_tune_leaves_no_hidden_folder(server_mod, tmp_path, monkeypatch):
    def broken_tune(vocal, out, *a, **kw):  # project.ustx까지 쓰고 preview.json · report.json 전에 실패
        (out / "project.ustx").write_text("x")
        raise RuntimeError("멈춤")
    monkeypatch.setattr(server_mod, "tune", broken_tune)
    monkeypatch.setattr(server_mod.game, "ensure_model", lambda *a, **k: None)
    vocal = tmp_path / "v.wav"
    _wav(vocal)
    job = server_mod._new_job()
    server_mod._run(job, str(vocal), None, "", str(vocal), None, server_mod.Options(refine="off"))
    assert server_mod.jobs[job]["status"] == "error"
    assert [p for p in (tmp_path / "out").iterdir() if p.name != ".work"] == []


def test_oversized_input_is_refused(server_mod, tmp_path, monkeypatch):
    monkeypatch.setitem(server_mod.MAX_INPUT_BYTES, "음표 파일", 10)
    big = tmp_path / "n.mid"
    big.write_bytes(b"x" * 100)
    with pytest.raises(Exception) as e:
        server_mod._input_file(str(big), "음표 파일")
    assert getattr(e.value, "status_code", None) == 400


# ---------------------------------------------------------------- GAME 모델
def _zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, b"x")
    return zipfile.ZipFile(path)


def test_game_zip_checks(tmp_path):
    game._check_zip(_zip(tmp_path / "ok.zip", ["game/config.json", "game/encoder.onnx"]))
    for bad in (["../evil.onnx"], ["/abs.onnx"], ["game/run.sh"], ["a\\..\\b.onnx"]):
        with pytest.raises(ValueError):
            game._check_zip(_zip(tmp_path / "bad.zip", bad))
    with zipfile.ZipFile(tmp_path / "link.zip", "w") as z:
        info = zipfile.ZipInfo("game/link.onnx")
        info.external_attr = (0o120777 << 16)
        z.writestr(info, "/etc/passwd")
    with pytest.raises(ValueError):
        game._check_zip(zipfile.ZipFile(tmp_path / "link.zip"))


def test_game_download_checks_hash_and_size(tmp_path, monkeypatch):
    import hashlib
    src = tmp_path / "m.oudep"
    src.write_bytes(b"model" * 100)
    url = src.as_uri()
    good = hashlib.sha256(src.read_bytes()).hexdigest()
    game._download(url, tmp_path / "a", good)
    assert (tmp_path / "a").read_bytes() == src.read_bytes()
    with pytest.raises(ValueError):
        game._download(url, tmp_path / "b", "0" * 64)
    assert not (tmp_path / "b").exists()
    monkeypatch.setattr(game, "MAX_DOWNLOAD", 10)
    with pytest.raises(ValueError):
        game._download(url, tmp_path / "c", good)
    assert not (tmp_path / "c").exists()


def test_game_model_is_pinned():
    assert game.MODEL_URL.startswith("https://") and len(game.MODEL_SHA256) == 64


# ---------------------------------------------------------------- 구간 다시 조교 · 되돌리기
def test_splice_wav_at_song_start(tmp_path):
    sr = 44100
    old, new = tmp_path / "old.wav", tmp_path / "new.wav"
    _wav(old, 2.0, 0.1)
    _wav(new, 1.0, 0.5)
    retune._splice_wav(old, new, -0.05, 0.6, 0.0)  # 첫 음표가 50ms 안에서 시작하는 곡
    y, _ = sf.read(str(old))
    assert len(y) == 2 * sr
    assert abs(y[int(0.3 * sr)] - 0.5) < 1e-3 and abs(y[int(1.5 * sr)] - 0.1) < 1e-3
    # 끝이 wav보다 뒤 · 새 렌더가 늦게 시작 (offset > a)
    retune._splice_wav(old, new, 1.5, 2.5, 1.7)
    y2, _ = sf.read(str(old))
    assert len(y2) == 2 * sr and abs(y2[int(1.6 * sr)]) < 0.11


def _project(folder):
    notes = [{"position": 480, "duration": 480, "tone": 60, "lyric": "あ"}]
    proj = {"name": "t", "tempos": [{"position": 0, "bpm": 120.0}], "voice_parts": [{"notes": notes, "curves": []}]}
    (folder / "project.ustx").write_text(yaml.safe_dump(proj, allow_unicode=True))
    (folder / "preview.json").write_text(json.dumps({"notes": [], "flags": []}))


def test_undo_restores_preview_wav(tmp_path):
    _project(tmp_path)
    _wav(tmp_path / "teto_preview.wav", value=0.1)
    edit.snapshot(tmp_path, wav=True)
    _wav(tmp_path / "teto_preview.wav", value=0.4)
    (tmp_path / "project.ustx").write_text((tmp_path / "project.ustx").read_text().replace("あ", "い"))
    prev = edit.undo(tmp_path)
    y, _ = sf.read(str(tmp_path / "teto_preview.wav"))
    assert abs(y[100] - 0.1) < 1e-3 and prev["notes"][0][3] == "あ" and not prev.get("teto_stale")


def test_history_keeps_few_wavs(tmp_path):
    _project(tmp_path)
    _wav(tmp_path / "teto_preview.wav", 0.1)
    for _ in range(edit.KEEP_WAV + 3):
        edit.snapshot(tmp_path, wav=True)
    assert len(list((tmp_path / ".history").glob("*.wav"))) == edit.KEEP_WAV


def test_atomic_write_keeps_old_file_on_error(tmp_path, monkeypatch):
    p = tmp_path / "preview.json"
    p.write_text("old")

    def boom(*a):
        raise OSError("디스크가 꽉 참")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        edit.write_text_atomic(p, "new")
    assert p.read_text() == "old" and [f.name for f in tmp_path.iterdir()] == ["preview.json"]


def test_render_error_is_reported(tmp_path):
    from tetotune import refine

    def broken(ustx):
        raise RuntimeError("OpenUtau로 불러 보지 못했어요: 가수를 찾지 못함")
    assert refine._render_checked(broken, tmp_path / "p.ustx") == (None, None)
    assert "가수를 찾지 못함" in refine.last_render_error


def test_silent_render_reason():
    from tetotune import openutau
    assert "찾지 못했어요" in openutau._why_silent("singer=teto found=False\nphonemes=10 phoneme_errors=0\nrendered")
    assert "바꾸지 못했어요" in openutau._why_silent("singer=teto found=True\nphonemizer=X renderer=CLASSIC\nphonemes=10 phoneme_errors=9")
    assert "렌더러 CLASSIC" in openutau._why_silent("singer=teto found=True\nphonemizer=X renderer=CLASSIC\nphonemes=10 phoneme_errors=0")

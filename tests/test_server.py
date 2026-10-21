"""엔진 서버: 이 컴퓨터에서 온 요청만 받는지, 잘못된 입력을 알기 쉽게 거절하는지."""
import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("STEMMIXER_LIBRARY", str(tmp_path / "sm"))
    (tmp_path / "sm" / "song1").mkdir(parents=True)
    (tmp_path / "sm" / "song1" / "meta.json").write_text(json.dumps({"title": "곡", "stems": ["vocals"]}))
    import server
    monkeypatch.setattr(server, "OUTPUT", tmp_path / "out")
    return TestClient(server.app, base_url="http://127.0.0.1:8766")


def test_rejects_other_websites(client):
    r = client.get("/api/options", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.get("/api/options", headers={"Host": "evil.example"})
    assert r.status_code == 403
    assert client.get("/api/options").status_code == 200


def test_rejects_bad_inputs(client, tmp_path):
    assert client.post("/api/tune", data={}).status_code == 400
    assert client.post("/api/tune", data={"vocal": str(tmp_path)}).status_code == 400  # 폴더는 안 됨
    assert client.post("/api/tune", data={"song": "../sm/song1"}).status_code == 400
    assert client.post("/api/tune", data={"song": ".."}).status_code == 400
    vocal = tmp_path / "v.wav"
    vocal.write_bytes(b"RIFF")
    r = client.post("/api/tune", data={"vocal": str(vocal), "options": json.dumps({"breaths": "lots"})})
    assert r.status_code == 400 and "설정" in r.json()["detail"]
    r = client.post("/api/tune", data={"vocal": str(vocal), "options": json.dumps({"transpose": "high"})})
    assert r.status_code == 400


def test_options_are_clamped():
    import server
    o = server._options({"pitch_follow": 5, "vibrato": -1, "transpose": "7.4", "unknown": 1})
    assert o.pitch_follow == 1.0 and o.vibrato == 0.0 and o.transpose == "7"


def test_result_paths_stay_inside_output(client):
    assert client.get("/api/results/..%2F..%2Fetc/preview").status_code == 404
    assert client.delete("/api/results/..").status_code in (404, 405)


def test_token_required_when_set(client, monkeypatch):
    import server
    monkeypatch.setattr(server, "TOKEN", "s3cret")
    assert client.get("/api/options").status_code == 403
    assert client.get("/api/options", headers={"X-TetoTune-Token": "wrong"}).status_code == 403
    assert client.get("/api/options", headers={"X-TetoTune-Token": "s3cret"}).status_code == 200
    assert client.get("/api/options", headers={"X-TetoTune-Token": b"\xe9\xe9"}).status_code == 403


def test_unfinished_folders_are_swept(client, tmp_path):
    import server
    out = tmp_path / "out"
    dead, live, mine = out / "a 2026", out / ".work" / "job1", out / "b 2026"
    for d, pid in ((dead, 999999), (live, os.getppid())):
        d.mkdir(parents=True)
        (d / server.INCOMPLETE).write_text(str(pid))
    mine.mkdir()  # 표시 없는 폴더 (사용자 것)
    server.sweep_unfinished()
    assert not dead.exists() and live.exists() and mine.exists()


def test_folder_names(client, monkeypatch):
    import server
    a = server._folder_name("../.hidden:name")
    assert "/" not in a and not a.startswith(".") and ":" not in a
    (server.OUTPUT / a).mkdir(parents=True)
    assert server._folder_name("../.hidden:name") != a


def test_edit_and_undo_endpoints(client, tmp_path):
    import yaml
    import server
    folder = server.OUTPUT / "곡 2026"
    folder.mkdir(parents=True)
    notes = [{"position": 480, "duration": 240, "tone": 60, "lyric": "あ"},
             {"position": 720, "duration": 240, "tone": 62, "lyric": "+"}]
    (folder / "project.ustx").write_text(yaml.safe_dump({"tempos": [{"position": 0, "bpm": 120.0}],
                                                          "voice_parts": [{"notes": notes, "curves": []}]}))
    (folder / "report.json").write_text(json.dumps({"notes": 2, "breath_aliases": []}))
    (folder / "preview.json").write_text(json.dumps({"notes": []}))
    r = client.post(f"/api/results/{folder.name}/edit", data={"op": "lyric", "index": 1, "text": "い"})
    assert r.status_code == 200 and [n[3] for n in r.json()["notes"]] == ["あ", "い"] and r.json()["history"] == 1
    r = client.post(f"/api/results/{folder.name}/edit", data={"op": "move_left", "index": 0})
    assert r.status_code == 400 and "음절" in r.json()["detail"]
    r = client.post(f"/api/results/{folder.name}/undo")
    assert r.status_code == 200 and [n[3] for n in r.json()["notes"]] == ["あ", "+"]
    assert client.post(f"/api/results/{folder.name}/undo").status_code == 400
    listed = client.get("/api/results").json()[0]
    assert listed["teto"] is None and listed["backing"] is None
    r = client.post(f"/api/results/{folder.name}/retune", data={"start": 2, "end": 1})
    assert r.status_code == 400


def test_render_endpoint_refreshes_teto_preview(client, monkeypatch):
    import time as _time
    import yaml
    import server
    folder = server.OUTPUT / "곡 렌더"
    folder.mkdir(parents=True)
    notes = [{"position": 480, "duration": 480, "tone": 60, "lyric": "あ"}]
    (folder / "project.ustx").write_text(yaml.safe_dump({"tempos": [{"position": 0, "bpm": 120.0}],
                                                          "voice_parts": [{"notes": notes, "curves": []}]}))
    (folder / "report.json").write_text(json.dumps({"notes": 1, "breath_aliases": [], "options": {"vibrato": 0.5}}))
    (folder / "preview.json").write_text(json.dumps({"notes": []}))
    (folder / "teto_preview.wav").write_bytes(b"old")
    r = client.post(f"/api/results/{folder.name}/edit", data={"op": "lyric", "index": 0, "text": "い"})
    assert r.json()["teto_stale"] is True  # 고친 뒤에는 미리 듣기가 예전 것
    assert client.get(f"/api/results/{folder.name}/preview").json()["options"] == {"vibrato": 0.5}

    def fake_render(ustx, out_dir, name=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "render_x.wav").write_bytes(b"new")
        return out_dir / "render_x.wav"
    monkeypatch.setattr(server.openutau, "available", lambda: True)
    monkeypatch.setattr(server.openutau, "render", fake_render)
    job = client.post(f"/api/results/{folder.name}/render").json()["job_id"]
    for _ in range(100):
        s = client.get(f"/api/jobs/{job}").json()
        if s["status"] != "running":
            break
        _time.sleep(0.02)
    assert s["status"] == "done" and "teto_stale" not in s["result"]
    assert (folder / "teto_preview.wav").read_bytes() == b"new" and not (folder / ".render").exists()

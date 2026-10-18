"""엔진 서버: 이 컴퓨터에서 온 요청만 받는지, 잘못된 입력을 알기 쉽게 거절하는지."""
import json

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


def test_folder_names(client, monkeypatch):
    import server
    a = server._folder_name("../.hidden:name")
    assert "/" not in a and not a.startswith(".") and ":" not in a
    (server.OUTPUT / a).mkdir(parents=True)
    assert server._folder_name("../.hidden:name") != a

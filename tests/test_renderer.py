"""신경망 렌더(hifisampler): 모델 받기(oudeps) · 쓸 수 없으면 예전 렌더러로 · 렌더가 안 되면 되돌아가기."""
import hashlib
import zipfile

import pytest
import yaml

from tetotune import openutau, oudeps, refine, tune


def _oudep(path, pid, files):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("oudep.yaml", f'id: "{pid}"\nname: "{pid}"\nversion: "1"\nclass: X\n')
        for f in files:
            z.writestr(f, "x")
    return path


@pytest.fixture
def deps(tmp_path, monkeypatch):
    monkeypatch.setenv("TETOTUNE_OU_DEPS", str(tmp_path / "Dependencies"))
    return tmp_path / "Dependencies"


def _fake_packages(tmp_path, monkeypatch):
    """PACKAGES를 이 폴더의 작은 oudep 파일로 (주소 · SHA-256 · 있어야 하는 파일)."""
    pkgs = {}
    for pid, need in oudeps.PACKAGES.items():
        src = _oudep(tmp_path / f"{pid}.oudep", pid, list(need[2]) + ["LICENSE"])
        pkgs[pid] = (src.as_uri(), hashlib.sha256(src.read_bytes()).hexdigest(), need[2])
    monkeypatch.setattr(oudeps, "PACKAGES", pkgs)


def test_packages_are_pinned():
    for url, sha, need in oudeps.PACKAGES.values():
        assert url.startswith("https://github.com/") and len(sha) == 64 and need


def test_ensure_installs_like_openutau(tmp_path, monkeypatch, deps):
    _fake_packages(tmp_path, monkeypatch)
    assert sorted(oudeps.missing()) == sorted(oudeps.PACKAGES)
    got = oudeps.ensure()
    assert sorted(got) == sorted(oudeps.PACKAGES) and oudeps.missing() == []
    for pid, (_, _, need) in oudeps.PACKAGES.items():  # OpenUtau처럼 Dependencies/<id>/ 바로 아래에
        assert (deps / pid / "oudep.yaml").is_file() and all((deps / pid / f).is_file() for f in need)
    assert [p.name for p in deps.iterdir() if p.name.startswith(".")] == []  # 받던 임시 폴더는 지운다
    assert oudeps.ensure() == []  # 있으면 다시 받지 않는다


def test_ensure_refuses_wrong_hash_and_keeps_nothing(tmp_path, monkeypatch, deps):
    _fake_packages(tmp_path, monkeypatch)
    pid = next(iter(oudeps.PACKAGES))
    url, _, need = oudeps.PACKAGES[pid]
    monkeypatch.setitem(oudeps.PACKAGES, pid, (url, "0" * 64, need))
    with pytest.raises(ValueError):
        oudeps.ensure()
    assert not (deps / pid).exists()


def test_zip_checks(tmp_path):
    pid = "hnsep_240512"
    oudeps._check_zip(zipfile.ZipFile(_oudep(tmp_path / "ok.zip", pid, ["model.onnx", "hnsep.yaml", "LICENSE"])), pid)
    for bad in (["../evil.onnx"], ["/abs.onnx"], ["run.sh"], ["a\\..\\b.onnx"], ["a/b/c.onnx"]):
        with pytest.raises(ValueError):
            oudeps._check_zip(zipfile.ZipFile(_oudep(tmp_path / "bad.zip", pid, bad)), pid)
    with pytest.raises(ValueError):  # 다른 패키지
        oudeps._check_zip(zipfile.ZipFile(_oudep(tmp_path / "other.zip", "game", ["model.onnx"])), pid)
    with zipfile.ZipFile(tmp_path / "noyaml.zip", "w") as z:
        z.writestr("model.onnx", "x")
    with pytest.raises(ValueError):
        oudeps._check_zip(zipfile.ZipFile(tmp_path / "noyaml.zip"), pid)


def test_renderer_settings_fall_back(tmp_path, monkeypatch, deps):
    assert (tune.RENDERER, tune.RESAMPLER) == ("CLASSIC", "hifisampler")
    monkeypatch.setattr(openutau, "has_hifisampler", lambda: True)
    assert tune.renderer_settings() == {"renderer": tune.FALLBACK_RENDERER}  # 모델이 없다
    _fake_packages(tmp_path, monkeypatch)
    oudeps.ensure()
    assert tune.renderer_settings() == {"renderer": "CLASSIC", "resampler": "hifisampler"}
    monkeypatch.setattr(openutau, "has_hifisampler", lambda: False)  # 예전 OpenUtau
    assert tune.renderer_settings() == {"renderer": tune.FALLBACK_RENDERER}
    monkeypatch.setattr(openutau, "has_hifisampler", lambda: True)
    assert tune.renderer_settings("worldline") == {"renderer": tune.FALLBACK_RENDERER}  # 앱에서 예전 렌더러를 골랐다
    monkeypatch.setattr(tune, "RENDERER", "WORLDLINE-R1.1")
    monkeypatch.setattr(tune, "RESAMPLER", None)
    assert tune.renderer_settings() == {"renderer": "WORLDLINE-R1.1"}


def test_has_hifisampler_reads_core_dll(tmp_path, monkeypatch):
    app = tmp_path / "OpenUtau.app" / "Contents" / "MacOS"
    app.mkdir(parents=True)
    for f in ("OpenUtau", "OpenUtau.deps.json"):
        (app / f).write_text("x")
    (app / "OpenUtau.Core.dll").write_bytes(b"\0MZ old core\0")
    monkeypatch.setenv("TETOTUNE_OPENUTAU", str(tmp_path / "OpenUtau.app"))
    assert openutau.app_dir() == app and not openutau.has_hifisampler()
    (app / "OpenUtau.Core.dll").write_bytes(b"\0MZ\0HifisamplerResampler\0")
    assert openutau.has_hifisampler()


def test_prepare_renderer_messages(tmp_path, monkeypatch, deps):
    monkeypatch.setattr(openutau, "app_dir", lambda: None)
    assert tune.prepare_renderer() is None  # OpenUtau가 없으면 받지 않는다
    monkeypatch.setattr(openutau, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(openutau, "has_hifisampler", lambda: False)
    assert "0.1.576" in tune.prepare_renderer()
    monkeypatch.setattr(openutau, "has_hifisampler", lambda: True)
    monkeypatch.setattr(oudeps, "ensure", lambda progress: (_ for _ in ()).throw(OSError("인터넷 없음")))
    assert "받지 못해서" in tune.prepare_renderer()
    monkeypatch.setattr(oudeps, "ensure", lambda progress: [])
    assert tune.prepare_renderer() is None
    monkeypatch.setattr(oudeps, "ensure", lambda progress: (_ for _ in ()).throw(AssertionError("받으면 안 돼요")))
    assert tune.prepare_renderer(choice="worldline") is None  # WORLDLINE-R을 골랐으면 모델을 받지 않는다


def test_region_retune_sings_with_the_song_renderer(tmp_path):
    from tetotune import edit, retune

    song, clip = tmp_path / "song", tmp_path / "clip"
    for d in (song, clip):
        d.mkdir()
    proj = {"name": "t", "tempos": [{"position": 0, "bpm": 120}], "voice_parts": [{"notes": []}],
            "tracks": [{"singer": "teto", "renderer_settings": {"renderer": "WORLDLINE-R"}}]}
    (song / "project.ustx").write_text(yaml.safe_dump(proj, allow_unicode=True), encoding="utf-8")
    _result(clip, {"renderer": "CLASSIC", "resampler": "hifisampler"})
    retune.same_renderer(edit.Project(song), clip / "project.ustx")  # 예전 판에서 만든 곡: 구간도 WORLDLINE-R로
    assert _track(clip) == {"renderer": "WORLDLINE-R"}
    proj["tracks"][0].pop("renderer_settings")  # DiffSinger 같은 가수: 건드리지 않는다
    (song / "project.ustx").write_text(yaml.safe_dump(proj, allow_unicode=True), encoding="utf-8")
    _result(clip, {"renderer": "CLASSIC", "resampler": "hifisampler"})
    retune.same_renderer(edit.Project(song), clip / "project.ustx")
    assert _track(clip) == {"renderer": "CLASSIC", "resampler": "hifisampler"}


def _result(folder, renderer):
    proj = {"name": "t", "tracks": [{"singer": "teto", "renderer_settings": renderer}], "voice_parts": []}
    (folder / "project.ustx").write_text(yaml.safe_dump(proj, allow_unicode=True), encoding="utf-8")
    (folder / "report.json").write_text('{"singer": "teto"}', encoding="utf-8")


def _track(folder):
    return yaml.safe_load((folder / "project.ustx").read_text(encoding="utf-8"))["tracks"][0]["renderer_settings"]


def test_refine_falls_back_when_hifisampler_cannot_render(tmp_path, monkeypatch):
    import json

    _result(tmp_path, {"renderer": "CLASSIC", "resampler": "hifisampler"})
    monkeypatch.setattr(openutau, "available", lambda: True)
    seen = []

    def fake_refine(folder, render, progress=None):
        rs = _track(folder)
        seen.append(rs)
        if rs.get("resampler") == "hifisampler":
            raise RuntimeError("Package \"pc_nsf_hifigan\" not found")
        return {"rounds": [], "best": 0, "preview": "teto_preview.wav"}
    monkeypatch.setattr(refine, "refine", fake_refine)
    w = refine.for_result(tmp_path)
    assert "hifisampler" in w and tune.FALLBACK_RENDERER in w
    assert _track(tmp_path) == {"renderer": tune.FALLBACK_RENDERER} and len(seen) == 2
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["teto_preview"] == "teto_preview.wav"


def test_refine_keeps_hifisampler_when_fallback_fails_too(tmp_path, monkeypatch):
    _result(tmp_path, {"renderer": "CLASSIC", "resampler": "hifisampler"})
    before = (tmp_path / "project.ustx").read_text(encoding="utf-8")
    monkeypatch.setattr(openutau, "available", lambda: True)

    def fake_refine(folder, render, progress=None):
        raise RuntimeError("OpenUtau가 가수를 찾지 못했어요")
    monkeypatch.setattr(refine, "refine", fake_refine)
    w = refine.for_result(tmp_path)
    assert "가수를 찾지 못했어요" in w and "다듬지 못했어요" in w
    assert (tmp_path / "project.ustx").read_text(encoding="utf-8") == before  # 렌더러를 몰래 바꾸지 않는다


def test_refine_error_without_hifisampler_is_unchanged(tmp_path, monkeypatch):
    _result(tmp_path, {"renderer": "WORLDLINE-R"})
    monkeypatch.setattr(openutau, "available", lambda: True)
    calls = []

    def fake_refine(folder, render, progress=None):
        calls.append(1)
        raise RuntimeError("렌더가 비었어요")
    monkeypatch.setattr(refine, "refine", fake_refine)
    assert "다듬지 못했어요" in refine.for_result(tmp_path) and len(calls) == 1

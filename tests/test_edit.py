"""조교 결과 고치기 (음절 옮기기 · 늘여 부르기 · 나누기 · 합치기 · 되돌리기)."""
import json

import numpy as np
import pytest
import yaml

from tetotune import edit


def _note(pos, dur, tone, lyric):
    return {"position": pos, "duration": dur, "tone": tone, "lyric": lyric,
            "pitch": {"data": [{"x": 0, "y": 0, "shape": "l"}, {"x": 5, "y": 0, "shape": "l"}], "snap_first": True},
            "vibrato": {"length": 0}, "phoneme_expressions": [], "phoneme_overrides": []}


@pytest.fixture
def folder(tmp_path):
    # 120 BPM: 480틱 = 0.5초. 소절 1: あ い + う, 숨, 소절 2: え お
    notes = [_note(480, 240, 60, "あ"), _note(720, 240, 62, "い"), _note(960, 240, 62, "+"), _note(1200, 480, 64, "う"),
             _note(1800, 120, 64, "息"),
             _note(1920, 240, 65, "え"), _note(2160, 240, 67, "お")]
    proj = {"name": "t", "tempos": [{"position": 0, "bpm": 120.0}],
            "voice_parts": [{"notes": notes, "curves": [{"abbr": "pitd", "xs": [0, 3000], "ys": [0, 0]}]}]}
    (tmp_path / "project.ustx").write_text(yaml.safe_dump(proj, allow_unicode=True, sort_keys=False))
    (tmp_path / "report.json").write_text(json.dumps({"breath_aliases": ["息"]}))
    (tmp_path / "preview.json").write_text(json.dumps({"notes": [], "flags": [[0.95, 1.05, 1, "lyric", "x"]]}))
    return tmp_path


def lyrics(prev):
    return [n[3] for n in prev["notes"]]


def test_phrases_skip_breath_notes(folder):
    p = edit.Project(folder)
    assert len(p.sung()) == 6
    assert p.phrase(1) == [0, 1, 2, 3] and p.phrase(4) == [4, 5]


def test_move_left_and_right(folder):
    prev = edit.apply(folder, "move_right", 1)  # い를 한 음표 뒤로: あ +  い う
    assert lyrics(prev) == ["あ", "+", "い", "う", "え", "お"]
    prev = edit.apply(folder, "move_left", 2)
    assert lyrics(prev) == ["あ", "い", "+", "う", "え", "お"]
    with pytest.raises(edit.EditError):
        edit.apply(folder, "move_left", 1)  # 앞 음표가 + 가 아니다


def test_extend_splits_last_note_when_full(folder):
    prev = edit.apply(folder, "extend", 5)  # お 자리에서 え를 늘여 부르고 お는 뒤로 → 마지막 음표를 나눈다
    assert lyrics(prev) == ["あ", "い", "+", "う", "え", "+", "お"]
    a, b = prev["notes"][5], prev["notes"][6]
    assert a[1] == b[0] and a[2] == b[2] == 67


def test_unextend_pulls_syllables(folder):
    prev = edit.apply(folder, "unextend", 2)
    assert lyrics(prev) == ["あ", "い", "う", "+", "え", "お"]
    assert prev["flags"] == []  # 고친 음표에 걸친 가사 표시는 지운다


def test_merge_keeps_sung_pitch(folder):
    from tetotune import oupitch
    p = edit.Project(folder)
    grid = np.arange(480, 1200)  # OpenUtau가 어느 틱에서 재도 (5틱 격자가 어디서 시작하든)
    before = oupitch.sung(p.notes, p._curve("pitd"), p.tps, grid)
    edit.apply(folder, "merge", 0)  # あ(60) + い(62): 길이가 같으면 앞 음표 높이, い는 뒤로 민다
    p = edit.Project(folder)
    after = oupitch.sung(p.notes, p._curve("pitd"), p.tps, grid)
    assert np.max(np.abs(after - before)) <= 2.5
    assert [p.notes[i]["lyric"] for i in p.sung()][:4] == ["あ", "い", "+", "う"]  # 음절은 한 음표씩 밀린다


def test_lyric_and_undo(folder):
    edit.apply(folder, "lyric", 0, text="か")
    prev = edit.apply(folder, "lyric", 1, text="")
    assert lyrics(prev)[:2] == ["か", "+"] and prev["history"] == 2
    prev = edit.undo(folder)
    assert lyrics(prev)[:2] == ["か", "い"]
    prev = edit.undo(folder)
    assert lyrics(prev)[:2] == ["あ", "い"]
    with pytest.raises(edit.EditError):
        edit.undo(folder)

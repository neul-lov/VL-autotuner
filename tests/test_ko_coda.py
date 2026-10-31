"""한글(KO to JA) 막히는 받침을 음표 끝 가까이로 옮기기 (tune.ko_coda_overrides)."""
import json

import yaml

from tetotune import edit, tune, voicebank


def _n(pos, dur, lyric):
    return {"position": pos, "duration": dur, "tone": 60, "lyric": lyric, "phoneme_overrides": []}


def test_stop_codas_move_and_index_follows_ko_to_ja():
    # 120 BPM: 90ms = 86틱. 각(앞 음절 받침 없음 → [VC, CV, 받침]) · 입(ㅇ 시작 → [CV, 받침]) · 밖(앞이 ㅂ 받침) · 산(ㄴ 받침은 그대로)
    ns = [_n(0, 480, "나"), _n(480, 480, "각"), _n(960, 480, "입"), _n(1440, 240, "밖"), _n(1680, 240, "+"),
          _n(1920, 480, "산"), _n(2880, 480, "떡")]
    assert tune.ko_coda_overrides(ns, 120.0) == 4
    by = {x["lyric"]: x["phoneme_overrides"] for x in ns}
    close = round(0.09 * 960)
    assert by["각"] == [{"index": 2, "offset": 180 - close}]
    assert by["입"] == [{"index": 1, "offset": 180 - close}]
    assert by["밖"] == [{"index": 1, "offset": 180 - close}]  # 늘인 음표(+)까지 한 음절로
    assert by["산"] == [] and by["나"] == []
    assert by["떡"] == [{"index": 1, "offset": 120 - close}]  # 뒤에 붙은 음표 없음 → KO to JA는 120틱


def test_short_notes_keep_half_and_zero_turns_off(monkeypatch):
    ns = [_n(0, 160, "각")]
    assert tune.ko_coda_overrides(ns, 120.0) == 0  # 이미 음표 절반 자리 (KO to JA가 len/2보다 앞에 두지 않는다)
    monkeypatch.setattr(tune, "KO_CODA_CLOSE_MS", 0)
    ns = [_n(0, 960, "각")]
    ns[0]["phoneme_overrides"] = [{"index": 1, "offset": 50}, {"index": 0, "phoneme": "x"}]
    assert tune.ko_coda_overrides(ns, 120.0) == 0
    assert ns[0]["phoneme_overrides"] == [{"index": 0, "phoneme": "x"}]  # 다른 고침은 그대로


def test_edit_rewrites_overrides_after_change(tmp_path):
    ns = [_n(480, 480, "나"), _n(960, 960, "각"), _n(1920, 480, "다")]
    for x in ns:
        x.update({"pitch": {"data": [{"x": 0, "y": 0, "shape": "l"}, {"x": 5, "y": 0, "shape": "l"}], "snap_first": True},
                  "vibrato": {"length": 0}, "phoneme_expressions": []})
    tune.ko_coda_overrides(ns, 120.0)
    proj = {"name": "t", "tempos": [{"position": 0, "bpm": 120.0}],
            "tracks": [{"singer": "", "phonemizer": voicebank.KO_TO_JA}],
            "voice_parts": [{"notes": ns, "curves": [{"abbr": "pitd", "xs": [0, 3000], "ys": [0, 0]}]}]}
    (tmp_path / "project.ustx").write_text(yaml.safe_dump(proj, allow_unicode=True, sort_keys=False))
    (tmp_path / "report.json").write_text(json.dumps({"breath_aliases": []}))
    (tmp_path / "preview.json").write_text(json.dumps({"notes": [], "flags": []}))
    edit.apply(tmp_path, "split", 1)  # 각을 둘로: 각 + (+) → 같은 음절 길이, 받침은 그대로 옮겨진다
    p = edit.Project(tmp_path)
    over = [x["phoneme_overrides"] for x in p.notes if x["lyric"] == "각"][0]
    assert over == [{"index": 2, "offset": 180 - round(0.09 * 960)}]
    assert all(not x["phoneme_overrides"] for x in p.notes if x["lyric"] == "+")

"""소리를 내지 않는 시험: 합성한 '노래'를 조교해서 음표 · 가사 · 프로젝트 파일이 맞는지 본다."""
import json

import numpy as np
import soundfile as sf
import yaml

from tetotune import lyrics
from tetotune.tune import Options, tune

SR = 44100


def _sing(path, tones, dur=0.5, gap_after=None):
    """배음이 있는 소리로 음표를 차례로 부른 wav (gap_after 번째 음 뒤에 0.4초 쉼)."""
    out = []
    for i, m in enumerate(tones):
        t = np.arange(int(SR * dur)) / SR
        f = 440 * 2 ** ((m - 69) / 12) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t))
        ph = 2 * np.pi * np.cumsum(f) / SR
        y = sum(np.sin(k * ph) / k for k in range(1, 8))
        env = np.minimum(1, np.minimum(t / 0.02, (dur - t) / 0.03))
        out.append(0.2 * y * env)
        if gap_after is not None and i == gap_after:
            out.append(np.zeros(int(SR * 0.4)))
    y = np.concatenate([np.zeros(SR // 4), *out, np.zeros(SR // 4)])
    sf.write(path, y, SR)


def test_split_lyrics():
    lines, warnings = lyrics.split("あいたいよー\nキャット\n보고 싶어")
    assert lines == [["あ", "い", "た", "い", "よ", "+"], ["きゃ", "っ", "と"], ["보", "고", "싶", "어"]]
    assert warnings == []
    assert lyrics.language("보고 싶어") == "ko"


def test_tune_writes_project(tmp_path):
    vocal = tmp_path / "vocal.wav"
    _sing(vocal, [60, 62, 64, 65, 67, 69], gap_after=2)
    res = tune(str(vocal), tmp_path / "out", "らりる\nれろら", options=Options(transpose="0", breaths="off"))
    project = yaml.safe_load((tmp_path / "out" / "project.ustx").read_text(encoding="utf-8"))
    notes = project["voice_parts"][0]["notes"]
    assert [n["tone"] for n in notes] == [60, 62, 64, 65, 67, 69]
    assert [n["lyric"] for n in notes] == ["ら", "り", "る", "れ", "ろ", "ら"]
    assert res.notes == 6
    curves = {c["abbr"] for c in project["voice_parts"][0]["curves"]}
    assert {"pitd", "dyn"} <= curves
    preview = json.loads((tmp_path / "out" / "preview.json").read_text())
    assert len(preview["notes"]) == 6 and preview["phrases"] == 2


def test_auto_transpose_lifts_low_voice(tmp_path):
    vocal = tmp_path / "low.wav"
    _sing(vocal, [48, 50, 52])
    res = tune(str(vocal), tmp_path / "out", options=Options(breaths="off"))
    assert res.transpose == 12

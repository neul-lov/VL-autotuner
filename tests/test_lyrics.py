"""가사 나누기 · 읽는 법 · 반주 복사 시험 (fugashi가 있을 때와 없을 때 둘 다)."""
import json

import numpy as np
import pytest
import soundfile as sf
import yaml

from tetotune import audio, lyrics, voicebank
from tetotune import tune as tune_mod
from tetotune.tune import Options, tune

SR = 44100
needs_tagger = pytest.mark.skipif(lyrics._tagger() is None, reason="fugashi · unidic-lite 없음")


def _flat(lines):
    return [s for line in lines for s in line]


@pytest.fixture
def no_tagger(monkeypatch):
    monkeypatch.setattr(lyrics, "_tagger", lambda: None)


# ---------------------------------------------------------------- 읽는 법 (fugashi)
@needs_tagger
def test_kanji_keeps_lines_and_spaces():
    lines, warnings = lyrics.split("君の名前を\n呼んでいた\n夜空に")
    assert lines == [["き", "み", "の", "な", "ま", "え", "を"], ["よ", "ん", "で", "い", "た"], ["よ", "ぞ", "ら", "に"]]
    assert warnings == []
    lines, _ = lyrics.split("hello world\n空")
    assert lines == [["hello", "world"], ["そ", "ら"]]


@needs_tagger
def test_particles_and_readings():
    assert _flat(lyrics.split("私は君が好き")[0]) == list("わたしわきみがすき")
    assert _flat(lyrics.split("明日へ")[0]) == list("あしたえ")
    assert _flat(lyrics.split("何もない")[0]) == list("なにもない")
    assert _flat(lyrics.split("何の話")[0]) == list("なんのはなし")  # 何の는 なん 그대로
    # 가나만 쓴 가사도 조사만 바뀐다: 낱말 속 は · へ, 줄 첫머리의 は는 그대로
    assert lyrics.split("わたしは\nはなはきれい\nはしれ")[0] == [list("わたしわ"), list("はなわきれい"), list("はしれ")]
    assert _flat(lyrics.split("へいわ そらへ")[0]) == list("へいわそらえ")
    assert _flat(lyrics.split("ははははは")[0]) == list("ははははは")  # 웃음 뒤의 は는 조사가 아니다


# ---------------------------------------------------------------- fugashi 없을 때
def test_without_tagger_kana_still_splits(no_tagger):
    lines, warnings = lyrics.split("わたしは\nキャット")
    assert lines == [list("わたしは"), ["きゃ", "っ", "と"]]
    assert warnings == []


def test_without_tagger_kanji_warns(no_tagger):
    lines, warnings = lyrics.split("君の\n名前")
    assert len(lines) == 2
    assert any("fugashi" in w for w in warnings)


# ---------------------------------------------------------------- 작은 글자 · 기호
def test_small_vowel_kana(no_tagger):
    assert _flat(lyrics.split("ねぇ あぁ もぅ いやだよぉ")[0]) == ["ね", "+", "あ", "+", "も", "+", "い", "や", "だ", "よ", "+"]
    combos = ["ふぁ", "てぃ", "うぃ", "ゔぁ", "ちぇ", "しぇ", "じぇ", "いぇ", "うぇ", "うぉ", "きゃ", "しゅ", "ちょ", "くゎ"]
    assert _flat(lyrics.split("".join(combos))[0]) == combos
    assert _flat(lyrics.split("ヴァイオリン かぃ")[0]) == ["ゔぁ", "い", "お", "り", "ん", "か", "い"]


def test_symbols_dropped_and_tilde_extends(no_tagger):
    lines, warnings = lyrics.split("ありがとう…\nさよなら〜\n“あい”♪、ねえ！\n사랑해~\nまた～ね")
    assert lines == [list("ありがとう"), ["さ", "よ", "な", "ら", "+"], list("あいねえ"), ["사", "랑", "해", "+"],
                     ["ま", "た", "+", "ね"]]
    assert warnings == []
    assert _flat(lyrics.split("ｱｲｼﾃﾙ")[0]) == list("あいしてる")  # 반각 가타카나


def test_digits_dropped_with_warning(no_tagger):
    lines, warnings = lyrics.split("らら1２3\nla la")
    assert lines == [["ら", "ら"], ["la", "la"]]
    assert any("숫자" in w for w in warnings)


# ---------------------------------------------------------------- tune()
def _sing(path, tones, gap_after=None, dur=0.5):
    out = []
    for i, m in enumerate(tones):
        t = np.arange(int(SR * dur)) / SR
        ph = 2 * np.pi * np.cumsum(440 * 2 ** ((m - 69) / 12) * np.ones_like(t)) / SR
        env = np.minimum(1, np.minimum(t / 0.02, (dur - t) / 0.03))
        out.append(0.2 * sum(np.sin(k * ph) / k for k in range(1, 8)) * env)
        if gap_after is not None and i == gap_after:
            out.append(np.zeros(int(SR * 0.4)))
    sf.write(path, np.concatenate([np.zeros(SR // 4), *out, np.zeros(SR // 4)]), SR)


def test_extend_marks_do_not_count_as_missing(tmp_path):
    vocal = tmp_path / "vocal.wav"
    _sing(vocal, [60, 62, 64, 65, 67, 69], gap_after=2)
    res = tune(str(vocal), tmp_path / "out", "らりー\nれろー", options=Options(transpose="0", breaths="off"))
    notes = yaml.safe_load((tmp_path / "out" / "project.ustx").read_text(encoding="utf-8"))["voice_parts"][0]["notes"]
    assert [n["lyric"] for n in notes] == ["ら", "り", "+", "れ", "ろ", "+"]
    assert not any("음절 중" in w for w in res.warnings)


def test_korean_note_lyrics_use_ko_to_ja_and_renderer(tmp_path, monkeypatch):
    vocal = tmp_path / "vocal.wav"
    _sing(vocal, [60, 62, 64])
    notes = tmp_path / "vocals.json"
    notes.write_text(json.dumps({"raw": [[0.25 + 0.5 * i, 0.7 + 0.5 * i, 60 + 2 * i, 100, s]
                                         for i, s in enumerate("사랑해")], "tempo": 120}))
    monkeypatch.setattr(voicebank, "find", lambda prefer=None: (None, None))
    res = tune(str(vocal), tmp_path / "a", "", str(notes), options=Options(transpose="0", breaths="off"))
    assert res.phonemizer == voicebank.KO_TO_JA
    track = yaml.safe_load((tmp_path / "a" / "project.ustx").read_text(encoding="utf-8"))["tracks"][0]
    assert track["renderer_settings"] == tune_mod.renderer_settings()  # hifisampler 또는 (쓸 수 없으면) WORLDLINE-R

    # DiffSinger 같은 가수: 렌더러는 OpenUtau가 고르게 비워 둔다
    info = {"phonemizer": "X", "breath_in": None, "breath_out": None, "japanese": False, "name": "ds", "classic": False}
    monkeypatch.setattr(voicebank, "find", lambda prefer=None: ("ds", info))
    res = tune(str(vocal), tmp_path / "b", "", str(notes), options=Options(transpose="0", breaths="off"))
    assert res.phonemizer == "X"
    track = yaml.safe_load((tmp_path / "b" / "project.ustx").read_text(encoding="utf-8"))["tracks"][0]
    assert "renderer_settings" not in track


@needs_tagger
def test_katakana_particles_and_greetings():
    assert _flat(lyrics.split("ワタシハ ドコヘ")[0]) == list("わたしわどこえ")
    assert _flat(lyrics.split("こんにちは")[0]) == list("こんにちわ")
    assert _flat(lyrics.split("ヨハネ ハハハ")[0]) == list("よはねははは")


def test_line_separators_and_apostrophes():
    assert lyrics.split("あ\u2028い")[0] == [["あ"], ["い"]]
    assert lyrics.split("don't stop I’m 'cause")[0] == [["don't", "stop", "I'm", "cause"]]


def test_korean_lyrics_warn_when_singer_cannot_read_them(tmp_path, monkeypatch):
    vocal = tmp_path / "vocal.wav"
    _sing(vocal, [60, 62, 64])
    info = {"phonemizer": "DiffSingerJapanesePhonemizer", "breath_in": None, "breath_out": None, "japanese": False,
            "korean": False, "name": "ds", "classic": False}
    monkeypatch.setattr(voicebank, "find", lambda prefer=None: ("ds", info))
    res = tune(str(vocal), tmp_path / "k", "사랑해", options=Options(transpose="0", breaths="off"))
    assert any("한글 가사를 부르지 못해요" in w for w in res.warnings)


# ---------------------------------------------------------------- 오디오
def test_backing_kept_as_is(tmp_path):
    y = (0.1 * np.random.default_rng(0).standard_normal((48000, 2))).astype(np.float32)
    src = tmp_path / "mix.FLAC"
    sf.write(src, y, 48000, subtype="PCM_24", format="FLAC")
    out, ms = audio.save_copy(src, tmp_path / "backing")
    assert out.name == "backing.flac" and out.read_bytes() == src.read_bytes()
    assert ms == pytest.approx(1000)
    ogg = tmp_path / "mix.ogg"
    sf.write(ogg, y, 48000, format="OGG", subtype="VORBIS")
    out, ms = audio.save_copy(ogg, tmp_path / "backing2")
    info = sf.info(out)
    assert out.name == "backing2.wav" and (info.channels, info.samplerate, info.subtype) == (2, 48000, "PCM_24")
    assert ms == pytest.approx(1000, abs=30)


def test_aifc_backing_becomes_wav(tmp_path):
    y = (0.1 * np.random.default_rng(1).standard_normal((44100, 2))).astype(np.float32)
    plain, little = tmp_path / "a.aiff", tmp_path / "b.aiff"
    sf.write(plain, y, 44100, subtype="PCM_16", format="AIFF")
    sf.write(little, y, 44100, subtype="PCM_16", format="AIFF", endian="LITTLE")  # AIFF-C sowt
    assert audio.save_copy(plain, tmp_path / "p")[0].name == "p.aiff"
    out = audio.save_copy(little, tmp_path / "q")[0]
    assert out.name == "q.wav" and sf.info(out).subtype == "PCM_24"


def test_tune_writes_backing_part(tmp_path):
    vocal = tmp_path / "vocal.wav"
    _sing(vocal, [60, 62, 64])
    backing = tmp_path / "mix.wav"
    sf.write(backing, np.zeros((SR * 2, 2), dtype=np.float32), SR, subtype="PCM_24")
    tune(str(vocal), tmp_path / "out", "らりる", backing=str(backing), options=Options(transpose="0", breaths="off"))
    part = yaml.safe_load((tmp_path / "out" / "project.ustx").read_text(encoding="utf-8"))["wave_parts"][1]
    assert part["relative_path"] == "backing.wav" and part["file_duration_ms"] == pytest.approx(2000)
    assert sf.info(tmp_path / "out" / "backing.wav").channels == 2


@pytest.mark.skipif(audio._ffmpeg() is None, reason="ffmpeg 없음")
def test_unreadable_audio_says_why(tmp_path):
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(np.random.default_rng(1).bytes(4000))
    with pytest.raises(ValueError, match="오디오를 읽지 못했어요") as e:
        audio.load(bad)
    assert "Command" not in str(e.value) and str(tmp_path) not in str(e.value)

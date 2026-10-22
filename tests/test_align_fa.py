"""가사 맞추기 모델(HubertFA)의 모델 없는 부분: 가사 → 음소, 디코더, 음표에 넣기."""
import numpy as np
import pytest

from tetotune import align_fa as fa
from tetotune import notes as nt

# HubertFA v0.0.7 일본어 음소 (vocab.json의 ja/…)
JA = set("AP N SP a b by ch cl d dy e f g gw gy h hy i j k kw ky m my n ny o p py r ry s sh t ts ty u v w y z".split())


@pytest.mark.parametrize("kana,romaji", [
    ("か", "ka"), ("し", "shi"), ("つ", "tsu"), ("ん", "n"), ("っ", "cl"), ("きゃ", "kya"), ("しゃ", "sha"),
    ("ちょ", "cho"), ("じゅ", "ju"), ("ふぁ", "fa"), ("てぃ", "ti"), ("でゅ", "dyu"), ("うぃ", "wi"), ("くぁ", "kwa"),
    ("つぁ", "tsa"), ("ゔぁ", "va"), ("いぇ", "ye"), ("ka", None), ("かき", None),
])
def test_kana_romaji(kana, romaji):
    assert fa.kana_romaji(kana) == romaji


@pytest.mark.parametrize("syl,prev,phones", [
    ("가", None, ["k", "a"]), ("가", "나", ["g", "a"]), ("가", "각", ["k", "a"]), ("각", None, ["k", "a", "cl"]),
    ("안", None, ["a", "N"]), ("일", None, ["i", "r"]), ("시", None, ["sh", "i"]), ("샤", None, ["sh", "a"]),
    ("티", None, ["t", "i"]), ("과", None, ["kw", "a"]), ("야", None, ["y", "a"]), ("쟈", None, ["ch", "a"]),
    ("자", "아", ["j", "a"]), ("의", None, ["u", "i"]), ("뷰", None, ["py", "u"]), ("뷰", "아", ["by", "u"]),
])
def test_hangul_phones(syl, prev, phones):
    assert fa.hangul_phones(syl, prev) == phones


def test_every_hangul_syllable_uses_known_phones():
    for code in range(0xAC00, 0xD7A4):
        for prev in (None, "아"):
            ph = fa.hangul_phones(chr(code), prev)
            assert ph and set(ph) <= JA, (chr(code), ph)
            assert any(p in fa.VOWELS for p in ph), chr(code)


def _decode_vocab():
    names = ["SP", "AP", "ja/a", "ja/k", "ja/i", "ja/cl", "ja/N"]
    return {"vocab": {p: i for i, p in enumerate(names)}, "vocab_size": len(names)}


def test_decoder_finds_phoneme_edges():
    """음소마다 프레임 확률이 높은 구간이 있으면 디코더가 그 경계를 찾는다 (프레임 10ms)."""
    vocab = _decode_vocab()
    v = vocab["vocab"]
    truth = [("SP", 10), ("ja/k", 6), ("ja/a", 30), ("SP", 4), ("ja/k", 5), ("ja/i", 25), ("SP", 20)]
    T = sum(n for _, n in truth)
    logits = np.full((1, len(v), T), -4.0, np.float32)
    edges = np.full((1, T), -6.0, np.float32)
    t = 0
    for p, n in truth:
        logits[0, v[p], t:t + n] = 4.0
        if t:
            edges[0, t] = 6.0
        t += n
    ph_seq = ["SP", "ja/k", "ja/a", "SP", "ja/k", "ja/i", "SP"]
    times, conf = fa._decode(vocab, 441, 44100, T * 441, logits, edges, ph_seq)
    want = np.cumsum([0] + [n for _, n in truth])[:-1] * 0.01
    got = {k: a for a, _, k in times}
    for k, w in enumerate(want):
        assert abs(got[k] - w) <= 0.011, (ph_seq[k], got[k], w)
    assert 0 < conf <= 1
    syl = fa._syllable_times(times, ph_seq, [-1, 0, 0, -1, 1, 1, -1], 2, "ja/")
    assert syl[0][1] == pytest.approx(0.16, abs=0.011) and syl[1][1] == pytest.approx(0.55, abs=0.011)


def test_decoder_may_skip_silence_but_not_phonemes():
    """SP는 건너뛸 수 있고 (숨 안 쉬고 이어 부름), 가사 음소는 소리가 약해도 한 프레임 이상 남는다."""
    vocab = _decode_vocab()
    v = vocab["vocab"]
    T = 40
    logits = np.full((1, len(v), T), -4.0, np.float32)
    logits[0, v["ja/a"], :20] = 4.0
    logits[0, v["ja/i"], 20:] = 4.0
    edges = np.full((1, T), -6.0, np.float32)
    edges[0, 20] = 6.0
    ph_seq = ["SP", "ja/a", "SP", "ja/k", "ja/i", "SP"]
    times, _ = fa._decode(vocab, 441, 44100, T * 441, logits, edges, ph_seq)
    ks = [k for _, _, k in times]
    assert 1 in ks and 3 in ks and 4 in ks
    syl = fa._syllable_times(times, ph_seq, [-1, 0, -1, 1, 1, -1], 2, "ja/")
    assert syl is not None and abs(syl[1][1] - 0.2) <= 0.03


def _n(a, b, tone, lyric, strength=10.0):
    return nt.Note(a, b, tone, lyric, strength)


def test_place_moves_syllable_to_measured_vowel_onset():
    """가사를 한 음 늦게 넣었으면 (음 바뀌는 곳이 실제 음절 시작): 모델 시각으로 옮기고 뒤 음표는 이어 부르기."""
    g = [_n(0.0, 0.5, 60, "か"), _n(0.5, 1.0, 62, "+", 1.0), _n(1.0, 1.6, 62, "き", 3.0)]
    out = fa._place(g, [0.03, 0.55], ["か", "き"])
    assert [(round(x.start, 3), x.lyric) for x in out] == [(0.03, "か"), (0.55, "き"), (1.0, "+")]
    assert out[0].end == pytest.approx(0.55)
    assert out[0].strength == 10.0  # 프레이즈 나눔은 그대로


def test_place_splits_note_for_same_pitch_syllable():
    g = [_n(0.0, 1.0, 60, "か"), _n(1.0, 1.5, 60, "き", 3.0)]
    out = fa._place(g, [0.0, 0.6], ["か", "き"])
    assert [(round(x.start, 3), round(x.end, 3), x.lyric) for x in out] == \
        [(0.0, 0.6, "か"), (0.6, 1.0, "き"), (1.0, 1.5, "+")]
    assert all(x.tone == 60 for x in out)


def test_place_after_gap_moves_only_the_start():
    g = [_n(0.0, 0.5, 60, "か"), _n(0.7, 1.2, 64, "き", 5.0)]
    out = fa._place(g, [0.0, 0.66], ["か", "き"])
    assert out[0].end == pytest.approx(0.5) and out[1].start == pytest.approx(0.66)


def test_place_gives_up_when_notes_would_be_too_short():
    g = [_n(0.0, 0.5, 60, "か"), _n(0.5, 1.0, 60, "き", 3.0)]
    assert fa._place(g, [0.0, 0.98], ["か", "き"]) is None  # 끝에서 20ms: 너무 짧다
    assert fa._place(g, [0.0, 2.0], ["か", "き"]) is None  # 음표 밖


def test_refine_without_model_keeps_notes(tmp_path, monkeypatch):
    monkeypatch.setenv("TETOTUNE_MODELS", str(tmp_path))
    g = [_n(0.0, 0.5, 60, "か"), _n(0.5, 1.0, 62, "き", 3.0)]
    out, fixed = fa.refine(g, np.zeros(44100, np.float32), 44100)
    assert out is g and fixed == 0

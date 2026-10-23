"""가사 맞추기 모델(HubertFA)의 모델 없는 부분: 가사 → 음소, 디코더, 음표에 넣기."""
import numpy as np
import pytest

from tetotune import align_fa as fa
from tetotune import notes as nt

# HubertFA v0.0.7 일본어 음소 (vocab.json의 ja/…)
JA = set("AP N SP a b by ch cl d dy e f g gw gy h hy i j k kw ky m my n ny o p py r ry s sh t ts ty u v w y z".split())


@pytest.fixture(autouse=True)
def _plain(monkeypatch):
    """기본 동작부터 시험한다: 시간 보정 · 덜 버리기 · 틈 당기기 · 함께 맞추기는 끈 채로 (그 시험들이 켠다)."""
    for k, v in (("OFFSET", 0.0), ("SOFT_SPLIT", False), ("GAP_EXTEND", False), ("JOINT", 0.0)):
        monkeypatch.setattr(fa, k, v)


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
    names = ["SP", "AP", "ja/a", "ja/k", "ja/i", "ja/N"]
    v = {p: i for i, p in enumerate(names)}
    v["cl"] = 0  # 실제 vocab.json처럼 っ(cl)는 쉼과 한 무리
    return {"vocab": v, "vocab_size": len(names)}


def _aligner(prefix="ja/"):
    """모델 없이 음소 바꾸기만 (실제 v0.0.7 vocab.json처럼: 일본어 음소는 ja/…, 쉼 무리 SP · cl · AP …는 0)."""
    al = fa.Aligner.__new__(fa.Aligner)
    lex = sorted(JA - {"SP", "AP", "cl"})
    al.ids = {p: 0 for p in ("", "<SP>", "AP", "CL", "EP", "GS", "SP", "cl", "pau")}
    al.ids.update({prefix + p: i + 1 for i, p in enumerate(lex)})
    al.prefix, al.dict, al.sil = prefix, {}, 0
    return al


@pytest.mark.parametrize("syl,prev,phones", [
    ("각", None, ["ja/k", "ja/a"]), ("안", None, ["ja/a", "ja/N"]), ("ん", None, ["ja/N"]), ("てぃ", None, ["ja/t", "ja/i"]),
    ("くぁ", None, ["ja/kw", "ja/a"]), ("し", None, ["ja/sh", "ja/i"]), ("っ", None, None), ("ka", None, None),
])
def test_phones_use_model_vocab(syl, prev, phones):
    """っ · 받침 막힘(cl)은 모델에서 쉼과 같은 음소라 빼고 (음절 사이 SP가 맡음), 사전에 없는 로마자는 자음 + 모음으로."""
    assert _aligner().phones(syl, prev) == phones


def test_every_hangul_syllable_has_model_phones():
    al = _aligner()
    for code in range(0xAC00, 0xD7A4):
        assert al.phones(chr(code), "아"), chr(code)
    seq, word = al.sequence(["각", "시", "ん"])
    assert seq == ["SP", "ja/k", "ja/a", "SP", "ja/sh", "ja/i", "SP", "ja/N", "SP"]
    assert word == [-1, 0, 0, -1, 1, 1, -1, 2, -1]


def test_korean_finals_and_eo_from_other_languages_fall_back_to_japanese(monkeypatch):
    """KO_FINALS "en" · KO_EO "zh": 모델에 그 음소가 있으면 쓰고, 없으면 일본어 그대로."""
    monkeypatch.setattr(fa, "KO_FINALS", "en")
    monkeypatch.setattr(fa, "KO_EO", "zh")
    assert fa.hangul_phones("안") == ["a", "en/n|N"]
    assert fa.hangul_phones("멍") == ["m", "zh/e|o", "en/ng|N"]
    assert fa.hangul_phones("별") == ["py", "zh/e|o", "en/l|r"]
    al = _aligner()
    assert al.phones("안") == ["ja/a", "ja/N"]  # 영어 음소가 없는 모델
    al.ids.update({"en/n": 90, "en/ng": 91, "zh/e": 92})
    assert al.phones("안") == ["ja/a", "en/n"]
    assert al.phones("멍") == ["ja/m", "zh/e", "en/ng"]
    times = fa._syllable_times([(0.0, 0.1, 1), (0.1, 0.3, 2), (0.3, 0.4, 3)], ["SP", "ja/m", "zh/e", "en/ng", "SP"],
                               [-1, 0, 0, 0, -1], 1, "ja/")
    assert times == [(0.0, 0.1, 0.4)]  # 모음 시작 = zh/e 시작


def test_decoder_finds_phoneme_edges():
    """음소마다 프레임 확률이 높은 구간이 있으면 디코더가 그 경계를 찾는다 (프레임 10ms)."""
    vocab = _decode_vocab()
    v = vocab["vocab"]
    truth = [("SP", 10), ("ja/k", 6), ("ja/a", 30), ("SP", 4), ("ja/k", 5), ("ja/i", 25), ("SP", 20)]
    T = sum(n for _, n in truth)
    logits = np.full((1, vocab["vocab_size"], T), -4.0, np.float32)
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
    logits = np.full((1, vocab["vocab_size"], T), -4.0, np.float32)
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


def test_place_gap_extend_pulls_note_start_to_vowel(monkeypatch):
    """모음 시작이 음표 앞 틈(150ms 안)에 있으면: 기본은 음표 시작 그대로, GAP_EXTEND면 모음 시작으로 당긴다."""
    g = [_n(0.0, 0.5, 60, "か"), _n(0.7, 1.2, 64, "き", 5.0)]
    assert fa._place(g, [0.0, 0.58], ["か", "き"])[1].start == pytest.approx(0.7)
    monkeypatch.setattr(fa, "GAP_EXTEND", True)
    out = fa._place(g, [0.0, 0.58], ["か", "き"])
    assert out[1].start == pytest.approx(0.58) and out[0].end == pytest.approx(0.5)
    assert g[1].start == 0.7  # 원래 음표는 그대로


def test_place_soft_split_uses_next_note_instead_of_giving_up(monkeypatch):
    g = [_n(0.0, 0.5, 60, "か"), _n(0.62, 1.0, 62, "き", 5.0)]
    assert fa._place(g, [0.0, 0.48], ["か", "き"]) is None  # 앞 음표 끝 20ms 안: 프레이즈를 버린다
    monkeypatch.setattr(fa, "SOFT_SPLIT", True)
    out = fa._place(g, [0.0, 0.48], ["か", "き"])
    assert [(x.start, x.lyric) for x in out] == [(0.0, "か"), (0.62, "き")]
    g = [_n(0.0, 0.5, 60, "か")]  # 뒤 음표가 없으면 40ms 앞에서 나눈다
    out = fa._place(g, [0.0, 0.48], ["か", "き"])
    assert [(round(x.start, 3), x.lyric) for x in out] == [(0.0, "か"), (0.46, "き")]


def test_place_offset_shifts_every_onset(monkeypatch):
    monkeypatch.setattr(fa, "OFFSET", 0.02)
    g = [_n(0.0, 1.0, 60, "か"), _n(1.0, 1.5, 60, "き", 3.0)]
    out = fa._place(g, [0.0, 0.6], ["か", "き"])
    assert [round(x.start, 3) for x in out] == [0.02, 0.62, 1.0]


def test_closure_notes_are_merged_then_split_again():
    """っ 음표는 앞 음절 끝의 막힘이라 모델에 넣지 않고, 음절 자리를 맞춘 뒤 같은 방법으로 다시 나눈다."""
    g = [_n(0.0, 0.41, 60, "き"), _n(0.41, 0.5, 60, "っ", 3.0), _n(0.5, 1.0, 62, "と", 3.0)]
    core, closures = fa._without_closures(g)
    assert [(x.start, x.end, x.lyric) for x in core] == [(0.0, 0.5, "き"), (0.5, 1.0, "と")] and closures == {0}
    assert g[0].end == 0.41  # 원래 음표는 그대로 (맞추지 못하면 원래대로 쓴다)
    out = nt._insert_closures(fa._place(core, [0.0, 0.55], ["き", "と"]), closures)
    assert [(round(x.start, 3), round(x.end, 3), x.lyric) for x in out] == \
        [(0.0, 0.46, "き"), (0.46, 0.55, "っ"), (0.55, 1.0, "と")]


def test_refine_without_model_keeps_notes(tmp_path, monkeypatch):
    monkeypatch.setenv("TETOTUNE_MODELS", str(tmp_path))
    g = [_n(0.0, 0.5, 60, "か"), _n(0.5, 1.0, 62, "き", 3.0)]
    out, fixed = fa.refine(g, np.zeros(44100, np.float32), 44100)
    assert out is g and fixed == 0


class _FakeAligner:
    """음절마다 정해 둔 모음 시작 (곡 시각)을 돌려주는 가짜 모델. 받은 구간 시작을 빼서 돌려준다."""
    def __init__(self, vowel_at, sr=100):
        self.vowel_at, self.sr, self.calls = vowel_at, sr, []

    def sequence(self, syls):
        return ["SP"], [-1]

    def align(self, y, sr, syls):
        a = self.offset
        self.calls.append(list(syls))
        return [(self.vowel_at[s] - a - 0.05, self.vowel_at[s] - a, self.vowel_at[s] - a + 0.1) for s in syls], 0.7


def test_joint_alignment_moves_syllable_across_phrases(monkeypatch):
    """가사 넣기가 「く」를 둘째 프레이즈에 넣었는데 소리로는 첫 프레이즈 끝: 한꺼번에 맞추면 제자리로 간다."""
    g = [_n(0.0, 0.4, 60, "か"), _n(0.4, 1.0, 62, "き", 5.0),
         _n(1.5, 1.8, 64, "く"), _n(1.8, 2.4, 65, "け", 5.0)]
    fake = _FakeAligner({"か": 0.0, "き": 0.4, "く": 0.75, "け": 1.5})
    monkeypatch.setattr(fa, "model_dir", lambda: "x")
    monkeypatch.setattr(fa, "Aligner", lambda folder: fake)
    orig_window = fa._fit

    def fit(al, notes, a, b, y, sr, tag):
        fake.offset = a
        return orig_window(al, notes, a, b, y, sr, tag)
    monkeypatch.setattr(fa, "_fit", fit)
    y = np.zeros(300)
    out, fixed = fa.refine(g, y, 100)  # 프레이즈마다: く는 둘째 프레이즈 첫 음표에 그대로
    assert (1.5, "く") in [(round(x.start, 2), x.lyric) for x in out]
    monkeypatch.setattr(fa, "JOINT", 30.0)
    out, fixed = fa.refine(g, y, 100)
    assert fixed == 2
    assert [(round(x.start, 2), x.lyric) for x in out if x.lyric != "+"] == [(0.0, "か"), (0.4, "き"), (0.75, "く"), (1.5, "け")]
    assert fake.calls[-1] == ["か", "き", "く", "け"]


def test_tta_takes_median_per_syllable(monkeypatch):
    """구간 시작을 밀어 여러 번 돌리면 음절마다 가운데 값: 한 번 크게 틀려도 나머지 둘이 맞으면 맞는 값."""
    answers = iter([[(0.10, 0.15, 0.4), (0.40, 0.45, 0.8)],
                    [(0.10, 0.15, 0.4), (0.70, 0.75, 0.8)],  # 둘째 음절을 300ms 틀림
                    [(0.10, 0.15, 0.4), (0.40, 0.45, 0.8)]])

    class Al:
        sr, hop = 44100, 441

        def align(self, y, sr, syls):
            return next(answers), 0.6
    monkeypatch.setattr(fa, "TTA", 3)
    times, conf = fa._align_tta(Al(), np.zeros(44100), 44100, 0.0, 1.0, ["か", "き"])
    assert conf == pytest.approx(0.6)
    assert times[1][1] == pytest.approx(0.45 + 2 * 0.01 / 3, abs=1e-3)  # 가운데 값 (밀어 돌린 만큼 더한 뒤)

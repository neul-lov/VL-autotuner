"""정확도 개선(모음 시작 · GAME 음표)의 작은 부분 시험. 모델 없이 돈다."""
import numpy as np

from tetotune import game, notes as nt
from tetotune.pitch import HOP


def test_voiced_onset():
    assert nt._voiced_onset("ま") and nt._voiced_onset("ら") and nt._voiced_onset("ぎゃ")
    assert not nt._voiced_onset("か") and not nt._voiced_onset("あ") and not nt._voiced_onset("+")
    assert nt._voiced_onset("나") and nt._voiced_onset("말") and not nt._voiced_onset("아")
    assert nt._voiced_onset("na") and not nt._voiced_onset("ka")


def test_shift_to_vowels_moves_start_past_quiet_nasal():
    n = int(1.0 / HOP)
    db = np.full(n, -60.0)
    db[int(0.2 / HOP):int(0.28 / HOP)] = -30.0  # ㅁ: 모음보다 작다
    db[int(0.28 / HOP):int(0.8 / HOP)] = -15.0  # 모음
    prev = nt.Note(0.0, 0.2, 60, "あ")
    x = nt.Note(0.2, 0.8, 62, "ま")
    out = nt.shift_to_vowels([prev, x], db)
    assert abs(out[1].start - 0.28) < 0.02 and out[0].end == out[1].start
    y = nt.Note(0.2, 0.8, 62, "か")  # 울리지 않는 자음은 그대로
    assert nt.shift_to_vowels([y], db)[0].start == 0.2


def test_from_game_phrases_and_overlap():
    voiced = np.ones(400, dtype=bool)
    midi = np.full(400, 60.2)
    midi[int(0.5 / HOP):] = 62.2
    midi[int(1.5 / HOP):] = 64.2
    out, off = nt.from_game([(0.0, 0.5, 60.1), (0.48, 1.0, 62.0), (1.5, 2.0, 64.0)], midi, voiced)
    assert [x.tone for x in out] == [60, 62, 64]
    assert out[0].end == out[1].start == 0.48  # 겹치면 앞 음표를 자른다
    assert out[1].strength == 5.0 and out[2].strength == 10.0  # 쉼 뒤는 프레이즈 시작


def test_game_chunks_cut_in_rests():
    voiced = np.zeros(int(50 / HOP), dtype=bool)
    for a in range(0, 50, 5):  # 4초 노래 + 1초 쉼
        voiced[int(a / HOP):int((a + 4) / HOP)] = True
    ch = game.chunks(voiced, HOP)
    assert ch[0][0] == 0 and abs(ch[-1][1] - 50) < 0.01
    assert all(b - a <= game.MAX_CHUNK * 1.5 for a, b in ch)
    for a, b in ch[1:]:
        assert not voiced[int(a / HOP)]  # 노래 중간에서 자르지 않는다


def _sung(spans, n, gaps=()):
    """노래 프레임: spans 안은 소리 (−15dB), gaps(자음 틈)는 무성 · 작게."""
    voiced = np.zeros(n, dtype=bool)
    db = np.full(n, -60.0)
    for a, b in spans:
        voiced[int(a / HOP):int(b / HOP)] = True
        db[int(a / HOP):int(b / HOP)] = -15.0
    for a, b in gaps:
        voiced[int(a / HOP):int(b / HOP)] = False
        db[int(a / HOP):int(b / HOP)] = -35.0
    return voiced, db


def test_game_melisma_keeps_syllables_on_consonants(monkeypatch):
    """あ い(+ +) し て(+) る: GAME이 8음표로 나눠도 음절은 자음 틈이 있는 곳에서 시작한다 (예전: 시간 순서대로 앞에 몰림)."""
    monkeypatch.setattr(nt, "GAME_EVIDENCE", True)  # 기본은 꺼짐 (notes.py)
    n = int(4.0 / HOP)
    bounds = [0.0, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4, 2.8]
    tones = [60, 62, 64, 62, 60, 62, 64, 60]
    # し · て · る 앞에 자음 틈 (ㅅ · ㅌ · ㄹ은 짧게 소리가 꺼진다)
    voiced, db = _sung([(0.0, 2.8)], n, gaps=[(1.17, 1.2), (1.57, 1.6), (2.37, 2.4)])
    midi = np.full(n, np.nan)
    for a, b, t in zip(bounds, bounds[1:], tones):
        midi[int(a / HOP):int(b / HOP)] = t
    midi = np.where(voiced, midi, np.nan)
    gn = [(a, b, float(t)) for a, b, t in zip(bounds, bounds[1:], tones)]
    out, _ = nt.from_game(gn, midi, voiced, db, None)
    out = nt.align(out, [["あ", "い", "し", "て", "る"]])
    lyr = [x.lyric for x in out]
    start = {x.lyric: round(x.start, 2) for x in out if x.lyric != "+"}
    assert start["し"] == 1.2 and start["て"] == 1.6 and start["る"] == 2.4, lyr
    assert start["い"] == 0.4, lyr


def test_small_tsu_is_a_closure_not_a_syllable_start():
    """きっと를 두 음(き · と)으로 부르면 と는 둘째 음 시작에, っ는 き 끝의 짧은 음표로."""
    n = int(2.0 / HOP)
    voiced, db = _sung([(0.0, 0.4), (0.45, 1.2)], n)
    midi = np.where(voiced, 60.0, np.nan)
    out, _ = nt.from_game([(0.0, 0.4, 60.0), (0.45, 1.2, 62.0)], midi, voiced, db, None)
    out = nt.align(out, [["き", "っ", "と"]])
    got = [(x.lyric, round(x.start, 2), round(x.end, 2)) for x in out]
    assert got == [("き", 0.0, 0.31), ("っ", 0.31, 0.4), ("と", 0.45, 1.2)], got


def test_plus_after_gap_is_not_silent():
    a = nt.Note(0.0, 0.5, 60, "か")
    b = nt.Note(0.6, 0.9, 62, "+")  # 짧은 틈: 앞 음표를 늘인다
    c = nt.Note(2.0, 2.5, 62, "+")  # 긴 쉼 뒤: 앞 모음으로
    d = nt.Note(2.5, 2.8, 64, "+")  # 붙어 있으면 그대로
    e = nt.Note(3.0, 3.3, 64, "+")  # 숨 뒤: 앞 모음으로
    assert nt.fix_plus([a, b, c, d, e], breaths=[(2.85, 2.95)]) == 3
    assert a.end == 0.6 and b.lyric == "+"
    assert c.lyric == "あ" and d.lyric == "+" and e.lyric == "あ"
    k = nt.Note(0.0, 0.5, 60, "강")
    m = nt.Note(1.5, 2.0, 60, "+")
    nt.fix_plus([k, m])
    assert m.lyric == "아"


def test_game_tone_follows_steady_sung_pitch():
    """GAME이 한 반음 틀리게 읽은 음표는 가운데에서 고르게 읽힌 음 높이로. 옥타브 차이 · 흔들리는 곳은 그대로."""
    n = int(3.0 / HOP)
    voiced = np.ones(n, dtype=bool)
    midi = np.full(n, 62.0)
    midi[int(1.0 / HOP):int(2.0 / HOP)] = 72.0  # 둘째 음: 옥타브 차이 (음 높이 읽기가 틀렸을 수 있다)
    midi[int(2.0 / HOP):] = 60 + 3 * np.sin(np.arange(n - int(2.0 / HOP)) * HOP * 2 * np.pi * 2)  # 크게 흔들림
    out, _ = nt.from_game([(0.0, 1.0, 61.0), (1.0, 2.0, 60.0), (2.0, 3.0, 61.0)], midi, voiced)
    assert [x.tone for x in out] == [62, 60, 61]

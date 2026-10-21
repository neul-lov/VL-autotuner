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

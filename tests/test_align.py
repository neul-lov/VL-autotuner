"""가사 줄과 프레이즈 수가 다를 때: 줄 경계를 지켜서 가사가 다음 줄로 밀리지 않는지."""
from tetotune import notes as nt


def _phrase(t, n, first=10.0):
    out = []
    for i in range(n):
        out.append(nt.Note(t, t + 0.3, 60, strength=first if i == 0 else 5.0))
        t += 0.3
    return out, t + 0.5


def _song(sizes):
    notes, t = [], 0.0
    for n in sizes:
        p, t = _phrase(t, n)
        notes += p
    return notes


def test_extra_break_inside_a_line_keeps_lines():
    # 줄 3개 (4 · 5 · 4음절)인데 둘째 줄이 숨 아닌 곳에서 끊겨 프레이즈가 4개 (4 · 2 · 3 · 4음)
    lines = [list("あいうえ"), list("かきくけこ"), list("さしすせ")]
    out = nt.align(_song([4, 2, 3, 4]), lines)
    assert [n.lyric for n in out] == list("あいうえかきくけこさしすせ")


def test_two_lines_sung_in_one_breath():
    lines = [list("あい"), list("うえ"), list("かきくけ")]
    out = nt.align(_song([4, 4]), lines)
    assert [n.lyric for n in out] == list("あいうえかきくけ")


def test_one_line_still_distributes():
    out = nt.align(_song([3, 3]), [list("あいうえおか")])
    assert [n.lyric for n in out] == list("あいうえおか")


def test_consonant_fragment_joins_previous_note():
    ns = [nt.Note(0.0, 0.4, 60), nt.Note(0.4, 0.5, 62, strength=5.0), nt.Note(0.5, 0.9, 64, strength=5.0)]
    out = nt._merge_onsets(ns, lambda x: True)
    assert [(n.start, n.end, n.tone) for n in out] == [(0.0, 0.5, 60), (0.5, 0.9, 64)]


def test_short_sung_note_is_kept():
    ns = [nt.Note(0.0, 0.4, 60), nt.Note(0.4, 0.5, 62, strength=5.0), nt.Note(0.5, 0.9, 64, strength=5.0)]
    assert len(nt._merge_onsets(ns, lambda x: False)) == 3


def test_consonant_fragment_at_phrase_start_is_dropped():
    ns = [nt.Note(0.0, 0.1, 62), nt.Note(0.1, 0.5, 64, strength=5.0)]
    out = nt._merge_onsets(ns, lambda x: True)
    assert [(n.start, n.strength) for n in out] == [(0.1, 10.0)]

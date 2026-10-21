"""음표 파일 읽기 (MIDI · ust · json), 가사 나누기(_distribute), 음원 폴더 읽기 시험."""
import json
import random
import time

import mido

from tetotune import notes as nt
from tetotune import voicebank as vb


# ---------------------------------------------------------------- MIDI
def _track(events, name=None):
    """[(틱, 메시지)] → mido 트랙 (같은 틱이면 적은 순서대로)."""
    tr = mido.MidiTrack()
    if name:
        tr.append(mido.MetaMessage("track_name", name=name.encode("utf-8").decode("latin-1"), time=0))
    last = 0
    for tick, msg in sorted(events, key=lambda e: e[0]):
        tr.append(msg.copy(time=tick - last))
        last = tick
    return tr


def _on(note, ch=0):
    return mido.Message("note_on", note=note, velocity=100, channel=ch)


def _off(note, ch=0):
    return mido.Message("note_off", note=note, velocity=0, channel=ch)


def _lyric(text):
    return mido.MetaMessage("lyrics", text=text.encode("utf-8").decode("latin-1"))


def _save(path, tracks, type=1):
    mid = mido.MidiFile(type=type, ticks_per_beat=480)
    mid.tracks.extend(tracks)
    mid.save(str(path))
    return path


def test_midi_same_pitch_on_before_off(tmp_path):
    """같은 음 셋을 이어서: 같은 틱에서 다음 note_on이 앞 note_off보다 먼저 온다."""
    ev = [(0, _on(60)), (480, _on(60)), (480, _off(60)), (960, _on(60)), (960, _off(60)), (1440, _off(60))]
    notes, bpm = nt.read_file(_save(tmp_path / "a.mid", [_track(ev)]))
    assert bpm == 120
    assert [(n.start, n.end, n.tone) for n in notes] == [(0, 0.5, 60), (0.5, 1.0, 60), (1.0, 1.5, 60)]


def test_midi_legato_overlap(tmp_path):
    ev = [(0, _on(60)), (480, _on(60)), (490, _off(60)), (960, _on(62)), (970, _off(60)), (1440, _off(62))]
    notes, _ = nt.read_file(_save(tmp_path / "a.mid", [_track(ev)]))
    assert [(n.start, n.end, n.tone) for n in notes] == [(0, 0.5, 60), (0.5, 1.0, 60), (1.0, 1.5, 62)]


def _vocal(ch=0, lyrics="らりるれろらりる"):
    ev = []
    for i, ly in enumerate(lyrics):
        ev += [(i * 480, _lyric(ly)), (i * 480, _on(60 + i, ch)), (i * 480 + 470, _off(60 + i, ch))]
    return ev


def _drums(n=32):
    return [e for i in range(n) for e in ((i * 120, _on(36 + i % 2 * 6, 9)), (i * 120 + 60, _off(36 + i % 2 * 6, 9)))]


def _piano(n=24, ch=1):
    return [e for i in range(n) for e in ((i * 160, _on(48 + i % 5, ch)), (i * 160 + 150, _off(48 + i % 5, ch)))]


def test_midi_picks_vocal_track_with_lyrics(tmp_path):
    path = _save(tmp_path / "song.mid", [_track(_drums(), "Drums"), _track(_piano(), "Piano"), _track(_vocal(), "Vocal")])
    notes, _ = nt.read_file(path)
    assert [n.tone for n in notes] == list(range(60, 68))
    assert "".join(n.lyric for n in notes) == "らりるれろらりる"


def test_midi_type0_picks_lyric_channel(tmp_path):
    """한 트랙에 모든 채널: 가사가 붙은 채널만 (드럼이 가사보다 먼저 와도 가사는 노래 음표에)."""
    path = _save(tmp_path / "song0.mid", [_track(_drums() + _vocal(ch=2) + _piano(ch=1))], type=0)
    notes, _ = nt.read_file(path)
    assert [n.tone for n in notes] == list(range(60, 68))
    assert "".join(n.lyric for n in notes) == "らりるれろらりる"


def test_midi_track_name_before_note_count(tmp_path):
    melody = [e for i in range(4) for e in ((i * 480, _on(64)), (i * 480 + 400, _off(64)))]
    path = _save(tmp_path / "m.mid", [_track(_piano(), "Piano"), _track(melody, "メロディ Melody")])
    notes, _ = nt.read_file(path)
    assert [n.tone for n in notes] == [64] * 4


def test_midi_cp932_lyrics(tmp_path):
    ev = [(0, mido.MetaMessage("lyrics", text="か①".encode("cp932").decode("latin-1"))), (0, _on(60)), (480, _off(60))]
    notes, _ = nt.read_file(_save(tmp_path / "a.mid", [_track(ev)]))
    assert [n.lyric for n in notes] == ["か①"]


def test_ust_cp932(tmp_path):
    p = tmp_path / "a.ust"
    p.write_bytes("[#SETTING]\nTempo=120\n[#0000]\nLength=480\nLyric=か①\nNoteNum=60\n"
                  "[#0001]\nLength=480\nLyric=息\nNoteNum=62\n".encode("cp932"))
    notes, _ = nt.read_file(p)
    assert [(n.lyric, n.tone) for n in notes] == [("か①", 60), ("息", 62)]


# ---------------------------------------------------------------- 겹친 음 (read_file)
def _json(path, raw):
    path.write_text(json.dumps({"raw": raw, "tempo": 120}))
    return path


def test_chord_keeps_one_note(tmp_path):
    """같은 때 시작하는 화음은 높은 음 하나만 → 음절이 길이 0인 음표에 들어가지 않는다."""
    raw = [[0, 0.5, 60, 100], [0.5, 1.0, 62, 100], [0.5, 1.0, 59, 100], [1.0, 1.5, 64, 100], [1.5, 2.0, 65, 100]]
    notes, _ = nt.read_file(_json(tmp_path / "c.json", raw))
    assert [n.tone for n in notes] == [60, 62, 64, 65]
    assert all(n.end - n.start >= 0.03 for n in notes)
    notes = nt.align(notes, [list("かきくけ")])
    assert [n.lyric for n in notes] == list("かきくけ")


def test_near_simultaneous_note_keeps_lyric(tmp_path):
    raw = [[0, 0.5, 60, 100], [0.5, 1.0, 59, 100, "き"], [0.51, 1.0, 62, 100], [0.8, 0.81, 70, 100]]
    notes, _ = nt.read_file(_json(tmp_path / "n.json", raw))
    assert [(n.tone, n.lyric) for n in notes] == [(60, ""), (62, "き")]
    assert notes[1].strength == 5.0 and notes[1].end == 1.0  # 너무 짧은 음(70)은 앞 음표를 자르지 않는다


def test_chord_with_lyric_on_upper_note_drops_lower(tmp_path):
    raw = [[0, 0.5, 60, 100], [0.5, 1.0, 59, 100], [0.5, 1.0, 64, 100, "き"], [1.0, 1.5, 65, 100]]
    notes, _ = nt.read_file(_json(tmp_path / "u.json", raw))
    assert [(n.tone, n.lyric) for n in notes] == [(60, ""), (64, "き"), (65, "")]


# ---------------------------------------------------------------- _distribute
def _utility_old(ev, k):
    top = ev[:max(0, k - 1)]
    return sum(e[0] - nt.NEUTRAL for e in top) - 6.0 * max(0, k - 1 - len(ev))


def _distribute_old(groups, total):
    """예전 구현 (G × S × S 파이썬 반복). 새 구현이 똑같은 답을 내는지 견준다."""
    evs = [nt._evidence(g) for g in groups]
    G = len(groups)
    NEG = -1e18
    best = [[NEG] * (total + 1) for _ in range(G + 1)]
    back = [[0] * (total + 1) for _ in range(G + 1)]
    best[0][0] = 0.0
    for g in range(G):
        for used in range(total + 1):
            if best[g][used] == NEG:
                continue
            for k in range(1, total - used + 1):
                v = best[g][used] + _utility_old(evs[g], k)
                if v > best[g + 1][used + k]:
                    best[g + 1][used + k] = v
                    back[g + 1][used + k] = k
    if best[G][total] == NEG:
        return [1 if i < total else 0 for i in range(G)]
    counts, used = [], total
    for g in range(G, 0, -1):
        k = back[g][used]
        counts.append(k)
        used -= k
    return counts[::-1]


def _groups(rng, G, max_notes=6, max_dips=3):
    pick = [1.0, 1.5, 2.5, 3.0, 4.0, 5.0, 10.0]  # 같은 값이 많아야 점수가 같은 경우를 시험한다
    out, t = [], 0.0
    for _ in range(G):
        g = []
        for i in range(rng.randint(1, max_notes)):
            s = 10.0 if i == 0 else (rng.choice(pick) if rng.random() < 0.6 else rng.uniform(1.0, 9.0))
            n = nt.Note(t, t + 0.5, 60, strength=s)
            n.dips = [(t + 0.1 * (j + 1), rng.choice(pick) if rng.random() < 0.5 else rng.uniform(1.5, 8.0))
                      for j in range(rng.randint(0, max_dips))]
            g.append(n)
            t += 0.5
        out.append(g)
    return out


def test_distribute_matches_old():
    rng = random.Random(7)
    for trial in range(400):
        G = rng.randint(0, 7)
        groups = _groups(rng, G)
        total = rng.randint(0, 35) if trial % 4 else rng.randint(0, max(G, 1))  # 음절이 프레이즈보다 적은 경우도
        assert nt._distribute(groups, total) == _distribute_old(groups, total), (trial, G, total)


def test_distribute_fast():
    groups = _groups(random.Random(3), 80, max_notes=8, max_dips=8)
    t = time.perf_counter()
    counts = nt._distribute(groups, 600)
    assert time.perf_counter() - t < 5
    assert sum(counts) == 600 and min(counts) >= 1


# ---------------------------------------------------------------- 음원 폴더
def _bank(folder, aliases, name="重音テト", yaml=None, enc="cp932"):
    folder.mkdir(parents=True)
    (folder / "character.txt").write_bytes(f"name={name}\n".encode(enc))
    oto = "".join(f"{i}.wav={a},0,100,-200,50,20\n" for i, a in enumerate(aliases))
    (folder / "oto.ini").write_bytes(oto.encode(enc))
    if yaml is not None:
        (folder / "character.yaml").write_text(yaml, encoding="utf-8")
    return folder


KANA = "かきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもらりるれろがぎぐげござじずぜぞ"


def test_cp932_oto(tmp_path):
    """①은 shift_jis에 없고 cp932에만 있다: 하나 때문에 파일 전체가 깨지면 안 된다."""
    bank = _bank(tmp_path / "teto①", ["あ", "い", "息", "吸", "あ①"], name="重音テト①")
    info = vb.inspect(bank)
    assert info["name"] == "重音テト①"
    assert (info["breath_in"], info["breath_out"], info["japanese"]) == ("吸", "息", True)
    assert info["phonemizer"] == vb.PRESAMP
    assert info["classic"] is True and info["kind"] == "CV"
    assert [s["name"] for s in vb.list_singers([tmp_path])] == ["重音テト①"]


def test_text_file_encoding(tmp_path):
    bank = _bank(tmp_path / "a", ["あ", "息"], yaml="text_file_encoding: shift_jis\n")
    assert vb.inspect(bank)["breath_out"] == "息"
    bank = _bank(tmp_path / "b", ["あ", "息"], name="한국어", yaml="text_file_encoding: cp949\n", enc="cp949")
    info = vb.inspect(bank)
    assert info["name"] == "한국어" and info["breath_out"] == "息"


def test_diffsinger_is_not_classic(tmp_path):
    _bank(tmp_path / "重音テト", ["あ", "息"])
    _bank(tmp_path / "Kasane Teto DiffSinger", [], name="Kasane Teto DiffSinger", yaml="singer_type: diffsinger\n")
    _bank(tmp_path / "Teto ENUNU", [], name="Teto ENUNU", yaml="singer_type: enunu\n")
    singers = vb.list_singers([tmp_path])
    assert [(s["name"], s["classic"]) for s in singers] == [
        ("重音テト", True), ("Kasane Teto DiffSinger", False), ("Teto ENUNU", False)]
    assert vb.inspect(tmp_path / "Kasane Teto DiffSinger")["classic"] is False


def test_find_prefers_classic(tmp_path, monkeypatch):
    _bank(tmp_path / "Kasane Teto DiffSinger", [], name="Kasane Teto DiffSinger", yaml="singer_type: diffsinger\n")
    _bank(tmp_path / "重音テト", ["あ", "息"], yaml="singer_type: utau\n")
    monkeypatch.setattr(vb, "singer_roots", lambda: [tmp_path])
    singer_id, info = vb.find()
    assert singer_id == "重音テト" and info["classic"] is True


def test_kind(tmp_path):
    cv = [k for k in KANA] + ["あ", "い", "う", "え", "お"]
    heads = ["- " + k for k in KANA]
    vc = [f"{v} {c}" for v in "aiueon" for c in ("k", "s", "t", "n", "h", "m", "r", "g", "z", "sh", "ch", "ts")]
    vv = [f"{v} {k}" for v in "aiueon" for k in "あいうえお"]
    assert vb.inspect(_bank(tmp_path / "cvvc", cv + heads + vc + vv))["kind"] == "CVVC"
    assert vb.inspect(_bank(tmp_path / "cvvc_pitch", cv + heads + [a + "C4" for a in vc]))["kind"] == "CVVC"
    vcv = [f"{v} {k}" for v in "aiueon" for k in KANA]
    assert vb.inspect(_bank(tmp_path / "vcv", heads + vcv + ["a 息", "a R"]))["kind"] == "VCV"
    romaji = [f"{v} {c}a" for v in "aiueon" for c in "kstnhmrgzbp"]
    assert vb.inspect(_bank(tmp_path / "vcv_romaji", romaji))["kind"] == "VCV"
    assert vb.inspect(_bank(tmp_path / "cv", cv + heads))["kind"] == "CV"

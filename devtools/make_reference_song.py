"""정답을 아는 시험 곡 만들기: 음표 · 가사 · 숨 위치를 정해 둔 ustx (OpenUtau로 렌더링해서 '원래 보컬'로 쓴다).
사용: python make_reference_song.py <ustx 경로> <정답 json 경로>"""
import json
import random
import sys

import yaml

random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 7)
TICKS, BPM = 480, 100
vowels = "あいうえお"
scale = [48, 50, 52, 53, 55, 57, 59, 60]  # C3 장음계
phrases = []
pos = TICKS * 2
notes, truth = [], {"bpm": BPM, "notes": [], "breaths": []}
for p in range(4):
    if p:
        # 숨: 다음 프레이즈 앞 0.4박
        b = int(TICKS * 0.6)
        notes.append({"position": pos - b - 60, "duration": b, "tone": 55, "lyric": "吸"})
        truth["breaths"].append([(pos - b - 60) / TICKS * 60 / BPM, (pos - 60) / TICKS * 60 / BPM])
    k = random.randint(2, 5)
    for i in range(7):
        dur = random.choice([240, 240, 480, 480, 720, 960])
        k = max(0, min(len(scale) - 1, k + random.choice([-2, -1, 0, 1, 1, 2])))
        ly = random.choice(vowels)
        notes.append({"position": pos, "duration": dur, "tone": scale[k], "lyric": ly})
        truth["notes"].append([pos / TICKS * 60 / BPM, (pos + dur) / TICKS * 60 / BPM, scale[k], ly])
        pos += dur
    pos += TICKS * 2
vnotes = []
for n in notes:
    long = n["duration"] >= 720 and n["lyric"] != "吸"
    vnotes.append({**n,
                   "pitch": {"data": [{"x": -40, "y": 0, "shape": "io"}, {"x": 40, "y": 0, "shape": "io"}], "snap_first": True},
                   "vibrato": {"length": 60 if long else 0, "period": 180, "depth": 30, "in": 20, "out": 20,
                               "shift": 0, "drift": 0, "vol_link": 0},
                   "phoneme_expressions": [], "phoneme_overrides": []})
proj = {"name": "reference", "comment": "", "output_dir": "Vocal", "cache_dir": "UCache", "ustx_version": "0.6", "key": 0,
        "time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}],
        "tempos": [{"position": 0, "bpm": BPM}],
        "tracks": [{"singer": "TestVowels", "phonemizer": "OpenUtau.Plugin.Builtin.JapanesePresampPhonemizer",
                    "renderer_settings": {"renderer": "WORLDLINE-R"}, "track_name": "ref", "track_color": "Red",
                    "mute": False, "solo": False, "volume": 0, "pan": 0}],
        "voice_parts": [{"name": "ref", "comment": "", "track_no": 0, "position": 0, "duration": pos + TICKS,
                         "notes": vnotes, "curves": []}],
        "wave_parts": []}
open(sys.argv[1], "w", encoding="utf-8").write(yaml.safe_dump(proj, allow_unicode=True, sort_keys=False))
json.dump(truth, open(sys.argv[2], "w"), ensure_ascii=False)
print(len(truth["notes"]), "notes", "".join(n[3] for n in truth["notes"]))

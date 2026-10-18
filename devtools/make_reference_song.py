"""정답을 아는 시험 곡 만들기: 음표 · 가사 · 숨 위치를 정해 둔 ustx (OpenUtau로 렌더링해서 '원래 보컬'로 쓴다).
사용: python make_reference_song.py <ustx 경로> <정답 json 경로> [시드] [--teto] [--lyrics 가사.txt]
  --teto: 설치된 테토 음원으로 부르고(숨은 음원의 숨 별칭), 가사는 모음 대신 か · さ · た … 같은 음절로"""
import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("ustx")
ap.add_argument("truth")
ap.add_argument("seed", nargs="?", type=int, default=7)
ap.add_argument("--teto", action="store_true")
ap.add_argument("--lyrics", help="한 줄에 한 소절씩 가사를 써 둘 파일")
args = ap.parse_args()

random.seed(args.seed)
TICKS, BPM = 480, 100
vowels = "あいうえお"
singer, breath_lyric = "TestVowels", "吸"
if args.teto:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tetotune import voicebank  # noqa: E402

    singer, info = voicebank.find()
    if not singer:
        sys.exit("테토 음원을 찾지 못했어요")
    breath_lyric = info["breath_in"] or info["breath_out"] or "息"
    vowels = list("あいうえおかきくけこさしすせそたてとなにぬねのはひふへほまみむめもらりるれろやゆよわん")
scale = [48, 50, 52, 53, 55, 57, 59, 60]  # C3 장음계
if args.teto:
    scale = [t + 12 for t in scale]  # 테토 음역 (C4 장음계)
phrases = []
pos = TICKS * 2
notes, truth = [], {"bpm": BPM, "notes": [], "breaths": []}
for p in range(4):
    if p:
        # 숨: 다음 프레이즈 앞 0.4박
        b = int(TICKS * 0.6)
        notes.append({"position": pos - b - 60, "duration": b, "tone": scale[4], "lyric": breath_lyric})
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
    long = n["duration"] >= 720 and n["lyric"] != breath_lyric
    vnotes.append({**n,
                   "pitch": {"data": [{"x": -40, "y": 0, "shape": "io"}, {"x": 40, "y": 0, "shape": "io"}], "snap_first": True},
                   "vibrato": {"length": 60 if long else 0, "period": 180, "depth": 30, "in": 20, "out": 20,
                               "shift": 0, "drift": 0, "vol_link": 0},
                   "phoneme_expressions": [], "phoneme_overrides": []})
proj = {"name": "reference", "comment": "", "output_dir": "Vocal", "cache_dir": "UCache", "ustx_version": "0.6", "key": 0,
        "time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}],
        "tempos": [{"position": 0, "bpm": BPM}],
        "tracks": [{"singer": singer, "phonemizer": "OpenUtau.Plugin.Builtin.JapanesePresampPhonemizer",
                    "renderer_settings": {"renderer": "WORLDLINE-R"}, "track_name": "ref", "track_color": "Red",
                    "mute": False, "solo": False, "volume": 0, "pan": 0}],
        "voice_parts": [{"name": "ref", "comment": "", "track_no": 0, "position": 0, "duration": pos + TICKS,
                         "notes": vnotes, "curves": []}],
        "wave_parts": []}
open(args.ustx, "w", encoding="utf-8").write(yaml.safe_dump(proj, allow_unicode=True, sort_keys=False))
json.dump(truth, open(args.truth, "w"), ensure_ascii=False)
if args.lyrics:
    lines = ["".join(n[3] for n in truth["notes"][i:i + 7]) for i in range(0, len(truth["notes"]), 7)]
    Path(args.lyrics).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(truth["notes"]), "notes", "".join(n[3] for n in truth["notes"]))

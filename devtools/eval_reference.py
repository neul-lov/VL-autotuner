"""정답을 아는 시험 곡으로 자동 조교를 채점한다: 음표 수 · 시작 시각 · 음 높이 · 가사 · 숨 위치.
사용: python eval_reference.py <정답 json> <결과 폴더>"""
import json
import sys

import numpy as np

truth = json.load(open(sys.argv[1]))
prev = json.load(open(sys.argv[2] + "/preview.json"))
est = [n for n in prev["notes"]]
syll = [n for n in est if n[3] != "+"]
tn = truth["notes"]
print(f"음표: 정답 {len(tn)} / 만든 음표 {len(est)} (음절 {len(syll)})")
on_err, tone_ok, lyric_ok = [], 0, 0
for s, e, tone, ly in tn:
    j = int(np.argmin([abs(x[0] - s) for x in syll]))
    on_err.append(abs(syll[j][0] - s) * 1000)
    tone_ok += syll[j][2] - prev["shift"] == tone
    lyric_ok += syll[j][3] == ly
on_err = np.array(on_err)
print(f"시작 시각 오차: 중앙 {np.median(on_err):.0f}ms, 90% {np.percentile(on_err, 90):.0f}ms, 50ms 안 {np.mean(on_err < 50):.0%}")
print(f"음 높이 맞음 {tone_ok}/{len(tn)}, 가사 맞음 {lyric_ok}/{len(tn)}")
hit = 0
for a, b in truth["breaths"]:
    hit += any(min(b, y) - max(a, x) > 0.3 * (b - a) for x, y in prev["breaths"])
extra = sum(1 for x, y in prev["breaths"] if not any(min(b, y) - max(a, x) > 0 for a, b in truth["breaths"]))
print(f"숨: 정답 {len(truth['breaths'])}개 중 {hit}개 찾음, 엉뚱한 곳 {extra}개")

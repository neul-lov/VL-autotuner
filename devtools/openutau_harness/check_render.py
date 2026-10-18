"""OpenUtau가 내보낸 트랙별 wav(<이름>_<트랙>.wav)에 소리가 있는지 본다 (재생 없음).
사용: python3 check_render.py <render.wav로 준 경로>   · 모두 소리가 있으면 0, 없거나 빈 파일이 있으면 1"""
import array
import sys
import wave
from pathlib import Path

out = Path(sys.argv[1])
files = sorted(out.parent.glob(f"{out.stem}_*.wav"))
ok = bool(files)
for f in files:
    with wave.open(str(f), "rb") as w:
        samples = array.array("h", w.readframes(w.getnframes()))
        seconds = w.getnframes() / w.getframerate()
    peak = max(map(abs, samples), default=0) / 32768
    print(f"render_check {f.name} {seconds:.1f}s peak={peak:.4f}", file=sys.stderr)
    ok = ok and peak > 1e-3
if not files:
    print(f"render_check {out.stem}_*.wav 없음", file=sys.stderr)
sys.exit(0 if ok else 1)

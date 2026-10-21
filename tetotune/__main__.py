"""명령줄: python -m tetotune 보컬.wav 결과폴더 [--lyrics 가사.txt] [--notes 음표.ustx] ..."""
import argparse
import json
from dataclasses import asdict, fields

from .tune import Options, tune


def main():
    p = argparse.ArgumentParser(prog="tetotune")
    p.add_argument("vocal")
    p.add_argument("out")
    p.add_argument("--lyrics", help="가사 글 파일 (UTF-8)")
    p.add_argument("--notes", help="MIDI · ustx · ust")
    p.add_argument("--backing", help="반주 오디오")
    for f in fields(Options):
        p.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    a = p.parse_args()
    opts = Options(**{f.name: getattr(a, f.name) for f in fields(Options)})
    text = open(a.lyrics, encoding="utf-8").read() if a.lyrics else ""
    if not a.notes:
        from . import game
        try:
            game.ensure_model(print)
        except Exception as e:
            print(f"GAME 모델을 받지 못했어요 (원래 방법으로 음표를 나눠요): {e}")
    res = tune(a.vocal, a.out, text, a.notes, a.backing, opts, progress=lambda v, s: print(f"{v:4.0%} {s}", flush=True))
    if opts.refine != "off":
        from . import refine
        w = refine.for_result(a.out, lambda v, s: print(f"다듬기 {v:4.0%} {s}", flush=True))
        if w:
            res.warnings.append(w)
    print(json.dumps(asdict(res), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

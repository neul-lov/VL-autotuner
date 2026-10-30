"""테토 렌더 보코더 미세 조정 실험 (CI Mac 한 작업 안에서만. 아무것도 올리지 않는다).

테토 음원 녹음으로 신경망 렌더 보코더(pc-nsf-hifigan 2025.02, OpenUtau hifisampler가 쓰는 것)를 조금 더 배워서
같은 프로젝트를 원래 보코더 · 테토 보코더로 렌더해 견준다 (vocoder_ab.py). 테토 약관은 테토 목소리로 학습한 것의
배포를 금지해서, 배운 모델 · 그 렌더는 이 기계에만 두고 숫자만 찍는다 (저장소 · 캐시 · 결과 파일 어디에도 넣지 않는다).

SingingVocoders(openvpi, 학습 코드)를 --sv 폴더에 받아 두고 그 코드를 그대로 쓴다. 바꾸는 것은 설정뿐:
공식 미세 조정 설정(configs/ft_hifigan.yaml)에 CPU · 시간 한도 · 배울 파일 섞기 · MPD 고정(README 권장)을 더한다.
내보내기는 DiffSinger 내보내기(NSFHiFiGANONNX)와 같은 모양 (mel [1, T, 128] · f0 [1, T] → waveform [1, T·512]).

사용:
  python vocoder_ft.py prep --sv SingingVocoders --voicebank 테토폴더 --data 작업폴더 --pretrained 미세조정용.ckpt
  python vocoder_ft.py probe --sv SingingVocoders --data 작업폴더 --out 고른것.txt [--min-rate 0.3]
      (LADDER 설정을 실제 학습 몇 걸음씩 재서 메모리에 맞는 첫 설정을 고른다)
  python vocoder_ft.py train --sv SingingVocoders --data 작업폴더 --plan 고른것.txt --until 유닉스초
  python vocoder_ft.py export --sv SingingVocoders --ckpt 학습결과.ckpt --out 이름.onnx
  python vocoder_ft.py check a.onnx b.onnx --data 작업폴더 [--require-snr 40] (같은 입력에 두 ONNX가 얼마나 같은지)
  python vocoder_ft.py copysyn 이름=a.onnx 이름=b.onnx --data 작업폴더 [--json out.json]
      배우지 않은 테토 녹음(검증용 10개)을 다시 만들어 원래와 견주고, 음 높이를 바꿔 불러도 따라가는지 잰다
"""
import argparse
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path

from ci_status import post as progress  # devtools/ (이 파일 옆)

CONFIG = "configs/teto_ft.yaml"
NAME = "teto_ft"
HOP, SR = 512, 44100
SNAP = 250  # 학습 중 생성기를 따로 남기는 걸음 (SNAP · 2 SNAP · 4 SNAP …): 얼마나 배워야 나아지는지 보려고
LADDER = [(2, 16, True), (4, 16, True), (4, 16, False), (4, 32, True), (4, 32, False)]
# 배울 설정 (배치, 조각 길이 mel 틀, 생성기 중간값 다시 계산하기). 앞일수록 메모리를 덜 쓴다.
# probe는 앞에서부터 올라가며 재고 (메모리 오류 · 디스크 넘침이 나면 멈춘다) 그중 가장 빠른 것을 고르고,
# train은 메모리가 모자라면 한 칸 앞으로 내려가 이어서 배운다.
# 공식 미세 조정 설정(배치 10 · 32틀)은 7GB CI 맥에 안 맞는다: 배치 4 · 32틀도 GPU 메모리 한도 없이 배우니 7.2GB를 넘겨
# 멈췄다 (CPU 프로세스 최대 메모리로 재면 배치 4 · 32틀 7.8GB, 4 · 16틀 5.7GB, 다시 계산하면 5.0GB, 2 · 16틀 4.2GB,
# 설정과 상관없는 몫이 3.4GB쯤). 판별기 중간값이 많아서 생성기만 다시 계산하면 3할쯤 준다.
# pc 증강(음 높이를 바꿔 부르는 학습, 배치보다 생성기를 1.5배 더 돌린다)은 hifisampler가 바로 그렇게 쓰므로 끄지 않는다.
# 14GB 맥(권장 9.3GB)에서 실제로 재면 GPU 드라이버 메모리가 2 · 16틀 다시 계산 4.4GB, 4 · 16틀 다시 계산 5.3GB, 4 · 16틀 6.2GB.
# 7GB 맥(권장 4.7GB)은 가장 작은 설정도 한도 4.2GB를 넘어서(4.3GB) 워크플로가 처음에 멈추고 다시 돈다.
MPS_LIMIT = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.9", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.7"}
# 맥 GPU가 권장 메모리의 0.9배(7GB 기계에서 4.2GB쯤)보다 더 쓰려 하면 바로 메모리 오류가 나게 한다: 그 위로는
# 디스크로 넘쳐 한 걸음이 몇 분씩 걸렸다.
GPU_ERROR = ("command buffer exited with error status", "kIOGPUCommandBufferCallbackError")
# 맥 GPU(Metal) 오류 줄. 한 번 나면 GPU가 그 프로세스의 계산을 무시해서 손실이 0이 되고 가중치가 망가지는데 학습은
# 멈추지 않는다 (2026-10-30 실행: 30분쯤 배운 뒤 오류가 나서 4시간을 헛배우고 소리 없는 보코더가 나왔다). 그래서 보이면
# 바로 멈추고, 그 뒤로 손실이 멀쩡한 것을 본(확인된) 저장점에서 새 프로세스로 이어서 배운다.
BAD_LOSS = 3  # train1이 손실이 0 · NaN인 것을 보면 이 종료 코드로 끝난다
CONFIRM = 30  # 저장점 · 생성기를 남기고 이 초가 지나 손실이 멀쩡하면 확인된 것으로 친다


def enter(sv):
    """SingingVocoders 코드는 자기 폴더를 기준으로 설정 · 결과 경로를 쓴다 (configs/ · experiments/)."""
    sv = Path(sv).resolve()
    os.chdir(sv)
    sys.path.insert(0, str(sv))
    os.environ.setdefault("MPLBACKEND", "Agg")
    return sv


def write_config(data, pretrained, device, batch, val, freeze_mpd, workers):
    import yaml

    data = Path(data).resolve()
    cfg = {
        "base_config": ["configs/ft_hifigan.yaml"],
        "data_input_path": [str(data / "wavs")],
        "data_out_path": [str(data / "npz")],
        "DataIndexPath": str(data / "index"),
        "val_num": 10,
        "finetune_ckpt_path": str(Path(pretrained).resolve()),
        "pl_trainer_accelerator": device,
        "pl_trainer_devices": 1,
        "batch_size": batch,
        "ds_workers": workers,
        "val_check_interval": val,
        "num_ckpt_keep": 1,  # 하나에 1GB쯤 (판별기 · 옵티마이저까지)
        "num_sanity_val_steps": 1,
        "log_interval": 50,
        "max_updates": 1000000,  # 시간 한도로 멈춘다
        "freezing_enabled": bool(freeze_mpd),
        "frozen_params": ["discriminator.mpd"] if freeze_mpd else [],
    }
    Path(CONFIG).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("설정:", json.dumps({k: cfg[k] for k in ("pl_trainer_accelerator", "batch_size", "val_check_interval",
                                                "freezing_enabled")}, ensure_ascii=False), flush=True)


def prep(args):
    """음원 폴더의 wav → 작업폴더/wavs (링크) → SingingVocoders 전처리 (mel · f0 npz) → 배울 순서를 섞는다."""
    import subprocess

    import soundfile as sf

    data = Path(args.data).resolve()
    src = Path(args.voicebank).resolve()
    (data / "wavs").mkdir(parents=True, exist_ok=True)
    rates, secs, n = {}, 0.0, 0
    for i, w in enumerate(sorted(src.rglob("*.wav"))):
        try:
            info = sf.info(str(w))
        except RuntimeError:
            continue
        rates[info.samplerate] = rates.get(info.samplerate, 0) + 1
        if info.samplerate < 44100 or info.frames < 512 * 40:  # 44.1kHz 아래는 전처리가 버린다 · 0.5초 아래는 자를 곳이 없다
            continue
        link = data / "wavs" / f"{i:05d}.wav"
        if not link.exists():
            os.symlink(w, link)
        secs += info.frames / info.samplerate
        n += 1
    folders = sorted({str(w.parent.relative_to(src)) for w in src.rglob("*.wav")})
    print(f"테토 녹음 {n}개 · {secs / 3600:.2f}시간 (표본율 {rates}), 폴더 {len(folders)}개", flush=True)
    sv = enter(args.sv)
    write_config(data, args.pretrained, args.device, args.batch, args.val, not args.train_mpd, args.workers)
    t = time.time()
    subprocess.run([sys.executable, "process.py", "--config", CONFIG, "--num_cpu", str(os.cpu_count() or 3), "--strx", "1"],
                   cwd=sv, check=True, env={**os.environ, "TQDM_MININTERVAL": "30"})
    train_idx = data / "index" / "train"
    lines = [x for x in train_idx.read_text(encoding="utf-8").splitlines() if x.strip()]
    random.Random(7).shuffle(lines)  # 데이터로더가 섞지 않아서 (폴더 순서대로 배우지 않게)
    train_idx.write_text("\n".join(lines) + "\n", encoding="utf-8")
    valid = [x for x in (data / "index" / "valid").read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"전처리 {time.time() - t:.0f}초: 배울 것 {len(lines)}개 · 검증 {len(valid)}개", flush=True)
    progress("보코더 실험", f"준비: 테토 녹음 {n}개 {secs / 3600:.2f}시간, 배울 것 {len(lines)} · 검증 {len(valid)}")


def describe(plan):
    b, crop, rec = plan
    return f"배치 {b} · {crop}틀" + (" · 다시 계산" if rec else "")


def swap_mb():
    """맥이 메모리를 디스크로 넘긴 양(스왑) MB. 모르면 0."""
    import re
    import subprocess

    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def error_line(tail):
    err = [x for x in tail if "Error" in x] or tail or ["?"]
    return err[-1][:200]


def why(rc):
    """따로 띄운 프로세스가 어떻게 끝났는지."""
    import signal

    if rc is None:
        return "시간 안에 못 끝냄"
    if rc < 0:
        try:
            return f"신호 {signal.Signals(-rc).name}로 꺼짐"
        except ValueError:
            return f"신호 {-rc}로 꺼짐"
    return f"종료 코드 {rc}"


def brief(tail, n=12):
    """따로 띄운 프로세스 기록의 끝 몇 줄 (설정 · 모델 구조를 찍은 줄은 빼고)."""
    keep = [x for x in tail if x.strip() and not x.startswith((" ", "\t", "|")) and x.strip() != ")"
            and "\x1b[0;33m" not in x]
    return keep[-n:]


def gpu_reason(tail):
    """맥 GPU 오류 줄의 까닭 (예: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory))."""
    import re

    for x in tail:
        m = re.search(r"^\s*(.*?)\s*\((\w+:kIOGPU\w+)\)", x)
        if m:
            return f"{m.group(1)} ({m.group(2)})".strip()
    return "까닭 줄 없음"


def run_child(args, plan, extra, timeout, quiet=False, limit=True):
    """train1을 따로 띄운 프로세스에서 돌린다 (GPU 메모리 한도 MPS_LIMIT를 걸고, 죽으면 파이썬 위치를 찍게,
    os._exit로 끝나도 기록이 남게 버퍼 없이). 맥 GPU 오류 줄(GPU_ERROR)이 보이면 까닭 줄까지 받고 3초 뒤 멈춘다.
    (종료 코드, 마지막 줄들, GPU 오류가 처음 보인 시각 또는 None); timeout초가 지나 멈췄으면 종료 코드가 None."""
    import collections
    import subprocess
    import threading

    b, crop, rec = plan
    cmd = [sys.executable, "-X", "faulthandler", "-W", "ignore", os.path.abspath(__file__), "train1", "--sv", str(args.sv),
           "--data", str(args.data), "--batch", str(b), "--crop", str(crop), *(["--recompute"] if rec else []), *extra]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **(MPS_LIMIT if limit else {})}
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    fired = threading.Event()

    def stop():
        fired.set()
        p.kill()

    timer = threading.Timer(timeout, stop)
    timer.daemon = True
    timer.start()
    tail, gpu, kill = collections.deque(maxlen=300), None, None
    for line in p.stdout:
        tail.append(line.rstrip("\n"))
        if not quiet:
            print(line, end="", flush=True)
        if gpu is None and any(s in line for s in GPU_ERROR):
            gpu = time.time()
            kill = threading.Timer(3, p.kill)
            kill.daemon = True
            kill.start()
    p.wait()
    timer.cancel()
    if kill:
        kill.cancel()
    return (None if fired.is_set() else p.returncode), list(tail), gpu


def probe(args):
    """LADDER를 작은 설정부터 실제 학습 몇 걸음씩 돌려 보며 올라간다 (따로 띄운 프로세스, GPU 메모리 한도를 걸고).
    메모리에 맞고 디스크로 넘치지 않은 것 중 1초에 녹음을 가장 많이 배우는 설정을 고른다. 그게 --min-rate초보다 짧으면
    실패한다: 느린 맥에 걸리면 몇 시간 배워도 조금뿐이라, 잡을 다시 돌려 다른 맥에서 돈다.
    CI 맥은 같은 설정도 기계마다 10배 넘게 달랐다 (배치 4 · 32틀 한 걸음 1.7초 · 15초 · 39초).
    이 프로세스는 torch를 올리지 않는다 (메모리를 재는 학습 프로세스 몫을 덜 뺏게)."""
    import shutil

    print(f"메모리 {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30:.1f}GB, 스왑 {swap_mb():.0f}MB, "
          f"GPU 메모리 한도 {MPS_LIMIT}", flush=True)
    best, limit, done = None, True, []
    for i, plan in enumerate(LADDER):
        for again in (False, True):
            rc, tail, gpu = run_child(args, plan, ["--probe", "6"], timeout=600, quiet=True, limit=limit)
            shutil.rmtree(Path(args.sv) / "experiments" / "teto_probe", ignore_errors=True)
            line = next((x for x in tail[::-1] if x.startswith('{"sec"')), None)
            if line and not rc and not gpu:
                break
            how = f"맥 GPU 오류: {gpu_reason(tail)}" if gpu else why(rc)
            print(f"{describe(plan)}{'' if limit else ' (메모리 한도 없이)'}: 못 씀 ({how}), 기록 끝:", flush=True)
            for x in brief(tail):
                print("    " + x[:300], flush=True)
            if again or not limit or rc is None or gpu or any("out of memory" in x for x in tail):
                break
            limit = False  # 메모리 오류 말고 다른 까닭으로 죽었으면, 한도가 문제인지 한도 없이 한 번 더 본다
        if not line or rc or gpu:
            break
        r = {**json.loads(line), "index": i, "limit": limit}
        r["rate"] = plan[0] * plan[1] * HOP / SR / r["sec"]
        print(f"{describe(plan)}: 한 걸음 {r['sec']:.2f}초 (처음 {r['first']:.0f}초), 1초에 녹음 {r['rate']:.2f}초, "
              f"GPU {r['mps_gb']:.1f}GB (드라이버 {r['driver_gb']:.1f}GB, 권장 {r['rec_gb']:.1f}GB), "
              f"스왑 {r['swap_mb']:+.0f}MB", flush=True)
        if r["swap_mb"] >= 300:
            print("디스크로 넘쳐서 더 큰 설정은 재지 않아요", flush=True)
            best = best or r
            break
        best = r
        done.append(r)
    if best is None:
        progress("보코더 실험", f"가장 작은 설정({describe(LADDER[0])})으로도 몇 걸음을 못 배움")
        raise SystemExit("가장 작은 설정으로도 몇 걸음을 못 배웠어요")
    best = max((r for r in done if r["swap_mb"] < 300), key=lambda r: r["rate"], default=best)  # 가장 빠른 것
    plan = LADDER[best["index"]]
    print(f"고른 설정: {describe(plan)}", flush=True)
    progress("보코더 실험", f"고른 설정 {describe(plan)}: 한 걸음 {best['sec']:.1f}초, 1초에 녹음 {best['rate']:.2f}초, "
             f"GPU {best['mps_gb']:.1f}GB")
    if args.min_rate and best["rate"] < args.min_rate:
        raise SystemExit(f"이 맥은 1초에 녹음 {best['rate']:.2f}초어치밖에 못 배워요 (기준 {args.min_rate}). "
                         "잡을 다시 돌리면 다른 맥에서 돌아요")
    Path(args.out).write_text(f"{best['index']} {'limit' if best['limit'] else 'nolimit'}\n", encoding="utf-8")


def book_of(work):
    """저장점마다 배운 걸음 · 조각 수 · 녹음 길이(초) · 남긴 시각 · 확인됐는지 (experiments/teto_ft/done.json)."""
    try:
        return json.loads((Path(work) / "done.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_book(work, book):
    tmp = Path(work) / "done.json.tmp"
    tmp.write_text(json.dumps(book), encoding="utf-8")
    tmp.replace(Path(work) / "done.json")


def confirm_snaps(work, before=float("inf")):
    """학습 중 남긴 생성기 snap/gen_N.pt.tmp → gen_N.pt (before 시각보다 먼저 남긴 것만)."""
    for q in (Path(work) / "snap").glob("*.pt.tmp"):
        if q.stat().st_mtime < before:
            q.replace(q.with_suffix(""))


def drop_unconfirmed(work):
    """GPU 오류 · 손실 이상으로 멈춘 뒤: 확인되지 않은 저장점 · 생성기 · final.ckpt를 지운다 (망가졌을 수 있어서).
    다음 프로세스는 남은 저장점 중 가장 늦은 것에서 (없으면 미세 조정 시작점에서) 배운다."""
    work = Path(work)
    book = book_of(work)
    for q in work.glob("model_ckpt_steps_*.ckpt"):
        if not book.get(q.name, {}).get("ok"):
            q.unlink()
            book.pop(q.name, None)
            print(f"확인 안 된 저장점 지움: {q.name}", flush=True)
    write_book(work, book)
    for q in [*(work / "snap").glob("*.tmp"), *(work / "snap").glob("*.part"), *work.glob("*.part")]:
        q.unlink()
    (work / "final.ckpt").unlink(missing_ok=True)


def ckpt_ok(path):
    """저장점의 생성기 가중치가 모두 유한하고, 절반 넘게 0뿐이지는 않은지 (CPU에서 읽어서)."""
    import torch

    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:  # 없거나 덜 써진 파일
        print(f"{Path(path).name}: 못 읽음 ({type(e).__name__}: {str(e)[:100]})", flush=True)
        return False
    gen = [v for k, v in ck.get("state_dict", {}).items() if k.startswith("generator.") and v.is_floating_point()]
    bad = sum(1 for v in gen if not bool(torch.isfinite(v).all()))
    zero = sum(1 for v in gen if not bool(v.abs().sum() > 0))
    ok = bool(gen) and not bad and zero <= len(gen) // 2
    print(f"{Path(path).name}: 생성기 텐서 {len(gen)}개 중 유한하지 않은 것 {bad}개 · 0뿐인 것 {zero}개 → "
          f"{'멀쩡' if ok else '망가짐'}", flush=True)
    return ok


def train(args):
    """probe가 고른 설정으로 배운다 (따로 띄운 프로세스). 저장점은 --ckpt-every초마다 남기고, 그 뒤 손실이 멀쩡하면
    확인된 것으로 친다. 도중에 GPU 메모리가 모자라거나 프로세스가 죽으면 LADDER의 한 칸 작은 설정으로 마지막
    저장점에서, 맥 GPU 오류 · 손실 이상이면 확인 안 된 것을 지우고 같은 설정으로 한 번 더 (같은 설정에서 또 나면 한 칸
    작은 설정으로) 확인된 저장점에서 이어서 배운다. --until이 되면 final.ckpt를 남기고 끝난다. 이어서 배울 시간이
    없으면 멀쩡한 마지막 저장점을 final.ckpt로 쓴다."""
    import shutil

    i, how = Path(args.plan).read_text(encoding="utf-8").split()[:2]
    i, limit = int(i), how == "limit"
    work = Path(args.sv) / "experiments" / NAME
    fake_oom, fake_gpu, first, gpu_errors = args.fake_oom, args.fake_gpu, True, {}
    while True:
        left = args.until - time.time()
        if left < (300 if first else 90):
            if first:
                raise SystemExit(f"배울 시간이 모자라요 ({left:.0f}초)")
            break
        extra = ["--until", str(args.until), "--ckpt-every", str(args.ckpt_every)]
        if fake_oom and i > 0:  # 짧은 시험: 이어서 배우기가 되는지 한 번 일부러 멈춰 본다
            extra += ["--fake-oom", str(fake_oom)]
            fake_oom = 0
        elif fake_gpu and not first:  # 짧은 시험: 그다음 프로세스는 맥 GPU 오류 줄을 일부러 찍어 본다
            extra += ["--fake-gpu", str(fake_gpu)]
            fake_gpu = 0
        first = False
        print(f"배울 설정: {describe(LADDER[i])}, 남은 {left / 60:.0f}분", flush=True)
        rc, tail, gpu = run_child(args, LADDER[i], extra, timeout=min(left + 1800, 86400), limit=limit)
        bad = gpu is not None or rc == BAD_LOSS
        if rc == 0 and not bad:
            if ckpt_ok(work / "final.ckpt"):
                confirm_snaps(work)
                for q in work.glob("model_ckpt_steps_*.ckpt"):  # 하나에 1GB쯤, final.ckpt만 남긴다
                    q.unlink()
                return
            bad = True  # 오류 줄 없이 마지막 가중치가 망가졌으면 확인된 저장점으로
        oom = not bad and any("out of memory" in x for x in tail)
        if not (oom or bad) and (rc is None or rc > 0):
            raise SystemExit(f"학습이 멈췄어요 ({why(rc)}: {error_line(tail)})")
        if bad:
            drop_unconfirmed(work)
            gpu_errors[i] = gpu_errors.get(i, 0) + 1
            again = gpu_errors[i] < 2
            how = (f"맥 GPU 오류 ({gpu_reason(tail)})" if gpu else "손실 이상 (0 · NaN)" if rc == BAD_LOSS
                   else "마지막 가중치가 망가짐")
            then = "같은 설정으로 한 번 더" if again else "한 칸 작은 설정으로"
            print(f"::warning::보코더 학습 {describe(LADDER[i])}: {how}, 확인된 저장점에서 {then}", flush=True)
            progress("보코더 실험", f"{describe(LADDER[i])}: {how[:80]}, 확인된 저장점에서 {then}")
            if again:
                continue
        else:
            how = "GPU 메모리가 모자라" if oom else why(rc)
            print(f"{describe(LADDER[i])}: {how} 멈춤", flush=True)
            if not oom:
                for x in brief(tail):
                    print("    " + x[:300], flush=True)
            confirm_snaps(work)  # 메모리 오류 · 프로세스가 죽은 것은 가중치를 망가뜨리지 않는다
            progress("보코더 실험", f"{describe(LADDER[i])}: {how} 멈춤, 한 칸 작은 설정으로 이어서")
        if i == 0:
            break
        i -= 1
    saved = sorted(work.glob("model_ckpt_steps_*.ckpt"), key=lambda q: int(q.stem.rsplit("_", 1)[1]), reverse=True)
    good = next((q for q in saved if ckpt_ok(q)), None)
    if good is None:
        raise SystemExit("학습이 멈췄고, 멀쩡한 저장점도 없어요")
    shutil.copy(good, work / "final.ckpt")
    done = book_of(work).get(good.name, {})
    summary = (f"학습 끝 (마지막 저장점 {good.name}): {done.get('steps', '?')}걸음, "
               f"배운 녹음 {done.get('audio', 0) / 60:.0f}분어치")
    print(summary, flush=True)
    progress("보코더 실험", summary)


def train1(args):
    """probe · train이 따로 띄우는 것: SingingVocoders train.py를 설정 하나로 맥 GPU(MPS)에서 돌린다.
    experiments/teto_ft에 저장점이 있으면 거기서 이어서 배운다. --probe N이면 experiments/teto_probe에서
    N걸음(처음 두 걸음 빼고)만 재서 JSON 한 줄을 찍고 끝난다 (저장하지 않는다)."""
    enter(args.sv)
    import lightning.pytorch as pl
    import torch
    import yaml
    from lightning.pytorch.callbacks import Callback, ProgressBar

    name = "teto_probe" if args.probe else NAME
    dev = "mps" if torch.backends.mps.is_available() else "cpu"  # CPU는 이 코드를 맥 밖에서 시험할 때만
    cfg = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
    cfg.update({"pl_trainer_accelerator": dev, "batch_size": args.batch, "crop_mel_frames": args.crop,
                # 검증은 녹음을 통째로 넣는데 MPS conv1d는 길이 65536을 못 넘는다. 배우지 않은 녹음 채점은 copysyn이 한다
                "num_sanity_val_steps": 0, "val_check_interval": 10 ** 9})
    config = f"configs/{name}.yaml"
    Path(config).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    work = Path("experiments") / name
    from utils.training_utils import get_latest_checkpoint_path

    resume = get_latest_checkpoint_path(work)
    done = book_of(work).get(Path(resume).name, {}) if resume else {}
    n_train = sum(1 for x in (Path(args.data) / "index" / "train").read_text(encoding="utf-8").splitlines() if x.strip())
    plan = (args.batch, args.crop, args.recompute)
    rec_gb = torch.mps.recommended_max_memory() / 2 ** 30 if dev == "mps" else 0.0
    print(f"설정: {describe(plan)}, torch {torch.__version__} {dev} (권장 메모리 {rec_gb:.1f}GB)"
          + (f", 이어서 배움: {Path(resume).name}" if resume else ""), flush=True)
    _load = torch.load

    def load_cpu(*a, **k):  # 미세 조정 시작점은 CUDA에서 저장돼 있어서 (맥에는 CUDA가 없다) CPU로 먼저 읽는다
        k.setdefault("map_location", "cpu")
        return _load(*a, **k)

    torch.load = load_cpu
    from training.base_task_gan import GanBaseTask

    load = GanBaseTask.load_finetune_ckpt

    def load_checked(self, state_dict):  # strict=False로 불러서, 이름이 안 맞으면 처음부터 배우게 된다
        have = set(self.state_dict())  # 생성기 · 판별기 둘 다 맞아야 한다 (판별기가 처음부터면 GAN이 생성기를 망친다)
        miss = [k for k in have - set(state_dict) if k.startswith(("generator.", "discriminator."))]
        print(f"미세 조정 시작점: 맞는 가중치 {len(have & set(state_dict))}개 · 모델에만 {len(have - set(state_dict))}개 "
              f"· 체크포인트에만 {len(set(state_dict) - have)}개", flush=True)
        if miss or not any(k.startswith("generator.") for k in state_dict):
            raise SystemExit(f"미세 조정 시작점에 없는 생성기 · 판별기 가중치 {len(miss)}개 (예: {miss[:3]})")
        return load(self, state_dict)

    GanBaseTask.load_finetune_ckpt = load_checked

    def train_loader(self):  # 일꾼 프로세스 없이 (일꾼마다 torch를 따로 올려 메모리를 먹는다. 한 걸음 읽기는 몇십 ms)
        return torch.utils.data.DataLoader(self.train_dataset, collate_fn=self.train_dataset.collater,
                                           batch_size=self.config["batch_size"], shuffle=True, num_workers=0)

    GanBaseTask.train_dataloader = train_loader
    if args.recompute:  # 생성기 중간값을 저장하지 않고 거꾸로 갈 때 다시 계산한다 (메모리가 훨씬 적고 계산이 3할쯤 늘어난다)
        from torch.utils.checkpoint import checkpoint

        from models.nsf_HiFigan.models import ResBlock1

        forward = ResBlock1.forward

        def recompute(self, x):
            if torch.is_grad_enabled() and x.requires_grad:
                return checkpoint(forward, self, x, use_reentrant=False)
            return forward(self, x)

        ResBlock1.forward = recompute
    clip = args.crop * HOP / SR  # 한 조각 녹음 길이 (초)

    def save(trainer, path):  # 쓰다가 꺼져도 반쪽 저장점이 남지 않게 (이어서 배울 때 그걸 읽다 멈춘다)
        part = path.with_name(path.name + ".part")
        trainer.save_checkpoint(str(part))
        part.replace(path)

    class Log(Callback):  # CI 기록에 50걸음마다 한 줄 (진행 막대 대신), 커밋 상태에 10분마다, 저장점은 --ckpt-every초마다
        def __init__(self):
            self.t0 = self.saved = time.time()
            self.last, self.times, self.said, self.mem, self.swap = None, [], 0.0, [0, 0], 0.0
            # 이어서 배우면 저장점까지 배운 걸음 · 조각 수 · 녹음 길이(초)부터 센다
            self.steps, self.items, self.audio = done.get("steps", 0), done.get("items", 0), done.get("audio", 0.0)
            self.faked = False

        def on_train_start(self, trainer, pl_module):  # 준비 시간은 빼고 잰다
            self.last = time.time()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            now = time.time()
            self.times.append(now - self.last)
            self.last = now
            self.steps += 1
            self.items += args.batch
            self.audio += args.batch * clip
            if dev == "mps":
                self.mem = [max(self.mem[0], torch.mps.current_allocated_memory()),
                            max(self.mem[1], torch.mps.driver_allocated_memory())]
            if len(self.times) == 2:
                self.swap = swap_mb()
            if args.probe:
                timed = self.times[2:]  # 처음 두 걸음은 GPU 준비(셰이더 만들기)라 뺀다
                if len(timed) >= args.probe or (len(timed) >= 2 and sum(timed) > 150):
                    print(json.dumps({"sec": sorted(timed)[len(timed) // 2], "first": self.times[0],
                                      "mps_gb": self.mem[0] / 2 ** 30, "driver_gb": self.mem[1] / 2 ** 30,
                                      "rec_gb": rec_gb, "swap_mb": swap_mb() - self.swap}), flush=True)
                    os._exit(0)
                return
            n = self.steps
            sec = sorted(self.times[-20:])[len(self.times[-20:]) // 2]
            if n % 50 == 0:
                loss = trainer.callback_metrics.get("loss")
                print(f"걸음 {n} · {now - self.t0:.0f}초 · 한 걸음 {sec:.2f}초 · GPU {self.mem[0] / 2 ** 30:.1f}GB"
                      + (f" · 손실 {float(loss):.4f}" if loss is not None else ""), flush=True)
                self.check(trainer, now)
            if now - self.saved > args.ckpt_every:
                self.saved = now
                self.check(trainer, now)
                path = work / f"model_ckpt_steps_{trainer.global_step}.ckpt"
                save(trainer, path)
                book = book_of(work)
                book[path.name] = {"steps": n, "items": self.items, "audio": self.audio, "time": time.time(), "ok": False}
                write_book(work, book)
                print(f"걸음 {n}: 저장점 {path.name} ({time.time() - now:.0f}초)", flush=True)
            if n % SNAP == 0 and (n // SNAP) & (n // SNAP - 1) == 0:  # 250 · 500 · 1000 · 2000 … 걸음
                self.snapshot(pl_module, n)
            if args.fake_oom and now - self.t0 > args.fake_oom:
                raise RuntimeError("MPS backend out of memory (연습: 이어서 배우기 시험)")
            if args.fake_gpu and not self.faked and now - self.t0 > args.fake_gpu:
                self.faked = True  # 맥 GPU 오류가 날 때 찍히는 줄과 같게 (train이 보고 이 프로세스를 멈추는지 시험)
                print("Error: command buffer exited with error status.\n\tThe Metal Performance Shaders operations "
                      "encoded on it may not have completed.\n\tError: \n\t(null)\n\tInsufficient Memory "
                      "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory) (연습)", file=sys.stderr, flush=True)
            if now > args.until:
                trainer.should_stop = True
            if len(self.times) == 5 or now - self.said > 600:
                self.said = now
                mem = (f" · GPU {torch.mps.current_allocated_memory() / 2 ** 30:.1f}GB"
                       f" (드라이버 {torch.mps.driver_allocated_memory() / 2 ** 30:.1f}GB)" if dev == "mps" else "")
                progress("보코더 실험", f"학습 걸음 {n} · 한 걸음 {sec:.1f}초 ({describe(plan)}){mem}, "
                         f"남은 {(args.until - now) / 60:.0f}분")

        def check(self, trainer, now):
            """손실이 0 · NaN이면 GPU가 계산을 무시하고 있다 (맥 GPU 오류 뒤): 바로 끝낸다 (BAD_LOSS).
            멀쩡하면 CONFIRM초 넘게 지난 저장점 · 생성기를 확인된 것으로 치고, 그보다 앞선 저장점은 지운다 (하나에 1GB쯤)."""
            loss = trainer.callback_metrics.get("loss")
            if loss is None:
                return
            loss = float(loss)
            if not math.isfinite(loss) or abs(loss) < 1e-6:
                print(f"손실 이상 ({loss}): GPU가 계산을 무시하고 있어요", flush=True)
                sys.stderr.flush()
                os._exit(BAD_LOSS)
            book = book_of(work)
            new = [k for k, v in book.items() if not v.get("ok") and now - v["time"] > CONFIRM]
            if new:
                for k in new:
                    book[k]["ok"] = True
                last = max(book[k]["steps"] for k in new)
                for q in work.glob("model_ckpt_steps_*.ckpt"):
                    if book.get(q.name, {}).get("ok") and book[q.name]["steps"] < last:
                        q.unlink()
                        book.pop(q.name)
                write_book(work, book)
            confirm_snaps(work, before=now - CONFIRM)

        def snapshot(self, pl_module, n):  # 생성기만 따로 남긴다 (60MB쯤): 학습 뒤 trend가 걸음마다 얼마나 나아졌는지 잰다
            try:  # 확인되면 (check) .tmp를 뗀다
                (work / "snap").mkdir(exist_ok=True)
                gen = {k[len("generator."):]: v.detach().cpu() for k, v in pl_module.state_dict().items()
                       if k.startswith("generator.")}
                part = work / "snap" / f"gen_{n:06d}.pt.part"  # 쓰다가 꺼져도 반쪽 파일이 남지 않게
                torch.save({"generator": gen}, str(part))
                part.replace(part.with_suffix(".tmp"))
            except Exception as e:  # 덤이라 학습은 멈추지 않는다
                print(f"걸음 {n}: 생성기 따로 남기기 실패 ({type(e).__name__}: {e})", flush=True)

        def on_exception(self, trainer, pl_module, exception):  # 정리 단계가 멈추지 않게 바로 끝낸다 (실패로)
            import traceback

            traceback.print_exception(exception)
            if not args.probe:
                progress("보코더 실험", f"학습 멈춤: {type(exception).__name__}: {str(exception)[:90]}")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

        def on_train_end(self, trainer, pl_module):
            if args.probe:
                print(f"잴 걸음을 다 하기 전에 학습이 끝났어요 ({len(self.times)}걸음)", flush=True)
                os._exit(1)
            self.check(trainer, time.time())
            save(trainer, work / "final.ckpt")
            n = self.steps
            sec = sorted(self.times[-50:])[len(self.times[-50:]) // 2] if self.times else 0.0
            print(f"배운 걸음 {n} → {work / 'final.ckpt'}", flush=True)
            summary = (f"학습 끝: {n}걸음, 마지막 설정 {describe(plan)} · 한 걸음 {sec:.1f}초, "
                       f"배운 녹음 {self.audio / 60:.0f}분어치 (학습 파일 {self.items / n_train:.1f}바퀴)")
            print(summary, flush=True)
            progress("보코더 실험", summary)
            sys.stderr.flush()
            os._exit(0)  # 저장했으면 바로 끝낸다: 학습 뒤 정리(데이터로더 일꾼 · 공유 메모리)에서 한 시간 넘게 멈춘 적이 있다

    class Trainer(pl.Trainer):  # train.py가 만드는 Trainer에 기록만 더한다 (isinstance가 되게 물려받아서)
        def __init__(self, *a, **k):
            k["limit_val_batches"] = 0  # 멈출 때 한 번 하는 검증도 끈다 (녹음 통째로 넣는 검증은 MPS에서 안 된다)
            k["enable_progress_bar"] = False
            k["log_every_n_steps"] = 50
            k["callbacks"] = [c for c in k.get("callbacks", []) if not isinstance(c, ProgressBar)] + [Log()]
            super().__init__(*a, **k)

    pl.Trainer = Trainer
    import train as sv_train

    sv_train.train.callback(config=config, exp_name=name, work_dir=None)
    raise SystemExit("학습이 끝 기록 없이 끝났어요")


def build_onnx_model(sv, ckpt):
    """학습 결과(.ckpt, state_dict의 generator.*) → DiffSinger NSFHiFiGANONNX와 같은 모듈 (가중치 정규화 풀고)."""
    import torch

    from models.nsf_HiFigan.models import AttrDict, Generator
    from utils.config_utils import read_full_config

    cfg = read_full_config(sv / "configs/ft_hifigan.yaml")
    h = AttrDict({**cfg["model_args"], "sampling_rate": cfg["audio_sample_rate"], "num_mels": cfg["audio_num_mel_bins"],
                  "hop_size": cfg["hop_size"]})
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if "state_dict" in ck:  # 학습 체크포인트 (lightning): generator.* · discriminator.*
        gen = {k[len("generator."):]: v for k, v in ck["state_dict"].items() if k.startswith("generator.")}
    else:  # export_ckpt.py로 내보낸 것: {"generator": ...}
        gen = ck["generator"]

    class NSFHiFiGANONNX(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.generator = Generator(h)

        def forward(self, mel, f0):
            return self.generator(mel.transpose(1, 2), f0).squeeze(1)

    m = NSFHiFiGANONNX().eval()
    m.generator.load_state_dict(gen, strict=True)
    m.generator.remove_weight_norm()
    return m


def export(args):
    export_onnx(enter(args.sv), Path(args.cwd, args.ckpt).resolve(), Path(args.cwd, args.out).resolve())


def export_onnx(sv, ckpt, out):
    import onnx
    import torch

    model = build_onnx_model(sv, ckpt)
    tmp = out.with_suffix(".raw.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    mel = torch.randn(1, 10, 128)
    f0 = torch.randn(1, 10) + 440.0
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}  # DiffSinger처럼
    with torch.no_grad():
        torch.onnx.export(model, (mel, f0), str(tmp), input_names=["mel", "f0"], output_names=["waveform"],
                          dynamic_axes={"mel": {1: "n_frames"}, "f0": {1: "n_frames"}, "waveform": {1: "n_samples"}},
                          opset_version=17, **kw)
    m = onnx.load(str(tmp))
    try:  # 있으면 그래프를 줄인다 (DiffSinger 내보내기처럼). 없어도 같은 소리 (Mac 파이썬 3.12 바퀴가 없을 수 있다)
        import onnxsim

        m, ok = onnxsim.simplify(m, include_subgraph=True)
        assert ok
        how = "onnxsim으로 줄임"
    except ImportError:
        how = "onnxsim 없음"
    onnx.save(m, str(out))
    tmp.unlink()
    print(f"내보냄 {out.name} ({out.stat().st_size / 1e6:.0f}MB, {how})", flush=True)


def valid_items(data, n=10):
    data = Path(data)
    import numpy as np

    out = []
    for line in (data / "index" / "valid").read_text(encoding="utf-8").splitlines()[:n]:
        if line.strip():
            z = np.load(line.strip())
            out.append((z["mel"].astype("float32"), z["f0"].astype("float32"), z["audio"].astype("float32"),
                        np.asarray(z["uv"]).astype(bool)))
    return out


def run_onnx(sess, mel, f0):
    return sess.run(None, {"mel": mel[None], "f0": f0[None]})[0][0]


def check(args):
    """두 ONNX에 같은 입력(검증용 테토 녹음의 mel · f0)을 넣어 출력 차이를 잰다 (내보내기가 맞는지)."""
    import numpy as np
    import onnxruntime as ort

    a, b = (ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in args.onnx[:2])
    worst, snr = 0.0, []
    for mel, f0, _, _ in valid_items(args.data, 3):
        ya, yb = run_onnx(a, mel, f0), run_onnx(b, mel, f0)
        for p, y in zip(args.onnx, (ya, yb)):  # 망가진 보코더로 몇십 분씩 렌더하지 않게 (OpenUtau는 NaN을 소리 없음으로 쓴다)
            if not np.isfinite(y).all():
                raise SystemExit(f"{Path(p).name}: 출력에 NaN · 무한대가 있어요")
            if np.abs(y).max() < 1e-4:
                raise SystemExit(f"{Path(p).name}: 출력에 소리가 없어요")
        n = min(len(ya), len(yb))
        d = np.abs(ya[:n] - yb[:n])
        worst = max(worst, float(d.max()))
        snr.append(10 * np.log10(np.sum(ya[:n] ** 2) / (np.sum(d ** 2) + 1e-20)))
    print(json.dumps({"max_abs_diff": worst, "snr_db_min": round(float(min(snr)), 1)}), flush=True)
    if args.require_snr and min(snr) < args.require_snr:
        raise SystemExit(f"두 ONNX가 다르게 소리를 내요 (SNR {min(snr):.1f}dB < {args.require_snr}dB)")


def logspec(y, n_fft, hop):
    import numpy as np

    w = np.hanning(n_fft)
    frames = [y[i:i + n_fft] * w for i in range(0, len(y) - n_fft, hop)]
    return np.log10(np.abs(np.fft.rfft(np.stack(frames), axis=1)) + 1e-5) if frames else np.zeros((0, n_fft // 2 + 1))


def f0_of(y, sr=44100, hop=512, lo=65, hi=1100):
    """SingingVocoders 전처리(get_pitch_parselmouth)와 같은 틀로 음 높이 (0 = 소리 없음)."""
    import numpy as np
    import parselmouth

    l_pad = int(np.ceil(1.5 / lo * sr))
    r_pad = hop * ((len(y) - 1) // hop + 1) - len(y) + l_pad + 1
    snd = parselmouth.Sound(np.pad(y, (l_pad, r_pad)).astype("float64"), sampling_frequency=sr)
    p = snd.to_pitch_ac(time_step=hop / sr, voicing_threshold=0.6, silence_threshold=0.01, pitch_floor=lo, pitch_ceiling=hi)
    return np.nan_to_num(p.selected_array["frequency"])


def copysyn(args):
    """배우지 않은 테토 녹음: mel · f0 → 각 보코더 → 원래 녹음과 로그 스펙트럼 거리 (작을수록 원래와 같다).
    음 높이를 ±4 · ±7반음 바꿔 넣으면 그 음 높이로 부르는지 (센트 오차 중앙값; 테토만 배워서 음 높이 따라가기를 잃지 않았는지)."""
    import onnxruntime as ort

    items = valid_items(args.data, 10)
    res = {}
    for spec in args.onnx:
        name, path = spec.split("=", 1)
        res[name] = measure(ort.InferenceSession(path, providers=["CPUExecutionProvider"]), items)
        print(name, json.dumps({k: round(v, 4) for k, v in res[name].items()}, ensure_ascii=False), flush=True)
    print("\n### 테토 녹음 다시 만들기 (배우지 않은 검증용 10개)\n")
    table(res, "보코더")
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def measure(sess, items):
    import numpy as np

    dist, cents = {512: [], 2048: []}, {k: [] for k in (-7, -4, 4, 7)}
    for mel, f0, audio, uv in items:
        y = run_onnx(sess, mel, f0)
        n = min(len(y), len(audio))
        for n_fft in dist:
            a, b = logspec(audio[:n], n_fft, n_fft // 4), logspec(y[:n], n_fft, n_fft // 4)
            loud = a.max(axis=1) > a.max() - 2  # 소리 나는 틀만 (가장 큰 틀에서 40dB 안)
            if loud.any():
                dist[n_fft].append(float(np.mean(np.abs(a[loud] - b[loud]))))
        for k in cents:
            target = f0 * 2 ** (k / 12)
            got = f0_of(run_onnx(sess, mel, target))
            m = min(len(got), len(target))
            ok = (got[:m] > 0) & ~uv[:m]  # 녹음에서 소리 나는(유성) 틀만
            if ok.sum() > 10:
                cents[k].append(float(np.median(np.abs(1200 * np.log2(got[:m][ok] / target[:m][ok])))))
    return {"logspec_512": float(np.mean(dist[512])), "logspec_2048": float(np.mean(dist[2048])),
            **{f"cents_{k:+d}": float(np.mean(v)) for k, v in cents.items() if v}}


def table(res, first):
    print(f"| {first} | 로그 스펙트럼 거리 (짧은 창) | (긴 창) | 음 높이 바꿔 부르기 센트 오차 −7 · −4 · +4 · +7반음 |")
    print("|---|---|---|---|")
    for name, r in res.items():
        c = " · ".join(f"{r.get(f'cents_{k:+d}', float('nan')):.1f}" for k in (-7, -4, 4, 7))
        print(f"| {name} | {r['logspec_512']:.4f} | {r['logspec_2048']:.4f} | {c} |", flush=True)


def trend(args):
    """배우는 동안 저장점마다 남긴 생성기(experiments/teto_ft/snap/gen_N.pt)를 ONNX로 내보내 copysyn과 같은 잣대로 잰다.
    몇 걸음쯤부터는 더 배워도 나아지지 않는지 보려고 (앱에서 얼마나 배울지 정할 때). 원래 보코더 · 마지막은 --copysyn 결과를
    그대로 쓴다. 많으면 --max-snaps개를 고르게 고르고, --deadline(유닉스 초)에 못 맞출 것 같으면 남은 것은 건너뛴다."""
    import onnxruntime as ort

    sv = enter(args.sv)
    snaps = sorted((Path("experiments") / NAME / "snap").glob("gen_*.pt"), key=lambda q: int(q.stem.split("_")[1]))
    if len(snaps) > args.max_snaps:  # 마지막 것은 늘 넣고 앞에서 고르게
        idx = sorted({round(i * (len(snaps) - 1) / (args.max_snaps - 1)) for i in range(args.max_snaps)})
        snaps = [snaps[i] for i in idx]
    old = Path(args.cwd, args.copysyn or "-")
    old = json.loads(old.read_text(encoding="utf-8")) if args.copysyn and old.exists() else {}
    items = valid_items(Path(args.cwd, args.data), 10)
    if "원래 보코더" in old:
        res = {"원래 보코더 (0걸음)": old["원래 보코더"]}
    elif args.base:  # copysyn을 못 했으면 (테토 보코더가 망가졌으면) 원래 보코더는 여기서 잰다
        base = ort.InferenceSession(str(Path(args.cwd, args.base)), providers=["CPUExecutionProvider"])
        res = {"원래 보코더 (0걸음)": measure(base, items)}
    else:
        res = {}
    tmp, per, skipped = Path(args.cwd, args.out or "trend_onnx").resolve(), 60.0, []
    tmp.mkdir(parents=True, exist_ok=True)
    for q in snaps:
        if time.time() + per * 1.5 > args.deadline:
            skipped.append(q)
            continue
        t = time.time()
        out = tmp / (q.stem + ".onnx")
        try:
            export_onnx(sv, q.resolve(), out)
            r = measure(ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]), items)
        except Exception as e:  # 망가진 생성기 하나 때문에 나머지를 못 재지 않게
            print(q.stem, f"못 잼 ({type(e).__name__}: {str(e)[:150]})", flush=True)
            continue
        finally:
            out.unlink(missing_ok=True)
        res[f"{int(q.stem.split('_')[1])}걸음"] = r
        per = time.time() - t
        print(q.stem, json.dumps({k: round(v, 4) for k, v in r.items()}), f"({per:.0f}초)", flush=True)
    if "테토 보코더" in old:
        res["마지막 (final.ckpt)"] = old["테토 보코더"]
    if skipped:
        print(f"시간이 모자라 {len(skipped)}개 건너뜀: {[q.stem for q in skipped]}", flush=True)
    print("\n### 배우는 동안 테토 녹음 다시 만들기 (걸음마다, 배우지 않은 검증용 10개)\n")
    table(res, "배운 걸음")
    if args.json:
        Path(args.cwd, args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "probe", "train", "train1", "export", "check", "copysyn", "trend"])
    ap.add_argument("onnx", nargs="*")
    ap.add_argument("--sv")
    ap.add_argument("--voicebank")
    ap.add_argument("--data")
    ap.add_argument("--pretrained")
    ap.add_argument("--device", default="cpu", help="prep: 설정에 쓸 장치 (전처리만 해서 아무거나)")
    ap.add_argument("--batch", type=int, default=10, help="prep: 설정에 쓸 배치, train1: 배울 배치")
    ap.add_argument("--crop", type=int, default=32, help="train1: 조각 길이 (mel 틀)")
    ap.add_argument("--recompute", action="store_true", help="train1: 생성기 중간값을 거꾸로 갈 때 다시 계산")
    ap.add_argument("--probe", type=int, default=0, help="train1: 이만큼 걸음만 재고 끝 (저장하지 않음)")
    ap.add_argument("--plan", help="train: probe가 고른 설정 (LADDER 번호)")
    ap.add_argument("--ckpt-every", type=float, default=600, help="train: 이 초마다 이어서 배울 저장점")
    ap.add_argument("--fake-oom", type=float, default=0, help="train: 짧은 시험에서 이 초 뒤 한 번 일부러 멈춰 이어서 배우기 시험")
    ap.add_argument("--fake-gpu", type=float, default=0,
                    help="train: 짧은 시험에서 두 번째 프로세스가 이 초 뒤 맥 GPU 오류 줄을 일부러 찍는다")
    ap.add_argument("--val", type=int, default=200, help="prep: 설정에 쓸 검증 간격 (MPS에서는 끈다)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--train-mpd", action="store_true", help="MPD도 배운다 (기본은 README 권장대로 고정)")
    ap.add_argument("--until", type=float, default=float("inf"), help="유닉스 초. 이때까지만 배운다")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--require-snr", type=float, help="check: 두 ONNX 출력 SNR이 이보다 낮으면 실패 (dB)")
    ap.add_argument("--min-rate", type=float, help="probe: 1초에 배우는 녹음이 이 초보다 짧으면 실패 (느린 맥)")
    ap.add_argument("--copysyn", help="trend: copysyn --json 결과 (원래 보코더 · 마지막 줄로)")
    ap.add_argument("--base", help="trend: copysyn 결과가 없으면 원래 보코더 줄을 이 ONNX로 잰다")
    ap.add_argument("--max-snaps", type=int, default=8, help="trend: 잴 생성기 수 (고르게 고른다)")
    ap.add_argument("--deadline", type=float, default=float("inf"), help="trend: 유닉스 초. 못 맞출 것은 건너뛴다")
    args = ap.parse_args()
    if args.cmd == "copysyn" and args.onnx and "=" not in args.onnx[0]:
        ap.error("copysyn은 이름=경로.onnx 로")
    if args.cmd == "train" and not args.plan:
        ap.error("train은 --plan (probe가 고른 것)과 함께")
    args.cwd = os.getcwd()
    {"prep": prep, "probe": probe, "train": train, "train1": train1, "export": export, "check": check,
     "copysyn": copysyn, "trend": trend}[args.cmd](args)


if __name__ == "__main__":
    main()

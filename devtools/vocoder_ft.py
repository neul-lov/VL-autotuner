"""테토 렌더 보코더 미세 조정 실험 (CI Mac 한 작업 안에서만. 아무것도 올리지 않는다).

테토 음원 녹음으로 신경망 렌더 보코더(pc-nsf-hifigan 2025.02, OpenUtau hifisampler가 쓰는 것)를 조금 더 배워서
같은 프로젝트를 원래 보코더 · 테토 보코더로 렌더해 견준다 (vocoder_ab.py). 테토 약관은 테토 목소리로 학습한 것의
배포를 금지해서, 배운 모델 · 그 렌더는 이 기계에만 두고 숫자만 찍는다 (저장소 · 캐시 · 결과 파일 어디에도 넣지 않는다).

SingingVocoders(openvpi, 학습 코드)를 --sv 폴더에 받아 두고 그 코드를 그대로 쓴다. 바꾸는 것은 설정뿐:
공식 미세 조정 설정(configs/ft_hifigan.yaml)에 CPU · 시간 한도 · 배울 파일 섞기 · MPD 고정(README 권장)을 더한다.
내보내기는 DiffSinger 내보내기(NSFHiFiGANONNX)와 같은 모양 (mel [1, T, 128] · f0 [1, T] → waveform [1, T·512]).

사용:
  python vocoder_ft.py prep --sv SingingVocoders --voicebank 테토폴더 --data 작업폴더 --pretrained 미세조정용.ckpt
  python vocoder_ft.py bench --sv SingingVocoders --data 작업폴더            (CPU · MPS 한 걸음 시간)
  python vocoder_ft.py train --sv SingingVocoders --data 작업폴더 [--device cpu --until 유닉스초]
  python vocoder_ft.py export --sv SingingVocoders --ckpt 학습결과.ckpt --out 이름.onnx
  python vocoder_ft.py check a.onnx b.onnx --data 작업폴더 [--require-snr 40] (같은 입력에 두 ONNX가 얼마나 같은지)
  python vocoder_ft.py copysyn 이름=a.onnx 이름=b.onnx --data 작업폴더 [--json out.json]
      배우지 않은 테토 녹음(검증용 10개)을 다시 만들어 원래와 견주고, 음 높이를 바꿔 불러도 따라가는지 잰다
"""
import argparse
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

from ci_status import post as progress  # devtools/ (이 파일 옆)

CONFIG = "configs/teto_ft.yaml"
NAME = "teto_ft"


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


def bench(args):
    """한 걸음(배치 하나: 판별기 · 생성기 한 번씩, pc 증강 포함)과 비슷한 계산을 CPU와 MPS에서 재 본다."""
    import torch

    enter(args.sv)
    from models.nsf_HiFigan.models import AttrDict, Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
    from utils.config_utils import read_full_config

    cfg = read_full_config(Path("configs/ft_hifigan.yaml"))
    h = AttrDict({**cfg["model_args"], "sampling_rate": 44100, "num_mels": 128, "hop_size": 512})
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    print(f"torch {torch.__version__}, 스레드 {torch.get_num_threads()}, MPS {torch.backends.mps.is_available()}", flush=True)
    best = None
    for dev in devices:
        try:
            g = Generator(h).to(dev)
            d = torch.nn.ModuleList([MultiScaleDiscriminator(), MultiPeriodDiscriminator(periods=h.discriminator_periods)]).to(dev)
            og, od = torch.optim.AdamW(g.parameters(), 1e-5), torch.optim.AdamW(d.parameters(), 1e-5)
            b, frames = args.batch, 32
            mel = torch.randn(b, 128, frames, device=dev)
            f0 = 200 + 300 * torch.rand(b, frames, device=dev)
            real = torch.randn(b, 1, frames * 512, device=dev) * 0.1
            times = []
            for k in range(4):
                t = time.time()
                fake = torch.cat([g(mel, f0), g(mel[:4], f0[:4] * 1.2), g(mel[:4], f0[:4]), g(mel[:4], f0[:4] * 0.9)])
                lo = sum(o.mean() for D in d for o in D(fake.detach())[0]) - sum(o.mean() for D in d for o in D(real)[0])
                od.zero_grad()
                lo.backward()
                od.step()
                lg = -sum(o.mean() for D in d for o in D(fake)[0]) + (fake[:b] - real).abs().mean() * 45
                og.zero_grad()
                lg.backward()
                og.step()
                if dev == "mps":
                    torch.mps.synchronize()
                times.append(time.time() - t)
            s = sorted(times[1:])[len(times[1:]) // 2]
            print(f"{dev}: 한 걸음 {s:.2f}초 (배치 {b})", flush=True)
            if best is None or s < best[1]:
                best = (dev, s)
        except Exception as e:  # MPS가 있다고 나와도 가상 기계에서는 안 될 수 있다
            print(f"{dev}: 못 씀 ({type(e).__name__}: {str(e)[:200]})", flush=True)
    print(f"고른 장치: {best[0]}", flush=True)
    if args.out:
        Path(args.out).write_text(best[0], encoding="utf-8")


def train(args):
    enter(args.sv)
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, ProgressBar

    if args.device:  # bench가 고른 장치로 설정을 다시 쓴다
        import yaml

        cfg = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
        cfg["pl_trainer_accelerator"] = args.device
        Path(CONFIG).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    left = int(args.until - time.time()) if args.until else None
    if left is not None and left < 300:
        raise SystemExit(f"배울 시간이 모자라요 ({left}초)")
    work = Path("experiments") / NAME
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

    class Log(Callback):  # CI 기록에 50걸음마다 한 줄 (진행 막대 대신), 커밋 상태에 10분마다
        def __init__(self):
            self.t, self.n, self.said, self.val = time.time(), 0, 0.0, []

        def on_train_start(self, trainer, pl_module):  # 준비 · 처음 검증 시간은 빼고 잰다
            self.t = time.time()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.n += 1
            if self.n % 50 == 0:
                loss = trainer.callback_metrics.get("loss")
                print(f"걸음 {self.n} · {time.time() - self.t:.0f}초 · 손실 {float(loss):.4f}" if loss is not None else
                      f"걸음 {self.n} · {time.time() - self.t:.0f}초", flush=True)
            if self.n == 5 or time.time() - self.said > 600:
                self.said = time.time()
                rest = f", 남은 {(args.until - time.time()) / 60:.0f}분" if args.until else ""
                val = f", 검증 손실 {self.val[0]:.4f}→{self.val[-1]:.4f}" if self.val else ""
                progress("보코더 실험", f"학습 걸음 {self.n} · 한 걸음 {(time.time() - self.t) / self.n:.1f}초{val}{rest}")

        def on_validation_end(self, trainer, pl_module):
            v = trainer.callback_metrics.get("val_loss")
            if v is not None:
                self.val.append(float(v))
                print(f"걸음 {self.n} 검증 손실 {float(v):.4f}", flush=True)

        def on_train_end(self, trainer, pl_module):  # 시간 한도로 멈춰도 마지막 것을 남긴다
            trainer.save_checkpoint(str(work / "final.ckpt"))
            print(f"배운 걸음 {self.n} ({time.time() - self.t:.0f}초) → {work / 'final.ckpt'}", flush=True)

    class Trainer(pl.Trainer):  # train.py가 만드는 Trainer에 시간 한도 · 기록만 더한다 (isinstance가 되게 물려받아서)
        def __init__(self, *a, **k):
            if left is not None:
                k["max_time"] = {"seconds": left}
            k["enable_progress_bar"] = False
            k["log_every_n_steps"] = 50
            k["callbacks"] = [c for c in k.get("callbacks", []) if not isinstance(c, ProgressBar)] + [Log()]
            super().__init__(*a, **k)

    pl.Trainer = Trainer
    print(f"배울 시간 한도 {left / 3600:.2f}시간" if left else "시간 한도 없음", flush=True)
    import train as sv_train

    sv_train.train.callback(config=CONFIG, exp_name=NAME, work_dir=None)


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
    import onnx
    import torch

    out = Path(args.out).resolve()
    sv = enter(args.sv)
    model = build_onnx_model(sv, Path(args.ckpt).resolve() if Path(args.ckpt).is_absolute() else Path(args.cwd, args.ckpt))
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
    import numpy as np
    import onnxruntime as ort

    items = valid_items(args.data, 10)
    res = {}
    for spec in args.onnx:
        name, path = spec.split("=", 1)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
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
        res[name] = {"logspec_512": float(np.mean(dist[512])), "logspec_2048": float(np.mean(dist[2048])),
                     **{f"cents_{k:+d}": float(np.mean(v)) for k, v in cents.items() if v}}
        print(name, json.dumps({k: round(v, 4) for k, v in res[name].items()}, ensure_ascii=False), flush=True)
    print("\n### 테토 녹음 다시 만들기 (배우지 않은 검증용 10개)\n")
    print("| 보코더 | 로그 스펙트럼 거리 (짧은 창) | (긴 창) | 음 높이 바꿔 부르기 센트 오차 −7 · −4 · +4 · +7반음 |")
    print("|---|---|---|---|")
    for name, r in res.items():
        c = " · ".join(f"{r.get(f'cents_{k:+d}', float('nan')):.1f}" for k in (-7, -4, 4, 7))
        print(f"| {name} | {r['logspec_512']:.4f} | {r['logspec_2048']:.4f} | {c} |")
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "bench", "train", "export", "check", "copysyn"])
    ap.add_argument("onnx", nargs="*")
    ap.add_argument("--sv")
    ap.add_argument("--voicebank")
    ap.add_argument("--data")
    ap.add_argument("--pretrained")
    ap.add_argument("--device", default="")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--val", type=int, default=200, help="몇 걸음마다 검증 · 저장")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--train-mpd", action="store_true", help="MPD도 배운다 (기본은 README 권장대로 고정)")
    ap.add_argument("--until", type=float, help="유닉스 초. 이때까지만 배운다")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--require-snr", type=float, help="check: 두 ONNX 출력 SNR이 이보다 낮으면 실패 (dB)")
    args = ap.parse_args()
    if args.cmd == "copysyn" and args.onnx and "=" not in args.onnx[0]:
        ap.error("copysyn은 이름=경로.onnx 로")
    args.cwd = os.getcwd()
    if args.cmd == "prep" and not args.device:
        args.device = "cpu"
    {"prep": prep, "bench": bench, "train": train, "export": export, "check": check, "copysyn": copysyn}[args.cmd](args)


if __name__ == "__main__":
    main()

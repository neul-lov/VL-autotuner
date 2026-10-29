"""한국어 가사 맞추기 모델(HubertFA의 FA 모델) 학습 · ONNX 내보내기 (CI용, GPU 없이 CPU로).

HubertFA(wolfgitpr, Apache-2.0)를 --hfa 폴더에 받아 두고 그 코드를 그대로 쓴다. 바꾸는 것은 설정뿐:
CPU · 32비트, 학습 시간 한도(--until), 그림 안 그리기, 한국어 사전 하나 (음소 앞에 ko/).
내보내기는 인코더(chinese-hubert-base) + FA를 한 ONNX로 묶어 앱의 align_fa.Aligner가 읽는 모양
(model.onnx · vocab.json · config.json)으로 둔다. 숨 소리 모델(NLL)은 앱이 쓰지 않아서 배우지 않는다.

사용 (gtsinger_hfa.py 결과 폴더 data에 train/ · eval/ · ko_phones.txt):
  python hfa_train.py binarize --hfa HubertFA --data data [--binary 폴더 --batch 100 --steps 6000 --val 500]
  python hfa_train.py train --hfa HubertFA [--until 유닉스초]
  python hfa_train.py export --hfa HubertFA --out out/hfa_ko [--ckpt best|last]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

CONFIG = "configs/fa_ko.yaml"
NAME = "ko_fa"


def enter(hfa):
    """HubertFA 코드는 자기 폴더를 기준으로 경로를 쓴다 (configs/ · dictionaries/ · dependencies/ · ckpt/)."""
    hfa = Path(hfa).resolve()
    os.chdir(hfa)
    sys.path.insert(0, str(hfa))
    import torch

    torch.set_num_threads(os.cpu_count() or 4)
    return hfa


def write_config(hfa, data, binary, batch, steps, val):
    import yaml

    data = Path(data).resolve()
    cfg = yaml.safe_load((hfa / "configs/force_alignment.yaml").read_text(encoding="utf-8"))
    shutil.copy(data / "ko_phones.txt", hfa / "dictionaries/ko_phones.txt")
    sets = [{"raw_data_dir": str(data / "train"), "label_type": "full", "language": "ko", "test_prefixes": []}]
    if (data / "eval/transcriptions.csv").exists() and len((data / "eval/transcriptions.csv").read_text().splitlines()) > 1:
        sets.append({"raw_data_dir": str(data / "eval"), "label_type": "evaluate", "language": "ko", "test_prefixes": []})
    (hfa / "configs/datasets_ko.yaml").write_text(yaml.safe_dump({"datasets": sets}, allow_unicode=True), encoding="utf-8")
    cfg.update({
        "model_name": NAME,
        "dictionaries": {"ko": "dictionaries/ko_phones.txt"},
        "datasets_config_paths": ["configs/datasets_ko.yaml"],
        "binary_folder": str(Path(binary).resolve()) if binary else "data/binary_ko",
        "multiprocess_works": 0,
        "dataloader_workers": 2,
        "batch_max_length": batch,
        "num_valid_plots": 0,
        "draw_evaluate": False,
        "accelerator": "cpu",
        "devices": 1,
        "precision": "32-true",
        "val_check_interval": val,
        "save_every_steps": val,
        "merged_phoneme": False,
    })
    cfg["optimizer_config"]["total_steps"] = steps
    (hfa / CONFIG).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps({k: cfg[k] for k in ("batch_max_length", "val_check_interval", "precision")} |
                     {"total_steps": steps, "datasets": len(sets)}, ensure_ascii=False), flush=True)


def binarize(args):
    binary = Path(args.binary).resolve() if args.binary else None  # enter가 폴더를 옮기기 전에
    data = Path(args.data).resolve()
    hfa = enter(args.hfa)
    write_config(hfa, data, binary, args.batch, args.steps, args.val)
    from binarize import ForcedAlignmentBinarizer

    t = time.time()
    ForcedAlignmentBinarizer(CONFIG).process()
    print(f"미리 계산 {time.time() - t:.0f}초", flush=True)


def train(args):
    enter(args.hfa)
    import lightning

    if args.until:
        left = int(args.until - time.time())
        if left < 600:
            raise SystemExit(f"학습할 시간이 모자라요 ({left}초)")
        orig = lightning.Trainer

        def trainer(*a, **k):  # 시간 한도: 다 못 배워도 그때까지 배운 것을 저장한다
            k["max_time"] = {"seconds": left}
            k["enable_progress_bar"] = False
            k["log_every_n_steps"] = 50
            return orig(*a, **k)
        lightning.Trainer = trainer
        print(f"학습 시간 한도 {left / 3600:.2f}시간", flush=True)
    from lightning.pytorch.callbacks import Callback

    class Log(Callback):  # CI 기록에 50걸음마다 한 줄 (진행 막대 대신)
        def __init__(self):
            self.t = time.time()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            s = trainer.global_step
            if s % 50 == 0:
                m = trainer.callback_metrics
                loss = m.get("train/total_loss")
                print(f"걸음 {s} · {time.time() - self.t:.0f}초 · 손실 {float(loss):.4f}" if loss is not None else
                      f"걸음 {s} · {time.time() - self.t:.0f}초", flush=True)

        def on_validation_end(self, trainer, pl_module):
            m = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()
                 if k.startswith(("unseen_evaluate/", "valid_evaluate/total", "valid/total"))}
            print(f"걸음 {trainer.global_step} 채점 {json.dumps(m)}", flush=True)

    import train as hfa_train

    hfa_train.StepProgressBar = Log  # train.py가 넣는 진행 막대 대신 (진행 막대를 껐다)
    t = time.time()
    hfa_train.ForcedAlignmentTrainer(CONFIG).train()
    print(f"학습 {time.time() - t:.0f}초", flush=True)


def pick_ckpt(hfa, which):
    folder = hfa / "ckpt" / NAME
    if which == "last":
        return hfa / "ckpt" / f"{NAME}.ckpt"
    best = sorted(folder.glob("best-step=*.ckpt"), key=lambda p: float(p.stem.split("=")[-1]))
    return best[0] if best else hfa / "ckpt" / f"{NAME}.ckpt"


def export(args):
    hfa = enter(args.hfa)
    import onnx
    import onnxsim
    import torch
    import yaml
    from networks.task.forced_alignment_task import LitForcedAlignmentTask
    from scripts.onnx_exporter import change_input_node_name, change_output_node_name, export_encoder

    ckpt = pick_ckpt(hfa, args.ckpt)
    print(f"내보낼 학습 결과: {ckpt.relative_to(hfa)}", flush=True)
    folder = hfa / "ckpt" / NAME
    cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    hubert, mel = cfg["hubert_config"], cfg["mel_spec_config"]
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(args.cwd) / out
    out.mkdir(parents=True, exist_ok=True)
    enc = export_encoder(out, hubert, mel)

    class FA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fa = LitForcedAlignmentTask.load_from_checkpoint(str(ckpt), map_location="cpu")

        def forward(self, x, curves):
            return self.fa(x, curves)

    pred = str(out / "predict.onnx")
    model = FA().eval()
    n = 1000
    with torch.no_grad():
        torch.onnx.export(model, (torch.randn(1, hubert["channel"], n), torch.randn(1, 1, n)), pred,
                          input_names=["input_feature", "curves"],
                          output_names=["ph_frame_logits", "ph_edge_logits", "ctc_logits"],
                          dynamic_axes={"input_feature": {2: "n_samples"}, "curves": {2: "n_samples"},
                                        "ph_frame_logits": {2: "n_samples"}, "ph_edge_logits": {1: "n_samples"},
                                        "ctc_logits": {2: "n_samples"}},
                          opset_version=17)
    m, ok = onnxsim.simplify(pred, include_subgraph=True)
    assert ok
    onnx.save(m, pred)
    a = onnx.compose.add_prefix(onnx.load(enc), prefix="encoder/")
    b = onnx.compose.add_prefix(onnx.load(pred), prefix="predict/")
    merged = onnx.compose.merge_models(a, b, io_map=[("encoder/units_aligned", "predict/input_feature"),
                                                     ("encoder/curves", "predict/curves")])
    change_input_node_name(merged, ["waveform"])
    change_output_node_name(merged, ["ph_frame_logits", "ph_edge_logits", "ctc_logits"])
    path = str(out / "model.onnx")
    onnx.save(merged, path)
    m, ok = onnxsim.simplify(path, include_subgraph=True)
    assert ok
    onnx.save(m, path)
    os.remove(enc)
    os.remove(pred)
    vocab = yaml.safe_load((folder / "vocab.yaml").read_text(encoding="utf-8"))
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({"hubert_config": hubert, "mel_spec_config": mel}, ensure_ascii=False,
                                                indent=1), encoding="utf-8")
    shutil.copy(folder / "ko_phones.txt", out / "ko_phones.txt")
    print(f"내보냄: {out} ({os.path.getsize(path) / 1e6:.0f}MB, 음소 {vocab['vocab_size']}개)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["binarize", "train", "export"])
    ap.add_argument("--hfa", required=True)
    ap.add_argument("--data")
    ap.add_argument("--binary", help="미리 계산한 특징을 둘 폴더 (크다: 소리 1시간에 2GB쯤)")
    ap.add_argument("--batch", type=float, default=100, help="초. 한 걸음에 배우는 소리 길이 합 (HubertFA 기본 200)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--until", type=float, help="유닉스 초. 이때까지만 배운다")
    ap.add_argument("--out")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    args = ap.parse_args()
    args.cwd = os.getcwd()
    {"binarize": binarize, "train": train, "export": export}[args.cmd](args)


if __name__ == "__main__":
    main()

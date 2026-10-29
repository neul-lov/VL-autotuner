"""한국어 가사 맞추기 모델(HubertFA의 FA 모델) 학습 · ONNX 내보내기 (CI용, GPU 없이 CPU로).

HubertFA(wolfgitpr, Apache-2.0)를 --hfa 폴더에 받아 두고 그 코드를 그대로 쓴다. 바꾸는 것은 설정뿐:
CPU · 32비트, 학습 시간 한도(--until), 그림 안 그리기, 한국어 사전 하나 (음소 앞에 ko/).
내보내기는 인코더(chinese-hubert-base) + FA를 한 ONNX로 묶어 앱의 align_fa.Aligner가 읽는 모양
(model.onnx · vocab.json · config.json)으로 둔다. 숨 소리 모델(NLL)은 앱이 쓰지 않아서 배우지 않는다.

처음부터 배우면 GTSinger(6.7시간, 세 사람)로는 지금 모델(여러 말 노래로 배운 것)보다 못해서 (CSD 모음 시작 50ms 안 0.70 vs 0.81),
지금 모델 무게에서 시작해 이어 배울 수 있게 했다 (official → train --init): 지금 모델은 ONNX로만 공개돼 있어서
같은 모양 HubertFA 모델을 똑같이 내보낸 ONNX와 짝지어 무게를 꺼내고, 마지막 층(음소마다 한 줄)은 한국어 음소마다
지금 모델이 쓰는 일본어 음소 줄(KO_INIT, align_fa.hangul_phones와 같은 뜻)에서 시작한다.

사용 (gtsinger_hfa.py 결과 폴더 data에 train/ · eval/ · ko_phones.txt):
  python hfa_train.py official --hfa HubertFA --src models/hfa --out official_fa.pt
  python hfa_train.py binarize --hfa HubertFA --data data [--binary 폴더 --batch 100 --steps 6000 --val 500]
  python hfa_train.py train --hfa HubertFA [--until 유닉스초] [--init official_fa.pt --lr 0.0001]
  python hfa_train.py export --hfa HubertFA --out out/hfa_ko [--ckpt best|last|init]
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

CONFIG = "configs/fa_ko.yaml"
NAME = "ko_fa"
OUTS = ["ph_frame_logits", "ph_edge_logits", "ctc_logits"]
PARTS = ("backbone.", "curves_edge_fusion.", "head.")  # FA 무게 (손실 함수 것은 빼고)
# 한국어 음소 → 지금 모델의 일본어 음소 (마지막 층을 이 줄에서 시작한다). 받침 막힘은 쉼(cl = SP)
KO_INIT = {"ɐ": "a", "ɛ": "e", "e": "e", "ʌ": "o", "o": "o", "u": "u", "ɨ": "u", "i": "i",
           "j": "y", "w": "w", "ɥ": "w", "ɰ": "u",
           "k": "k", "kʰ": "k", "k͈": "k", "ɡ": "g", "t": "t", "tʰ": "t", "t͈": "t", "d": "d",
           "p": "p", "pʰ": "p", "p͈": "p", "b": "b", "sʰ": "s", "s͈": "s", "ɕʰ": "sh", "ɕ͈": "sh",
           "h": "h", "ç": "h", "ɦ": "h", "tɕ": "ch", "tɕʰ": "ch", "tɕ͈": "ch", "dʑ": "j",
           "n": "n", "m": "m", "ŋ": "N", "ɲ": "ny", "ɾ": "r", "ɭ": "r", "k̚": "cl", "t̚": "cl", "p̚": "cl"}


def note(title, msg):
    """GitHub Actions 주석으로도 (작업이 도는 동안 진행을 볼 수 있게, 단계마다 10개까지만 보인다)."""
    print(f"::notice title={title}::{msg}" if os.environ.get("GITHUB_ACTIONS") else f"{title}: {msg}", flush=True)


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
    note("미리 계산", f"{time.time() - t:.0f}초")


def _consts(m):
    """그래프의 상수 무게 (initializer · Constant 노드) 이름 → TensorProto."""
    out = {i.name: i for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    out[n.output[0]] = a.t
    return out


def _match(ours, theirs, co, ct):
    """같은 계산인 두 그래프를 출력에서 거슬러 올라가며 노드끼리 짝짓는다 (노드 순서가 달라도 된다).
    → (우리 상수 이름 → 그쪽 상수 이름, 짝지은 노드 수, 안 맞는 곳 [설명])."""
    from onnx import helper

    po = {o: n for n in ours.graph.node for o in n.output}
    pt = {o: n for n in theirs.graph.node for o in n.output}
    pairs, bad, seen, nodes = {}, [], set(), 0
    stack = [(o, o) for o in OUTS]
    while stack:
        a, b = stack.pop()
        if not a or (a, b) in seen:
            continue
        seen.add((a, b))
        if a in co or b in ct:
            if a in co and b in ct:
                if pairs.setdefault(a, b) != b:
                    bad.append(f"상수 {a}가 둘과 짝: {pairs[a]} · {b}")
            else:
                bad.append(f"한쪽만 상수: {a} ↔ {b} ({pt[b].op_type if b in pt else '-'})")
            continue
        na, nb = po.get(a), pt.get(b)
        if na is None:  # 우리 그래프 입력 (그쪽은 인코더 출력)
            continue
        if nb is None or na.op_type != nb.op_type or len(na.input) != len(nb.input) or \
                list(na.output).index(a) != list(nb.output).index(b):
            bad.append(f"노드가 다름: {na.op_type} ↔ {nb.op_type if nb else '-'} ({a} ↔ {b})")
            continue
        nodes += 1
        aa = {x.name: helper.get_attribute_value(x) for x in na.attribute}
        ab = {x.name: helper.get_attribute_value(x) for x in nb.attribute}
        if na.op_type != "Constant" and aa != ab:
            bad.append(f"{na.op_type} 속성이 다름: {aa} ↔ {ab}")
        ia, ib = list(na.input), list(nb.input)
        if na.op_type in ("Add", "Mul") and len(ia) == 2 and \
                [x in co for x in ia] == [x in ct for x in ib][::-1] != [x in ct for x in ib]:
            ib = ib[::-1]  # 상수가 앞 · 뒤 다른 자리에 (torch 판에 따라)
        stack += list(zip(ia, ib))
    return pairs, nodes, bad


def official(args):
    """지금 앱의 가사 맞추기 모델(HubertFA v0.0.7, ONNX로만 공개)에서 FA 무게를 꺼내 HubertFA 모델 모양으로 저장한다.
    같은 모양 모델에 무게마다 다른 값을 넣어 똑같이 내보낸 ONNX와 출력에서부터 노드끼리 짝지어 어느 상수가
    어느 무게인지 찾고, 꺼낸 무게로 PyTorch에서 돌린 값이 지금 모델 ONNX와 같은지 확인한다 (다르면 멈춘다)."""
    src = Path(args.src).resolve()
    out = Path(args.out) if Path(args.out).is_absolute() else Path(args.cwd) / args.out
    hfa = enter(args.hfa)
    import hashlib

    import numpy as np
    import onnx
    import onnxruntime as ort
    import onnxsim
    import torch
    import yaml
    from onnx import numpy_helper
    from networks.task.forced_alignment_task import LitForcedAlignmentTask

    vocab = json.loads((src / "vocab.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((hfa / "configs/force_alignment.yaml").read_text(encoding="utf-8"))
    cfg.update({k: v for k, v in json.loads((src / "config.json").read_text(encoding="utf-8")).items()
                if k in ("hubert_config", "mel_spec_config")})
    cfg["aug_num"] = 1
    ch = int(cfg["hubert_config"]["channel"])
    big = onnx.load(str(src / "model.onnx"))
    nodes = list(big.graph.node)
    prod = {o: k for k, n in enumerate(nodes) for o in n.output}
    bound = {"encoder/units_aligned", "encoder/curves"}
    by_name = bound <= set(prod)  # 인코더와 FA 사이 값 이름이 남아 있으면 거기서 끊는다 (없으면 predict/ 접두 노드만)
    keep, todo = set(), list(OUTS)
    while todo:
        k = prod.get(todo.pop())
        if k is None or k in keep or (by_name and set(nodes[k].output) & bound) or \
                (not by_name and not nodes[k].name.startswith("predict/")):
            continue
        keep.add(k)
        todo += [x for x in nodes[k].input if x]
    fa = [n for k, n in enumerate(nodes) if k in keep]
    cb = _consts(big)
    made = {o for n in fa for o in n.output}
    ext = sorted({x for n in fa for x in n.input if x and x not in made and x not in cb})
    print(f"지금 모델: FA 노드 {len(fa)}개 / 전체 {len(nodes)}, 밖에서 받는 값 {ext}, 음소 {vocab['vocab_size']}개",
          flush=True)
    head = next((cb[x] for x in cb if x.endswith("fa.head.weight")), None)
    if head is not None:
        cfg["fa_arg"]["hidden_dims"] = int(head.dims[1])
        assert int(head.dims[0]) == vocab["vocab_size"] + 2, f"마지막 층 모양이 음소 수와 다름: {list(head.dims)}"
    print(f"FA 크기 {cfg['fa_arg']}", flush=True)

    torch.manual_seed(0)
    model = LitForcedAlignmentTask(vocab, cfg).eval()
    with torch.no_grad():
        for p in model.parameters():  # 무게마다 다른 값 (내보낸 상수를 값으로 찾는다)
            p.copy_(torch.randn_like(p))

    class FA(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.fa = m  # 이름 fa.… (지금 모델을 내보낸 onnx_exporter.PredictModel과 같게)

        def forward(self, x, curves):
            return self.fa(x, curves)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out.parent / "same_shape_fa.onnx")
    n = 1000  # onnx_exporter와 같은 길이로 (UNet이 길이에 따라 다르게 내보내지지 않게)
    with torch.no_grad():
        torch.onnx.export(FA(model), (torch.randn(1, ch, n), torch.randn(1, 1, n)), tmp,
                          input_names=["input_feature", "curves"], output_names=OUTS,
                          dynamic_axes={"input_feature": {2: "n_samples"}, "curves": {2: "n_samples"},
                                        "ph_frame_logits": {2: "n_samples"}, "ph_edge_logits": {1: "n_samples"},
                                        "ctc_logits": {2: "n_samples"}},
                          opset_version=17)
    ours, ok = onnxsim.simplify(tmp, include_subgraph=True)
    assert ok
    os.remove(tmp)
    co = _consts(ours)
    pairs, matched, bad = _match(ours, big, co, cb)
    print(f"노드 {matched}개 짝지음, 상수 {len(pairs)}개, 안 맞는 곳 {len(bad)}개", flush=True)
    for x in bad[:15]:
        print("  " + x[:300], flush=True)

    def h(a):
        return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()

    ours_hash, used = {}, set()
    for x, t in co.items():
        a = numpy_helper.to_array(t)
        if a.dtype.kind == "f":
            ours_hash.setdefault(h(a), x)
    sd, new, how, missing = model.state_dict(), {}, Counter(), []
    for k, p in sd.items():
        if not k.startswith(PARTS):
            continue
        a = p.numpy()
        x, tr = ours_hash.get(h(a)), False
        if x is None and a.ndim == 2:
            x, tr = ours_hash.get(h(a.T)), True
        y = pairs.get(x) if x else None
        if y is None or tuple(cb[y].dims) != tuple(co[x].dims):
            missing.append(k)
            continue
        used.add(x)
        b = numpy_helper.to_array(cb[y])
        new[k] = torch.from_numpy(np.ascontiguousarray(b.T if tr else b).reshape(a.shape).astype(np.float32))
        how["이름 같음" if y == "predict/" + x else "이름 다름"] += 1
    # 무게가 아닌 상수 (모양 · 축 · eps …)는 두 그래프에서 같아야 같은 구조다
    diff = []
    for x, y in pairs.items():
        if x in used:
            continue
        u, v = numpy_helper.to_array(co[x]), numpy_helper.to_array(cb[y])
        if u.shape != v.shape or not np.allclose(u, v, rtol=1e-5, atol=1e-7):
            diff.append(f"{x} {list(u.shape)} {u.ravel()[:6].tolist()} ↔ {y} {list(v.shape)} {v.ravel()[:6].tolist()}")
    print(f"무게 {len(new)}개 짝 찾음 ({dict(how)}), 못 찾음 {missing}, 값이 다른 상수 {len(diff)}개", flush=True)
    for x in diff[:15]:
        print("  " + x[:300], flush=True)
    model.load_state_dict(new, strict=False)

    # 지금 모델을 소리 하나로 돌려 FA가 받는 값(인코더 출력)과 FA 출력을 같이 꺼내 PyTorch 결과와 견준다
    try:
        vi = {v.name: v for v in onnx.shape_inference.infer_shapes(big).graph.value_info}
    except Exception as e:  # 모양을 못 알아내면 float로 둔다 (인코더 출력은 float)
        print(f"모양 추론 못 함: {type(e).__name__}", flush=True)
        vi = {}
    for x in ext:
        big.graph.output.append(vi[x] if x in vi else onnx.helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT, None))
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(big.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    sr = int(cfg["mel_spec_config"]["sample_rate"])
    t = np.arange(sr * 6) / sr
    rng = np.random.default_rng(0)
    wav = 0.3 * np.sin(2 * np.pi * np.cumsum(220 + 110 * np.sin(2 * np.pi * 0.5 * t)) / sr) * (t % 1.5 < 1.2)
    wav = (wav + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    res = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"waveform": wav[None, :]})))
    units = next(res[x] for x in ext if res[x].ndim == 3 and res[x].shape[1] == ch)
    curves = next(res[x] for x in ext if res[x].ndim == 3 and res[x].shape[1] == 1)
    # 짝이 맞는지 따로 본다: 같은 모양 ONNX의 상수를 짝지은 지금 모델 값으로 바꿔 돌리면 지금 모델과 같아야 한다
    swapped = onnx.ModelProto()
    swapped.CopyFrom(ours)
    for i in swapped.graph.initializer:
        if i.name in pairs:
            i.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(cb[pairs[i.name]]), i.name))
    for nd in swapped.graph.node:
        if nd.op_type == "Constant" and nd.output[0] in pairs:
            for at in nd.attribute:
                if at.name == "value":
                    at.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(cb[pairs[nd.output[0]]])))
    try:
        r2 = ort.InferenceSession(swapped.SerializeToString(), opts, providers=["CPUExecutionProvider"]).run(
            None, {"input_feature": units.astype(np.float32), "curves": curves.astype(np.float32)})
        for name, a in zip(OUTS, r2):
            ref = res[name]
            d = float(np.abs(a - ref).max()) / max(1.0, float(np.abs(ref).max()))
            print(f"(상수만 바꾼 같은 모양 ONNX) {name}: 가장 큰 차이 {d:.2e}", flush=True)
    except Exception as e:
        print(f"상수만 바꾼 ONNX를 못 돌림: {type(e).__name__}: {str(e)[:300]}", flush=True)
    with torch.no_grad():
        mine = model(torch.from_numpy(units).float(), torch.from_numpy(curves).float())
    worst = 0.0
    for name, a in zip(OUTS, mine):
        ref = res[name]
        d = float(np.abs(a.numpy() - ref).max()) / max(1.0, float(np.abs(ref).max()))
        worst = max(worst, d)
        print(f"{name}: 모양 {list(ref.shape)}, 가장 큰 차이 {d:.2e} (값 크기에 견줘)", flush=True)
    if missing or worst > 1e-3:
        raise SystemExit("지금 모델 무게를 제대로 꺼내지 못했어요 (위 차이 · 못 찾은 무게)")
    torch.save({"state_dict": new, "vocab": vocab, "fa_arg": cfg["fa_arg"]}, out)
    note("지금 모델 무게", f"FA 무게 {len(new)}개, PyTorch와 ONNX 차이 {worst:.1e} → {out.name}")


def load_init(model, path):
    """official이 꺼낸 지금 모델 무게를 한국어 모델에 넣는다. 마지막 층은 한국어 음소마다 KO_INIT의 일본어 음소 줄로."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, ids = ck["state_dict"], ck["vocab"]["vocab"]
    own = model.state_dict()
    new = {k: v for k, v in sd.items() if not k.startswith("head.") and k in own and own[k].shape == v.shape}
    skip = [k for k in sd if not k.startswith("head.") and k not in new]
    w, b = sd["head.weight"], sd["head.bias"]
    hw, hb = own["head.weight"].clone(), own["head.bias"].clone()
    hw[:2], hb[:2] = w[:2], b[:2]  # 0: 음소 경계, 1: CTC 빈칸
    done, miss = set(), []
    for name, i in sorted(model.vocab["vocab"].items(), key=lambda kv: kv[1]):
        if i in done:
            continue
        p = name.split("/", 1)[1] if "/" in name else name
        q = "SP" if i == 0 else KO_INIT.get(p)
        j = next((ids[x] for x in (f"ja/{q}", q) if q and x in ids), None)
        if j is None:
            miss.append(p)
            continue
        hw[i + 2], hb[i + 2] = w[j + 2], b[j + 2]
        done.add(i)
    new["head.weight"], new["head.bias"] = hw, hb
    r = model.load_state_dict(new, strict=False)
    left = [k for k in r.missing_keys if k.startswith(PARTS)]
    if skip or left:
        raise SystemExit(f"지금 모델 무게가 한국어 모델과 모양이 달라요: 뺀 것 {skip}, 빈 것 {left}")
    note("이어 배우기", f"지금 모델 무게 {len(new)}개에서 시작, 마지막 층 음소 {len(done)}/{model.vocab['vocab_size']}개"
         + (f" (일본어 음소 없음: {' '.join(miss)})" if miss else ""))


def train(args):
    if args.init and not Path(args.init).is_absolute():
        args.init = str(Path(args.cwd) / args.init)
    hfa = enter(args.hfa)
    import lightning
    import yaml

    if args.init or args.lr:  # 이어 배우기: FA 크기는 지금 모델과 같게, 학습률은 낮게
        cfg = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
        if args.init:
            import torch

            cfg["fa_arg"].update(torch.load(args.init, map_location="cpu", weights_only=False)["fa_arg"])
        if args.lr:
            cfg["optimizer_config"]["lr"] = args.lr
        Path(CONFIG).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(json.dumps({"fa_arg": cfg["fa_arg"], "lr": cfg["optimizer_config"]["lr"]}, ensure_ascii=False), flush=True)

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

    class Log(Callback):  # CI 기록에 50걸음마다 한 줄 (진행 막대 대신), 주석은 처음과 30분마다
        def __init__(self):
            self.t = self.noted = time.time()
            self.last = {}

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            s = trainer.global_step
            if s % 50 == 0:
                m = trainer.callback_metrics
                loss = m.get("train/total_loss")
                line = (f"걸음 {s} · {time.time() - self.t:.0f}초 · 손실 {float(loss):.4f}" if loss is not None else
                        f"걸음 {s} · {time.time() - self.t:.0f}초")
                print(line, flush=True)
                if s == 50 or time.time() - self.noted > 1800:
                    self.noted = time.time()
                    note("학습", f"{line} · 채점 {json.dumps(self.last)}")

        def on_validation_end(self, trainer, pl_module):
            m = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()
                 if k.startswith(("unseen_evaluate/", "valid_evaluate/total", "valid/total"))}
            self.last = {k.split("/")[-1] if k.startswith("unseen") else k: v for k, v in m.items()}
            print(f"걸음 {trainer.global_step} 채점 {json.dumps(m)}", flush=True)

    from networks.task.forced_alignment_task import LitForcedAlignmentTask

    forward = LitForcedAlignmentTask.forward

    def forward32(self, x, curves):  # 미리 계산한 특징은 float16 (HubertFA 기본 bf16-mixed용). CPU 32비트로 배우니 맞춘다
        return forward(self, x.float(), curves.float())

    LitForcedAlignmentTask.forward = forward32
    import train as hfa_train

    hfa_train.StepProgressBar = Log  # train.py가 넣는 진행 막대 대신 (진행 막대를 껐다)
    t = time.time()
    trainer = hfa_train.ForcedAlignmentTrainer(CONFIG)
    if args.init:
        import torch

        load_init(trainer.model, args.init)
        torch.save({"state_dict": trainer.model.state_dict(), "hyper_parameters": dict(trainer.model.hparams),
                    "pytorch-lightning_version": lightning.__version__},
                   hfa / "ckpt" / NAME / "init.ckpt")  # 배우기 전 (내보내 견주기용, --ckpt init)
    trainer.train()
    note("학습 끝", f"{time.time() - t:.0f}초")


def pick_ckpt(hfa, which):
    folder = hfa / "ckpt" / NAME
    if which == "last":
        return hfa / "ckpt" / f"{NAME}.ckpt"
    if which == "init":
        return folder / "init.ckpt"
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
    ap.add_argument("cmd", choices=["official", "binarize", "train", "export"])
    ap.add_argument("--hfa", required=True)
    ap.add_argument("--data")
    ap.add_argument("--binary", help="미리 계산한 특징을 둘 폴더 (크다: 소리 1시간에 2GB쯤)")
    ap.add_argument("--batch", type=float, default=100, help="초. 한 걸음에 배우는 소리 길이 합 (HubertFA 기본 200)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--until", type=float, help="유닉스 초. 이때까지만 배운다")
    ap.add_argument("--src", help="official: 지금 가사 맞추기 모델 폴더 (model.onnx · vocab.json · config.json)")
    ap.add_argument("--init", help="train: official이 꺼낸 무게에서 시작")
    ap.add_argument("--lr", type=float, help="train: 학습률 (HubertFA 기본 0.0003)")
    ap.add_argument("--out")
    ap.add_argument("--ckpt", default="best", choices=["best", "last", "init"])
    args = ap.parse_args()
    args.cwd = os.getcwd()
    {"official": official, "binarize": binarize, "train": train, "export": export}[args.cmd](args)


if __name__ == "__main__":
    main()

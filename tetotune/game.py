"""GAME (OpenVPI의 노래 → 음표 모델, MIT): 노래를 음표로 나눈다. OpenUtau에 GAME을 설치해 두었으면 그 모델을,
없으면 앱이 처음 조교할 때 OpenUtau 패키지 목록에서 받아 둔 모델(models_home())을 쓴다.
실제 테토 · 사람 노래 비교(devtools/accuracy.py)에서 원래 방법보다 가사 · 음표가 더 잘 맞았다.

OpenUtau의 GameOnnxBackend와 같은 순서로 모델 넷을 돌린다:
encoder → segmenter (D3PM 8번) → bd2dur → estimator. 긴 곡은 쉼표(조용한 곳)에서 20초 안쪽 조각으로 잘라 넣는다.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import audio

FILES = ("encoder.onnx", "segmenter.onnx", "estimator.onnx", "bd2dur.onnx")
STEPS, SEG_THRESHOLD, SEG_RADIUS, EST_THRESHOLD = 8, 0.2, 2, 0.2
MAX_CHUNK = 20.0
SEED = 20251021


REGISTRY = "https://openutau.github.io/svs-index/registry/v1/softwares/all.json"


def models_home():
    """앱이 받은 모델을 두는 곳 (TETOTUNE_MODELS로 바꿀 수 있다)."""
    if os.environ.get("TETOTUNE_MODELS"):
        return Path(os.environ["TETOTUNE_MODELS"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "TetoTune" / "models"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "AutoTuner" / "models"


def _complete(d):
    return (d / "config.json").exists() and all((d / f).exists() for f in FILES)


def model_dir():
    """쓸 GAME 모델 폴더: OpenUtau 의존성 폴더(OpenUtau → 의존성 설치) → 앱이 받아 둔 것. onnxruntime이 없거나 모델이 없으면 None."""
    import importlib.util

    if importlib.util.find_spec("onnxruntime") is None:
        return None
    for home in (Path.home() / "Library" / "OpenUtau", Path.home() / ".local" / "share" / "OpenUtau"):
        if _complete(home / "Dependencies" / "game"):
            return home / "Dependencies" / "game"
    return models_home() / "game" if _complete(models_home() / "game") else None


def ensure_model(progress=lambda s: None):
    """GAME 모델이 없으면 OpenUtau 패키지 목록(svs-index)에서 받아 models_home()/game에 푼다 → 폴더 또는 None (못 받음)."""
    import importlib.util
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    if importlib.util.find_spec("onnxruntime") is None:
        return None
    found = model_dir()
    if found:
        return found
    dest = models_home() / "game"
    progress("음표 나누기 모델(GAME) 받는 중 (처음 한 번)")
    req = urllib.request.Request(REGISTRY, headers={"User-Agent": "OpenUtau", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        items = json.load(r)["items"]
    sw = next(s for s in items if s.get("id") == "game")
    url = sw["versions"][-1]["mirrors"][0]["url"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        zpath = Path(tmp) / "game.oudep"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OpenUtau"}), timeout=60) as r, \
                open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        out = Path(tmp) / "game"
        with zipfile.ZipFile(zpath) as z:
            for m in z.namelist():  # 압축 밖으로 나가는 경로는 풀지 않는다
                if m.startswith("/") or ".." in Path(m).parts:
                    raise ValueError("GAME 모델 압축 파일이 이상해요")
            z.extractall(out)
        if not _complete(out):  # 폴더 하나 안에 들어 있는 압축
            inner = next((d for d in out.iterdir() if d.is_dir() and _complete(d)), None)
            if inner is None:
                raise ValueError("GAME 모델 파일이 모자라요")
            out = inner
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(out), str(dest))
    return dest


class Game:
    def __init__(self, folder):
        import onnxruntime as ort

        self.cfg = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8"))
        if hasattr(ort, "set_seed"):  # 나누기 모델(D3PM)이 난수를 써서: 같은 노래면 늘 같은 음표가 나오게
            ort.set_seed(SEED)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.s = {f[:-5]: ort.InferenceSession(str(Path(folder) / f), opts, providers=["CPUExecutionProvider"])
                  for f in FILES}
        self.sr = int(self.cfg.get("samplerate", 44100))

    def _run(self, name, feeds):
        sess = self.s[name]
        names = {i.name for i in sess.get_inputs()}
        out = sess.run(None, {k: v for k, v in feeds.items() if k in names})
        return dict(zip([o.name for o in sess.get_outputs()], out))

    def transcribe(self, y, language=None):
        """y: self.sr 모노 → [(시작초, 끝초, 음 높이 midi 소수 또는 None(쉼))]."""
        wav = np.clip(y, -1, 1).astype(np.float32)[None, :]
        enc = self._run("encoder", {"waveform": wav, "duration": np.array([len(y) / self.sr], np.float32)})
        x_seg, x_est, mask_t = enc["x_seg"], enc["x_est"], enc["maskT"]
        T = x_seg.shape[1]
        known = np.zeros((1, T), bool)
        bounds = np.zeros((1, T), bool)
        feeds = {"x_seg": x_seg, "known_boundaries": known, "maskT": mask_t,
                 "threshold": np.array(SEG_THRESHOLD, np.float32), "radius": np.array(SEG_RADIUS, np.int64)}
        langs = self.cfg.get("languages")
        if langs is not None:
            feeds["language"] = np.array([langs.get(language, 0) if language else 0], np.int64)
        if self.cfg.get("loop", True):
            for i in range(STEPS):
                bounds = self._run("segmenter", {**feeds, "prev_boundaries": bounds,
                                                 "t": np.array([i / STEPS], np.float32)})["boundaries"]
        else:
            bounds = self._run("segmenter", feeds)["boundaries"]
        bd = self._run("bd2dur", {"boundaries": bounds, "maskT": mask_t})
        durs, mask_n = bd["durations"], bd["maskN"]
        est = self._run("estimator", {"x_est": x_est, "boundaries": bounds, "maskT": mask_t, "maskN": mask_n,
                                      "threshold": np.array(EST_THRESHOLD, np.float32)})
        out, t = [], 0.0
        for i in range(durs.shape[1]):
            if not mask_n[0, i]:
                break
            d = float(durs[0, i])
            out.append((t, t + d, float(est["scores"][0, i]) if est["presence"][0, i] else None))
            t += d
        return out


def chunks(voiced, hop, max_len=MAX_CHUNK, min_gap=0.3):
    """노래하지 않는 곳(min_gap초 넘는 무성)의 가운데에서 자른 [(시작초, 끝초)]. 조각은 max_len초 안쪽."""
    from .pitch import _runs

    n = len(voiced)
    cuts = [0.0]
    gaps = [((a + b) / 2 * hop, (b - a) * hop) for a, b in _runs(~voiced) if (b - a) * hop >= min_gap]
    last = 0.0
    for i, (c, _) in enumerate(gaps):
        nxt = gaps[i + 1][0] if i + 1 < len(gaps) else n * hop
        if nxt - last > max_len and c - last > 1.0:
            cuts.append(c)
            last = c
    cuts.append(n * hop)
    out = []
    for a, b in zip(cuts, cuts[1:]):  # 쉼 없이 너무 긴 곳은 그냥 잘게
        while b - a > max_len * 1.5:
            out.append((a, a + max_len))
            a += max_len
        out.append((a, b))
    return out


def transcribe(y, sr, voiced, hop, language=None):
    """→ [(시작초, 끝초, midi 소수)] 쉼은 뺀다. GAME이 없으면 None."""
    folder = model_dir()
    if folder is None:
        return None
    g = Game(folder)
    ys = audio.resample(y, sr, g.sr)
    out = []
    for a, b in chunks(voiced, hop):
        seg = ys[int(a * g.sr):int(b * g.sr)]
        if len(seg) < g.sr * 0.2:
            continue
        for s, e, p in g.transcribe(seg, language):
            if p is not None and e > s:
                out.append((a + s, a + e, p))
    return out

"""align_bench.py의 무거운 엔진들 (쓸 때만 불러온다). register(align_bench 모듈)로 ENGINES에 이름을 넣는다.

- mfa: Montreal Forced Aligner (MFA_CMD, 없으면 PATH의 mfa 명령, 한국어 korean_mfa 음향 모델). 음절 하나를 낱말
  하나로 주고 발음은 ko_g2p.phrase (연음 · 비음화 … 노래할 때 소리 나는 대로)로 사전을 만든다
- mms: torchaudio MMS_FA (wav2vec2 CTC, 1100여 말). 음절마다 로마자(uroman, 없으면 국어 로마자 표기)로 맞춘다
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

import ko_g2p
from tetotune import audio

ab = None  # align_bench 모듈 (register)


# ---------------------------------------------------------------- MFA
def _strip(p):
    """MFA 사전에 없는 음소는 덧붙은 표시(ʰ · ͈ · ̚ · ː)를 떼어 가까운 것으로."""
    base = "".join(c for c in unicodedata.normalize("NFD", p) if not unicodedata.combining(c))
    return base.replace("ʰ", "").replace("ː", "")


def _textgrid(path):
    """긴 TextGrid → {층 이름: [(시작, 끝, 글)]} (MFA가 쓰는 꼴만)."""
    tiers, cur, xmin, xmax = {}, None, None, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("name ="):
            cur = s.split("=", 1)[1].strip().strip('"')
            tiers[cur] = []
        elif s.startswith("xmin =") and cur is not None:
            xmin = float(s.split("=", 1)[1])
        elif s.startswith("xmax =") and cur is not None:
            xmax = float(s.split("=", 1)[1])
        elif s.startswith("text =") and cur is not None:
            tiers[cur].append((xmin, xmax, s.split("=", 1)[1].strip().strip('"')))
    return tiers


def _tok(k, s):
    """조각 k의 음절 s를 가리키는 낱말 (글자만: MFA가 숫자 · 기호를 다르게 다룰 수 있어서)."""
    abc = str.maketrans("0123456789", "abcdefghij")
    return f"w{k:05d}x{s:03d}".translate(abc)


class MFA:
    name = "mfa"
    extra = []

    def __init__(self):
        # conda 환경 안의 mfa는 그 환경을 켜야 돈다 (MFA_CMD="micromamba run -n mfa mfa")
        self.exe = shlex.split(os.environ["MFA_CMD"]) if os.environ.get("MFA_CMD") else [shutil.which("mfa")]
        if not self.exe[0]:
            raise RuntimeError("mfa 명령이 없어요 (conda-forge montreal-forced-aligner)")
        self.acoustic = os.environ.get("MFA_ACOUSTIC", "korean_mfa")
        root = Path(os.environ.get("MFA_ROOT_DIR", Path.home() / "Documents" / "MFA"))
        self.phones = set()
        for d in (root / "pretrained_models" / "dictionary").glob("korean_mfa*.dict"):
            for line in d.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    self.phones.update(parts[-1].split())
        print(f"  mfa: {' '.join(self.exe)}, 음향 모델 {self.acoustic}, 사전 음소 {len(self.phones)}개", flush=True)
        self.res = {}

    def _phone(self, p):
        if not self.phones or p in self.phones:
            return p
        q = _strip(p)
        return q if q in self.phones else p

    def prepare(self, items):
        tmp = Path(tempfile.mkdtemp(prefix="mfa_"))
        corpus, out = tmp / "corpus", tmp / "out"
        corpus.mkdir()
        lines, missing = [], set()
        for k, it in enumerate(items):
            y, _ = ab.clip(it, 16000)
            audio.write(corpus / f"u{k:05d}.wav", y, 16000)
            toks = []
            for s, ph in enumerate(ko_g2p.phrase(it["syls"])):
                tok = _tok(k, s)
                toks.append(tok)
                ph = [self._phone(p) for p in (ph or ["ɐ"])]
                missing.update(p for p in ph if self.phones and p not in self.phones)
                lines.append(f"{tok}\t{' '.join(ph)}")
            (corpus / f"u{k:05d}.lab").write_text(" ".join(toks), encoding="utf-8")
        if missing:
            print(f"  mfa: 사전에 없는 음소 {sorted(missing)}", flush=True)
        (tmp / "dict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        jobs = str(max(1, (os.cpu_count() or 2)))
        cmd = self.exe + ["align", str(corpus), str(tmp / "dict.txt"), self.acoustic, str(out), "--clean", "--single_speaker",
               "--beam", os.environ.get("MFA_BEAM", "100"), "--retry_beam", os.environ.get("MFA_RETRY_BEAM", "400"),
               "-j", jobs, "--output_format", "json"] + self.extra
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  mfa align: 끝남 {r.returncode}", flush=True)
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-3000:], flush=True)
        got = 0
        for k, it in enumerate(items):
            words, phones = self._read(out, f"u{k:05d}")
            if words is None:
                continue
            res = []
            for s in range(len(it["syls"])):
                tok = _tok(k, s)
                w = next((x for x in words if x[2] == tok), None)
                if w is None:
                    res = None
                    break
                inside = [p for p in phones if p[0] >= w[0] - 1e-4 and p[1] <= w[1] + 1e-4]
                vow = next((p[0] for p in inside if _strip(p[2]) in {"ɐ", "ɛ", "e", "ʌ", "o", "u", "ɨ", "i"}), w[0])
                res.append((it["a"] + w[0], it["a"] + vow))
            if res is not None:
                self.res[k] = res
                got += 1
        print(f"  mfa: 조각 {len(items)}개 중 {got}개 맞춤", flush=True)
        self.index = {id(it): k for k, it in enumerate(items)}
        shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _read(out, stem):
        js = next(out.rglob(stem + ".json"), None)
        if js is not None:
            d = json.loads(js.read_text(encoding="utf-8"))
            tiers = d.get("tiers", {})

            def entries(name):
                t = tiers.get(name) or next((v for k, v in tiers.items() if k.endswith(name)), None)
                return [(float(a), float(b), str(c)) for a, b, c in (t or {}).get("entries", [])]
            return entries("words"), entries("phones")
        tg = next(out.rglob(stem + ".TextGrid"), None)
        if tg is None:
            return None, None
        tiers = _textgrid(tg)
        words = next((v for k, v in tiers.items() if k.endswith("words")), [])
        phones = next((v for k, v in tiers.items() if k.endswith("phones")), [])
        return words, phones

    def align(self, item):
        return self.res.get(self.index.get(id(item)))


class MFAFine(MFA):
    """MFA + --fine_tune (경계를 1ms 단위로 한 번 더 다듬는다)."""
    name = "mfa_ft"
    extra = ["--fine_tune"]


# ---------------------------------------------------------------- MMS_FA
_RR_CHO = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_RR_JUNG = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu",
            "eu", "ui", "i"]
_RR_JONG = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t", "ng",
            "t", "t", "k", "t", "p", "t"]


def romanize(syl):
    """한글 한 음절 → 로마자 (uroman이 있으면 그것, 없으면 국어의 로마자 표기법). 글자는 a~z만."""
    try:
        import uroman
        if not hasattr(romanize, "_u"):
            romanize._u = uroman.Uroman()
        r = romanize._u.romanize_string(syl)
    except Exception:
        k = ord(syl) - 0xAC00
        r = _RR_CHO[k // 588] + _RR_JUNG[(k % 588) // 28] + _RR_JONG[k % 28]
    r = re.sub(r"[^a-z]", "", r.lower())
    return r or "a"


class MMS:
    name = "mms"

    def __init__(self):
        import torch
        import torchaudio

        self.torch = torch
        b = torchaudio.pipelines.MMS_FA
        self.model = b.get_model(with_star=False).eval()
        self.tok, self.aligner = b.get_tokenizer(), b.get_aligner()
        print(f"  mms: torchaudio {torchaudio.__version__}, 로마자 예: 사랑 → {romanize('사')}{romanize('랑')}", flush=True)

    def prepare(self, items):
        pass

    def align(self, item):
        torch = self.torch
        y, sr = ab.clip(item, 16000)
        words = [romanize(s) for s in item["syls"]]
        wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            em, _ = self.model(wav)
        spans = self.aligner(em[0], self.tok(words))
        fs = wav.size(1) / em.size(1) / 16000
        out = []
        for w, sp in zip(words, spans):
            vi = next((k for k, ch in enumerate(w) if ch in "aeiou"), 0)
            vi = min(vi, len(sp) - 1)
            out.append((item["a"] + sp[0].start * fs, item["a"] + sp[vi].start * fs))
        return out


def register(bench):
    global ab
    ab = bench
    bench.ENGINES.update({"mfa": MFA, "mfa_ft": MFAFine, "mms": MMS})

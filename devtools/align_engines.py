"""align_bench.py의 무거운 엔진들 (쓸 때만 불러온다). register(align_bench 모듈)로 ENGINES에 이름을 넣는다.

- mfa: Montreal Forced Aligner (MFA_CMD, 없으면 PATH의 mfa 명령, 한국어 korean_mfa 음향 모델). 음절 하나를 낱말
  하나로 주고 발음은 ko_g2p.phrase (연음 · 비음화 … 노래할 때 소리 나는 대로)로 사전을 만든다
- mms: torchaudio MMS_FA (wav2vec2 CTC, 1100여 말). 음절마다 로마자(uroman, 없으면 국어 로마자 표기)로 맞춘다
  (mms_star: 앞뒤에 아무 소리 토큰)
- w2vko: 한국어 음소 wav2vec2 CTC (slplab, 말소리 108시간), w2vsyl: 한글 음절 CTC (kresnik, WhisperX 한국어 모델)
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
    star = False

    def __init__(self):
        import torch
        import torchaudio

        self.torch = torch
        b = torchaudio.pipelines.MMS_FA
        self.model = b.get_model(with_star=self.star).eval()
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
        toks = self.tok(words)
        if self.star:  # 앞뒤에 <star> 낱말
            star = self.tok(["*"])[0]
            toks = [star] + toks + [star]
        spans = self.aligner(em[0], toks)
        if self.star:
            spans = spans[1:-1]
        fs = wav.size(1) / em.size(1) / 16000
        out = []
        for w, sp in zip(words, spans):
            vi = next((k for k, ch in enumerate(w) if ch in "aeiou"), 0)
            vi = min(vi, len(sp) - 1)
            out.append((item["a"] + sp[0].start * fs, item["a"] + sp[vi].start * fs))
        return out


class MMSStar(MMS):
    """MMS_FA + 앞뒤 <star> (조각 앞뒤의 숨 · 반주 · 잡음을 아무 소리로 받아 준다, ctc-forced-aligner처럼)."""
    name = "mms_star"
    star = True


# ---------------------------------------------------------------- 한국어 wav2vec2 CTC (Hugging Face transformers)
# slplab/wav2vec2-xls-r-300m_phone-mfa_korean: 한국어 음소 40개 (A EO EU …, 이중 모음은 iA · oA처럼 한 토큰, 받침 k p t)
_W2V_CONS = {"k": "G", "ɡ": "G", "k͈": "GG", "kʰ": "Kh", "t": "D", "d": "D", "t͈": "DD", "tʰ": "Th", "p": "B", "b": "B",
             "p͈": "BB", "pʰ": "Ph", "tɕ": "J", "dʑ": "J", "tɕ͈": "JJ", "tɕʰ": "CHh", "sʰ": "S", "ɕʰ": "S", "s͈": "SS",
             "ɕ͈": "SS", "h": "H", "ɦ": "H", "ç": "H", "m": "M", "n": "N", "ɲ": "N", "ŋ": "NG", "ɾ": "R", "ɭ": "L",
             "k̚": "k", "t̚": "t", "p̚": "p"}
_W2V_VOW = {"ɐ": "A", "ɛ": "E", "e": "E", "ʌ": "EO", "o": "O", "u": "U", "ɨ": "EU", "i": "I"}
_W2V_GLIDE = {("j", "ɐ"): "iA", ("j", "ɛ"): "iE", ("j", "e"): "iE", ("j", "ʌ"): "iEO", ("j", "o"): "iO", ("j", "u"): "iU",
              ("w", "ɐ"): "oA", ("w", "ɛ"): "oE", ("w", "e"): "oE", ("w", "ʌ"): "uEO", ("w", "i"): "uI", ("ɥ", "i"): "uI",
              ("ɰ", "i"): "euI"}
_W2V_IVOW = {"ɐ": "iA", "ɛ": "iE", "e": "iE", "ʌ": "iEO", "o": "iO", "u": "iU"}  # ㄴ + 반모음 j (ko_g2p는 ɲ으로 반모음을 뺀다)


def w2v_phones(ph):
    """ko_g2p 음소 목록 → slplab 토큰 목록 + 모음 토큰 자리."""
    out, vi, k = [], None, 0
    while k < len(ph):
        p = ph[k]
        if p in ko_g2p.GLIDES and k + 1 < len(ph) and (p, ph[k + 1]) in _W2V_GLIDE:
            vi = len(out) if vi is None else vi
            out.append(_W2V_GLIDE[(p, ph[k + 1])])
            k += 2
            continue
        if p in _W2V_VOW:
            vi = len(out) if vi is None else vi
            out.append(_W2V_IVOW[p] if k and ph[k - 1] == "ɲ" and p in _W2V_IVOW else _W2V_VOW[p])
        elif p in _W2V_CONS:
            out.append(_W2V_CONS[p])
        k += 1
    return out, (vi or 0)


class W2VKo:
    """한국어로 배운 wav2vec2 CTC로 강제 정렬 (torchaudio.functional.forced_align). 말소리로 배웠다."""
    name = "w2vko"
    repo = "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"

    def __init__(self):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(self.repo)
        self.model = AutoModelForCTC.from_pretrained(self.repo).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id
        print(f"  {self.name}: {self.repo}, 토큰 {len(self.vocab)}개, 빈칸 {self.blank}", flush=True)

    def prepare(self, items):
        pass

    def tokens(self, syls):
        """음절마다 (토큰 번호 목록, 모음 토큰 자리)."""
        out = []
        for ph in ko_g2p.phrase(syls):
            toks, vi = w2v_phones(ph or ["ɐ"])
            out.append(([self.vocab[t] for t in toks], vi))
        return out

    def align(self, item):
        import torchaudio.functional as taf

        torch = self.torch
        y, sr = ab.clip(item, 16000)
        feats = self.proc(y, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(feats.input_values).logits
        lp = torch.log_softmax(logits, dim=-1)
        per = self.tokens(item["syls"])
        flat = [t for toks, _ in per for t in toks]
        if lp.size(1) < len(flat):
            return None
        ali, sc = taf.forced_align(lp, torch.tensor([flat], dtype=torch.int32), blank=self.blank)
        spans = taf.merge_tokens(ali[0], sc[0].exp(), blank=self.blank)  # 토큰마다 (시작, 끝) 프레임
        fs = len(y) / lp.size(1) / 16000
        out, k = [], 0
        for toks, vi in per:
            sp = spans[k:k + len(toks)]
            k += len(toks)
            out.append((item["a"] + sp[0].start * fs, item["a"] + sp[min(vi, len(sp) - 1)].start * fs))
        return out


class W2VSyl(W2VKo):
    """한글 음절 토큰 CTC (WhisperX가 한국어 맞추기에 쓰는 모델). 음절 하나가 토큰 하나라 모음 시작 = 음절 시작."""
    name = "w2vsyl"
    repo = "kresnik/wav2vec2-large-xlsr-korean"

    def tokens(self, syls):
        unk = self.proc.tokenizer.unk_token_id
        return [([self.vocab.get(s, unk)], 0) for s in syls]


def register(bench):
    global ab
    ab = bench
    bench.ENGINES.update({"mfa": MFA, "mfa_ft": MFAFine, "mms": MMS, "mms_star": MMSStar, "w2vko": W2VKo, "w2vsyl": W2VSyl})

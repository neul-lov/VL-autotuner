"""노래 가사 맞추기 모델 (HubertFA, 로컬): 프레이즈마다 가사 음절의 모음이 시작하는 곳을 소리에서 직접 찾는다.

지금까지는 음표 경계(GAME) 중 소리 증거(자음 틈 · 크기 꺼짐)가 센 곳에 음절을 넣었는데, 한 모음을 여러 음으로 부르거나
같은 음으로 다른 음절을 부르면 틀렸다. 가사 맞추기 모델은 가사 음소 순서를 알고 소리에 맞추니까 음절마다 시작(모음)을
10ms 단위로 준다. 그 시각으로 음표 경계를 옮기거나 음표를 나눈다 (UTAU처럼 음표 시작 = 모음 시작, 자음은 그 앞).

모델: wolfgitpr/HubertFA v0.0.7 일본어 노래 모델 (처음 쓸 때 한 번 받는다, 판 고정 + SHA-256). 한글은 OpenUtau KO→JA처럼
일본어 음소로 바꿔서 맞춘다. 디코더는 HubertFA(Apache-2.0)의 tools/decoder.py를 옮겼다.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HFA_URL = "https://github.com/wolfgitpr/HubertFA/releases/download/v0.0.7/1218_hfa_model_new_dict.zip"
HFA_SHA256 = "48bd6dbcc293e47cc6cbcc556baf575e3d91d61d0ac88ddeccd6098f8f346fa2"
HFA_MAX = 400 << 20
USE_FA = True  # 모델이 있으면 가사를 넣은 뒤 프레이즈마다 맞춘다 (CI 시험에서 끄고 켜서 비교)
MIN_CONF = 0.0  # 프레이즈 맞추기 확신이 이보다 낮으면 그 프레이즈는 원래대로 (CI에서 값을 보고 정한다)
NEAR = 0.08  # 초. 음절 시작 가까이(이 안) 음표 경계가 있으면 그 경계를 옮기고, 없으면 음표를 나눈다
MAX_MOVE = 0.0  # 초. 음절 하나라도 이보다 멀리 옮기면 그 프레이즈는 원래대로 (0 = 제한 없음, CI에서 정한다)
MAX_PHRASE = 40.0  # 초. 모델이 배운 길이(45초)보다 긴 프레이즈는 맞추지 않는다
OFFSET = 0.0  # 초. 모델이 찾은 모음 시작에 더한다 (사람 노래 정답과 견줘 정한다)
FIRST_MAX = 0.2  # 초. 프레이즈 첫 음절은 이 안에서만 옮긴다
GAP_EXTEND = False  # 모음 시작이 음표 앞 틈에 있으면 그 음표 시작을 모음 시작으로 당긴다 (False면 음표 시작 그대로)
TTA = 1  # 프레이즈마다 모델을 몇 번 돌릴지: 구간 시작을 모델 한 칸(10ms)의 1/TTA씩 밀어 돌리고 음절마다 가운데 값을 쓴다
JOINT = 0.0  # 초. 이 길이 안의 이웃 프레이즈는 한꺼번에 맞춘다 (0 = 프레이즈마다)
SOFT_SPLIT = False  # 모음 시작이 음표 끝 40ms 안이면 프레이즈를 버리지 않고 뒤 음표(150ms 안)를 쓰거나 40ms 앞에서 나눈다
WIN_BEFORE, WIN_AFTER = 0.4, 0.3  # 초. 프레이즈 앞뒤로 모델에 더 주는 소리 (이웃 프레이즈와 겹치지 않게)
# 한글을 모델 음소로 바꾸는 법. 모델은 일본어 · 중국어 · 영어 노래로 배웠다 (음소 앞에 ja/ · zh/ · en/).
# KO_FINALS "en": 받침 ㄴ · ㅁ · ㅇ · ㄹ을 일본어 ん · r 대신 영어 n · m · ng · l로 (모델에 없으면 일본어 그대로)
# KO_EO "zh": ㅓ를 일본어 o 대신 중국어 e [ɤ]로
KO_FINALS = "ja"
KO_EO = "ja"
VOWELS = {"a", "i", "u", "e", "o", "N", "zh/e"}
LAST = []  # 마지막 조교의 프레이즈마다 {확신, 맞춤 여부, 옮긴 거리} (시험 기록용)


def _ort():
    return importlib.util.find_spec("onnxruntime") is not None


def _complete(d):
    return all((d / f).exists() for f in ("model.onnx", "vocab.json", "config.json"))


def model_dir():
    from .game import models_home

    d = models_home() / "hfa"
    return d if _ort() and _complete(d) else None


def ensure_model(progress=lambda s: None):
    import shutil
    import tempfile
    import zipfile

    from .game import _download, models_home

    if not _ort():
        return None
    found = model_dir()
    if found:
        return found
    dest = models_home() / "hfa"
    dest.parent.mkdir(parents=True, exist_ok=True)
    progress("가사 맞추기 모델(HubertFA) 받는 중 (처음 한 번)")
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        zpath = Path(tmp) / "hfa.zip"
        _download(HFA_URL, zpath, HFA_SHA256, HFA_MAX, "가사 맞추기 모델(HubertFA)")
        out = Path(tmp) / "hfa"
        with zipfile.ZipFile(zpath) as z:
            _check_zip(z)
            z.extractall(out)
        inner = next((p.parent for p in out.rglob("model.onnx") if _complete(p.parent)), None)
        if inner is None:
            raise ValueError("가사 맞추기 모델 파일이 모자라요")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(inner), str(dest))
    return dest


def _check_zip(z, max_unpacked=1 << 30):
    total = 0
    for m in z.infolist():
        parts = Path(m.filename).parts
        if m.filename.startswith(("/", "\\")) or ".." in parts or ":" in m.filename or "\\" in m.filename:
            raise ValueError("가사 맞추기 모델 압축 파일이 이상해요")
        if (m.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("가사 맞추기 모델 압축 파일이 이상해요")
        if m.is_dir():
            continue
        name = Path(m.filename).name
        if (Path(name).suffix.lower() not in {".onnx", ".json", ".txt", ".md", ".yaml"} and name != "VERSION") \
                or len(parts) > 4:
            raise ValueError(f"가사 맞추기 모델 압축 안에 모르는 파일이 있어요: {m.filename}")
        total += m.file_size
        if total > max_unpacked:
            raise ValueError("가사 맞추기 모델 압축을 풀면 너무 커요")


# ---------------------------------------------------------------- 가사 → 음소
_KANA = {}
for _row, _cons in (("あいうえお", ""), ("かきくけこ", "k"), ("さしすせそ", "s"), ("たちつてと", "t"), ("なにぬねの", "n"),
                    ("はひふへほ", "h"), ("まみむめも", "m"), ("らりるれろ", "r"), ("がぎぐげご", "g"), ("ざじずぜぞ", "z"),
                    ("だぢづでど", "d"), ("ばびぶべぼ", "b"), ("ぱぴぷぺぽ", "p")):
    for _k, _v in zip(_row, "aiueo"):
        _KANA[_k] = _cons + _v
_KANA.update({"し": "shi", "ち": "chi", "つ": "tsu", "ふ": "fu", "じ": "ji", "ぢ": "ji", "づ": "zu", "や": "ya", "ゆ": "yu",
              "よ": "yo", "わ": "wa", "を": "o", "ん": "n", "ゔ": "vu", "っ": "cl"})
_YOON = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
_SMALL = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}


def kana_romaji(syl):
    """가나 음절(か · きゃ · ふぁ) → 로마자 (사전 키). 모르면 None."""
    if not syl:
        return None
    if len(syl) == 1:
        return _KANA.get(syl)
    head, tail = _KANA.get(syl[0]), syl[1:]
    if head is None or len(tail) != 1:
        return None
    if tail in _YOON:
        if head in ("shi", "chi", "ji"):
            return head[:-1] + _YOON[tail]
        return head[:-1] + "y" + _YOON[tail]
    if tail in _SMALL:
        v = _SMALL[tail]
        if head in ("fu", "vu"):
            return head[0] + v
        if head in ("te", "de"):
            return head[0] + v
        if head in ("to", "do"):
            return head[0] + v
        if head == "u":
            return "w" + v
        if head == "i":
            return "y" + v
        if head in ("shi", "chi", "ji"):
            return head[:-1] + v
        if head == "tsu":
            return "ts" + v
        if head in ("ku", "gu"):
            return head[0] + "w" + v
        return head[:-1] + v
    return None


_CHO = ["k", "k", "n", "t", "t", "r", "m", "p", "p", "s", "s", "", "ch", "ch", "ch", "k", "t", "p", "h"]
_CHO_VOICED = {0: "g", 3: "d", 7: "b", 12: "j"}  # 모음 · ㄴ · ㄹ · ㅁ · ㅇ 뒤 ㄱ · ㄷ · ㅂ · ㅈ은 울린다
_JUNG = [["a"], ["e"], ["y", "a"], ["y", "e"], ["o"], ["e"], ["y", "o"], ["y", "e"], ["o"], ["w", "a"], ["w", "e"],
         ["w", "e"], ["y", "o"], ["u"], ["w", "o"], ["w", "e"], ["w", "i"], ["y", "u"], ["u"], ["u", "i"], ["i"]]
_JONG = [None, "cl", "cl", "cl", "N", "N", "N", "cl", "r", "cl", "N", "cl", "r", "r", "cl", "r", "N", "cl", "cl", "cl",
         "cl", "N", "cl", "cl", "cl", "cl", "cl", "cl"]
_JONG_EN = {4: "en/n", 5: "en/n", 6: "en/n", 10: "en/m", 16: "en/m", 21: "en/ng", 8: "en/l", 11: "en/l", 12: "en/l",
            13: "en/l", 15: "en/l"}
_PAL = {"k": "ky", "g": "gy", "n": "ny", "h": "hy", "m": "my", "r": "ry", "b": "by", "p": "py", "s": "sh", "t": "ch",
        "d": "j", "z": "j"}


def hangul_phones(syl, prev=None):
    """한글 한 글자 → 일본어 음소 (OpenUtau KO→JA처럼). prev: 앞 글자 (울림 판단)."""
    code = ord(syl) - 0xAC00
    cho, jung, jong = code // 588, (code % 588) // 28, code % 28
    c = _CHO[cho]
    if cho in _CHO_VOICED and prev is not None and "가" <= prev <= "힣" and _JONG[(ord(prev) - 0xAC00) % 28] in (None, "N", "r"):
        c = _CHO_VOICED[cho]
    v = list(_JUNG[jung])
    out = []
    if c and v[0] == "y":
        out.append(_PAL.get(c, c))
        v = v[1:]
    elif c == "s" and v[0] == "i":  # 시 = [ɕi]
        out.append("sh")
    elif c in ("k", "g") and v[0] == "w":
        out.append(c + "w")
        v = v[1:]
    elif c:
        out.append(c)
    if KO_EO == "zh" and jung in (4, 6, 14):  # ㅓ ㅕ ㅝ
        v = v[:-1] + ["zh/e|o"]
    out += v
    if KO_FINALS == "en" and jong in _JONG_EN:
        out.append(_JONG_EN[jong] + "|" + _JONG[jong])
    elif _JONG[jong]:
        out.append(_JONG[jong])
    return out


def _romaji_phones(r):
    """사전에 없는 로마자 → 자음 · 모음 (ti → t i, kwa → kw a, n → N)."""
    if r in ("n", "N"):
        return ["N"]
    if r == "cl":
        return ["cl"]
    if not r or r[-1] not in "aiueo":
        return None
    return [r[:-1], r[-1]] if len(r) > 1 else [r]


class Aligner:
    def __init__(self, folder):
        import onnxruntime as ort

        folder = Path(folder)
        self.vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.mel = cfg["mel_spec_config"]
        self.sr, self.hop = int(self.mel["sample_rate"]), int(self.mel["hop_size"])
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(folder / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.prefix = "ja/" if self.vocab.get("language_prefix") else ""
        self.dict = {}
        dpath = folder / (self.vocab.get("dictionaries") or {}).get("ja", "")
        if dpath.is_file():
            for line in dpath.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    k, v = line.split("\t", 1)
                    self.dict[k.strip()] = v.strip().split()
        self.ids = self.vocab["vocab"]
        self.sil = self.ids.get("SP", 0)

    def phones(self, syl, prev=None):
        """음절 → 모델 음소 목록 (없는 음소가 있으면 None). っ · 받침 막힘(cl)은 모델에서 쉼(SP)과 같은 음소라서
        (merged_phoneme_groups) 빼고, 음절 사이에 넣는 SP가 맡는다."""
        if len(syl) == 1 and "가" <= syl <= "힣":
            ph = hangul_phones(syl, prev)
        else:
            r = kana_romaji(syl)
            if r is None:
                return None
            ph = _romaji_phones(r) if r in ("n", "cl") else (self.dict.get(r) or _romaji_phones(r))
        if not ph:
            return None
        out = []
        for p in ph:
            q = next((q for alt in p.split("|") for q in (alt if "/" in alt else self.prefix + alt, alt)
                      if q in self.ids), None)  # "en/n|N": 앞의 것이 모델에 없으면 뒤의 것
            if q is None:
                return None
            if self.ids[q] != self.sil:
                out.append(q)
        return out or None

    def sequence(self, syls):
        """음절 목록 → (음소 순서 (앞뒤 · 사이에 SP), 음소마다 음절 번호 (SP는 −1)) 또는 None (모르는 음절)."""
        ph_seq, ph_word, prev = ["SP"], [-1], None
        for i, s in enumerate(syls):
            ph = self.phones(s, prev)
            prev = s
            if ph is None:
                return None
            ph_seq += ph + ["SP"]
            ph_word += [i] * len(ph) + [-1]
        return ph_seq, ph_word

    def align(self, y, sr, syls):
        """y: 프레이즈 소리 (앞뒤 조금 여유), syls: 음절 목록 → ([(시작, 모음 시작, 끝)] 음절마다 또는 None, 확신)."""
        from . import audio

        seq = self.sequence(syls)
        if seq is None:
            return None, 0.0
        ph_seq, ph_word = seq
        wav = audio.resample(y, sr, self.sr).astype(np.float32)
        outs = self.sess.run(None, {"waveform": wav[None, :]})
        res = dict(zip([o.name for o in self.sess.get_outputs()], outs))
        ph_times, conf = _decode(self.vocab, self.hop, self.sr, len(wav), res["ph_frame_logits"],
                                 res["ph_edge_logits"], ph_seq)
        return _syllable_times(ph_times, ph_seq, ph_word, len(syls), self.prefix), conf


def _syllable_times(ph_times, ph_seq, ph_word, n, prefix=""):
    """디코더 결과 [(시작, 끝, 음소 번호)] → 음절마다 (시작, 모음 시작, 끝). 한 음절이라도 빠지면 None."""
    out = [None] * n
    for a, b, k in ph_times:
        w = ph_word[k]
        if w < 0:
            continue
        name = ph_seq[k][len(prefix):] if prefix and ph_seq[k].startswith(prefix) else ph_seq[k]
        cur = out[w] or [a, None, b]
        cur[2] = b
        if cur[1] is None and (name in VOWELS or name == "cl"):
            cur[1] = a
        out[w] = cur
    if any(x is None for x in out):
        return None
    return [(a, v if v is not None else a, b) for a, v, b in out]


# ---------------------------------------------------------------- 디코더 (HubertFA tools/decoder.py, Apache-2.0)
def _softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _log_softmax(x, axis):
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def _decode(vocab, hop, sr, n_samples, frame_logits, edge_logits, ph_seq):
    """→ ([(시작초, 끝초, ph_seq 번호)], 확신)."""
    frame_logits, edge_logits = frame_logits[0], edge_logits[0]
    seq_id = np.array([vocab["vocab"][p] for p in ph_seq])
    mask = np.full(vocab["vocab_size"], 1e9)
    mask[seq_id], mask[0] = 0, 0
    frame = hop / sr
    T = min(frame_logits.shape[1], int((n_samples + 0.5) / hop))
    frame_logits, edge_logits = frame_logits[:, :T], edge_logits[:T]
    adj = frame_logits - mask[:, None]
    prob_log = _log_softmax(adj, 0).astype(np.float32)
    edge_pred = np.clip(1 / (1 + np.exp(-edge_logits)), 0, 1).astype(np.float32)
    edge_diff = np.concatenate((np.diff(edge_pred), [0]))
    edge_prob = (edge_pred + np.concatenate(([0], edge_pred[:-1]))).clip(0, 1)
    idx_seq, t_int, fconf = _viterbi(seq_id, prob_log, edge_prob)
    conf = float(np.exp(np.mean(np.log(fconf + 1e-6)) / 3))
    t_int = np.array(t_int, dtype=np.int64)
    frac = (edge_diff[t_int] / 2).clip(-0.5, 0.5)
    t = np.clip(frame * np.concatenate([t_int + frac, [T]]), 0, None)
    return [(float(t[i]), float(t[i + 1]), int(k)) for i, k in enumerate(idx_seq)], conf


def _viterbi(seq_id, ph_prob_log, edge_prob):
    S = len(seq_id)
    T = ph_prob_log.shape[1]
    prob_log = ph_prob_log[seq_id, :]
    cur_max = np.full(S, -np.inf)
    dp = np.full((S, T), -np.inf, dtype=np.float32)
    dp[0, 0] = prob_log[0, 0]
    cur_max[0] = prob_log[0, 0]
    if seq_id[0] == 0 and S > 1:
        dp[1, 0] = prob_log[1, 0]
        cur_max[1] = prob_log[1, 0]
    back = np.full_like(dp, -1, dtype=np.int32)
    e_log, ne_log = np.log(edge_prob + 1e-6), np.log(1 - edge_prob + 1e-6)
    reset = seq_id == 0
    pad = 2 if S >= 2 else 1
    p1 = np.empty(S, dtype=np.float32)
    p2 = np.full(S, -np.inf, dtype=np.float32)
    p3 = np.full(S, -np.inf, dtype=np.float32)
    iv = np.arange(pad, S)
    ia = np.clip(iv - pad + 1, 0, S - 1)
    cond3 = (ia >= S - 1) | (seq_id[ia] == 0)
    for t in range(1, T):
        pl = prob_log[:, t]
        prev = dp[:, t - 1]
        p1[:] = prev + pl + ne_log[t]
        p2[1:] = prev[:S - 1] + pl[:S - 1] + e_log[t] + cur_max[:S - 1] * (T / S)
        cand = prev[:S - pad] + pl[:S - pad] + e_log[t] + cur_max[:S - pad] * (T / S)
        p3[iv] = np.where(cond3, cand, -np.inf)
        st = np.vstack((p1, p2, p3))
        k = st.argmax(0)
        dp[:, t], back[:, t] = st[k, np.arange(S)], k
        stay = k == 0
        np.maximum(cur_max, pl, out=cur_max, where=stay)
        np.copyto(cur_max, pl, where=~stay)
        cur_max[reset] = 0.0
        p2[1:], p3[iv] = -np.inf, -np.inf
    s = 0 if S == 1 else (S - 2 if dp[-2, -1] > dp[-1, -1] and seq_id[-1] == 0 else S - 1)
    idx_seq, t_int, fconf = [], [], []
    for t in range(T - 1, -1, -1):
        fconf.append(dp[s, t])
        if back[s, t] != 0:
            idx_seq.append(s)
            t_int.append(t)
            if back[s, t] == 1:
                s -= 1
            elif back[s, t] == 2:
                s -= 2
    idx_seq.reverse()
    t_int.reverse()
    fconf.reverse()
    fconf = np.exp(np.diff(np.pad(fconf, (1, 0), "constant", constant_values=0.0), 1))
    return idx_seq, t_int, fconf


# ---------------------------------------------------------------- 음표에 넣기
def refine(notes, y, sr):
    """가사를 넣은 음표(notes.align 뒤) → 프레이즈마다 모델로 음절 모음 시작을 찾아 음표 경계를 옮기거나 나눈다.
    JOINT초 안의 이웃 프레이즈는 먼저 한꺼번에 맞춰서, 가사 넣기가 음절을 옆 프레이즈에 잘못 나눈 것도 고친다
    (못 맞추면 프레이즈마다). 모델이 없거나 맞추지 못한 프레이즈는 그대로. → (notes, 고친 프레이즈 수)."""
    from . import notes as nt

    LAST.clear()
    folder = model_dir() if USE_FA else None
    if folder is None:
        return notes, 0
    al = Aligner(folder)
    groups = nt._phrases(notes)

    def window(i, j):  # 프레이즈 i..j의 소리 구간 (앞뒤 조금 여유, 이웃 프레이즈와 겹치지 않게)
        a = max(0.0, groups[i][0].start - WIN_BEFORE, groups[i - 1][-1].end if i else 0.0)
        b = groups[j][-1].end + WIN_AFTER
        if j + 1 < len(groups):
            b = min(b, groups[j + 1][0].start)
        return a, b

    out, fixed, gi = [], 0, 0
    while gi < len(groups):
        gj = gi
        if JOINT > 0:
            while gj + 1 < len(groups) and window(gi, gj + 1)[1] - window(gi, gj + 1)[0] <= JOINT:
                gj += 1
        if gj > gi:
            new = _fit(al, [x for g in groups[gi:gj + 1] for x in g], *window(gi, gj), y, sr, f"{gi}-{gj}")
            if new is not None:
                out.extend(new)
                fixed += gj - gi + 1
                gi = gj + 1
                continue
            LAST.pop()  # 한꺼번에 못 맞춘 기록은 빼고 프레이즈마다 다시
        new = _fit(al, groups[gi], *window(gi, gi), y, sr, gi)
        if new is None:
            out.extend(groups[gi])
        else:
            out.extend(new)
            fixed += 1
        gi += 1
    return out, fixed


def _fit(al, g, a, b, y, sr, tag):
    """음표 g(프레이즈 하나 또는 이웃 여럿)의 음절을 소리 y[a:b]에서 맞춰 넣은 새 음표, 못 하면 None. LAST에 기록."""
    from . import notes as nt

    core_g, closures = _without_closures(g) if nt.CLOSURE else (g, set())
    syls = [x.lyric for x in core_g if x.lyric != "+"]
    if not syls or b - a > MAX_PHRASE:
        LAST.append({"phrase": tag, "syls": syls, "skip": "long" if syls else "empty"})
        return None
    if al.sequence(syls) is None:
        LAST.append({"phrase": tag, "syls": syls, "unknown": "".join(syls)[:40]})
        return None
    try:
        times, conf = _align_tta(al, y, sr, a, b, syls)
    except Exception as e:  # 모델이 이 소리를 못 다루면 이 프레이즈만 원래대로
        times, conf = None, 0.0
        LAST.append({"phrase": tag, "error": str(e)[:200]})
    why = []
    new = _place(core_g, [a + v for _, v, _ in times], syls, why=why) if times is not None and conf >= MIN_CONF else None
    if new is not None and closures:
        new = nt._insert_closures(new, closures)
    old = [x.start for x in g if x.lyric != "+"]
    moved = [abs(x.start - o) for x, o in zip([x for x in new if x.lyric != "+"], old)] if new else []
    if MAX_MOVE > 0 and moved and max(moved) > MAX_MOVE:  # 음절이 통째로 다른 자리로 가는 것은 모델이 틀릴 때가 많다
        new = None
    LAST.append({"phrase": tag, "conf": round(float(conf), 3), "n": len(syls), "placed": new is not None,
                 "moved_ms": round(1000 * float(np.median(moved)), 1) if moved else None,
                 "moved_max_ms": round(1000 * float(np.max(moved)), 1) if moved else None,
                 "before": old, "after": [x.start for x in new if x.lyric != "+"] if new else None,
                 "syls": syls, "raw": [a + v for _, v, _ in times] if times is not None else None,
                 "raw_start": [a + t for t, _, _ in times] if times is not None else None,
                 "why": why[0] if why else ("align" if times is None else ("conf" if conf < MIN_CONF else ""))})
    return new


def _align_tta(al, y, sr, a, b, syls):
    """al.align을 구간 시작을 조금씩 밀어 TTA번 돌려서 음절마다 (시작, 모음 시작, 끝)의 가운데 값 (구간 시작 a 기준).
    한 번은 틀려도 나머지가 맞으면 가운데 값은 맞는다. 맞춘 번이 없으면 (None, 확신)."""
    hop = getattr(al, "hop", 441) / getattr(al, "sr", 44100)
    runs, confs = [], []
    for k in range(max(1, TTA)):
        d = k * hop / max(1, TTA)
        times, conf = al.align(y[int((a + d) * sr):int(b * sr)], sr, syls)
        confs.append(conf)
        if times is not None:
            runs.append(np.array(times, dtype=float) + d)
    if not runs:
        return None, float(np.mean(confs))
    med = np.median(np.stack(runs), axis=0) if len(runs) > 1 else runs[0]
    return [tuple(float(v) for v in row) for row in med], float(np.mean(confs))


def _without_closures(g):
    """っ 음표(앞 음절 끝의 막힘, notes._insert_closures)를 앞 음표에 도로 붙인다 → (음표, っ 앞 음절 번호).
    모델로 음절 자리를 맞춘 뒤 같은 방법으로 다시 나눈다."""
    from . import notes as nt

    out, closures, n = [], set(), 0
    for x in g:
        if x.lyric == "っ" and n and abs(out[-1].end - x.start) < 1e-6:
            p = out[-1]
            out[-1] = nt.Note(p.start, x.end, p.tone, p.lyric, p.strength)
            closures.add(n - 1)
            continue
        out.append(x)
        if x.lyric != "+":
            n += 1
    return out, closures


def _place(g, onsets, syls, near=NEAR, why=None):
    """프레이즈 음표 g에 음절 syls를 onsets(모음 시작) 자리로 넣는다. 가까운 음표 경계(near 안)는 그 자리로 옮기고,
    없으면 그 자리에서 음표를 나눈다 (같은 음으로 다른 음절). 음 높이 · 길이가 너무 짧아지면 None (원래대로)."""
    from . import notes as nt

    g = [nt.Note(x.start, x.end, x.tone, "+", x.strength) for x in g]
    picked = []  # 음절마다 고른 음표
    for j, v in enumerate(onsets):
        v += OFFSET
        if picked:
            v = max(v, picked[-1].start + 0.04)
        if j == 0:
            x = g[0]
            if abs(v - x.start) <= FIRST_MAX and x.end - v > 0.05:
                x.start = v
            picked.append(x)
            continue
        last = g.index(picked[-1])
        cands = [i for i in range(last + 1, len(g)) if abs(g[i].start - v) <= near]
        if cands:
            k = min(cands, key=lambda i: abs(g[i].start - v))
            x, prev = g[k], g[k - 1]
            if abs(prev.end - x.start) < 1e-6:  # 붙은 경계: 두 음표 모두 30ms는 남게
                if v - prev.start > 0.03 and x.end - v > 0.03:
                    prev.end = x.start = v
            elif prev.end + 0.005 < v < x.end - 0.03:  # 틈 뒤 음표: 시작만
                x.start = v
            picked.append(x)
            continue
        k = next((i for i in range(last, len(g)) if g[i].start < v < g[i].end), None)
        if k is not None:
            x = g[k]
            if SOFT_SPLIT and x.end - v < 0.04 and x.end - x.start >= 0.12:
                nxt = k + 1 if k + 1 < len(g) and g[k + 1].start - v <= 0.15 else None
                if nxt is not None:
                    if GAP_EXTEND and g[nxt].start - x.end > 0.005:
                        g[nxt].start = x.end + 0.005
                    picked.append(g[nxt])
                    continue
                v = x.end - 0.0401
            if v - x.start < 0.04 or x.end - v < 0.04:
                if why is not None:
                    why.append(f"split{j}")
                return None
            g.insert(k + 1, nt.Note(v, x.end, x.tone, "+", 3.0))
            x.end = v
            picked.append(g[k + 1])
            continue
        # 틈 안: 바로 뒤 음표가 가까우면 그 음표
        k = next((i for i in range(last + 1, len(g)) if g[i].start >= v), None)
        if k is None or g[k].start - v > 0.15:
            if why is not None:
                why.append(f"gap{j}" if k is not None else f"end{j}")
            return None
        if GAP_EXTEND and k > 0 and v > g[k - 1].end + 0.005:
            g[k].start = v
        picked.append(g[k])
    for x, syl in zip(picked, syls):
        x.lyric = syl
    return g

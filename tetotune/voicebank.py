"""이 Mac의 OpenUtau 가수 찾기 (테토 우선), 그리고 음원에 어떤 소리가 있는지 (숨소리 별칭, 연속음인지).

OpenUtau의 가수 id는 Singers 폴더에서 character.txt가 있는 폴더까지의 상대 경로다.
"""
import json
import os
import re
from pathlib import Path

TETO = ("テト", "Teto", "teto", "TETO", "테토")
# 테토 음원이 여럿이면 기본 목소리를 먼저 (속삭임 · 영어 · 힘 준 소리 같은 특수 음원은 뒤로)
SPECIAL = ("囁", "ささやき", "sasayaki", "whisper", "English", "english", "力", "rikimi", "叫", "sakebi", "弱", "jaku",
           "エッジ", "edge", "滑", "nameraka", "extra", "Extra", "Lite", "lite", "beta", "ameya", "tetovoice")
# 숨소리 별칭 (음원마다 다름). 들숨을 먼저 찾는다.
INHALE = ("吸", "息吸", "br_in", "breath_in", "inhale")
EXHALE = ("息", "br", "breath", "R息", "hh", "Breath", "息1")

PRESAMP = "OpenUtau.Plugin.Builtin.JapanesePresampPhonemizer"
KO_TO_JA = "OpenUtau.Plugin.Builtin.KOtoJAPhonemizer"
DEFAULT = "OpenUtau.Core.DefaultPhonemizer"


def openutau_home():
    """OpenUtau 데이터 폴더: Mac은 ~/Library/OpenUtau, 리눅스(시험용)는 $XDG_DATA_HOME/OpenUtau."""
    mac = Path.home() / "Library" / "OpenUtau"
    if mac.exists():
        return mac
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "OpenUtau"


def singer_roots():
    home = openutau_home()
    roots = [home / "Singers", home / "Content" / "Singers"]
    try:
        prefs = json.loads((home / "prefs.json").read_text(encoding="utf-8-sig"))
        extra = prefs.get("AdditionalSingerPath")
        if extra:
            roots.append(Path(extra))
    except Exception:
        pass
    return [r for r in roots if r.exists()]


def _read_text(p):
    data = p.read_bytes()
    for enc in ("utf-8-sig", "shift_jis", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _name(folder):
    for f in ("character.yaml", "character.txt"):
        p = folder / f
        if p.exists():
            m = re.search(r"^\s*name\s*[:=]\s*(.+)$", _read_text(p), re.M)
            if m:
                return m.group(1).strip().strip("'\"")
    return folder.name


def list_singers(roots=None):
    """→ [{id, name, path, root}] 테토가 먼저."""
    out = []
    for root in roots or singer_roots():
        for c in root.rglob("character.txt"):
            folder = c.parent
            out.append({"id": str(folder.relative_to(root)), "name": _name(folder), "path": str(folder), "root": str(root)})
    def rank(s):
        text = s["name"] + " " + s["id"]
        return (not any(k in text for k in TETO), any(k in text for k in SPECIAL), s["name"])
    out.sort(key=rank)
    return out


def _otos(folder):
    """→ {별칭: (wav 경로, offset, cutoff)} (같은 별칭이 여럿이면 처음 것)."""
    out = {}
    for oto in Path(folder).rglob("oto.ini"):
        for line in _read_text(oto).splitlines():
            if "=" not in line:
                continue
            wav, rest = line.split("=", 1)
            parts = rest.split(",")
            alias = parts[0].strip() or Path(wav).stem
            try:
                offset, cutoff = float(parts[1] or 0), float(parts[3] or 0)
            except (IndexError, ValueError):
                offset, cutoff = 0.0, 0.0
            out.setdefault(alias, (oto.parent / wav.strip(), offset, cutoff))
    return out


def _level(entry):
    """OpenUtau가 실제로 내는 크기 (dB). OpenUtau는 샘플마다 (0.5 / 최댓값)^0.86 만큼 키워서(자동 음량, P=86)
    조용한 숨 샘플도 꽤 커진다. 그래서 "큰 쪽 80% 크기 − 0.86 × 최댓값"으로 비교한다."""
    import numpy as np
    import soundfile as sf

    path, offset, cutoff = entry
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    y = y.mean(axis=1)
    a = int(offset / 1000 * sr)
    b = a + int(-cutoff / 1000 * sr) if cutoff < 0 else len(y) - int(cutoff / 1000 * sr)
    seg = y[a:b]
    hop = int(sr * 0.01)
    if len(seg) < hop * 3:
        return None
    rms = np.sqrt(np.mean(seg[:len(seg) // hop * hop].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    peak = float(np.abs(seg).max()) + 1e-9
    return float(20 * np.log10(np.percentile(rms, 80)) - 0.86 * 20 * np.log10(peak))


def inspect(folder):
    """음원 폴더 → {name, phonemizer, kind, breath_in, breath_out, subbank_affixes}."""
    folder = Path(folder)
    otos = _otos(folder)
    aliases = set(otos)
    info = {"name": _name(folder), "path": str(folder), "aliases": len(aliases)}
    # 하위 음원(음 높이별 · 표정별) 접두사 · 접미사를 떼고 본다 (예: "息↑", "あ_D4")
    affixes = set()
    yaml_path = folder / "character.yaml"
    phonemizer = None
    if yaml_path.exists():
        import yaml
        try:
            cfg = yaml.safe_load(_read_text(yaml_path)) or {}
            phonemizer = cfg.get("default_phonemizer")
            for sb in cfg.get("subbanks") or []:
                affixes.add((sb.get("prefix") or "", sb.get("suffix") or ""))
        except Exception:
            pass
    affixes.add(("", ""))

    def has(alias):
        return any(f"{p}{alias}{s}" in aliases for p, s in affixes) or alias in aliases

    vcv = sum(1 for a in aliases if re.match(r"^[aiueon-] \S", a))
    info["kind"] = "VCV" if vcv > 50 else ("CVVC" if any(re.match(r"^[aiueo] [kgsztdnhbpmyrw]", a) for a in aliases) else "CV")
    info["breath_in"] = next((a for a in INHALE if has(a)), None)
    info["breath_out"] = next((a for a in EXHALE if has(a)), None)
    # 숨 샘플이 모음 샘플보다 얼마나 작은지 (숨 음표 크기를 원래 보컬의 숨 크기에 맞출 때 쓴다)
    info["breath_gap_db"] = None
    breath = info["breath_in"] or info["breath_out"]
    vowel = next((a for a in ("あ", "a", "- あ", "- a", "あ_C4", "a あ") if a in otos), None)
    if breath and vowel:
        def entry(alias):
            return otos.get(alias) or next(v for k, v in otos.items() if alias in k)
        lb, lv = _level(entry(breath)), _level(otos[vowel])
        if lb is not None and lv is not None:
            info["breath_gap_db"] = round(lb - lv, 1)
    info["phonemizer"] = phonemizer or (PRESAMP if any(re.search(r"[぀-ゟ]", a) for a in aliases) else DEFAULT)
    info["japanese"] = any(re.search(r"[぀-ゟ]", a) for a in aliases)
    return info


def find(prefer_path=None):
    """쓸 가수 → (id, 정보) 또는 (None, None)."""
    singers = list_singers()
    if prefer_path:
        p = Path(prefer_path)
        for s in singers:
            if Path(s["path"]) == p:
                return s["id"], inspect(p)
        if p.exists():  # Singers 폴더 밖: 폴더 이름만 쓴다 (OpenUtau에서 다시 골라야 할 수 있음)
            return p.name, inspect(p)
    if not singers:
        return None, None
    return singers[0]["id"], inspect(singers[0]["path"])

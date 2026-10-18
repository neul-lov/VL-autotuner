"""가사 글 → 음표 하나에 들어갈 음절 목록.

- 일본어: 가나 한 글자 (きゃ처럼 작은 글자는 앞 글자와 함께). 가타카나는 히라가나로 (테토 음원 별칭이 히라가나).
  ー는 앞 모음 늘이기라서 "+" (OpenUtau에서 앞 음절을 이어 부르는 표시).
  한자는 fugashi가 있으면 읽는 법(히라가나)으로 바꾼다.
- 한국어: 한 글자 (OpenUtau의 KO to JA 음소기가 일본어 음원으로 읽게 해 준다).
- 영어 · 로마자: 띄어쓰기 단위.
- "/"나 띄어쓰기로 직접 나눠도 되고, 줄 바꿈은 숨 쉬는 곳(프레이즈)으로 본다.
"""
import re

SMALL = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
KANA = re.compile(r"[぀-ゟ゠-ヿー]")
HANGUL = re.compile(r"[가-힣]")
KANJI = re.compile(r"[一-鿿々]")


def kata_to_hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def kanji_to_kana(text):
    """한자 섞인 일본어 → 히라가나. fugashi(unidic-lite)가 없으면 그대로 (경고는 부르는 쪽에서)."""
    if not KANJI.search(text):
        return text, True
    try:
        import fugashi
    except Exception:
        return text, False
    tagger = fugashi.Tagger()
    out = []
    for word in tagger(text):
        kana = getattr(word.feature, "kana", None) or getattr(word.feature, "pron", None)
        surface = word.surface
        out.append(kata_to_hira(kana) if KANJI.search(surface) and kana and kana != "*" else surface)
    return "".join(out), True


def language(text):
    if HANGUL.search(text):
        return "ko"
    if KANA.search(text) or KANJI.search(text):
        return "ja"
    return "en"


def split(text):
    """→ (줄마다 음절 목록, 경고 목록)."""
    warnings = []
    text, ok = kanji_to_kana(text)
    if not ok:
        warnings.append("한자를 읽는 법으로 바꾸지 못했어요 (fugashi 없음). 한자는 히라가나로 써 주세요")
    lines = []
    for raw in re.split(r"[\n\r]+", text):
        syl = []
        for chunk in re.split(r"[\s/,、。，.!?！？「」『』()（）\"']+", raw):
            if not chunk:
                continue
            if KANA.search(chunk) or HANGUL.search(chunk) or KANJI.search(chunk):
                for c in kata_to_hira(chunk):
                    if c in SMALL and syl:
                        syl[-1] += kata_to_hira(c)
                    elif c == "ー":
                        syl.append("+")
                    elif c.isspace():
                        continue
                    else:
                        syl.append(c)
            else:
                syl.append(chunk)
        if syl:
            lines.append(syl)
    return lines, warnings

"""가사 글 → 음표 하나에 들어갈 음절 목록.

- 일본어: 가나 한 글자 (きゃ · ふぁ처럼 작은 글자로 한 소리가 되면 앞 글자와 함께). 가타카나는 히라가나로 (테토 음원 별칭이 히라가나).
  ー · 〜 · ~ 는 앞 모음 늘이기라서 "+" (OpenUtau에서 앞 음절을 이어 부르는 표시). ねぇ · あぁ처럼 같은 모음의 작은 글자도 "+".
  fugashi가 있으면 한자는 읽는 법(히라가나)으로, 조사 は · へ는 소리 나는 대로(わ · え) 바꾼다.
- 한국어: 한 글자 (OpenUtau의 KO to JA 음소기가 일본어 음원으로 읽게 해 준다).
- 영어 · 로마자: 단어 단위.
- "/"나 띄어쓰기로 직접 나눠도 되고, 줄 바꿈은 숨 쉬는 곳(프레이즈)으로 본다. 그 밖의 문장 부호 · 기호 · 숫자는 뺀다.
"""
import re
import unicodedata

SMALL = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
KANA = re.compile(r"[぀-ゟ゠-ヿー]")
HANGUL = re.compile(r"[가-힣]")
KANJI = re.compile(r"[一-鿿々]")
SYLLABLE = re.compile(r"[ぁ-ゖゝゞァ-ヺヽヾ一-鿿々가-힣]")  # 한 글자가 한 음절 (・ ゛ 같은 기호는 빼고)
LONG = set("ー〜~+")  # 앞 모음 늘이기 (～는 NFKC로 ~가 된다)
# 작은 ぁぃぅぇぉ와 한 소리가 되는 앞 글자 (OpenUtau 기본 presamp 표에서 모음이 다른 것만).
# 나머지는 따로 부른다: 같은 모음이면 늘이기 (ねぇ → ね +), 아니면 큰 글자 (かぃ → か い)
COMBO = {"ぁ": "ふゔつ", "ぃ": "てでふゔうすずつ", "ぅ": "とどゔ", "ぇ": "いうしじちつふゔきぎにひびぴみり", "ぉ": "うふゔつ"}
VOWEL = {c: v for v, row in (("あ", "あかがさざただなはばぱまやらわぁゃゎ"), ("い", "いきぎしじちぢにひびぴみりぃゐ"),
                             ("う", "うくぐすずつづぬふぶぷむゆるぅゅゔ"), ("え", "えけげせぜてでねへべぺめれぇゑ"),
                             ("お", "おこごそぞとどのほぼぽもよろをぉょ")) for c in row}
# unidic 읽기가 노래와 다른 흔한 낱말 (私 → わたくし, 明日 → あす로 읽어서 음절 수까지 달라진다)
READING = {"私": "わたし", "明日": "あした"}
# 何 + 이 조사는 なに (unidic은 늘 なん). 何の · 何で · 何て는 なん이 맞아서 뺀다
NANI = {"を", "が", "も", "に", "か", "より"}


def kata_to_hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _tagger():
    """fugashi(unidic-lite) 형태소 분석기. 추가 패키지라서 없으면 None."""
    try:
        import fugashi
        return fugashi.Tagger()
    except Exception:
        return None


def kanji_to_kana(text):
    """일본어 → 부르는 대로의 히라가나: 한자는 읽는 법으로, 조사 は · へ는 わ · え로. 띄어쓰기 · 줄 바꿈은 그대로.
    fugashi(unidic-lite)가 없으면 그대로 (한자가 있으면 ok=False, 경고는 부르는 쪽에서)."""
    if not (KANA.search(text) or KANJI.search(text)):
        return text, True
    tagger = _tagger()
    if tagger is None:
        return text, not KANJI.search(text)
    # 가나만 쓴 가사도 돌린다 (わたしは → わたしわ). 가나 낱말은 그대로 두고 조사 は · へ만 바꾸므로
    # はな · はじめて · へいわ 같은 낱말은 は · へ로 남는다. 줄 첫머리 · 띄어쓰기 뒤의 は · へ는 조사로 보지 않는다
    # (はしれ를 は + しれ로 잘못 나누는 일이 있어서). 감탄사 뒤도 조사가 아니다 (ははははは = はははは + は로 나눈다)
    words = list(tagger(text))

    def pos(i):
        return getattr(words[i].feature, "pos1", None) if 0 <= i < len(words) else None

    out = []
    for i, word in enumerate(words):
        surface = word.surface
        kana = getattr(word.feature, "kana", None) or getattr(word.feature, "pron", None)
        pron = getattr(word.feature, "pron", None)
        if surface in READING:
            surface = READING[surface]
        elif surface == "何" and pos(i + 1) == "助詞" and words[i + 1].surface in NANI and not words[i + 1].white_space:
            surface = "なに"
        elif KANJI.search(surface):
            if kana and kana != "*":
                surface = kata_to_hira(kana)
        elif surface in ("は", "へ") and pos(i) == "助詞" and i and not word.white_space \
                and pos(i - 1) not in ("補助記号", "空白", "感動詞") and pron and pron != "*":
            surface = kata_to_hira(pron)
        out.append(word.white_space + surface)  # 띄어쓰기 · 줄 바꿈은 white_space에 있다
    return "".join(out), True


def language(text):
    if HANGUL.search(text):
        return "ko"
    if KANA.search(text) or KANJI.search(text):
        return "ja"
    return "en"


def _small(syl, c):
    """작은 ぁぃぅぇぉゃゅょゎ: 한 소리가 되면 앞 음절에 붙이고, 아니면 따로."""
    if c not in COMBO:  # ゃゅょゎ
        if syl and syl[-1] != "+":
            syl[-1] += c
        else:
            syl.append(c)
        return
    big = chr(ord(c) + 1)  # ぁ → あ
    prev = syl[-1] if syl else ""
    v = VOWEL.get(prev[-1]) if prev else None
    if len(prev) == 1 and prev in COMBO[c]:
        syl[-1] += c
    elif prev == "+" or v == big or (v == "お" and big == "う"):
        syl.append("+")  # ねぇ · あぁ · よぉ · もぅ(= もう): 앞 모음 늘이기
    else:
        syl.append(big)


def split(text):
    """→ (줄마다 음절 목록, 경고 목록)."""
    warnings = []
    text = unicodedata.normalize("NFKC", text)  # 반각 가타카나 · 전각 기호 · 숫자를 보통 글자로
    text, ok = kanji_to_kana(text)
    if not ok:
        warnings.append("한자를 읽는 법으로 바꾸지 못했어요 (fugashi 없음). 한자는 히라가나로 써 주세요")
    elif KANJI.search(text):
        warnings.append(f"한자 {''.join(dict.fromkeys(KANJI.findall(text)))}의 읽는 법을 찾지 못했어요. 히라가나로 써 주세요")
    lines, digits = [], False
    for raw in re.split(r"[\n\r]+", text):
        syl, word = [], ""
        for c in kata_to_hira(raw) + " ":
            if c.isalpha() and c not in LONG and not SYLLABLE.match(c):
                word += c  # 영어 · 로마자는 글자를 모아 한 음절
                continue
            if word:
                syl.append(word)
                word = ""
            if c in LONG:
                syl.append("+")
            elif c in SMALL:
                _small(syl, c)
            elif SYLLABLE.match(c):
                syl.append(c)
            elif c.isdigit():
                digits = True
            # 나머지 (띄어쓰기 · / · 문장 부호 · ♪ 같은 기호)는 음절 사이를 나눌 뿐
        if syl:
            lines.append(syl)
    if digits:
        warnings.append("가사의 숫자는 뺐어요. 부르는 대로 가나 · 한글로 써 주세요")
    return lines, warnings

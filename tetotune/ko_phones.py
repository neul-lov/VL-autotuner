"""한글 가사 → 발음 (음절마다 음소 목록). 한국어 가사 맞추기 모델(GTSinger로 배운 HubertFA) · MFA 한국어 모델에 줄 발음.

표기 그대로가 아니라 노래할 때 소리 나는 대로: 연음(먹어 → 머거), 비음화(국물 → 궁물), ㄹ 동화(신라 → 실라),
경음화(학교 → 학꾜), 거센소리(좋다 → 조타 · 입학 → 이팍), 울림소리 사이 예사소리(가방의 ㅂ → b).
음소 이름은 MFA korean_mfa 사전(IPA)과 같다: 모음 ɐ ɛ e ʌ o u ɨ i, 반모음 j w ɥ ɰ, 받침 k̚ t̚ p̚ n m ŋ ɭ.
"""

# 첫소리 (ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ)
CHO = ["k", "k͈", "n", "t", "t͈", "ɾ", "m", "p", "p͈", "sʰ", "s͈", "", "tɕ", "tɕ͈", "tɕʰ", "kʰ", "tʰ", "pʰ", "h"]
VOICED = {"k": "ɡ", "t": "d", "p": "b", "tɕ": "dʑ"}  # 울림소리 사이 예사소리
TENSE = {"k": "k͈", "t": "t͈", "p": "p͈", "sʰ": "s͈", "tɕ": "tɕ͈"}  # 막힌 받침 뒤
ASPIRATE = {"k": "kʰ", "t": "tʰ", "p": "pʰ", "tɕ": "tɕʰ"}  # ㅎ과 만나면
# 가운데 소리 (ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ)
JUNG = [["ɐ"], ["ɛ"], ["j", "ɐ"], ["j", "ɛ"], ["ʌ"], ["e"], ["j", "ʌ"], ["j", "e"], ["o"], ["w", "ɐ"], ["w", "ɛ"],
        ["w", "e"], ["j", "o"], ["u"], ["w", "ʌ"], ["w", "e"], ["ɥ", "i"], ["j", "u"], ["ɨ"], ["ɰ", "i"], ["i"]]
# 받침 (없음 ㄱ ㄲ ㄳ ㄴ ㄵ ㄶ ㄷ ㄹ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅁ ㅂ ㅄ ㅅ ㅆ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ):
# (받침으로 날 때 소리, 다음 음절이 ㅇ으로 시작하면 넘어가는 첫소리 번호 또는 None, 남는 받침 소리)
JONG = [None, ("k̚", 0, None), ("k̚", 1, None), ("k̚", 9, "k̚"), ("n", 2, None), ("n", 12, "n"), ("n", None, "n"),
        ("t̚", 3, None), ("ɭ", 5, None), ("k̚", 0, "ɭ"), ("m", 6, "ɭ"), ("ɭ", 7, "ɭ"), ("ɭ", 9, "ɭ"), ("ɭ", 16, "ɭ"),
        ("p̚", 17, "ɭ"), ("ɭ", None, "ɭ"), ("m", 6, None), ("p̚", 7, None), ("p̚", 9, "p̚"), ("t̚", 9, None),
        ("t̚", 10, None), ("ŋ", None, None), ("t̚", 12, None), ("t̚", 14, None), ("k̚", 15, None), ("t̚", 16, None),
        ("p̚", 17, None), (None, None, None)]
VOWELS = {"ɐ", "ɛ", "e", "ʌ", "o", "u", "ɨ", "i"}
GLIDES = {"j", "w", "ɥ", "ɰ"}
SONORANT = {"n", "m", "ŋ", "ɭ"}


def _jamo(c):
    k = ord(c) - 0xAC00
    return k // 588, (k % 588) // 28, k % 28


def is_hangul(c):
    return len(c) == 1 and "가" <= c <= "힣"


def phrase(syls):
    """한 프레이즈(이어 부르는 음절들) → 음절마다 음소 목록. 한글이 아닌 음절은 None."""
    js = [_jamo(s) if is_hangul(s) else None for s in syls]
    onset = [None if j is None else j[0] for j in js]  # 첫소리 번호 (연음으로 바뀔 수 있다)
    coda = [None if j is None else JONG[j[2]] for j in js]  # 받침 (소리, 넘어감, 남음)
    out = []
    for i, j in enumerate(js):
        if j is None:
            out.append(None)
            continue
        cho, jung, _ = j
        nxt = js[i + 1] if i + 1 < len(js) else None
        prev_coda = out[i - 1][-1] if i and out[i - 1] and out[i - 1][-1] not in VOWELS | GLIDES else None
        prev_open = i > 0 and out[i - 1] is not None and prev_coda is None  # 앞 음절이 모음으로 끝남
        c = onset[i]
        ph = []
        # 첫소리
        if c is not None and c != 11:
            p = CHO[c]
            if c == 18:  # ㅎ
                if prev_coda in ("k̚", "t̚", "p̚"):  # 입학 → 이팍: 앞 받침이 거센소리가 된다 (앞 음절에서 이미 뺐다)
                    p = None
                elif prev_open or prev_coda in SONORANT:
                    p = "ɦ"  # 울림소리 사이 ㅎ은 약하게
                elif JUNG[jung][0] in ("i", "j"):
                    p = "ç"
            elif c == 5:  # ㄹ
                p = "ɭ" if prev_coda in ("n", "ɭ") else "ɾ"
            elif c == 2 and prev_coda == "ɭ":  # ㄹ + ㄴ → ㄹㄹ
                p = "ɭ"
            elif p in TENSE and prev_coda in ("k̚", "t̚", "p̚"):
                p = TENSE[p]
            elif p in VOICED and (prev_open or prev_coda in SONORANT):
                p = VOICED[p]
            if p == "sʰ" and JUNG[jung][0] in ("i", "j", "ɥ"):
                p = "ɕʰ"
            elif p == "s͈" and JUNG[jung][0] in ("i", "j", "ɥ"):
                p = "ɕ͈"
            if c == 2 and JUNG[jung][0] in ("i", "j"):
                p = "ɲ"
            if p:
                ph.append(p)
        ph += JUNG[jung][1:] if ph and ph[-1] == "ɲ" and JUNG[jung][0] == "j" else JUNG[jung]
        # 받침
        cd = coda[i]
        if cd is not None:
            sound, moves, stays = cd
            ncho = nxt[0] if nxt is not None else None
            if ncho == 11 and moves is not None:  # 연음: 받침이 다음 음절 첫소리로
                # 구개음화: ㄷ · ㅌ 받침 + 이 → 지 · 치 (같이 → 가치)
                onset[i + 1] = {3: 12, 16: 14}.get(moves, moves) if nxt[1] == 20 else moves
                if stays:
                    ph.append(stays)
            elif ncho == 18 and sound in ("k̚", "t̚", "p̚"):  # 받침 + ㅎ → 다음 음절 거센소리
                onset[i + 1] = {"k̚": 15, "t̚": 16, "p̚": 17}[sound]
            elif sound is None:  # ㅎ 받침: 다음 ㄱ · ㄷ · ㅈ을 거세게, 아니면 소리 없음
                if ncho in (0, 3, 12):
                    onset[i + 1] = {0: 15, 3: 16, 12: 14}[ncho]
            else:
                if ncho in (2, 6) and sound in ("k̚", "t̚", "p̚"):  # 비음화
                    sound = {"k̚": "ŋ", "t̚": "n", "p̚": "m"}[sound]
                elif ncho == 5 and sound == "n":  # ㄴ + ㄹ → ㄹㄹ
                    sound = "ɭ"
                ph.append(sound)
        out.append(ph)
    return out


def vowel_index(ph):
    """음소 목록에서 첫 모음(반모음 다음)의 자리."""
    return next((k for k, p in enumerate(ph) if p in VOWELS), 0)

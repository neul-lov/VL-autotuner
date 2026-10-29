"""GTSinger 한국어 (사람이 고친 음소 시각) → HubertFA 학습 파일 (wavs/ + transcriptions.csv).

GTSinger는 MFA korean_mfa 음소(IPA)로 표시돼 있고 이음(kʷ · mʲ · c · ɸʷ · 긴 모음 …)까지 나눠 적었다.
앱이 맞출 때 쓰는 발음(tetotune.ko_phones: 노래할 때 소리 나는 대로)과 같은 음소로 바꿔서 배운다.
- 모음(ɐ ɛ e ʌ o u ɨ i)은 GTSinger 시각 그대로 (앱이 쓰는 것은 모음 시작이라 여기를 바꾸지 않는다)
- 모음과 모음 사이(앞 음절 받침 + 뒤 음절 첫소리 · 반모음)는 앱 발음의 음소를 GTSinger의 그 구간에 넣는다:
  음소 수가 같으면 GTSinger 경계 그대로, 다르면 고르게 나눈다 (연음 · 이음이 달라도 모음 시각은 그대로)
- 쉼(<SP>) · 숨(<AP>)은 SP. 한 음절에 모음이 하나가 아닌 낱말이 있는 프레이즈는 뺀다

소리 · 가사는 이 기계에만 두고 숫자만 찍는다 (GTSinger는 CC BY-NC-SA 4.0, 가사를 찍지 않는다).
사용: python gtsinger_hfa.py GTSinger/Korean out/hfa_data [--speech] [--eval-songs 6] [--json out/gts.json]
"""
import argparse
import csv
import json
import os
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tetotune import ko_phones as kp  # noqa: E402

SIL = {"<SP>", "<AP>", "_NONE", "breathe", "breath", "sil", "silence", "", "SP", "AP"}
MIN_PH = 0.02  # 초. 구간이 이보다 짧으면 앞 모음 끝에서 빌린다
MAX_LEN = 45.0  # 초. HubertFA가 이보다 긴 파일은 버린다


def vowel(ph):
    p = ph.replace("ː", "")
    return p if p in kp.VOWELS else None


def phrase_segments(words, why):
    """이어 부른 낱말들 → [(음소, 시작, 끝)] 또는 None (why에 까닭을 센다)."""
    syls, theirs = [], []
    for w in words:
        chars = [c for c in w["word"] if kp.is_hangul(c)]
        if not chars or any(c.isalnum() and not kp.is_hangul(c) for c in w["word"]):
            why["한글 아닌 낱말"] += 1
            return None
        ph = list(zip(w["ph"], w["ph_start"], w["ph_end"]))
        if any(p in SIL or p == "spn" for p, _, _ in ph):
            why["낱말 안 쉼 · 모르는 소리"] += 1
            return None
        k = 0
        while sum(1 for p, _, _ in ph if vowel(p)) > len(chars) and k + 1 < len(ph):
            # 한 모음을 음표마다 나눠 적은 것 (같은 모음이 잇달아): 하나로 합친다
            if vowel(ph[k][0]) and vowel(ph[k][0]) == vowel(ph[k + 1][0]):
                ph[k:k + 2] = [(ph[k][0], ph[k][1], ph[k + 1][2])]
                why["잇단 같은 모음 합침"] += 1
            else:
                k += 1
        if sum(1 for p, _, _ in ph if vowel(p)) != len(chars):
            why["음절과 모음 수가 다름"] += 1
            return None
        syls += chars
        theirs += ph
    ours = kp.phrase(syls)
    if any(o is None for o in ours):
        why["발음 못 만듦"] += 1
        return None
    vi = [k for k, (p, _, _) in enumerate(theirs) if vowel(p)]
    nuc = [[theirs[k][1], theirs[k][2]] for k in vi]
    if any(b <= a for a, b in nuc):
        why["모음 길이 0"] += 1
        return None
    on = [o[:kp.vowel_index(o)] for o in ours]
    co = [o[kp.vowel_index(o) + 1:] for o in ours]
    nv = [o[kp.vowel_index(o)] for o in ours]
    n = len(syls)
    start, end = theirs[0][1], theirs[-1][2]
    regions = []  # (우리 음소, GTSinger 음소 [(이름, 시작, 끝)], 구간 시작, 구간 끝)
    for k in range(n + 1):
        lo = vi[k - 1] + 1 if k else 0
        hi = vi[k] if k < n else len(theirs)
        a = nuc[k - 1][1] if k else start
        b = nuc[k][0] if k < n else end
        regions.append(((co[k - 1] if k else []) + (on[k] if k < n else []), theirs[lo:hi], a, b))
    out = []
    for k, (mine, their, a, b) in enumerate(regions):
        if not mine:
            if b - a > 1e-6:
                if 0 < k:  # 앞 모음을 늘린다 (뒤 모음 시작은 그대로)
                    nuc[k - 1][1] = b
                    out[-1] = (out[-1][0], out[-1][1], b)
                else:  # 프레이즈 첫 음절이 모음으로 시작하는데 그 앞에 소리가 있으면 쉼으로
                    out.append(("SP", a, b))
                why["우리 발음에 없는 자음 구간"] += 1
        else:
            need = MIN_PH * len(mine)
            if b - a < need:
                lend = need - (b - a)
                if k == 0 and a - lend >= 0:  # 프레이즈 앞 쉼에서 빌린다 (convert가 앞 SP를 줄인다)
                    a -= lend
                elif k and nuc[k - 1][1] - nuc[k - 1][0] >= lend + 0.05:
                    a -= lend
                    nuc[k - 1][1] = a
                    out[-1] = (out[-1][0], out[-1][1], a)
                else:
                    why["자음 넣을 자리 없음"] += 1
                    return None
                why["앞에서 빌림"] += 1
            if len(mine) == len(their) and their[0][1] >= a - 1e-6 and their[-1][2] <= b + 1e-6 \
                    and all(e - s >= 0.005 for _, s, e in their):
                times = [(max(a, s), min(b, e)) for _, s, e in their]
                times[0] = (a, times[0][1])
                times[-1] = (times[-1][0], b)
                why["구간 음소 수 같음"] += 1
            else:
                step = (b - a) / len(mine)
                times = [(a + i * step, a + (i + 1) * step) for i in range(len(mine))]
                why["구간 고르게 나눔"] += 1
            out += [(p, s, e) for p, (s, e) in zip(mine, times)]
        if k < n:
            out.append((nv[k], nuc[k][0], nuc[k][1]))
    return out


def word_list(data):
    """json 내용 → 낱말 목록 (낱말마다 word · ph · ph_start · ph_end) 또는 None."""
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)), None)
    if not isinstance(data, list) or not data or not all(isinstance(w, dict) for w in data):
        return None
    if not all(k in data[0] for k in ("word", "ph", "ph_start", "ph_end")):
        return None
    return data


def convert(words, wav_len, why):
    """GTSinger 파일 하나의 낱말 목록 → [(음소, 시작, 끝)] (빈틈 · 쉼은 SP) 또는 None."""
    segs, cur = [], []

    def flush():
        if cur:
            got = phrase_segments(cur, why)
            if got is None:
                return False
            segs.extend(got)
            why["프레이즈"] += 1
        cur.clear()
        return True

    for w in words:
        if w["word"].strip() in SIL or all(p in SIL for p in w["ph"]):
            if not flush():
                return None
        else:
            cur.append(w)
    if not flush():
        return None
    out, t = [], 0.0
    for p, s, e in segs:
        if p == "SP":
            continue
        if s > t + 1e-6:
            out.append(["SP", t, s])
        elif s < t - 1e-6:  # 겹침 (앞 프레이즈 끝을 빌렸을 때)
            if out and out[-1][0] == "SP" and out[-1][1] < s:
                out[-1][2] = s
            else:
                why["겹침"] += 1
                return None
        out.append([p, s, e])
        t = e
    if wav_len > t + 1e-6:
        out.append(["SP", t, wav_len])
    return [x for x in out if x[2] - x[1] > 1e-6]


def songs_of(root, speech):
    """→ {노래 열쇠: [(가수, 기법, 묶음, json, wav)]} (같은 노래의 여러 판은 같은 열쇠)."""
    out = {}
    for js in sorted(Path(root).rglob("*.json")):
        wav = js.with_suffix(".wav")
        if not wav.exists():
            continue
        rel = js.relative_to(root).parts
        if len(rel) < 5:
            continue
        singer, tech, song, group = rel[0], rel[1], rel[2], rel[3]
        if "speech" in group.lower() and not speech:
            continue
        out.setdefault((singer, song), []).append((singer, tech, group, js, wav))
    return out


def wav_seconds(path):
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="GTSinger의 Korean 폴더")
    ap.add_argument("out")
    ap.add_argument("--speech", action="store_true", help="말소리(Paired_Speech_Group)도 넣는다")
    ap.add_argument("--eval-songs", type=int, default=6, help="가수마다 고르게, 학습에 쓰지 않고 HubertFA 채점에 쓸 노래 수")
    ap.add_argument("--json")
    args = ap.parse_args()
    songs = songs_of(args.root, args.speech)
    keys = sorted(songs)
    rng = random.Random(7)
    by_singer = {}
    for key in keys:
        by_singer.setdefault(key[0], []).append(key)
    eval_keys = set()
    singers = sorted(by_singer)
    for i in range(args.eval_songs):
        pool = [k for k in by_singer[singers[i % len(singers)]] if k not in eval_keys]
        if pool:
            eval_keys.add(rng.choice(pool))
    why, phones, theirs, stats = Counter(), Counter(), Counter(), {}
    rows = {"train": [], "eval": []}
    out = Path(args.out)
    for split in rows:
        (out / split / "wavs").mkdir(parents=True, exist_ok=True)
    song_id = {k: i for i, k in enumerate(keys)}
    lens, keys_shown = [], False
    for key in keys:
        split = "eval" if key in eval_keys else "train"
        for singer, tech, group, js, wav in songs[key]:
            data = json.loads(js.read_text(encoding="utf-8"))
            words = word_list(data)
            if words is None:
                why["json 모양 모름"] += 1
                if why["json 모양 모름"] == 1:  # 열쇠 이름만 (값은 가사라서 찍지 않는다)
                    print("모르는 json 모양:", type(data).__name__,
                          sorted(data)[:20] if isinstance(data, dict) else sorted(data[0])[:20] if data and isinstance(data[0], dict) else "")
                continue
            if not keys_shown:
                print("json 낱말 열쇠:", sorted(words[0]))
                keys_shown = True
            for w in words:
                theirs.update(w.get("ph", []))
            sec = wav_seconds(wav)
            st = stats.setdefault(f"{singer} / {group}", {"files": 0, "sec": 0.0, "kept": 0, "kept_sec": 0.0})
            st["files"] += 1
            st["sec"] += sec
            lens.append(sec)
            if sec > MAX_LEN:
                why["45초보다 긺"] += 1
                continue
            segs = convert(words, sec, why)
            if segs is None:
                why["못 바꾼 파일"] += 1
                continue
            name = f"s{song_id[key]:03d}_{zlib.crc32(f'{tech}/{group}'.encode()) % 10 ** 6:06d}_{js.stem}"
            dst = out / split / "wavs" / f"{name}.wav"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(wav.resolve(), dst)
            rows[split].append((name, " ".join(p for p, _, _ in segs), " ".join(f"{e - s:.5f}" for _, s, e in segs)))
            phones.update(p for p, _, _ in segs)
            st["kept"] += 1
            st["kept_sec"] += sec
    for split, rr in rows.items():
        with open(out / split / "transcriptions.csv", "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["name", "ph_seq", "ph_dur"])
            wr.writerows(rr)
    inv = sorted(p for p in phones if p != "SP")
    (out / "ko_phones.txt").write_text("".join(f"{p}\t{p}\n" for p in inv), encoding="utf-8")
    hours = {s: round(v["kept_sec"] / 3600, 2) for s, v in stats.items()}
    print(f"노래 {len(keys)}곡 (채점용 {len(eval_keys)}곡), 파일 학습 {len(rows['train'])} · 채점 {len(rows['eval'])}")
    print("묶음마다 (파일 수, 쓴 파일, 시간):")
    for s, v in sorted(stats.items()):
        print(f"  {s}: {v['files']}개 · {v['kept']}개 · {v['kept_sec'] / 3600:.2f}시간 (전체 {v['sec'] / 3600:.2f})")
    if lens:
        ls = sorted(lens)
        print(f"파일 길이 초: 가운데 {ls[len(ls) // 2]:.1f}, 90% {ls[int(len(ls) * 0.9)]:.1f}, 가장 긴 것 {ls[-1]:.1f}")
    print("바꾸기 기록:", dict(why.most_common()))
    print("앱 발음 음소 (수):", dict(phones.most_common()))
    print("GTSinger 음소 (수):", dict(theirs.most_common()))
    if args.json:
        Path(args.json).write_text(json.dumps({"hours": hours, "why": why, "phones": phones, "theirs": theirs,
                                               "files": {k: len(v) for k, v in rows.items()}}, ensure_ascii=False,
                                              indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

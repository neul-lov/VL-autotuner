"""CI: 분리한 보컬(sep_…, 옆에 반주 bk_…)마다 다시 뽑은 보컬을 TETOTUNE_CLEAN_CACHE에 미리 만들어 둔다.

리눅스 CPU에서는 보컬 다시 분리가 곡 길이의 몇 배 걸려서, 방식마다 다시 돌리면 비교 잡 하나가 두 시간을 넘는다.
여러 잡이 곡을 나눠 한 번씩만 돌리고 (--shard i/n), 비교 잡들은 조교할 때 같은 파일이면 이것을 쓴다 (clean.vocals_cached).
python devtools/preclean.py out/sep out/csdsep --shard 0/10
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tetotune import audio, clean, tune  # noqa: E402


def shard_of(songs, n):
    """크기가 큰 곡부터 지금까지 가장 덜 맡은 조각에 (조각마다 걸리는 시간이 비슷하게)."""
    load, out = [0] * n, {}
    for p in sorted(songs, key=lambda p: (-p.stat().st_size, p.name)):
        k = load.index(min(load))
        out[p] = k
        load[k] += p.stat().st_size
    return out


def main(argv):
    shard, n = 0, 1
    if "--shard" in argv:
        i = argv.index("--shard")
        shard, n = map(int, argv[i + 1].split("/"))
        del argv[i:i + 2]
    if not clean.CACHE:
        sys.exit("TETOTUNE_CLEAN_CACHE를 주세요")
    if not clean.ensure("vocals", print):
        sys.exit("보컬 다시 분리 모델이 없어요")
    songs = [p for d in argv for p in sorted(Path(d).glob("sep_*.wav")) if p.with_name("bk_" + p.name[4:]).exists()]
    mine = [p for p, k in shard_of(songs, n).items() if k == shard]
    print(f"{len(songs)}곡 중 {len(mine)}곡 (조각 {shard}/{n})", flush=True)
    for wav in sorted(mine):
        bk = wav.with_name("bk_" + wav.name[4:])
        t = time.time()
        y, sr = audio.load(str(wav))
        warnings = []
        tune._for_analysis(y, sr, str(bk), warnings, lambda p, s: None, str(wav))
        f = clean.cache_file(str(wav), str(bk), sr)
        print(f"  {wav.parent.name}/{wav.name}: {len(y) / sr:.1f}초 → {time.time() - t:.0f}초, 저장 {f.exists()} {warnings or ''}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

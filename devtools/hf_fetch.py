"""Hugging Face 저장소에서 고른 파일만 받기 (CI용). 판(커밋)을 고정하고, 여러 개를 함께 받고,
너무 많은 요청(429) · 끊김이면 기다렸다 다시 받는다. 받기 전에 저장소 조건(gated · 라이선스)과 폴더 요약을 찍는다.
노래 제목이 든 경로는 찍지 않는다 (앞 세 단계 폴더 · 확장자마다 수와 크기만).

사용: python hf_fetch.py AaronZ345/GTSinger 받을폴더 --type dataset --include "Korean/**/*.wav" --include "Korean/**/*.json"
      [--list-only] [--revision 커밋]
"""
import argparse
import fnmatch
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    from huggingface_hub import HfApi, hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("--type", default="dataset")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--revision")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    api = HfApi()
    info = api.repo_info(args.repo, repo_type=args.type, revision=args.revision)
    card = info.card_data.to_dict() if getattr(info, "card_data", None) else {}
    print(f"{args.repo}: 커밋 {info.sha}, gated {getattr(info, 'gated', None)}, 라이선스 {card.get('license')}", flush=True)
    files = [f for f in api.list_repo_tree(args.repo, repo_type=args.type, revision=info.sha, recursive=True)
             if getattr(f, "size", None) is not None]
    top, ext, size_top, sub, groups = Counter(), Counter(), Counter(), {}, Counter()
    for f in files:
        parts = f.path.split("/")
        key = "/".join(parts[:min(2, len(parts) - 1)]) or "."
        top[key] += 1
        size_top[key] += f.size
        ext[Path(f.path).suffix.lower() or "(없음)"] += 1
        for d in range(2, len(parts) - 1):
            sub.setdefault((key, d), set()).add(parts[d])
            if "group" in parts[d].lower():
                groups[parts[d]] += 1
    print(f"파일 {len(files)}개, {sum(f.size for f in files) / 1e9:.2f}GB. 폴더(두 단계까지)마다:", flush=True)
    for k in sorted(top):
        deeper = " · ".join(f"{d + 1}단계 {len(v)}개" for (kk, d), v in sorted(sub.items()) if kk == k)
        print(f"  {k}: 파일 {top[k]}개 · {size_top[k] / 1e9:.2f}GB ({deeper or '아래 폴더 없음'})")
    print("묶음 폴더 이름:", dict(groups.most_common()), flush=True)
    print("확장자:", dict(ext.most_common()), flush=True)
    pick = [f for f in files if any(fnmatch.fnmatch(f.path, p) for p in args.include)]
    print(f"고른 파일 {len(pick)}개, {sum(f.size for f in pick) / 1e9:.2f}GB", flush=True)
    if args.list_only or not pick:
        return
    done = [0]
    t0 = time.time()

    def get(f):
        for k in range(10):
            try:
                hf_hub_download(args.repo, f.path, repo_type=args.type, revision=info.sha, local_dir=args.dest)
                break
            except Exception as e:  # 429 · 끊김: 점점 오래 기다린다
                if k == 9 or "404" in str(e) or "401" in str(e) or "403" in str(e):
                    raise
                time.sleep(min(120, 5 * 2 ** k))
        done[0] += 1
        if done[0] % 500 == 0:
            print(f"  {done[0]}/{len(pick)} ({time.time() - t0:.0f}초)", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(get, pick))
    print(f"받음 {len(pick)}개 ({time.time() - t0:.0f}초)", flush=True)


if __name__ == "__main__":
    main()

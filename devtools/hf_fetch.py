"""Hugging Face 저장소에서 고른 파일만 받기 (CI용). 판(커밋)을 고정하고, 여러 개를 함께 받고,
요청 한도(429)에 걸리면 한도가 풀릴 때까지 기다렸다 다시 받는다. 받기 전에 저장소 조건(gated · 라이선스)과 폴더 요약을 찍는다.
노래 제목이 든 경로는 찍지 않는다 (앞 두 단계 폴더 · 확장자마다 수와 크기만).
로그인 없이 받으면 IP마다 5분에 요청 수가 정해져 있어서, 파일마다 /resolve/ 주소에 요청 한 번만 쓴다
(hf_hub_download는 파일마다 두세 번 써서 몇백 개 만에 한도에 닿았다). 목록도 --path 폴더 안만 본다.

사용: python hf_fetch.py AaronZ345/GTSinger 받을폴더 --type dataset --path Korean --include "Korean/*.wav"
      [--exclude "*/Paired_Speech_Group/*"] [--list-only] [--revision 커밋]
"""
import argparse
import fnmatch
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def wait_seconds(r):
    """429 응답에서 한도가 풀릴 때까지 남은 초 (RateLimit: "resolvers";r=0;t=55 · Retry-After, 없으면 1분)."""
    m = re.search(r"\bt=(\d+)", r.headers.get("RateLimit", ""))
    if m:
        return int(m.group(1)) + 2
    after = r.headers.get("Retry-After", "")
    return int(after) + 2 if after.isdigit() else 60


def main():
    import requests
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("--type", default="dataset")
    ap.add_argument("--path", help="이 폴더 안만 본다 (목록 요청을 줄인다)")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--revision")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    api = HfApi()
    info = api.repo_info(args.repo, repo_type=args.type, revision=args.revision)
    card = info.card_data.to_dict() if getattr(info, "card_data", None) else {}
    print(f"{args.repo}: 커밋 {info.sha}, gated {getattr(info, 'gated', None)}, 라이선스 {card.get('license')}", flush=True)
    files = [f for f in api.list_repo_tree(args.repo, path_in_repo=args.path, repo_type=args.type, revision=info.sha,
                                           recursive=True)
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
    pick = [f for f in files if any(fnmatch.fnmatch(f.path, p) for p in args.include)
            and not any(fnmatch.fnmatch(f.path, p) for p in args.exclude)]
    print(f"고른 파일 {len(pick)}개, {sum(f.size for f in pick) / 1e9:.2f}GB", flush=True)
    if args.list_only or not pick:
        return
    headers = build_hf_headers()
    local = threading.local()
    lock = threading.Lock()
    state = {"done": 0, "said": 0.0, "waits": 0, "stop": False}
    t0 = time.time()

    def get(item):
        try:
            fetch(*item)
        except BaseException:
            state["stop"] = True  # 남은 파일은 받지 않고 바로 끝낸다
            raise

    def fetch(n, f):
        if state["stop"]:
            return
        dest = Path(args.dest) / f.path
        if dest.exists() and dest.stat().st_size == f.size:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        url = hf_hub_url(args.repo, f.path, repo_type=args.type, revision=info.sha)
        if not hasattr(local, "s"):
            local.s = requests.Session()
        errors = 0
        for _ in range(80):
            try:
                with local.s.get(url, headers=headers, stream=True, timeout=120) as r:
                    if r.status_code == 429:  # 한도: 풀릴 때까지 기다린다 (1분에 한 번만 알린다)
                        s = wait_seconds(r)
                        with lock:
                            state["waits"] += 1
                            if time.time() - state["said"] > 60:
                                state["said"] = time.time()
                                print(f"  요청 한도에 닿아서 {s}초 기다려요 ({state['done']}/{len(pick)} 받음)", flush=True)
                        time.sleep(s)
                        continue
                    if r.status_code in (401, 403, 404):
                        raise SystemExit(f"받을 수 없어요 ({r.status_code}): 파일 {n}번")
                    r.raise_for_status()
                    with open(part, "wb") as out:
                        for chunk in r.iter_content(1 << 20):
                            out.write(chunk)
                if part.stat().st_size != f.size:
                    raise IOError(f"크기가 달라요 ({part.stat().st_size} != {f.size})")
                part.replace(dest)
                break
            except (requests.RequestException, IOError) as e:  # 끊김 · 서버 오류: 점점 오래 기다린다
                errors += 1
                if errors > 12:
                    raise SystemExit(f"받지 못했어요: 파일 {n}번 ({type(e).__name__}: {str(e)[:200]})")
                time.sleep(min(120, 5 * 2 ** errors))
        else:
            raise SystemExit(f"요청 한도가 오래 안 풀려요: 파일 {n}번")
        with lock:
            state["done"] += 1
            if state["done"] % 500 == 0:
                print(f"  {state['done']}/{len(pick)} ({time.time() - t0:.0f}초)", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(get, enumerate(pick)))
    print(f"받음 {len(pick)}개 ({time.time() - t0:.0f}초, 한도 기다림 {state['waits']}번)", flush=True)


if __name__ == "__main__":
    main()

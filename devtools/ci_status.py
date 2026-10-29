"""긴 CI 작업이 도는 동안 진행을 커밋 상태(commit status)로 남긴다. Actions 기록 · 주석은 작업이 끝나야 읽을 수 있어서.
GITHUB_TOKEN(statuses: write) · GITHUB_REPOSITORY · GITHUB_SHA가 있을 때만 쓰고, 없거나 실패하면 찍기만 한다. 숫자만 쓴다.

사용: python ci_status.py "이름" "내용" [pending|success|failure]   또는   from ci_status import post
"""
import json
import os
import sys
import urllib.request


def post(context, text, state="pending"):
    print(f"[{context}] {text}", flush=True)
    token, repo, sha = (os.environ.get(k) for k in ("GITHUB_TOKEN", "GITHUB_REPOSITORY", "GITHUB_SHA"))
    if not (token and repo and sha):
        return False
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/statuses/{sha}", method="POST",
        data=json.dumps({"state": state, "context": context, "description": text[:140]}).encode(),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    try:
        urllib.request.urlopen(req, timeout=20).close()
        return True
    except Exception as e:  # 진행 알림이 안 돼도 작업은 계속한다
        print(f"진행 상태를 남기지 못함: {type(e).__name__}", flush=True)
        return False


if __name__ == "__main__":
    post(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "pending")

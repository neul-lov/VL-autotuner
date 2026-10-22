"""앱이 받는 GAME 모델(tetotune/game.py의 MODEL_URL · MODEL_SHA256)을 실제로 받아 확인한다 (CI).
고정한 판을 받지 못하거나 SHA-256이 다르면, OpenUtau 패키지 목록(svs-index)의 최신 판 주소 · SHA-256 · 압축 안 목록을
보여 주고 실패한다 (새 판으로 올릴 때 이 값을 game.py에 넣는다).
"""
import hashlib
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tetotune import game  # noqa: E402

REGISTRY = "https://openutau.github.io/svs-index/registry/v1/softwares/all.json"


def latest():
    req = urllib.request.Request(REGISTRY, headers={"User-Agent": "OpenUtau", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        sw = next(s for s in json.load(r)["items"] if s.get("id") == "game")
    url = sw["versions"][-1]["mirrors"][0]["url"]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "game.oudep"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OpenUtau"}), timeout=120) as r:
            p.write_bytes(r.read())
        print(f"최신 판: {sw['versions'][-1].get('version')}\nMODEL_URL = \"{url}\"\n"
              f"MODEL_SHA256 = \"{hashlib.sha256(p.read_bytes()).hexdigest()}\"\n크기 {p.stat().st_size}")
        with zipfile.ZipFile(p) as z:
            for m in z.infolist():
                print(f"  {m.filename} {m.file_size}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as home:
        import os
        os.environ["TETOTUNE_MODELS"] = home
        game.model_dir = lambda: None  # 이미 깔린 OpenUtau 모델이 있어도 받는 길을 시험한다
        try:
            d = game.ensure_model(print)
            ok = d is not None and game._complete(Path(d))
            print("GAME 모델 받기:", "됨" if ok else "안 됨", d)
        except Exception as e:
            ok = False
            print("GAME 모델 받기 실패:", e)
    if not ok:
        latest()
        sys.exit(1)

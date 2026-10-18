"""OpenUtau(자체 포함 빌드) 폴더에서 Harness를 OpenUtau.dll 자리에 넣고 돌릴 때 쓰는 런타임 목록을 만든다.
OpenUtau.deps.json의 runtimepack 부분만 Microsoft.NETCore.App.deps.json으로 옮긴다.
사용: python make_deps.py <OpenUtau 폴더>"""
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
deps = json.loads((folder / "OpenUtau.deps.json").read_text(encoding="utf-8-sig"))
target = deps["runtimeTarget"]["name"]
packs = {k: v for k, v in deps["targets"][target].items() if k.lower().startswith("runtimepack.")}
libs = {k: v for k, v in deps.get("libraries", {}).items() if k in packs}
out = {"runtimeTarget": deps["runtimeTarget"], "compilationOptions": {}, "targets": {target: packs}, "libraries": libs}
(folder / "Microsoft.NETCore.App.deps.json").write_text(json.dumps(out, indent=1))
print(f"{target}: {', '.join(packs)}")

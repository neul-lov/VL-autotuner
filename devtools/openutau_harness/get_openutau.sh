#!/bin/bash
# 최신 OpenUtau Mac 버전을 받아 ./OpenUtau.app 으로 꺼낸다 (CI용). 뒤에 rmvpe · game을 쓰면 OpenUtau 패키지 목록에서 그 모델도 받아 둔다.
# 새 OpenUtau dmg는 열 때 사용권 동의를 물어서 yes를 넣어 준다.
set -euo pipefail
tag=$(gh release view -R stakira/OpenUtau --json tagName -q .tagName)
gh release download "$tag" -R stakira/OpenUtau -p "*osx-arm64*.dmg" -O ou.dmg --clobber
(yes || true) | PAGER=cat hdiutil attach ou.dmg -nobrowse -mountpoint /tmp/ou > /dev/null
cp -R /tmp/ou/OpenUtau.app ./
hdiutil detach /tmp/ou > /dev/null
echo "OpenUtau $tag"
for id in "$@"; do (
  set -euo pipefail
  dep="$HOME/Library/OpenUtau/Dependencies/$id"
  url=$(curl -fsSL -H "User-Agent: OpenUtau" https://openutau.github.io/svs-index/registry/v1/softwares/all.json | python3 -c '
import json, sys
items = json.load(sys.stdin)["items"]
sw = next(s for s in items if s["id"] == sys.argv[1])
print(sw["versions"][-1]["mirrors"][0]["url"])' "$id")
  echo "$id: $url"
  curl -fL --retry 3 -o "$id.oudep" "$url"
  mkdir -p "$dep" && unzip -oq "$id.oudep" -d "$dep"
  find "$dep" -maxdepth 2 | head
) || echo "$id 모델을 받지 못했어요 (그 방식은 비교에서 빠져요)"; done

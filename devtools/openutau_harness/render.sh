#!/bin/bash
# OpenUtau 프로그램 폴더(Mac: OpenUtau.app/Contents/MacOS)를 복사해서 Harness로 ustx를 wav로 렌더한다. 소리는 재생하지 않는다.
# 사용: render.sh <OpenUtau 폴더> <작업 폴더> <project.ustx> <render.wav>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$1"; work="$2"; ustx="$3"; out="$4"
if [[ ! -f "$work/OpenUtau.dll" || ! -f "$work/.harness" ]]; then
  rm -rf "$work"; mkdir -p "$work"
  cp -R "$src"/. "$work"/
  refs=()
  for f in "$work"/*.dll; do
    case "$(basename "$f")" in *Native*|Vortice*|SharpGen*) ;; *) refs+=("-r:$f");; esac
  done
  # C# 컴파일러: CSC_CMD (실행 파일) 또는 CSC (dotnet SDK의 csc.dll)
  if [[ -n "${CSC_CMD:-}" ]]; then compile=("$CSC_CMD"); else compile=(dotnet "${CSC:?CSC=csc.dll 경로가 필요해요}"); fi
  "${compile[@]}" -nologo -target:exe -out:"$work/Harness.dll" -nullable:disable "${refs[@]}" "$here/Harness.cs"
  python3 "$here/make_deps.py" "$work"
  rm "$work/OpenUtau.deps.json"
  cp "$work/Harness.dll" "$work/OpenUtau.dll"
  touch "$work/.harness"
fi
# 런타임 목록을 바꿔 둔 OpenUtau 실행 파일은 dotnet처럼 dll을 실행한다.
# OpenUtau가 가끔 소리 없는 wav를 내보내서, 그러면 렌더 캐시를 비우고 두 번까지 다시 렌더한다 (CI 로그에 남김)
cd "$work"
for attempt in 1 2 3; do
  ./OpenUtau Harness.dll "$ustx" "$out" 40
  if python3 "$here/check_render.py" "$out"; then exit 0; fi
  echo "render.sh: 소리 없는 렌더 (${attempt}번째), 캐시를 비우고 다시 해요: $ustx" >&2
  rm -rf "$HOME/Library/Caches/OpenUtau" "${XDG_CACHE_HOME:-$HOME/.cache}/OpenUtau"
done
echo "render.sh: 세 번 렌더해도 소리가 없어요: $ustx" >&2
exit 3

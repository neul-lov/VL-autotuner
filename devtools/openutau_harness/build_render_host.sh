#!/bin/bash
# 앱에 넣는 RenderHost.dll 만들기 (tetotune/render_host/RenderHost.cs). dotnet SDK가 있어야 해요.
# .NET 6 참조로 만들어서 .NET 6 이상으로 만든 OpenUtau면 어느 버전이든 돌아요 (없으면 가장 낮은 버전 참조로).
# 사용: build_render_host.sh <RenderHost.dll 경로>
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
out="$1"
dotnet_bin="$(command -v dotnet)"
root="${DOTNET_ROOT:-$(cd "$(dirname "$(readlink -f "$dotnet_bin")")" && pwd)}"
csc=$(ls -d "$root"/sdk/*/Roslyn/bincore/csc.dll | tail -1)
ref=$(ls -d "$root"/packs/Microsoft.NETCore.App.Ref/6.*/ref/net6.0 2>/dev/null | tail -1 || true)
if [[ -z "$ref" ]]; then ref=$(ls -d "$root"/packs/Microsoft.NETCore.App.Ref/*/ref/net* | head -1); fi
echo "RenderHost: $(basename "$(dirname "$(dirname "$ref")")") 참조, $(basename "$(dirname "$(dirname "$(dirname "$csc")")")") 컴파일러"
refs=()
for f in "$ref"/*.dll; do refs+=("-r:$f"); done
mkdir -p "$(dirname "$out")"
dotnet "$csc" -nologo -target:exe -optimize+ -nullable:disable -langversion:latest -out:"$out" "${refs[@]}" \
  "$here/tetotune/render_host/RenderHost.cs"

#!/bin/zsh
# Builds AutoTuner.app into ./build. Pass --install to copy it to /Applications as well.
#
# The app carries its own copy of the engine (Resources/engine) and installs its Python packages into
# ~/Library/Application Support/TetoTune on first launch. When this checkout already has a .venv
# (start-mac.command), the app uses this folder's server.py and .venv instead, so engine edits show up
# without rebuilding; keep the folder where it is in that case.
set -euo pipefail
cd "$(dirname "$0")"
BACKEND="$(cd .. && pwd)"

# Apple Silicon + Intel in one app (needs Xcode); with only the Command Line Tools, this Mac's kind only.
ARCHS=(--arch arm64 --arch x86_64)
if ! swift build -c release "${ARCHS[@]}"; then
    echo "Universal build failed (Xcode is needed for it); building for this Mac only."
    ARCHS=()
    swift build -c release
fi

APP=build/AutoTuner.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/engine/tetotune"
cp "$(swift build -c release "${ARCHS[@]}" --show-bin-path)/AutoTuner" "$APP/Contents/MacOS/AutoTuner"
lipo -archs "$APP/Contents/MacOS/AutoTuner"
cp Resources/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
cp ../LICENSE ../THIRD_PARTY_NOTICES.txt "$APP/Contents/Resources/"

ENGINE="$APP/Contents/Resources/engine"
cp ../server.py ../requirements.txt ../requirements-extra.txt "$ENGINE/"
cp ../tetotune/*.py "$ENGINE/tetotune/"
# pyworld wheels built by CI (PyPI has none for Mac), used on first launch so no compiler is needed.
# constraints.txt pins pyworld to the bundled version, so a newer source-only release on PyPI is never built instead.
if ls wheels/*.whl >/dev/null 2>&1; then
    mkdir -p "$ENGINE/wheels"
    cp wheels/*.whl "$ENGINE/wheels/"
    PYWORLD=$(ls wheels/pyworld-*.whl | head -n 1 | sed -E 's/.*pyworld-([^-]+)-.*/\1/')
    echo "pyworld==$PYWORLD" > "$ENGINE/wheels/constraints.txt"
fi

# 테토로 불러 보고 다듬는 도구 (OpenUtau로 렌더): CI가 만든 RenderHost.dll, 없으면 dotnet SDK가 있을 때 여기서 만든다.
# 둘 다 없으면 이 기능만 빠지고 조교는 그대로 된다.
if [[ ! -f RenderHost.dll ]] && command -v dotnet >/dev/null 2>&1; then
    ../devtools/openutau_harness/build_render_host.sh RenderHost.dll || echo "RenderHost를 만들지 못했어요 (테토로 불러 보기만 빠져요)"
fi
if [[ -f RenderHost.dll ]]; then
    mkdir -p "$ENGINE/tetotune/render_host"
    cp RenderHost.dll "$ENGINE/tetotune/render_host/RenderHost.dll"
    # 이 폴더의 엔진을 쓰는 앱(start-mac.command, 아래 TetoTuneBackend)도 쓰게
    if [[ -x "$BACKEND/.venv/bin/python" ]]; then cp RenderHost.dll "$BACKEND/tetotune/render_host/RenderHost.dll"; fi
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>AutoTuner</string>
    <key>CFBundleDisplayName</key><string>AutoTuner</string>
    <key>CFBundleIdentifier</key><string>com.neul.tetotune</string>
    <key>CFBundleExecutable</key><string>AutoTuner</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.3</string>
    <key>CFBundleVersion</key><string>3</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.music</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
# Added with plutil so a path with spaces, & or < stays valid XML.
if [[ -x "$BACKEND/.venv/bin/python" ]]; then
    plutil -insert TetoTuneBackend -string "$BACKEND" "$APP/Contents/Info.plist"
fi

codesign --force --deep --sign - "$APP"
echo "Built $APP"

if [[ "${1:-}" == "--install" ]]; then
    rm -rf /Applications/AutoTuner.app
    # 예전 이름(TetoTune)으로 깐 같은 앱은 지운다 (설정 · 가사 기억은 그대로 남는다)
    if [[ -d /Applications/TetoTune.app ]]; then
        osascript -e 'quit app "TetoTune"' 2>/dev/null || true
        rm -rf /Applications/TetoTune.app
    fi
    cp -R "$APP" /Applications/
    echo "Installed to /Applications/AutoTuner.app"
fi

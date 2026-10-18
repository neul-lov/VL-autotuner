#!/bin/zsh
# Builds TetoTune.app into ./build. Pass --install to copy it to /Applications as well.
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

APP=build/TetoTune.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/engine/tetotune"
cp "$(swift build -c release "${ARCHS[@]}" --show-bin-path)/TetoTune" "$APP/Contents/MacOS/TetoTune"
lipo -archs "$APP/Contents/MacOS/TetoTune"
cp Resources/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"

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

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>TetoTune</string>
    <key>CFBundleDisplayName</key><string>TetoTune</string>
    <key>CFBundleIdentifier</key><string>com.neul.tetotune</string>
    <key>CFBundleExecutable</key><string>TetoTune</string>
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
    rm -rf /Applications/TetoTune.app
    cp -R "$APP" /Applications/
    echo "Installed to /Applications/TetoTune.app"
fi

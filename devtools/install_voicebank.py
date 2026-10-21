"""UTAU 음원 zip을 OpenUtau 가수 폴더에 푼다. 일본 음원은 zip 안 파일 이름이 Shift-JIS라서 그것까지 맞춘다.
사용: python install_voicebank.py <음원.zip> [가수 폴더 (기본 ~/Library/OpenUtau/Singers)]"""
import sys
import zipfile
from pathlib import Path

src = Path(sys.argv[1])
dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "Library" / "OpenUtau" / "Singers"
dest.mkdir(parents=True, exist_ok=True)
root = dest.resolve()
with zipfile.ZipFile(src) as z:
    for info in z.infolist():
        name = info.filename
        if not info.flag_bits & 0x800:  # UTF-8 표시가 없으면 cp437로 읽힌 Shift-JIS
            try:
                name = name.encode("cp437").decode("cp932")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        name = name.replace("\\", "/").lstrip("/")
        if name.startswith("__MACOSX/") or ".." in Path(name).parts:
            continue
        target = dest / name
        if not target.resolve().is_relative_to(root):  # 절대 경로 · 드라이브 이름 · 링크로 가수 폴더 밖에 쓰지 않게
            continue
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(info))
for c in sorted(dest.rglob("character.txt")):
    print(c.parent.relative_to(dest))

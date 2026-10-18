#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "처음 실행이라 필요한 프로그램을 설치합니다. 몇 분 걸려요..."
  python3 -m venv .venv || { echo "Python이 필요해요: https://www.python.org/downloads/"; read; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "설치 실패"; read; exit 1; }
fi
if [ ! -d /Applications/TetoTune.app ] && command -v swift >/dev/null; then
  echo "Mac 앱을 만듭니다..."
  (cd mac && ./build.sh --install) || echo "앱을 만들지 못했어요. Xcode(또는 Command Line Tools)가 필요해요: xcode-select --install"
fi
echo "준비 끝! 응용 프로그램 폴더의 TetoTune을 여세요. 이 창은 닫아도 돼요."

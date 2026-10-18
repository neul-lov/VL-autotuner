#!/bin/bash
cd "$(dirname "$0")"
# 지난번 설치가 중간에 끊겼거나(창을 닫음 · 설치 실패) 맞지 않는 Python으로 만든 .venv는 지우고 다시 설치한다
if [ -d .venv ] && ! .venv/bin/python -c 'import numpy, scipy, soundfile, pyworld, fastapi, uvicorn' 2>/dev/null; then
  echo "이전 설치가 덜 끝나서 다시 설치해요"
  rm -rf .venv
fi
if [ ! -d .venv ]; then
  echo "처음 실행이라 필요한 프로그램을 설치합니다. 몇 분 걸려요..."
  # 필요한 프로그램이 다 나와 있는 Python을 고른다 (Homebrew의 python3는 3.15라 아직 안 돼요)
  PY=""
  for v in 3.13 3.12 3.11 3.10 3.14 3.9; do
    for c in /opt/homebrew/bin/python$v /usr/local/bin/python$v /Library/Frameworks/Python.framework/Versions/$v/bin/python$v; do
      [ -x "$c" ] && { PY="$c"; break 2; }
    done
  done
  if [ -z "$PY" ]; then
    for c in python3 /Library/Developer/CommandLineTools/usr/bin/python3; do
      "$c" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 14) else 1)' 2>/dev/null && { PY="$c"; break; }
    done
  fi
  [ -n "$PY" ] || { echo "Python 3.9~3.14가 필요해요: https://www.python.org/downloads/ (또는 xcode-select --install)"; read; exit 1; }
  echo "Python: $PY"
  "$PY" -m venv .venv || { echo "Python 환경을 만들지 못했어요"; read; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { rm -rf .venv; echo "설치 실패"; read; exit 1; }
  .venv/bin/python -m pip install -r requirements-extra.txt || echo "추가 프로그램(RMVPE · 한자 읽기)은 설치하지 못했어요. 없어도 조교는 돼요."
fi
if [ ! -d /Applications/TetoTune.app ] && command -v swift >/dev/null; then
  echo "Mac 앱을 만듭니다..."
  (cd mac && ./build.sh --install) || echo "앱을 만들지 못했어요. Xcode(또는 Command Line Tools)가 필요해요: xcode-select --install"
fi
echo "준비 끝! 응용 프로그램 폴더의 TetoTune을 여세요. 이 창은 닫아도 돼요."

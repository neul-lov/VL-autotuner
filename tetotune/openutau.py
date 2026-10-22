"""사용자의 OpenUtau로 조교 결과를 실제로 불러 보기 (렌더, 소리는 재생하지 않고 wav로만).

OpenUtau 프로그램 폴더(Mac: OpenUtau.app/Contents/MacOS)를 앱 데이터 폴더에 복사해 두고(Mac은 APFS 복제라 자리를
거의 차지하지 않는다) OpenUtau.dll 자리에 render_host/RenderHost.dll을 넣어 돌린다. OpenUtau의 음원 · 설정은
OpenUtau가 쓰던 것을 그대로 쓰고, 렌더 캐시만 AutoTuner 전용 폴더(cache_dir())를 쓴다. OpenUtau 앱 자체는 건드리지 않는다.
RenderHost.dll은 앱을 만들 때 컴파일해서 넣는다 (devtools/openutau_harness/build_render_host.sh).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT = 1800  # 초. 아주 긴 곡도
TRACK = "테토 (자동 조교)"  # tune._write_ustx의 첫 트랙 이름
_last_error = None


def host_dll():
    p = os.environ.get("TETOTUNE_RENDER_HOST")
    p = Path(p) if p else HERE / "render_host" / "RenderHost.dll"
    return p if p.is_file() else None


def app_dir():
    """OpenUtau 프로그램 폴더 (OpenUtau 실행 파일과 OpenUtau.Core.dll이 있는 곳) 또는 None."""
    cands = []
    if os.environ.get("TETOTUNE_OPENUTAU"):
        cands.append(Path(os.environ["TETOTUNE_OPENUTAU"]))
    if sys.platform == "darwin":
        cands += [Path("/Applications/OpenUtau.app"), Path.home() / "Applications" / "OpenUtau.app"]
    for c in cands:
        for d in (c / "Contents" / "MacOS", c):
            if (d / "OpenUtau.Core.dll").is_file() and (d / "OpenUtau").is_file() and (d / "OpenUtau.deps.json").is_file():
                return d
    return None


def support_dir():
    if os.environ.get("TETOTUNE_RENDER_WORK"):
        return Path(os.environ["TETOTUNE_RENDER_WORK"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "TetoTune" / "render-host"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "AutoTuner" / "render-host"


def available():
    return host_dll() is not None and app_dir() is not None


def _stamp(src, dll):
    """OpenUtau가 바뀌었거나(업데이트) RenderHost가 바뀌면 다시 준비한다."""
    s = (src / "OpenUtau.Core.dll").stat()
    d = dll.stat()
    return f"{src}|{s.st_size}|{int(s.st_mtime)}|{d.st_size}|{int(d.st_mtime)}"


def _copy_tree(src, dst):
    if sys.platform == "darwin":  # APFS 복제 (cp -c): 바로 끝나고 자리도 거의 안 쓴다
        r = subprocess.run(["/bin/cp", "-cR", str(src) + "/.", str(dst)], capture_output=True)
        if r.returncode == 0:
            return
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
    shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)


def _make_deps(folder):
    """OpenUtau.deps.json의 런타임 부분만 Microsoft.NETCore.App.deps.json으로 (OpenUtau.dll 자리의 다른 프로그램을 돌리려고)."""
    deps = json.loads((folder / "OpenUtau.deps.json").read_text(encoding="utf-8-sig"))
    target = deps["runtimeTarget"]["name"]
    packs = {k: v for k, v in deps["targets"][target].items() if k.lower().startswith("runtimepack.")}
    libs = {k: v for k, v in deps.get("libraries", {}).items() if k in packs}
    out = {"runtimeTarget": deps["runtimeTarget"], "compilationOptions": {}, "targets": {target: packs}, "libraries": libs}
    (folder / "Microsoft.NETCore.App.deps.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (folder / "OpenUtau.deps.json").unlink()


def prepare():
    """렌더할 폴더를 준비해서 그 경로 (이미 되어 있으면 그대로). 못 하면 RuntimeError."""
    src, dll = app_dir(), host_dll()
    if src is None:
        raise RuntimeError("OpenUtau를 찾지 못했어요 (응용 프로그램 폴더에 OpenUtau.app이 있어야 해요)")
    if dll is None:
        raise RuntimeError("이 앱에는 테토로 불러 보는 도구가 없어요")
    root = support_dir()
    host = root / "host"
    stamp = _stamp(src, dll)
    if (host / ".stamp").is_file() and (host / ".stamp").read_text() == stamp:
        return host
    tmp = root / "host.new"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    _copy_tree(src, tmp)
    _make_deps(tmp)
    shutil.copy(dll, tmp / "RenderHost.dll")
    shutil.copy(dll, tmp / "OpenUtau.dll")
    if sys.platform == "darwin":  # 복사본에 붙은 "인터넷에서 받음" 표시만 뗀다 (OpenUtau 앱은 그대로)
        subprocess.run(["/usr/bin/xattr", "-dr", "com.apple.quarantine", str(tmp)], capture_output=True)
    (tmp / ".stamp").write_text(stamp)
    shutil.rmtree(host, ignore_errors=True)
    tmp.rename(host)
    return host


def _has_sound(path):
    import soundfile as sf

    try:  # 소리가 나는 곳을 찾을 때까지만 조금씩 읽는다 (다듬기가 같은 파일을 또 읽으니 전체를 두 번 읽지 않게)
        info = sf.info(str(path))
        if info.frames <= info.samplerate * 0.2:
            return False
        return any(float(abs(b).max()) > 1e-3 for b in sf.blocks(str(path), blocksize=1 << 16, dtype="float32") if len(b))
    except Exception:
        return False


def cache_dir():
    """렌더할 때 OpenUtau가 쓰는 캐시 (AutoTuner 전용, RenderHost가 TETOTUNE_OU_CACHE로 받는다).
    사용자 OpenUtau 캐시(~/Library/Caches/OpenUtau)는 쓰지도 지우지도 않는다."""
    return support_dir() / "cache"


def _why_silent(log):
    """RenderHost 기록(singer= found= / phonemes= phoneme_errors= / phonemizer= renderer=)으로 소리가 없는 까닭을 짐작한다."""
    import re

    info = dict(re.findall(r"(\w+)=(\S+)", log))
    m = re.search(r"singer=(.*?) found=", log)  # 가수 id에는 띄어쓰기가 있을 수 있다
    singer = m.group(1) if m else info.get("singer", "?")
    if info.get("found") == "False" or singer == "null":
        return f"OpenUtau가 가수를 찾지 못했어요 ({singer}). OpenUtau를 한 번 열어 가수 목록에 테토가 있는지 확인해 주세요"
    total, errors = int(info.get("phonemes", "0") or 0), int(info.get("phoneme_errors", "0") or 0)
    if info.get("exported") == "0":
        return (f"OpenUtau가 렌더는 끝냈다는데 내보낸 파일이 없어요 (OpenUtau {info.get('openutau_core', '?')}, "
                f"렌더러 {info.get('renderer', '?')}). OpenUtau에서 project.ustx를 열어 재생 · 내보내기가 되는지 확인해 주세요")
    if total == 0 or errors >= max(1, total // 2):
        return (f"가사를 테토 소리로 바꾸지 못했어요 (음소 {total}개 중 {errors}개 오류, 음소 변환기 {info.get('phonemizer', '?')}). "
                "고른 가수 · 가사 언어가 맞는지 확인해 주세요")
    return (f"OpenUtau가 소리 없는 파일을 냈어요 (가수 {singer}, 음소 {total}개 · 오류 {errors}개, "
            f"렌더러 {info.get('renderer', '?')}). OpenUtau에서 이 가수로 소리가 나는지 확인해 주세요")


def _env(own_cache=True):
    env = dict(os.environ)
    if own_cache:
        env["TETOTUNE_OU_CACHE"] = str(cache_dir())
    else:
        env.pop("TETOTUNE_OU_CACHE", None)
    return env


def render(ustx, out_dir, name=TRACK, tries=3):
    """ustx → 그 트랙(이름, 기본은 조교한 테토 트랙)의 wav 경로.
    OpenUtau가 가끔 소리 없는 파일을 내서, 그러면 (AutoTuner 전용) 캐시를 비우고 다시 하고,
    마지막 번에는 전용 캐시 없이 OpenUtau가 원래 쓰는 캐시로 한다 (그 캐시는 지우지 않는다)."""
    global _last_error
    host = prepare()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log, silent = "", False
    for attempt in range(tries):
        for f in out_dir.glob("render_*.wav"):
            f.unlink()
        try:
            r = subprocess.run(["./OpenUtau", "RenderHost.dll", str(Path(ustx).resolve()), str((out_dir / "render.wav").resolve())],
                               cwd=host, capture_output=True, text=True, timeout=TIMEOUT,
                               env=_env(own_cache=attempt < tries - 1 or tries == 1))
            log = (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired:
            log = "렌더가 너무 오래 걸려요"
            continue
        files = sorted(out_dir.glob("render_*.wav"))
        if name:  # 트랙 이름으로 (render_<트랙 이름>.wav)
            files = [f for f in files if f.stem == "render_" + name] or files
        if r.returncode == 0 and files and _has_sound(files[0]):
            _last_error = None
            return files[0]
        silent = r.returncode == 0
        what = ("내보낸 파일 없음" if not files else "소리 없는 파일") if silent else f"실패 (코드 {r.returncode})"
        lines = [ln for ln in log.strip().splitlines() if "not on main thread" not in ln]
        print(f"OpenUtau 렌더 {attempt + 1}/{tries}: {what} / " + " / ".join(lines[-12:]), flush=True)
        if attempt == 0 and r.returncode == 0:  # 소리 없는 렌더: (AutoTuner 전용) 캐시를 비우고 다시
            shutil.rmtree(cache_dir(), ignore_errors=True)
    _last_error = log.strip().splitlines()[-3:] if log else ["렌더 기록 없음"]
    if silent:
        _last_error = [_why_silent(log)]
    raise RuntimeError("OpenUtau로 불러 보지 못했어요: " + " / ".join(_last_error))

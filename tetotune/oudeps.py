"""신경망 렌더(hifisampler)가 쓰는 OpenUtau 패키지 둘을 받아 OpenUtau 의존성 폴더에 둔다.

- pc_nsf_hifigan_44.1k_hop512_128bin_2025.02: 소리를 만드는 신경망 보코더 (OpenVPI DiffSinger 커뮤니티 보코더,
  CC BY-NC-SA 4.0: 비영리로만 쓸 수 있다)
- hnsep_240512: 배음 · 잡음 나누기 모델 (yxlllc/vocal-remover, MIT). 숨 섞인 소리 · 힘 곡선을 부를 때 쓴다
OpenUtau 0.1.576부터 든 hifisampler(CLASSIC 렌더러의 리샘플러)가 이 둘을 찾는다. OpenUtau 패키지 관리자가 설치하는
것과 같은 곳(~/Library/OpenUtau/Dependencies/<id>), 같은 모양으로 풀어서 OpenUtau · EasyUtau도 그대로 쓴다.
판을 고정하고 SHA-256으로 확인한다 (OpenUtau 패키지 목록 svs-index에 있는 판).

실제 테토 렌더 비교(README "렌더 엔진")에서 자연스러움 AI 점수(SingMOS-Pro)가 WORLDLINE-R보다 모든 시험 곡 묶음에서 높았다.
"""
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PACKAGES = {  # id → (주소, SHA-256, 있어야 하는 파일)
    "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02": (
        "https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/"
        "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.oudep",
        "ba7d43142d41f6900c8264b5662ca7125a50feb8760bb8b9615c61a8f5e6902e",
        ("vocoder.yaml", "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")),
    "hnsep_240512": (
        "https://github.com/stakira/vocal-remover/releases/download/hnsep_240512-oudep/hnsep_240512.oudep",
        "8f8c1046d0cacd40363e6f71d48bc1663e06a01d4625af0569055648d771ff92",
        ("hnsep.yaml", "model.onnx")),
}
WHAT = "신경망 렌더 모델"
MAX_DOWNLOAD = 256 << 20  # 하나 받는 크기 상한 (지금 둘 다 55MB쯤)
MAX_UNPACKED = 512 << 20
ALLOWED = {".onnx", ".yaml", ".yml", ".txt", ".md"}  # 압축 안에 있어도 되는 파일 (LICENSE는 따로)


def deps_dir():
    """OpenUtau 의존성 폴더 (OpenUtau PathManager.DependencyPath와 같은 곳). TETOTUNE_OU_DEPS로 바꿀 수 있다 (시험)."""
    if os.environ.get("TETOTUNE_OU_DEPS"):
        return Path(os.environ["TETOTUNE_OU_DEPS"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "OpenUtau" / "Dependencies"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "OpenUtau" / "Dependencies"


def installed(pid):
    return all((deps_dir() / pid / f).is_file() for f in PACKAGES[pid][2])


def missing():
    return [pid for pid in PACKAGES if not installed(pid)]


def ensure(progress=lambda s: None):
    """없는 패키지를 받아 푼다 → 받은 id 목록. 받지 못하면 ValueError · OSError (이미 받은 것은 그대로 둔다)."""
    todo = missing()
    for i, pid in enumerate(todo, start=1):
        progress(f"{WHAT} 받는 중 ({i}/{len(todo)}, 처음 한 번)")
        _install(pid)
    return todo


def _install(pid):
    from .game import _download

    url, sha256, _ = PACKAGES[pid]
    root = deps_dir()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".autotuner-") as tmp:
        zpath = Path(tmp) / f"{pid}.oudep"
        _download(url, zpath, sha256, MAX_DOWNLOAD, WHAT)
        out = Path(tmp) / pid
        with zipfile.ZipFile(zpath) as z:
            _check_zip(z, pid)
            z.extractall(out)
        if not all((out / f).is_file() for f in PACKAGES[pid][2]):
            raise ValueError(f"{WHAT} 파일이 모자라요 ({pid})")
        dest = root / pid
        if dest.exists():  # 반쯤 풀린 예전 것 (OpenUtau도 설치할 때 예전 폴더를 지운다)
            shutil.rmtree(dest)
        out.rename(dest)


def _check_zip(z, pid):
    """OpenUtau 패키지 모양(맨 위에 id가 맞는 oudep.yaml)이고, 압축 밖으로 나가는 경로 · 링크 · 모르는 종류의 파일 ·
    너무 큰 내용이 없어야 푼다."""
    import yaml

    total = 0
    for m in z.infolist():
        parts = Path(m.filename).parts
        if m.filename.startswith(("/", "\\")) or ".." in parts or ":" in m.filename or "\\" in m.filename:
            raise ValueError(f"{WHAT} 압축 파일이 이상해요")
        if (m.external_attr >> 16) & 0o170000 == 0o120000:  # 심볼릭 링크
            raise ValueError(f"{WHAT} 압축 파일이 이상해요")
        if m.is_dir():
            continue
        name = Path(m.filename).name
        if (Path(name).suffix.lower() not in ALLOWED and name != "LICENSE") or len(parts) > 2:
            raise ValueError(f"{WHAT} 압축 안에 모르는 파일이 있어요: {m.filename}")
        total += m.file_size
        if total > MAX_UNPACKED:
            raise ValueError(f"{WHAT} 압축을 풀면 너무 커요")
    try:
        meta = yaml.safe_load(z.read("oudep.yaml").decode("utf-8-sig"))
    except (KeyError, UnicodeDecodeError, yaml.YAMLError):
        raise ValueError(f"{WHAT} 압축에 oudep.yaml이 없어요") from None
    if not isinstance(meta, dict) or meta.get("id") != pid:
        raise ValueError(f"{WHAT} 압축이 다른 패키지예요 ({pid})")


if __name__ == "__main__":  # CI · 손으로: 받아 두기
    print("받음:", ensure(print) or "이미 있음", "→", deps_dir())

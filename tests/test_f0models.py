"""로컬 AI 음 높이 읽기의 작은 부분 (모델 없이)."""
import numpy as np
import pytest

from tetotune import f0models


def test_mel_matches_librosa():
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
    assert np.abs(f0models.mel_basis() - ref).max() < 1e-6


def test_decode_local_average():
    sal = np.zeros((3, 360), np.float32)
    sal[0, 100] = 0.9  # 한 칸만: 그 칸의 센트
    sal[1, 100] = sal[1, 101] = 0.8  # 두 칸 사이
    sal[2, 50] = 0.01  # 확신 낮음 → 소리 없음
    hz, conf = f0models._decode(sal)
    c = 1200 * np.log2(hz[:2] / 10)
    assert abs(c[0] - f0models.CENTS[100]) < 1e-6 and abs(c[1] - (f0models.CENTS[100] + 10)) < 1e-6
    assert hz[2] == 0 and conf[2] == pytest.approx(0.01)


def test_fuse_keeps_agreeing_trackers_and_drops_disagreement():
    a4 = 440.0
    r = np.array([a4, a4, a4 * 2, 0.0, a4, 0.0])
    s = np.array([a4 * 1.01, 0.0, a4, a4, a4 * 1.5, 0.0])
    h = np.array([a4, a4, a4, 0.0, a4 * 1.3, a4])
    out = f0models.fuse([(r, 0.55), (s, 0.3), (h, 0.3)])
    assert abs(1200 * np.log2(out[0] / a4)) < 10  # 셋이 같다
    assert out[1] == pytest.approx(a4)  # 둘이 같다
    assert out[2] == pytest.approx(a4)  # RMVPE 옥타브 틀림: 나머지 둘을 따른다
    assert out[3] == 0  # SwiftF0 혼자 (무게 0.5 아래)
    assert out[4] > 0  # 셋 다 다르면 RMVPE 혼자 (무게 0.5 넘음)
    assert out[5] == 0  # Harvest 혼자

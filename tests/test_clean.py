"""분석용 보컬 깨끗하게 하기(clean.py)의 모델 없는 부분: STFT · 겹쳐 돌리기 · 입력 모양."""
import numpy as np
import pytest

from tetotune import clean, tune


def test_stft_matches_librosa_and_inverts():
    librosa = pytest.importorskip("librosa")
    x = np.random.default_rng(0).standard_normal((2, 352800))
    s = clean.stft(x, 2048, 441)
    assert s.shape == (2, 1025, 801)  # 모델 입력과 같은 모양
    ref = librosa.stft(x[0], n_fft=2048, hop_length=441, window="hann", center=True, pad_mode="reflect")
    assert np.abs(s[0] - ref).max() < 1e-9
    assert np.abs(clean.istft(s, 2048, 441, x.shape[1]) - x).max() < 1e-9
    assert clean.stft(np.zeros((1, 384000)), 2048, 512).shape == (1, 1025, 751)  # 잔향 제거 모델 입력


@pytest.mark.parametrize("layout", ["sc", "cs"])
def test_pack_unpack(layout):
    s = clean.stft(np.random.default_rng(1).standard_normal((2, 44100)), 2048, 441)
    packed = clean._pack(s, layout)
    assert packed.shape == (1, 4, 1025, s.shape[-1]) and packed.dtype == np.float32
    assert np.abs(clean._unpack(packed, 2, layout) - s).max() < 1e-4


def test_overlapping_chunks_rebuild_the_whole_song():
    """모델이 소리를 그대로 돌려주면 조각을 겹쳐 더한 결과도 원래 소리 (가장자리 · 끝 조각 포함)."""
    m = clean.SpecModel.__new__(clean.SpecModel)
    m.cfg = clean.MODELS["vocals"]
    m._run = lambda part: 0.5 * part
    for n in (1000, 352800 // 2 + 7, 352800 * 3 + 12345):
        x = np.random.default_rng(n).standard_normal((2, n))
        assert np.abs(m(x) - 0.5 * x).max() < 1e-9, n


def test_analysis_audio_is_the_vocal_when_off(monkeypatch):
    monkeypatch.setattr(clean, "SEPARATE", "off")
    monkeypatch.setattr(clean, "DEREVERB", "off")
    y = np.zeros(44100)
    assert tune._for_analysis(y, 44100, None, [], lambda p, s: None) is y

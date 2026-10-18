"""시험용 UTAU 음원 만들기 (모음 あいうえお + 숨 息 · 吸). 렌더링 시험 전용이라 소리는 사람 녹음 모음 그대로다.

모음: vocobox/human-voice-dataset (MIT) 의 C3 모음 녹음
숨: vocadito (CC BY 4.0) 14번 곡의 들숨 한 번
사용: python make_test_voicebank.py <vocobox 폴더> <vocadito_14.wav> <만들 음원 폴더>
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

vocobox, vocadito, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
src = vocobox / "data/voices/martin/voyels/exports/mono"
lines = []
for kana, roma, file in [("あ", "a", "_-a-c3.wav"), ("い", "i", "_-i-c3.wav"), ("う", "u", "_-ou-c3.wav"),
                         ("え", "e", "_-e-c3.wav"), ("お", "o", "_-o-c3.wav")]:
    y, sr = sf.read(src / file, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    hop = int(sr * 0.005)
    rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop, hop)])
    db = 20 * np.log10(rms)
    on = int(np.argmax(db > db.max() - 25)) * 5
    off = (len(db) - int(np.argmax(db[::-1] > db.max() - 25))) * 5
    sf.write(out / f"{roma}.wav", y, sr, subtype="PCM_16")
    for alias in (kana, roma):
        # offset, consonant(고정), cutoff(음수 = offset부터 길이), preutter, overlap
        lines.append(f"{roma}.wav={alias},{max(0, on - 20)},{120},{-(off - on - 40)},{40},{15}")
y, sr = sf.read(vocadito, dtype="float32")
breath = y[int(3.42 * sr):int(3.62 * sr)]
fade = int(0.02 * sr)
breath[:fade] *= np.linspace(0, 1, fade)
breath[-fade:] *= np.linspace(1, 0, fade)
breath = np.concatenate([np.zeros(int(0.05 * sr)), breath, np.zeros(int(0.05 * sr))]) * 3
sf.write(out / "breath.wav", breath, sr, subtype="PCM_16")
for alias in ("息", "吸"):
    lines.append(f"breath.wav={alias},40,150,-200,30,10")
(out / "oto.ini").write_text("\n".join(lines) + "\n", encoding="utf-8")
(out / "character.txt").write_text("name=TestVowels\n", encoding="utf-8")
(out / "character.yaml").write_text("text_file_encoding: utf-8\nportrait_opacity: 0.67\ndefault_phonemizer: OpenUtau.Plugin.Builtin.JapanesePresampPhonemizer\n", encoding="utf-8")
print("ok", out)

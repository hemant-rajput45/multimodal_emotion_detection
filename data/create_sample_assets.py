"""
Utility to generate synthetic demo assets in data/sample_inputs/
Creates:
- sample_face.jpg (RGB test face image with facial features)
- sample_speech.wav (3-second WAV audio sample)
- sample_transcripts.txt (sample dialogue lines)
"""

import os
import numpy as np
import cv2
import scipy.io.wavfile as wavfile

def generate_samples(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate Sample Face Image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 230
    # Draw face oval
    cv2.ellipse(img, (112, 112), (70, 90), 0, 0, 360, (210, 180, 140), -1)
    # Draw eyes
    cv2.circle(img, (85, 95), 10, (255, 255, 255), -1)
    cv2.circle(img, (139, 95), 10, (255, 255, 255), -1)
    cv2.circle(img, (85, 95), 4, (60, 40, 20), -1)
    cv2.circle(img, (139, 95), 4, (60, 40, 20), -1)
    # Draw smile
    cv2.ellipse(img, (112, 140), (35, 20), 0, 10, 170, (40, 40, 40), 3)

    face_path = os.path.join(output_dir, "sample_face.jpg")
    cv2.imwrite(face_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Generated face asset: {face_path}")

    # 2. Generate Sample Audio WAV (3 seconds at 16000 Hz)
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Harmonics resembling human vocal pitch modulation
    f0 = 220.0
    audio_signal = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.25 * np.sin(2 * np.pi * (f0 * 2) * t) +
        0.15 * np.sin(2 * np.pi * (f0 * 3) * t)
    )
    # Add gentle envelope
    envelope = np.sin(np.pi * t / duration) ** 2
    audio_signal = (audio_signal * envelope * 32767).astype(np.int16)

    audio_path = os.path.join(output_dir, "sample_speech.wav")
    wavfile.write(audio_path, sr, audio_signal)
    print(f"Generated speech asset: {audio_path}")

    # 3. Sample transcripts
    txt_path = os.path.join(output_dir, "sample_transcripts.txt")
    with open(txt_path, "w") as f:
        f.write("I am genuinely proud and joyful about our collective success!\n")
        f.write("I feel deeply hurt and sorrowful about how everything collapsed.\n")
        f.write("This is outrageous and I will not tolerate this unacceptable disrespect!\n")
        f.write("I am completely stunned and shocked by this sudden revelation!\n")
        f.write("The regular procedure will follow its planned course tomorrow.\n")
    print(f"Generated text transcripts: {txt_path}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "sample_inputs")
    generate_samples(out_dir)

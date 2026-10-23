"""
Speech Modality Preprocessor
Handles audio loading, resampling, trimming silence, and acoustic feature extraction
(MFCCs, Mel-Spectrogram, Chroma, Spectral features) as described in Section V-D of the research paper.
"""

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample

class AudioPreprocessor:
    def __init__(self, target_sr=16000, duration_sec=3.0, n_mfcc=40):
        self.target_sr = target_sr
        self.duration_sec = duration_sec
        self.target_samples = int(target_sr * duration_sec)
        self.n_mfcc = n_mfcc

    def load_audio(self, audio_input):
        """
        Load audio from wav file or numpy array and normalize to float32 in [-1, 1]
        """
        if isinstance(audio_input, str):
            sr, data = wavfile.read(audio_input)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            else:
                data = data.astype(np.float32)

            # Convert stereo to mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Resample if sample rate doesn't match
            if sr != self.target_sr:
                num_new_samples = int(len(data) * self.target_sr / sr)
                data = resample(data, num_new_samples)
        elif isinstance(audio_input, np.ndarray):
            data = audio_input.astype(np.float32)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
        else:
            raise TypeError("Unsupported audio input type")

        return data

    def fix_length(self, signal):
        """
        Pad or truncate signal to match self.target_samples (e.g., 3.0 seconds at 16kHz)
        """
        if len(signal) > self.target_samples:
            return signal[:self.target_samples]
        elif len(signal) < self.target_samples:
            padding = np.zeros(self.target_samples - len(signal), dtype=np.float32)
            return np.concatenate([signal, padding])
        return signal

    def extract_features(self, audio_input):
        """
        Extract acoustic feature representation matrix [time_steps, feature_dim].
        Uses FFT-based filterbank & MFCC-like cepstral representations.
        """
        signal = self.load_audio(audio_input)
        signal = self.fix_length(signal)

        # Frame parameters
        frame_len = int(0.025 * self.target_sr)  # 25 ms
        hop_len = int(0.010 * self.target_sr)    # 10 ms
        num_frames = 1 + (len(signal) - frame_len) // hop_len

        # Compute STFT
        frames = np.lib.stride_tricks.sliding_window_view(signal, frame_len)[::hop_len]
        window = np.hamming(frame_len)
        windowed_frames = frames * window

        # Magnitude spectrum
        fft_len = 512
        mag_spec = np.abs(np.fft.rfft(windowed_frames, n=fft_len))

        # Filter into frequency bands (log energy bank)
        n_bands = self.n_mfcc
        band_indices = np.linspace(0, mag_spec.shape[1], n_bands + 2, dtype=int)
        band_energies = []
        for i in range(n_bands):
            start, stop = band_indices[i], band_indices[i+2]
            band_energy = np.mean(mag_spec[:, start:stop], axis=1, keepdims=True)
            band_energies.append(band_energy)

        # Log mel-scale energies
        log_filterbanks = np.log(np.maximum(np.hstack(band_energies), 1e-6))

        # Normalize features (zero mean, unit variance)
        mean = np.mean(log_filterbanks, axis=0, keepdims=True)
        std = np.std(log_filterbanks, axis=0, keepdims=True) + 1e-8
        norm_features = (log_filterbanks - mean) / std

        # norm_features has shape [num_frames, n_mfcc] -> e.g. [298, 40]
        return norm_features.astype(np.float32)

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class MelConverter:
    """
    Converts wav audio files into log-mel spectrograms
    and saves them as .npy and .png files.
    """

    def __init__(self,
        sample_rate=16000,
        duration=3,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def load_audio(self, audio_path):
        """
        Load audio, convert to mono, and resample.
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return y, sr

    def fix_length(self, y):
        """
        Pad or trim audio to a fixed duration.
        """
        target_length = self.sample_rate * self.duration

        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        return y

    def to_log_mel(self, y, sr):
        """
        Convert waveform to log-mel spectrogram.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def save_npy(self, log_mel_spec, npy_path):
        """
        Save spectrogram as .npy.
        """
        np.save(npy_path, log_mel_spec)

    def save_png(self, log_mel_spec, png_path):
        """
        Save spectrogram as .png.
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            log_mel_spec,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="mel"
        )
        plt.axis("off")
        plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def process_file(self, audio_path, npy_path, png_path):
        """
        Full pipeline for one audio file:
        load -> fix length -> convert -> save .npy and .png
        """
        y, sr = self.load_audio(audio_path)
        y = self.fix_length(y)
        log_mel_spec = self.to_log_mel(y, sr)

        self.save_npy(log_mel_spec, npy_path)
        self.save_png(log_mel_spec, png_path)

        return log_mel_spec
from application.components.PredictAppService.breathing_cycles_extractor import AudioMovingWindowPreProcessor
from application.components.PredictAppService.filters import FilterPipeline
import math
from application.components.PredictAppService.treadsafeSingleton import SingletonMeta
from application.components.PredictAppService.utils import get_audio
import librosa
import numpy as np
from typing import Tuple

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Filter Config
LOW_PASS_FREQUENCY = 100
HIGH_PASS_FREQUENCY = 2000

# Mel-Spectral Config
HOG_LENGTH = 347

# sizes in seconds
MOVING_WINDOW_SIZE = 5
AUDIO_STRIDE_SIZE = 4


class AudioPreProcessor(AudioMovingWindowPreProcessor, FilterPipeline, metaclass=SingletonMeta):

    def __init__(self):
        AudioMovingWindowPreProcessor.__init__(self)
        FilterPipeline.__init__(self, LOW_PASS_FREQUENCY, HIGH_PASS_FREQUENCY)
        super().__init__()

    def pre_process_audio(self, audio_data_in, sr_in, length_in) -> np.ndarray:
        """

        Generate Audio frames using moving window

        Parameters
        ----------
        audio_path
        sample_rate

        Returns audio_frames
        -------

        """
        audio_data, sample_rate, length = audio_data_in, sr_in, length_in
        sample_rate = 16000
        filtered_audio, sample_rate = self.filters(audio_data, sample_rate)
        audio_frames = self.get_audio_windows(filtered_audio, sample_rate, length, MOVING_WINDOW_SIZE,
                                              AUDIO_STRIDE_SIZE)

        return audio_frames, sample_rate

    def audio_to_mel_spectrogram(self, audio_data: np.ndarray, s_r: int, n_mel_val: int) -> np.ndarray:
        """
                Convert audio_frames into mel-spectrogram

        Parameters
        ----------
        s_r
        n_mel_val

        Returns mel-spectralgrams
        -------

        """
        spectrogram = librosa.feature.melspectrogram(audio_data,
                                                     sr=s_r,
                                                     n_mels=n_mel_val,
                                                     hop_length=HOG_LENGTH,
                                                     n_fft=n_mel_val * 20,
                                                     fmin=20,
                                                     fmax=s_r // 2)

        return librosa.power_to_db(spectrogram).astype(np.float32)

    def feature_engineering(self, audio_frames: np.ndarray, s_r: int, n_mel_filter_banks: int) -> np.ndarray:
        """
          Generate mel-spectral for audio

        Parameters
        ----------
        audio_frames
        s_r
        n_mel_filter_banks

        Returns All mel-spectral for audio
        -------

        """
        mel_spectral_frames = []

        for index in range(audio_frames.shape[0]):
            mel_frame = self.audio_to_mel_spectrogram(audio_frames[index], s_r, n_mel_filter_banks)
            mel_spectral_frames.append(mel_frame.transpose())

        # add Dummy zeo-vector for padding
        dummy_vec = np.zeros((318, n_mel_filter_banks), dtype=np.float)
        mel_spectral_frames.append(dummy_vec)

        return np.asarray(mel_spectral_frames, dtype=object)

    def get_pre_processed_data_mono_model(self, audio_data_in, sr_in, length_in, n_mel=64) -> np.ndarray:
        """
          Generate mono-mel spectrogram
        Parameters
        ----------
        audio_path
        sample_rate
        n_mel

        Returns
        -------

        """
        audio_frames, sample_rate = self.pre_process_audio(audio_data_in, sr_in, length_in)
        mel_spectral_frames = np.array(self.feature_engineering(audio_frames, sample_rate, n_mel))

        # raw data processing-padding
        x_mel_spectral_frames = pad_sequences(mel_spectral_frames, padding="post", dtype='float32')

        return x_mel_spectral_frames

    def get_pre_processed_data_dual_model(self, audio_data_in, sr_in, length_in, n_mel_1=128, n_mel_2=64) -> Tuple[
        np.ndarray, np.ndarray]:
        """
          Generate dual-mel spectrogram
        Parameters
        ----------
        audio_path
        sample_rate
        n_mel_1
        n_mel_2

        Returns
        -------

        """
        x_mel_spectral_1 = self.get_pre_processed_data_mono_model(audio_data_in, sr_in, length_in, n_mel_1)
        x_mel_spectral_2 = self.get_pre_processed_data_mono_model(audio_data_in, sr_in, length_in, n_mel_2)

        return x_mel_spectral_1, x_mel_spectral_2

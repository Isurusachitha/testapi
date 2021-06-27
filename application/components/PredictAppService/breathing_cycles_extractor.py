from .utils import get_audio
import numpy as np
from numpy.lib.stride_tricks import as_strided
from .treadsafeSingleton import SingletonMeta


class AudioMovingWindowPreProcessor(metaclass=SingletonMeta):

    def __init__(self) -> None:
        """
         init class
        """
        pass

    def get_audio_windows(self, audio: np.ndarray, sr: int, length: int, window_size: int, stride: int) -> np.ndarray:
        """
           Generate audio frames using  sliding window with stride -memory safe
        """
        no_frames = int((length - window_size) / stride) + 1
        window_size = window_size * sr
        stride = stride * sr

        audio_frames = []

        for index in range(no_frames):

            if (stride * index + window_size) < len(audio):
                frame = audio[stride * index:(stride * index + window_size)]
                audio_frames.append(frame)

            else:
                break

        return np.array(audio_frames)

    def get_audio_windows_numpy_vectorized(self, audio: np.ndarray, sr: int, length: int, window_size: int,
                                           stride: int) -> np.ndarray:
        """
           Generate audio frames using  sliding window with stride Numpy - non-memory safe
        """
        no_frames = int((length - window_size) / stride) + 1
        audio_frames = as_strided(audio, shape=(no_frames, window_size * sr), strides=(stride * sr, 1))

        audio_frames = audio_frames[:-2]

        return audio_frames

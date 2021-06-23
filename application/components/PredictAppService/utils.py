from typing import Tuple
import librosa
import numpy as np
import io


def get_audio(file_path) -> Tuple[np.ndarray, int]:
    print(type(librosa.load(io.BytesIO(file_path))))

    audio_data, sr = librosa.load(file_path)
    length = len(audio_data) / sr

    return audio_data, sr, length

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import stats
import IPython.display as ipd
import librosa
import librosa.display
from .treadsafeSingleton import SingletonMeta

from skimage.restoration import denoise_wavelet



class Wavelet_Filter:

    def wavelet_filter(self, filteredSignal, samplerate):
        x_den = denoise_wavelet(filteredSignal, method='VisuShrink', mode='soft', wavelet_levels=5, wavelet='coif2',
                                rescale_sigma='True')

        return x_den, samplerate


class Filter_BW_HP:

    def __init__(self, high_pass):
        self.high_pass = high_pass

    def BW_highpass(self, newdata, samplerate):
        b, a = signal.butter(4, 100 / (22050 / 2), btype='highpass')

        filteredSignal = signal.lfilter(b, a, newdata)

        return filteredSignal, samplerate


class FIlter_BW_LP:

    def __init__(self, low_pass):
        self.low_pass = low_pass

    def BW_lowpass(self, filteredSignal, samplerate):
        c, d = signal.butter(4, 2000 / (22050 / 2), btype='lowpass')
        newFilteredSignal = signal.lfilter(c, d, filteredSignal)

        return newFilteredSignal, samplerate


class FilterPipeline(metaclass=SingletonMeta):

    def __init__(self, low_pass, high_pass):

        self.low_pass = low_pass
        self.high_pass = high_pass

        self.lp_filter = FIlter_BW_LP(low_pass)
        self.hp_filter = Filter_BW_HP(high_pass)
        self.wavelet = Wavelet_Filter()

    def filters(self, audio_signal, sample_rate):
        filtered_output, sr = self.lp_filter.BW_lowpass(audio_signal, sample_rate)
        filtered_output, sr = self.hp_filter.BW_highpass(filtered_output, sr)
        filtered_output, sr = self.wavelet.wavelet_filter(filtered_output, sr)

        return filtered_output, sr


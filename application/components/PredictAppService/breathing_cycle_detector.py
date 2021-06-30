
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import numpy.matlib
import pdb
import numpy as np

# apply fourier transform

import scipy
def fft_plot(amplitudes, sampling_rate):
    n = len(amplitudes)
    T = 1/ sampling_rate
    yf = scipy.fft.fft(amplitudes)
    xf = np.linspace(0, 1 // (2 * T), n // 2)
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / n * np.abs(yf[:n // 2]))
    plt.grid()
    plt.xlabel('Frequency')
    plt.ylabel('magnitude')
    return plt.show()


N = 2048
L = 512
Frame_size = N
Hop_size = L


# move hamming window by L samples

def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
    frames = frame(signal, window_length, hop_length);

    window = periodic_hann(window_length);
    windowed_frames = frames * window;
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


# def of mean energy

def sigma(first, last, const):
    sum = 0
    for i in range(first, last + 1):
        sum += const * i
    return sum


# calculation of mean energy
# smoothing of mean energy

def smooth(y):
    [n1, n2] = y.shape
    n = n1 * n2

    u = []
    w = []

    for i in range(0, n2):
        u.append(round(math.cos(i * math.pi / n2), 3))

    uu = np.multiply(2, np.asarray(u))

    for j in range(0, n1):
        w.append(round(math.cos(j * math.pi / n1), 3))

    ww = np.multiply(2, (np.asarray(w)).T)
    ww1 = ww.reshape(n1, 1)

    # lamda = numpy.add(np.matlib.repmat( -2+2*y*math.pi/n2 ,n1 ,1) , -2+2*math.cos(numpy.transpose(0:n1-1)*math.pi/n1))
    lamda = np.add(np.matlib.repmat(-2 + uu, n1, 1), np.matlib.repmat(-2 + ww1, 1, n2))

    DCTy = scipy.fftpack.dct(y)

    p = 5

    def GCVscore(self):
        s = np.power(10, p)

        # Gamma = 1./(1 + s*Lamda^2)
        k = 1 + np.power(s * lamda, 2)
        Gamma = np.divide(1, k)

        # RSS = numpy.linalg.norm((DCTy(:).*(Gamma(:)-1)))^2
        RSS = np.linalg.norm(np.multiply(DCTy, Gamma - 1)) ** 2

        TrH = sum(Gamma)

        GCVs = RSS / n / (1 - TrH / n) ** 2

        return GCVs

    scipy.optimize.fminbound(GCVscore, -15, 38)

    z = scipy.fftpack.idct(Gamma * DCTy)

    return z



# find local minimum points in smoothed array
def findLocalMinima(n, arr):
    # Empty lists to store points of local minima
    mn = []

    # Checking whether the first point is local minima
    if (arr[0] < arr[1]):
        mn.append(0)

        # Iterating over all points to check local minima
        for i in range(1, n - 1):

            if (arr[i - 1] > arr[i] < arr[i + 1]):
                mn.append(i)

                # Checking whether the last point is local minima
    if (arr[-1] < arr[-2]):
        mn.append(n - 1)



float
Lmax = 5.5;
float
Lmin = 1.25;

# determining boundries of all models
def determining boundries():
     while i > 0:
        if i == 1:
            if ((mInd[j] > 1) & (mInd[j] < 2 * Lmax)):
                Llim[i - 1] = mInd[0]  # Llim = Lower limit

        elif i > 1:
            if ((mInd[j] > 1) & (mInd[j] < 2 * Lmax)):
                Llim[i - 1] = Ulim[i - 2]  # Ulim = Upper limit

        # j = i + 1

        for j in mInd:
            if j <= i:
                continue
            x = mInd[j] - mInd[i]
            if ((x > Lmin) & (x < Lmax)):
                if ((mInd[j] > 1) & (mInd[j] < 2 * Lmax)):
                    Ulim[i - 1] = mInd[j]

        i = i + 1

        if i == ln:
            break


#
def get_samples(arr, low, high):
    for i in sme:
        if i == low:
            x1 = sme.index(i)
            break

    for j in sme:
        if j == high:
            x2 = sme.index(j)
            break

    # get values in  low and high
    samples = sme[x1:x2 + 1]

    # decimation
    Ns = 44100  # not sure
    len(samples)
    des_samples = scipy.signal.decimate(samples, ...)  # downsampling factor depends on length

    return des_samples, samples


# getting model sample values for all models
def get_all():
    for i in Llim, Ulim:
        n1 = Llim[i]
        n2 = Ulim[i]
        # des_samples = decimated samples
        des_samples[i, :] = get_samples(sme, n1, n2)
        samples[i, :] = get_samples(sme, n1, n2)

        # first row of des_samples contain the sample values of the first model.

def get_coef():
    for i in range(1, Ns):  # here the Xs value is confusing
        max = numpy.amax(samples[i - 1, :])
        input = numpy.round(Llim[i - 1] + i * ((Ulim[i] - Llim[i]) / (Ns - 1)))
        Mi[i] = (1 / max) * sme[input]



p = 10;
a = 0;

# DTW
# determine lower limit of test patterns for all models
# all subtest patterns have the same lower limit

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def dtw_moves():

    while (a >= 0):

        for i in range(1, p):
            if p == 1:
                Llim_test[i - 1] = Ulim[a]
            elif p > 1:
                Llim_test[i - 1] = Ulim_test[i - 2]

            # determining upper limitis of subtest patterns
            r = ....
            for n in range(1, r):
                for j in sme:
                    w = sme[j] - Llim_test[i - 1]
                    if (w > Lmin) & (w < Lmax):
                        Ulim_subtest[n - 1] = sme[j]
                        # getting samples for each subtest pattern
                        subtest_samples[i, n] = sme[Llim_test[i - 1]: ULim_subtest[n - 1]]
                        # should have Ns amount of samples
                        subtest_samples_des[i, n] = scipy.signal.decimate(subtest_samples[i, n], )

                        distance[i, n], path[i, n] = fastdtw(subtest_samples_des[], Mi[a], dist=euclidean)

            # finding the subtest with minimum DTW
            test_pattern[i] = numpy.amin(distance[i])

        a = a + 1

        if a == len(Llim):
            break
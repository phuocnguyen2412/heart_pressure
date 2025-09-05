import argparse
import shutil
import traceback

import matplotlib

matplotlib.use("agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, resample
from scipy.signal import lfilter, filtfilt
import scipy.fftpack

from setting import SAMPLE_RATE

class SignalPreprocessor():

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.shorter_names = {
            "hpf": "butter_highpass_filter",
            "lpf": "butter_lowpass_filter",
            "maf": "moving_average_flat",
            "diff_pad": "minus_with_pad",
            "fft": "fft",
            "roll_avg": "rolling_average",
            "sub": "subtract",
            "bandpass": "butter_bandpass_filter",
            "imf": "increase_main_freq",
            "cut_start": "cut_start",
            "bpf_bpm": "bandpass_bpm"
        }

    def bandpass_bpm(self, signal, multiplier, mincut, order, **kwargs):

        no_nan_signal = np.array(signal)
        n_nan = 0
        if np.any(np.isnan(signal)):
            n_nan = signal[np.isnan(signal)].shape[0]
            no_nan_signal = signal[~np.isnan(signal)]

        T = 1.0 / self.sample_rate
        N = no_nan_signal.shape[0]
        signal_fft = np.abs(scipy.fftpack.fft(no_nan_signal))[:N // 2]
        signal_fft = signal_fft / signal_fft.max()

        freq_x = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

        max_cutoff = freq_x[np.argmax(signal_fft)] * multiplier
        # Thêm điều kiện kiểm tra và sửa lỗi, kẹp vào (0, Nyquist)
        nyq = 0.5 * self.sample_rate
        original_cutoff = max_cutoff
        # Kẹp cận dưới theo mincut, tránh 0
        max_cutoff = max(max_cutoff, max(mincut, 0.001))
        # Kẹp cận trên để không vượt Nyquist
        if max_cutoff >= nyq:
            max_cutoff = 0.99 * nyq
        if original_cutoff != max_cutoff:
            print(f"Warning: cutoff adjusted from {original_cutoff:.6f} to {max_cutoff:.6f} (Nyquist={nyq:.6f})")
        y = self.butter_highpass_filter(no_nan_signal, max_cutoff, order)
        y = np.concatenate((np.full(n_nan, np.nan), y), axis=0)

        return y

    def cut_start(self, signal, seconds, **kwargs):
        n_frames = self.sample_rate * seconds
        return np.concatenate((np.full(n_frames, np.nan), signal[n_frames:]), axis=0)

    def rolling_average(self, signal, **kwargs):
        window_size_seconds = kwargs["window_size_seconds"]
        window_size = int(window_size_seconds * self.sample_rate)
        if window_size % 2 == 0:
            window_size += 1
        y = np.convolve(signal, np.ones(window_size), 'valid') / window_size
        y = np.pad(y, [((window_size - 1) // 2, (window_size - 1) // 2)], mode='edge')
        return y

    def butter_bandpass_filter(self, signal, lowcut, highcut, order, **kwargs):
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, signal)
        return y

    def subtract(self, signal, **kwargs):
        original_signal = kwargs["prev_x"]
        assert signal.shape == original_signal.shape
        y = original_signal-signal
        return y

    def butter_highpass_filter(self, signal, cutoff, order, **kwargs):
        no_nan_signal = np.array(signal)
        n_nan = 0
        if np.any(np.isnan(signal)):
            n_nan = signal[np.isnan(signal)].shape[0]
            no_nan_signal = signal[~np.isnan(signal)]
        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyq
        # Kẹp giá trị chuẩn hóa vào (0,1) để tránh lỗi từ scipy
        if not np.isfinite(normal_cutoff):
            normal_cutoff = 0.001
        normal_cutoff = min(max(normal_cutoff, 1e-6), 0.999)
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, no_nan_signal)
        y = np.concatenate((np.full(n_nan, np.nan), y), axis=0)
        return y

    def butter_lowpass_filter(self, signal, low, filter_order, **kwargs):
        nyq = 0.5 * self.sample_rate
        normal_cutoff = low / nyq
        b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, signal)
        return y

    def minus_with_pad(self, x, pad, **kwargs):
        prev_x_padded = np.pad(kwargs["prev_x"], [(pad, 0)], mode='constant', constant_values=0)
        x_padded = np.pad(x, [(0, pad)], mode='constant', constant_values=0)
        y = prev_x_padded - x_padded
        y = y[pad:]
        return y

    def moving_average_flat(self, x, window_size=10, **kwargs):
        assert window_size % 2 == 1, "Odd number for window size in moving average pls"
        conv = np.convolve(x, np.ones(window_size), 'valid') / window_size
        return np.pad(conv, [((window_size-1)//2, (window_size-1)//2)], mode='constant', constant_values=0)

    def fft(self, signal, **kwargs):
        fourierTransform = np.fft.fft(signal) / len(signal)  # Normalize amplitude
        fourierTransform = fourierTransform[range(int(len(signal) / 2))]  # Exclude sampling frequency

        tpCount = len(signal)
        values = np.arange(int(tpCount / 2))
        timePeriod = tpCount / self.sample_rate
        frequencies = values / timePeriod

        return np.abs(fourierTransform[:200])

    def increase_main_freq(self, signal, **kwargs):
        pad = kwargs["pad"]
        mult = kwargs["mult"]
        fourierTransform = np.fft.fft(signal)  # Normalize amplitude
        # fourierTransform = fourierTransform[range(int(len(signal) / 2))]  # Exclude sampling frequency

        i_of_max = np.argmax(fourierTransform[10:50])+10

        fourierTransform[i_of_max-pad:i_of_max+pad] = fourierTransform[i_of_max-pad:i_of_max+pad] * mult
        y = np.fft.ifft(fourierTransform)
        return y


def visualize_signal(signals, labels, output_fname, title="", source="r_ch_mean"):
    plt.figure()
    plt.title(title)
    l_tp = [
        f"{source}",
        f"{source}>cut_start",
        f"{source}>cut_start>hpf",
        f"{source}>cut_start>hpf>bpf_bpm",
        f"{source}>roll_avg>sub>lpf>cut_start"
    ]
    n_plots = len(l_tp)
    fig, ax = plt.subplots(nrows=n_plots, figsize=(6, int(n_plots*2)))
    plot_c = 0
    printed_already = []

    for i, signal in enumerate(signals):
        # for visualization normalize signal
        # print(labels[i], printed_already)
        if labels[i] in l_tp and labels[i] not in printed_already:

            # to_plot = (signal-signal.mean())/ signal.std()
            to_plot = signal
            ax[plot_c].plot(range(len(signal)), to_plot, label=labels[i])
            printed_already.append(labels[i])
            ax[plot_c].grid(linestyle='dashed',)
            ax[plot_c].legend(loc="lower right")
            ax[plot_c].set_xlim((-10, len(signal)+10))
            plot_c += 1

    plt.tight_layout()
    plt.savefig(output_fname, bbox_inches="tight")
    plt.close("all")
    return True
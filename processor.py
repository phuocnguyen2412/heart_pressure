from scipy.signal import medfilt
import numpy as np
from scipy.signal import butter, filtfilt
from setting import TARGET_SAMPLES

class Processor:
    def median_filter(self, signal, kernel_size=5):
        return medfilt(signal, kernel_size)

    def zscore_outlier_impute(self, signal, threshold=3):
        signal = np.array(signal).flatten()
        mean, std = np.mean(signal), np.std(signal)
        z_scores = np.abs((signal - mean) / std)
        clean_signal = signal.copy()
        for i in range(len(signal)):
            if z_scores[i] > threshold:
                clean_signal[i] = clean_signal[i - 1] if i > 0 else mean
        return clean_signal

    def clip_to_range(self, signal, min_val, max_val):
        return np.clip(signal, min_val, max_val)

    def butter_lowpass_filter(self, signal, fs, cutoff=5):
        b, a = butter(4, cutoff / (0.5 * fs), btype='low')
        return filtfilt(b, a, signal)

    def min_max_scaler(self, signal, min_val, max_val):
        signal = np.array(signal)
        signal = self.clip_to_range(signal, min_val, max_val)
        signal = (signal - min_val) / (max_val - min_val)
        signal = signal.reshape(1, TARGET_SAMPLES, 1)
        return signal

    def inverse_min_max_scaler(self, signal, min_val, max_val):
        signal = np.array(signal)
        signal = signal * (max_val - min_val) + min_val
        return signal
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from setting import BASE_DIR
from evaluate import predicting_ABP_waveform
from models import UNetDS64, MultiResUNet1D

# Load metadata
meta = pickle.load(open(os.path.join(BASE_DIR, 'data', 'meta9.p'), 'rb'))
min_ppg, max_ppg = meta['min_ppg'], meta['max_ppg']
min_abp, max_abp = meta['min_abp'], meta['max_abp']

# Refine ABP waveform
mdl2 = MultiResUNet1D(1024)
mdl2.load_weights(os.path.join(BASE_DIR, 'models', 'RefinementNetwork.h5'))

 # Predict approximate ABP waveform
mdl1 = UNetDS64(1024)
mdl1.load_weights(os.path.join(BASE_DIR, 'models', 'ApproximateNetwork.h5'))
# ================= Utility Functions ================= #
def clip_signal(signal, min_val, max_val):
    return np.clip(signal, min_val, max_val)

def remove_spikes(signal, threshold=3):
    signal = np.array(signal).flatten()
    mean, std = np.mean(signal), np.std(signal)
    z_scores = np.abs((signal - mean) / std)
    clean_signal = signal.copy()
    for i in range(len(signal)):
        if z_scores[i] > threshold:
            clean_signal[i] = clean_signal[i - 1] if i > 0 else mean
    return clean_signal

def median_filter(signal, kernel_size=5):
    return medfilt(signal, kernel_size)

def calculate_heart_rate(signal, fs=125):
    signal = np.array(signal).flatten()
    peaks, _ = find_peaks(signal, distance=0.4 * fs)
    return (len(peaks) / (len(signal) / fs)) * 60

def extract_sbp_dbp(abp_signal, distance=30):
    abp_signal = np.array(abp_signal).flatten()
    sbp_idx, _ = find_peaks(abp_signal, distance=distance)
    dbp_idx, _ = find_peaks(-abp_signal, distance=distance)
    return abp_signal[sbp_idx], abp_signal[dbp_idx], sbp_idx, dbp_idx

from scipy.signal import butter, filtfilt

def lowpass_filter(signal, fs, cutoff=5):
    b, a = butter(4, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal)

def plot_abp_with_sbp_dbp(abp_signal, sbp_idx, dbp_idx):
    abp_signal = np.array(abp_signal).flatten()
    plt.plot(abp_signal, label="ABP")
    plt.plot(sbp_idx, abp_signal[sbp_idx], 'ro', label="SBP")
    plt.plot(dbp_idx, abp_signal[dbp_idx], 'go', label="DBP")
    plt.legend()
    plt.title("ABP waveform with SBP and DBP points")
    plt.xlabel("Time (samples)")
    plt.ylabel("Pressure (mmHg)")    
    plt.grid()
    plt.savefig(os.path.join('abp_with_sbp_dbp.png'))
    # plt.show()  # Commented out for server environment

# ================= Main Prediction Function ================= #
def predict_test_data(x_test):
    # Normalize and reshape PPG input
    x_test = np.array(x_test)
    x_test = clip_signal(x_test, min_ppg, max_ppg)
    ppg_norm = (x_test - min_ppg) / (max_ppg - min_ppg)
    ppg_norm = ppg_norm.reshape(1, 1024, 1)

   
    approx_abp = mdl1.predict(ppg_norm, verbose=1)
    refined_abp = mdl2.predict(approx_abp[0], verbose=1)

    # Denormalize ABP prediction
    abp_pred = refined_abp * (max_abp - min_abp) + min_abp
    abp_pred = lowpass_filter(abp_pred.flatten(), fs=125, cutoff=5)

    # Save prediction
    predicting_ABP_waveform(x_test, abp_pred)

    # Estimate heart rate (weighted avg of PPG & ABP HR)
    hr =  calculate_heart_rate(abp_pred)
    # # Tính distance tương ứng
    # beat_interval_sec = 60.0 / hr
    # distance = int(beat_interval_sec * 125 )  # Lấy 80% để tránh bỏ đỉnh gần
    distance = 125
    print(f"Estimated heart rate: {hr:.2f} bpm, distance: {distance} samples")
    distance = max(20, distance)  # đảm bảo không quá nhỏ

    # Extract SBP, DBP
    sbp_vals, dbp_vals, sbp_idx, dbp_idx = extract_sbp_dbp(abp_pred, distance=distance)
    sbp, dbp = np.median(sbp_vals), np.median(dbp_vals)

    map_val = (2 * dbp + sbp) / 3

    # Plot ABP with annotations
    plot_abp_with_sbp_dbp(abp_pred, sbp_idx, dbp_idx)

    result = {"hr": hr, "systolic": sbp, "diastolic": dbp, "mean": map_val}
    print(result)
    return result

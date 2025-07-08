"""
    Computes the outputs for test data
"""
import pickle

from scipy.signal import find_peaks

from evaluate import predicting_ABP_waveform
from models import UNetDS64, MultiResUNet1D
import os
import numpy as np
from setting import BASE_DIR

def calculate_heart_rate(signal, fs=125):
    """
    Tính nhịp tim từ tín hiệu PPG hoặc ABP.
    ---
    signal: mảng tín hiệu, có thể là dạng 1D hoặc 2D cần flatten
    fs: tần số lấy mẫu
    """
    signal = np.array(signal).flatten()  # đảm bảo là mảng 1D

    peaks, _ = find_peaks(signal, distance=fs*0.4)

    num_peaks = len(peaks)
    duration_sec = len(signal) / fs

    hr = (num_peaks / duration_sec) * 60
    return hr

def clip_signal(signal, min_val, max_val):
    return np.clip(signal, min_val, max_val)

def remove_spikes(signal, threshold=3):
    """
    Loại bỏ điểm nhiễu đột ngột bằng phương pháp dựa trên z-score.
    Các giá trị có z-score lớn hơn threshold sẽ được thay bởi giá trị trước đó.
    """
    signal = np.array(signal).flatten()
    mean = np.mean(signal)
    std = np.std(signal)
    z_scores = np.abs((signal - mean) / std)
    clean_signal = signal.copy()
    for i in range(len(signal)):
        if z_scores[i] > threshold:
            # Thay thế bằng giá trị trước đó (hoặc giá trị trung bình)
            clean_signal[i] = clean_signal[i-1] if i > 0 else mean
    return clean_signal

from scipy.signal import medfilt

def median_filter(signal, kernel_size=5):
    """
    Lọc nhiễu đột ngột bằng median filter.
    """
    return medfilt(signal, kernel_size)

def predict_test_data(x_test):
    """
        Computes the outputs for test data
        and saves them in order to avoid recomputing
    """
    #preprocessing
    dt = pickle.load(open(os.path.join(BASE_DIR, 'data', 'meta9.p'), 'rb'))			# loading metadata
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']
    print({
        "max_ppg": max_ppg,
        "min_ppg": min_ppg,
        "max_abp": max_abp,
        "min_abp": min_abp
    })
    # x_test = clip_signal(x_test, min_ppg, max_ppg)  # Lọc nhiễu đột ngột bằng median filter
    ppg_norm = np.array(x_test)      # Thêm dòng này để chắc chắn x_test là array

    ppg_norm = (ppg_norm - min_ppg) / (max_ppg - min_ppg)

    length = 1024               # length of signal

    ppg_norm = ppg_norm.reshape(1, 1024, 1)
    print("Shape of x_test:", ppg_norm.shape)
    mdl1 = UNetDS64(length)                                             # creating approximation network
    mdl1.load_weights(os.path.join(BASE_DIR, 'models','ApproximateNetwork.h5'))   # loading weights

    Y_test_pred_approximate = mdl1.predict(ppg_norm,verbose=1)        # predicting approximate abp waveform

    mdl2 = MultiResUNet1D(length)                                       # creating refinement network
    mdl2.load_weights(os.path.join(BASE_DIR, 'models','RefinementNetwork.h5'))    # loading weights

    Y_test_pred = mdl2.predict(Y_test_pred_approximate[0],verbose=1)    # predicting abp waveform
    print("Shape of Y_test_pred:", Y_test_pred.shape)

    abp_pred = Y_test_pred * (max_abp - min_abp) + min_abp
    predicting_ABP_waveform(x_test, abp_pred)
    #Khôi phục hồi lại giá trị huyết áp từ giá trị dự đoán

    SBP = np.max(abp_pred)  # Huyết áp tâm thu (mmHg)
    DBP = np.min(abp_pred)  # Huyết áp tâm trương (mmHg)
    MAP = np.mean(abp_pred)  # Huyết áp trung bình (mmHg)
    HR = (calculate_heart_rate(x_test) * 3 + calculate_heart_rate(abp_pred))/4
    return {
        "hr": HR,
        "systolic": SBP,
        "diastolic": DBP,
        "mean": MAP
    }
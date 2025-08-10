import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import resample
import torch
from setting import BASE_DIR, DEVICE
import os
model_path = os.path.join(BASE_DIR, 'trained_model', 'cnn-lstm', 'best_model.pth')

def visualize_abp_waveform(X_test: np.ndarray, Y_test_pred):
    """
        An interactive way to predict the ABP waveform from PPG signal
        from the test data.
        Ground truth, prediction from approximation network and refinement network
        are presented, and a comparison is also demonstrated
    """
    # Y_test_pred_approximate = np.array(Y_test_pred_approximate)

    ppg_signal = np.squeeze(X_test)          # (1024,)
    abp_signal_pred = np.squeeze(Y_test_pred) # (1024,)# series for time axis
    time_scale = np.arange(0, 8.192, 8.192/len(ppg_signal))
    print("time_scale shape:", time_scale.shape)
    print("ppg_signal shape:", ppg_signal.shape)
    print("abp_signal_pred shape:", abp_signal_pred.shape)

    plt.figure(figsize=(30, 15))

    plt.subplot(5, 1, 1)
    plt.plot(time_scale, ppg_signal, c='k', linewidth=2)
    plt.title('Input PPG Signal', fontsize=20)

    # plt.subplot(5, 1, 2)
    # plt.plot(time_scale, abp_signal_pred_approximate, c='r', linewidth=2)
    # plt.ylabel('ABP (mmHg)', fontsize=15)
    # plt.title('Output of Approximate Network', fontsize=20)

    plt.subplot(5, 1, 3)
    plt.plot(time_scale, abp_signal_pred, c='b', linewidth=2)
    plt.ylabel('ABP (mmHg)', fontsize=15)
    plt.title('Output of Refinement Network', fontsize=20)

    plt.tight_layout()
    plt.savefig('output.png')


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
    plt.show()
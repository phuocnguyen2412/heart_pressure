import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import resample
from cnn_lstm import ConvNet
import torch
from setting import BASE_DIR
import os
model_path = os.path.join(BASE_DIR, 'trained_model', 'cnn-lstm', 'best_model.pth')
model = ConvNet()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
def extract_ppg_from_video(video_path, plot_ppg=True):
    roi = (100, 100, 200, 200)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames_7s = int(7 * fps)
    start_frame = max(0, (total_frames - num_frames_7s) // 2)
    end_frame = start_frame + num_frames_7s

    ppg_signal = []
    current_frame = 0

    while True:
        ret, frame = cap.read()


        if not ret or current_frame >= end_frame:
                break
        if current_frame >= start_frame:
            # Chọn vùng trung tâm (50% giữa khung hình)
            h, w, _ = frame.shape
            x1, x2 = int(w * 0.25), int(w * 0.75)
            y1, y2 = int(h * 0.25), int(h * 0.75)
            roi = frame[y1:y2, x1:x2]

            frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            red_channel = frame_rgb[:, :, 0]
            mean_red = np.mean(red_channel)
            ppg_signal.append(mean_red)
        current_frame += 1

    cap.release()
    ppg_signal = np.array(ppg_signal)
    print("PPG Signal Shape:", ppg_signal.shape)

    # Resample về đúng 875 giá trị (125Hz x 7s)
    ppg_resampled = resample(ppg_signal, 875)
    print("PPG Vector Shape:", ppg_resampled.shape)

    print("Min value:", ppg_resampled.min(), "at index", ppg_resampled.argmin())
    print("Max value:", ppg_resampled.max(), "at index", ppg_resampled.argmax())

    if plot_ppg:
        plt.figure(figsize=(10,4))
        plt.plot(ppg_resampled)
        plt.title("Extracted PPG Signal (7s middle)")
        plt.xlabel("Sample (125Hz)")
        plt.ylabel("Mean ROI Value (Red Channel)")
        plt.show()

    return ppg_resampled


def predict_blood_pressure(ppg_signal):
    output = model(torch.tensor(ppg_signal, dtype=torch.float32).unsqueeze(0))
    output = output.item()
    print("Predicted Blood Pressure:", output)
    return output
    
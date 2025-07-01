import argparse
import os
import shutil
from pprint import pprint

import cv2
import matplotlib
from scipy.signal import resample

from setting import config

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import FastICA, PCA
from joblib import Parallel, delayed, cpu_count
import yaml


class SignalExtractor():

    def __init__(self, sample_rate, initial_skip_seconds=0):
        self.sample_rate = sample_rate
        self.initial_skip_seconds = initial_skip_seconds

    def red_channel_mean(self, frames, **kwargs):
        signal = []
        for frame_bgr in frames:
            mean_of_r_ch = frame_bgr[..., 2].mean()
            signal.append(mean_of_r_ch)
        signal = np.array(signal)
        samples_to_skip = self.initial_skip_seconds*self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal

    def green_channel_mean(self, frames, **kwargs):
        signal = []
        for frame_bgr in frames:
            mean_of_r_ch = frame_bgr[..., 1].mean()
            signal.append(mean_of_r_ch)
        signal = np.array(signal)
        samples_to_skip = self.initial_skip_seconds*self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal

    def green_channel_mean_upper_half(self, frames, **kwargs):
        signal = []
        for frame_bgr in frames:
            mean_of_r_ch = frame_bgr[:frame_bgr.shape[0]//2, : , 1].mean()
            signal.append(mean_of_r_ch)
        signal = np.array(signal)
        samples_to_skip = self.initial_skip_seconds*self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal

    def luma_component_mean(self, frames, **kwargs):
        signal = []
        for frame_bgr in frames:
            img_ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            mean_of_luma = img_ycrcb[..., 0].mean()
            signal.append(mean_of_luma)

        signal = np.array(signal)
        samples_to_skip = self.initial_skip_seconds * self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal

    def ica_decomposition(self, frames, **kwargs):
        s_r, s_g, s_b = [], [], []
        for frame_bgr in frames:
            b, g, r = frame_bgr.mean(axis=0).mean(axis=0)
            s_r.append(r)
            s_b.append(b)
            s_g.append(g)

        s_r = np.array(s_r).reshape(1, -1)
        s_b = np.array(s_b).reshape(1, -1)
        s_g = np.array(s_g).reshape(1, -1)

        fica = FastICA(n_components=1)
        stackd = np.concatenate((s_r, s_b, s_g), axis=0).T
        signal = fica.fit_transform(stackd).flatten()
        samples_to_skip = self.initial_skip_seconds * self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal

    def red_ch_threshold(self, frames, n_calib_frames=90, perc=80, **kwargs):
        # Average the per-frame <perc> percentile of the red channel over the first <calib> frames
        calib_vals = []
        calib_count = 0
        # jesus don't judge me for this
        while calib_count <= n_calib_frames:
            b, g, r = cv2.split(frames[calib_count])
            r = r.flatten()
            cval = np.percentile(r, perc)
            calib_vals.append(cval)
            calib_count += 1

        threshold = np.mean(calib_vals)
        signal = []
        img_h, img_w, _ = frames[0].shape
        for frame in frames:
            b, g, r = cv2.split(frame)
            mask_gt_threshold = r>threshold
            signal.append(mask_gt_threshold.astype(int).sum()/(img_h*img_w))

        signal = np.array(signal)
        signal = signal[self.initial_skip_seconds*self.sample_rate:]  # ignore first second because of auto exposure
        return signal

    def small_boxes_man(self, frames, **kwargs):
        n_boxes = kwargs["n_boxes"]

        frame_w = frames[0].shape[1]
        frame_h = frames[0].shape[0]

        assert frame_w / n_boxes == frame_w // n_boxes, frame_h / n_boxes == frame_h // n_boxes

        box_h, box_w = frame_h // n_boxes, frame_w // n_boxes

        signal = np.zeros((len(frames), n_boxes, n_boxes))
        for i, frame_bgr in enumerate(frames):
            img_ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            for j in range(n_boxes):
                for k in range(n_boxes):
                    cell = img_ycrcb[j * box_h:(j + 1) * box_h, k * box_w:(k + 1) * box_w, :]
                    cell = cell[..., 0].mean()
                    signal[i, j, k] = cell

        signal = signal.reshape(signal.shape[0], -1)

        pca = PCA(n_components=1)
        signal = pca.fit_transform(signal).flatten()

        samples_to_skip = self.initial_skip_seconds * self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal


def visualize_signal(signals, labels, output_fname, title=""):
    fig, ax = plt.subplots(nrows=len(signals), figsize=(16, 4*len(signals)))

    for i, signal in enumerate(signals):
        # for visualization normalize signal
        to_plot = (signal-signal.mean())/ signal.std()

        ax[i].plot(range(signal.shape[0]), to_plot, label=labels[i])
        ax[i].legend()
        ax[i].grid(linestyle='dashed',)

    plt.savefig(output_fname, bbox_inches="tight")
    plt.close(fig)
    return True


# lets make this one parallel
def extract_signal(filename, video_path, output_folder):
    sample_rate = 125
    segment_seconds = 7

    # Đọc video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    print(f"Video FPS: {fps}, Total frames: {total_frames}, Duration: {duration_seconds:.2f}s")


    target_samples = sample_rate * segment_seconds
    # Tính vị trí đoạn 7s ở giữa video (theo frame)
    center_frame = total_frames // 2
    half_seg = int((segment_seconds * fps) / 2)
    start_frame = max(center_frame - half_seg, 0)
    end_frame = min(center_frame + half_seg, total_frames)
    num_frames_to_read = end_frame - start_frame

    print(f"Cropping from frame {start_frame} to {end_frame} (~{num_frames_to_read} frames)")

    # Đọc các frame cần thiết
    list_of_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(num_frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        list_of_frames.append(frame)
    cap.release()

    se = SignalExtractor(int(fps))
    csv_fpath = os.path.join(output_folder, filename + ".csv")


    columns, extracted_s = [], []




    print("Extracting signal from", video_path, "with", len(list_of_frames))

    extractors = [{'functions': ['luma_component_mean'],
                   'name': 'luma_mean',
                   'parameters': {'initial_skip_seconds': 0}},
                  {'functions': ['red_channel_mean'],
                   'name': 'r_ch_mean',
                   'parameters': {'initial_skip_seconds': 0}}]
    for extractor in extractors:
        columns.append(extractor["name"])
        assert len(extractor["functions"]) == 1, "Only one extractor function is supported, check config.json"
        for fun_name in extractor["functions"]:
            fun = getattr(se, fun_name)
            f_output = fun(frames=list_of_frames, **extractor["parameters"])
            if len(f_output) != target_samples:
                f_output = resample(f_output, target_samples)
            print("Extracted signal with function", fun_name, "with shape", f_output.shape)
        extracted_s.append(f_output.tolist())
    extracted_s = np.array(extracted_s) * -1
    assert extracted_s.ndim == 2, "Different functions resulted in different length of extracted signal"
    df = pd.DataFrame(extracted_s.T, columns=columns)
    df.to_csv(csv_fpath, sep=",", float_format="%.4f", index=False)


    # Visualize the signals
    n_extractors = len(extractors)
    csv_fname = filename + ".csv"
    pdf_fname = filename + ".pdf"
    df = pd.read_csv(os.path.join(config.output_folder, filename, csv_fname), index_col=False)
    df.iloc[30:].plot(kind="line", subplots=True, figsize=(16, 4*n_extractors), layout=(n_extractors, 1), grid=True)
    plt.savefig(os.path.join(config.output_folder, filename, pdf_fname), bbox_inches="tight")
    plt.close()








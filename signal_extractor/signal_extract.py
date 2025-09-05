import argparse
import os
import shutil
from pprint import pprint

import cv2
import matplotlib
from scipy.signal import resample, find_peaks



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









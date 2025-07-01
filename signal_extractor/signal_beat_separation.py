import argparse
import json
import shutil

import matplotlib
import yaml

matplotlib.use("agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin
import os
import sys


class SignalBeatSeparator():

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def moving_average_flat(self, x, window_size=10, **kwargs):
        assert window_size % 2 == 1, "Odd number for window size in moving average pls"
        conv = np.convolve(x, np.ones(window_size), 'valid') / window_size
        return np.pad(conv, [((window_size - 1) // 2, (window_size - 1) // 2)], mode='constant', constant_values=0)

    def match_minimums_in_signal(self, target_signal, target_signal_idxs, filtered_signal_idxs, min_f_gap):
        """
        For every minimum in filtered_signal_idxs attempts to match a minimum of target signal (contained in target_signal_idxs).
        Requires that:
        - matching minimums in target_signal_idx are actually minimums in target_signal
        - finds matching minimum in a min_f_gap window around the minimum in filtered_signal_idxx

        :param target_signal:
        :param target_signal_idxs: index of minimums in target signal
        :param filtered_signal_idxs: index of minimums in filtered signal (smoothed)
        :param min_f_gap:
        :return:
        """
        t_indexes = []
        for minimum in filtered_signal_idxs:
            start_i = max(0, minimum - min_f_gap)
            actual_offset = minimum - min_f_gap if minimum - min_f_gap >= 0 else 0
            end_i = min(target_signal.shape[0], minimum + min_f_gap)
            subset = target_signal[start_i: end_i]
            argmin = np.argwhere(subset == subset.min())
            argmin = argmin[0, 0]
            idx_in_tar_signal = argmin + actual_offset
            if idx_in_tar_signal in target_signal_idxs:
                t_indexes.append(argmin + actual_offset)
        return np.array(t_indexes)

    def remove_consecutive_mins(self, indexes, signal, min_frame_gap):
        """
        Returns a filtered version of indexes that guarantees that consecutive minimums are at least min_frame_gap
        from each other.
        When two consecutive minimums are less than min_frame_gap elemnts from each other, constructs
        a series of minimum by searching in indexes for additional consecutive minimums that all lie within a window
        of min_frame_gap, counting from the first minimum in the series.
        Only the element that corresponds to the lowest minimum in signal is kept among the element in the series.

        :param indexes: indexes of minimum elements
        :param signal: array containing the signal where the minimums are
        :param min_frame_gap:
        :return:
        """

        findexes = []
        i = 0
        while i < len(indexes) - 1:
            curr_index = indexes[i]
            next_index = indexes[i + 1]
            if np.abs(curr_index - next_index) >= min_frame_gap:
                findexes.append(curr_index)
                i += 1
            else:
                streak = [curr_index]
                while np.abs(next_index - curr_index) < min_frame_gap and i < len(indexes) - 1:
                    streak.append(next_index)
                    i += 1
                    next_index = indexes[i]
                streak = np.array(streak)
                values_at_indx = signal[streak]
                minim_value_indx = np.argwhere(values_at_indx == values_at_indx.min())[0, 0]
                findexes.append(streak[minim_value_indx])
        findexes.append(indexes[-1])
        return findexes

    def hb_argrelmin(self, df, order, **kwargs):

        # min_bpm = kwargs["min_bpm"]
        max_bpm = kwargs["max_bpm"]
        processed_column_name = kwargs["processed_column_name"]
        frame_rate = self.sample_rate

        min_frame_gap = frame_rate / (max_bpm / 60)
        # max_frame_gap = frame_rate / (min_bpm/60)

        target_signal = df[processed_column_name].dropna().values
        # smoothed_signal = df["r_ch_mean>lpf>diff_pad>lpf>maf"].values
        smoothed_signal = self.moving_average_flat(target_signal, window_size=kwargs["smooth_window_size"])
        original_signal = df["luma_mean"].values
        indexes = argrelmin(smoothed_signal, order=order)[0]

        lines_to_plot = [[
            (np.arange(target_signal.shape[0]), target_signal, "processed", "blue"),
            (np.arange(smoothed_signal.shape[0]), smoothed_signal, "smoothed", "orange"),
            (np.arange(original_signal.shape[0]), original_signal, "original", "magenta")
        ]]
        scatters = [[(indexes, smoothed_signal[indexes], "mins", "o")]]

        # remove duplicates
        indexes = np.unique(indexes)

        findexes = self.remove_consecutive_mins(indexes, smoothed_signal, min_frame_gap)
        scatters[0].append((findexes, smoothed_signal[findexes], "mins>-cons", "^"))

        # remove duplicates
        findexes = np.unique(findexes)

        mins_in_target_signal = argrelmin(target_signal, order=order)[0]
        findexes = self.match_minimums_in_signal(target_signal, mins_in_target_signal, findexes, int(min_frame_gap / 2))

        # remove duplicates
        findexes = np.unique(findexes)

        scatters[0].append((findexes, target_signal[findexes], "mins>-cons>matched", "x"))

        ltp2 = []

        all_beats = []
        s_max, s_min = target_signal.max(), target_signal.min()

        for i, start_beat in enumerate(findexes[:-1]):
            end_beat = findexes[i + 1] + 1
            subsig = target_signal[start_beat:end_beat]
            all_beats.append(subsig)
            ltp2.append((np.arange(start_beat, end_beat), subsig, "valid" if i == 0 else "", "blue"))
            ltp2.append(([start_beat, start_beat], [s_min, s_max], "", "k"))

        lines_to_plot.append(ltp2)

        return all_beats, lines_to_plot, scatters


def plot_things(lines, scatters, filename):
    fig, ax = plt.subplots(nrows=len(lines), figsize=(12, 12))
    for i in range(len(lines)):
        # ith subplot
        lines_tp = lines[i]
        for j, _ in enumerate(lines_tp):
            # jth line
            x, y, label, color = lines_tp[j]
            ax[i].plot(x, y, label=label, c=color)

    for i in range(len(scatters)):
        # ith subplot
        scatters_tp = scatters[i]
        for j, _ in enumerate(scatters_tp):
            # jth line
            x, y, label, marker = scatters_tp[j]
            ax[i].scatter(x, y, label=label, marker=marker)

    for i in range(max(len(scatters), len(lines))):
        ax[i].legend(loc="upper right")
        ax[i].grid()

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    # plt.clf()
    plt.close(fig)

def run_beat_separation(filename, output_folder):
    bd = SignalBeatSeparator(30)

    filepath = os.path.join(output_folder, filename + ".csv")

    img_fpath = os.path.join(output_folder, filename + ".pdf")
    json_fpath = os.path.join(output_folder, filename + ".json")

    params = {'beat_separation': [{'name': 'hb_argrelmin',
                                   'params': {'max_bpm': 101,
                                              'order': 2,
                                              'processed_column_name': 'luma_mean>roll_avg>sub>lpf>cut_start',
                                              'smooth_window_size': 13}}]
              }
    extracted_s = pd.read_csv(filepath, index_col=False)
    beats = dict()
    ltp = []
    stp = []
    assert len(params["beat_separation"]) == 1, "I haven't implemented multiple beat separators"
    for beat_extractor in params["beat_separation"]:
        fun = getattr(bd, beat_extractor["name"])
        good_beats, lines_to_plot, scatters_to_plot = fun(
            df=extracted_s,
            **beat_extractor["params"]
        )

        img_fpath = os.path.join(output_folder, filename + ".pdf")

        # max_beat_length = np.array([x.shape[0] for x in good_beats]).max()
        beats[beat_extractor["name"]] = dict()
        for j, b in enumerate(good_beats):
            beats[beat_extractor["name"]][j] = b.tolist()

        ltp.append(lines_to_plot)
        stp.append(scatters_to_plot)

    json.dump(beats, open(json_fpath, "w"))
    plot_things(ltp[0], stp[0], filename=img_fpath)








import traceback
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample
import torch

from setting import config, segment_seconds, target_samples
from signal_extractor.signal_extract import SignalExtractor
from signal_extractor.signal_preprocessing import (
    SignalPreprocessor,
    visualize_signal,
)
from pyVHR.analysis.pipeline import Pipeline


class BPExtractor:
    def __init__(self, extract_config=None):
        """
        Initialize the BPExtractor class.
        """
        self.extract_config = extract_config or config
        self.fps = 30

    def read_video(self, video_path):
        """
        Read the video and return the list of frames.
        """
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video FPS: {self.fps}, Total frames: {total_frames}")
        duration_seconds = total_frames / self.fps
        print(f"Duration: {duration_seconds:.2f}s")

        center_frame = total_frames // 2
        half_seg = int((segment_seconds * self.fps) / 2)
        start_frame = max(center_frame - half_seg, 0)
        end_frame = min(center_frame + half_seg, total_frames)
        num_frames_to_read = end_frame - start_frame

        print(
            f"Cropping from frame {start_frame} to {end_frame} (~{num_frames_to_read} frames)"
        )

        # Đọc các frame cần thiết
        list_of_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(num_frames_to_read):
            ret, frame = cap.read()
            h, w, _ = frame.shape
            x1, x2 = int(w * 0.25), int(w * 0.75)
            y1, y2 = int(h * 0.25), int(h * 0.75)
            roi = frame[y1:y2, x1:x2]

            frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            if not ret:
                break
            list_of_frames.append(frame_rgb)
        cap.release()

        return list_of_frames

    def extract_signal(self, filename, video_path, output_folder):
        list_of_frames = self.read_video(video_path)

        se = SignalExtractor(int(self.fps))
        csv_fpath = os.path.join(output_folder, filename + ".csv")
        columns, extracted_s = [], []
        print("Extracting signal from", video_path, "with", len(list_of_frames))

        extractors = [
            {
                "functions": ["luma_component_mean"],
                "name": "luma_mean",
                "parameters": {"initial_skip_seconds": 0},
            },
            {
                "functions": ["red_channel_mean"],
                "name": "r_ch_mean",
                "parameters": {"initial_skip_seconds": 0},
            },
        ]

        for extractor in extractors:
            columns.append(extractor["name"])
            assert (
                len(extractor["functions"]) == 1
            ), "Only one extractor function is supported, check config.json"
            for fun_name in extractor["functions"]:
                fun = getattr(se, fun_name)
                f_output = fun(frames=list_of_frames, **extractor["parameters"])
                if len(f_output) != target_samples:
                    f_output = resample(f_output, target_samples)
                print(
                    "Extracted signal with function",
                    fun_name,
                    "with shape",
                    f_output.shape,
                )
            extracted_s.append(f_output.tolist())
        extracted_s = np.array(extracted_s) * -1
        assert (
            extracted_s.ndim == 2
        ), "Different functions resulted in different length of extracted signal"
        df = pd.DataFrame(extracted_s.T, columns=columns)
        df.to_csv(csv_fpath, sep=",", float_format="%.4f", index=False)

        # Visualize the signals
        n_extractors = len(extractors)
        csv_fname = filename + ".csv"
        pdf_fname = filename + ".pdf"
        df = pd.read_csv(
            os.path.join(config.output_folder, filename, csv_fname), index_col=False
        )
        df.iloc[30:].plot(
            kind="line",
            subplots=True,
            figsize=(16, 4 * n_extractors),
            layout=(n_extractors, 1),
            grid=True,
        )
        plt.savefig(
            os.path.join(config.output_folder, filename, pdf_fname), bbox_inches="tight"
        )
        plt.close()

    def process_single_signal_file(self, file_name, output_folder):
        """
        Xử lý 1 file tín hiệu (CSV), xuất CSV kết quả và PDF hình ảnh.
        """
        sp = SignalPreprocessor(125)
        filepath = os.path.join(output_folder, file_name + ".csv")

        csv_fpath = os.path.join(output_folder, file_name + "_preprocessed.csv")
        img_fpath = os.path.join(output_folder, file_name + "_preprocessed.pdf")
        params = {
            "preprocessor": {
                "filter_chains": [
                    {
                        "flist": [
                            {
                                "name": "roll_avg",
                                "params": {"window_size_seconds": 1.01},
                            },
                            {"name": "sub", "params": {}},
                            {"name": "lpf", "params": {"filter_order": 2, "low": 4}},
                            {"name": "cut_start", "params": {"seconds": 0}},
                        ],
                        "name": "chain2",
                    },
                    {
                        "flist": [
                            {"name": "cut_start", "params": {"seconds": 0}},
                            {"name": "hpf", "params": {"cutoff": 0.5, "order": 1}},
                            {
                                "name": "bpf_bpm",
                                "params": {"mincut": 0.01, "multiplier": 3, "order": 1},
                            },
                        ],
                        "name": "dynamic_bpm",
                    },
                ],
                "sources": ["luma_mean"],
            }
        }

        try:

            to_plot, to_plot_names = [], []
            extracted_s = pd.read_csv(
                filepath, index_col=False, encoding_errors="ignore"
            )
            preprocessed, columns = [], []

            for source in params["preprocessor"]["sources"]:
                assert source in extracted_s.columns.values, "%s not in columns %s" % (
                    source,
                    filepath,
                )
                for filter_chain in params["preprocessor"]["filter_chains"]:
                    fun_list = filter_chain["flist"]
                    signal_at_step_j = [extracted_s[source].values]
                    name_at_step_j = [source]
                    for j, fun_dict in enumerate(fun_list):
                        # apply function
                        fun = getattr(sp, sp.shorter_names[fun_dict["name"]])
                        filtered_j = fun(
                            signal_at_step_j[-1],
                            prev_x=(
                                signal_at_step_j[-2]
                                if len(signal_at_step_j) > 1
                                else None
                            ),
                            **fun_dict["params"],
                        )
                        new_name = "%s>%s" % (name_at_step_j[-1], fun_dict["name"])

                        if len(filtered_j) == len(extracted_s[source].values):
                            signal_at_step_j.append(
                                np.real(filtered_j)
                            )  # discard imaginary part if any
                            name_at_step_j.append(new_name)
                        else:
                            to_plot.append(filtered_j)
                            to_plot_names.append(new_name)

                    preprocessed.extend(signal_at_step_j)
                    columns.extend(name_at_step_j)

            preprocessed = np.array(preprocessed)
            assert (
                preprocessed.ndim == 2
            ), "Different functions resulted in different length of preprocessed signal"
            df = pd.DataFrame(preprocessed.T, columns=columns)
            df.to_csv(csv_fpath, sep=",", float_format="%.8f", index=False)

            everything_to_plot = [a.tolist() for a in preprocessed] + to_plot
            everything_to_plot_labels = columns + to_plot_names

            visualize_signal(
                everything_to_plot,
                labels=everything_to_plot_labels,
                output_fname=img_fpath,
                source="r_ch_mean",
            )
            return df["luma_mean>cut_start>hpf>bpf_bpm"].values
        except Exception as e:
            traceback.print_exc()
            print(e)

    def extract_ppg_from_video(self, video_path):
        video_name = os.path.basename(video_path)
        print("video_path:", video_path)
        output_folder = os.path.join(config.output_folder, video_name)
        os.makedirs(output_folder, exist_ok=True)

        self.extract_signal(video_name, video_path, output_folder)
        print("Signal extraction completed for", video_name)

        processed_signal = self.process_single_signal_file(video_name, output_folder)
        print("Signal preprocessing completed for", video_name)
        return processed_signal
        
        # pipeline = Pipeline()
        # method = "cpu_POS"
        # wsize = 8

        # roi_approach = "patches"
        # use_cuda = torch.cuda.is_available()

        # # Ước lượng fps để quyết định pre_filt
        # cap = cv2.VideoCapture(video_path)
        # fps = cap.get(cv2.CAP_PROP_FPS) or 30
        # nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # cap.release()
        # frames_per_win = int(fps * wsize)
        # duration = nframes / max(fps, 1)
        # wsize = max(3, min(6, int(duration // 2)))  # đảm bảo tạo được ≥1 window
        # print(f"Wsize: {wsize}")
        # pre_filt_flag = frames_per_win > 45  # đủ dài mới bật lọc trước

        # # params
        # roi_approach = 'holistic'   # 'holistic' or 'patches'
        # bpm_est = 'clustering'         # BPM final estimate, if patches choose 'medians' or 'clustering'
        # method = 'cpu_OMIT'       # one of the methods implemented in pyVHR
        # pipe = Pipeline()          # object to execute the pipeline

        # # run
        # bvps, timesES, bpmES = pipe.run_on_video(video_path,
        #                                         winsize=wsize,
        #                                         roi_method='convexhull',
        #                                         roi_approach=roi_approach,
        #                                         method=method,
        #                                         estimate=bpm_est,
        #                                         patch_size=40,
        #                                         RGB_LOW_HIGH_TH=(5,230),
        #                                         Skin_LOW_HIGH_TH=(5,230),
        #                                         pre_filt=True,
        #                                         post_filt=True,
        #                                         cuda=True,
        #                                         verb=True,
        # )
        # print(bvps)
        # print(timesES)
        # print(bpmES)
        # print(f"Nhịp tim ước lượng: {bpmES:.2f} bpm")
        # print(f"Độ dài tín hiệu rPPG: {len(bvps)}")

        # # Đảm bảo lấy đúng 1024 giá trị
        # if len(bvps) >= 1024:
        #     ppg_1024 = bvps[:1024]
        # else:
        #     # Nội suy nếu ít hơn 1024
        #     ppg_1024 = np.interp(
        #         np.linspace(0, len(bvps)-1, 1024),
        #         np.arange(len(bvps)),
        #         bvps
        #     )

        # print("Shape:", ppg_1024.shape)
        # return ppg_1024

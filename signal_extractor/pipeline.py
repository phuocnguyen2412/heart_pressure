import os
import shutil
from multiprocessing import cpu_count
from pprint import pprint

import yaml

from setting import config
from signal_extractor.signal_beat_separation import run_beat_separation
from signal_extractor.signal_extract import extract_signal
from signal_extractor.signal_fiducial_points_detection import run_fiducial_points_detection
from signal_extractor.signal_preprocessing import process_single_signal_file, SignalPreprocessor

if config.force_redo:
    shutil.rmtree(config.output_folder, ignore_errors=True)
    os.makedirs(config.output_folder, exist_ok=True)

params = yaml.load(open(config.params, "r"), Loader=yaml.FullLoader)
print("Parameters loaded from", pprint(params))


allowed_video_fmts = ["mov", "mp4"]
print("Allowed video formats", allowed_video_fmts)

allowed_video_fmts.extend(list(map(lambda x: x.upper(), allowed_video_fmts)))


def run_extract_signal(file_path):

    video_name = os.path.basename(file_path)
    video_path = file_path
    print("video_path:", video_path)
    output_folder = os.path.join(config.output_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)


    extract_signal(
        video_name, video_path, output_folder
    )
    print("Signal extraction completed for", video_name)


    process_single_signal_file(
        video_name, output_folder
    )
    print("Signal preprocessing completed for", video_name)

    import pandas as pd
    signal_file = os.path.join(str(output_folder), video_name + "_preprocessed.csv")
    df = pd.read_csv(signal_file, index_col=False)

    data = df["luma_mean>roll_avg>sub>lpf>cut_start"].values
    print("Data shape:", data.shape)
    return data.tolist()




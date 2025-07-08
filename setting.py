import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 100000
sample_rate = 125
segment_seconds = 8.192
target_samples = int(sample_rate * segment_seconds)

class config:
    filename = os.path.join(BASE_DIR, "data", "IMG_1157.MOV")
    video_directory = os.path.join(BASE_DIR, "data", "videos")
    output_folder = os.path.join(BASE_DIR, "data", "output")
    display = False
    force_redo = True
    number_of_cpus = 8
    params = os.path.join(BASE_DIR, "signal_extractor", "params.yaml")

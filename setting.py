import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 100000
SAMPLE_RATE = 30
TARGET_SAMPLES = 1024
SEGMENT_SECONDS = TARGET_SAMPLES / SAMPLE_RATE

class config:
    filename = os.path.join(BASE_DIR, "data", "IMG_1157.MOV")
    video_directory = os.path.join(BASE_DIR, "data", "videos")
    output_folder = os.path.join(BASE_DIR, "data", "output")
    display = False
    force_redo = True
    number_of_cpus = 8
    params = os.path.join(BASE_DIR, "signal_extractor", "params.yaml")

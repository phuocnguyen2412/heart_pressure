import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 100000


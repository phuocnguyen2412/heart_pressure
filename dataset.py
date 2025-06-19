import glob

import h5py
import torch

from torch.utils.data import Dataset, DataLoader

class PPGDataset(Dataset):
    def __init__(self, h5_files, label):
        """
        Khởi tạo dataset từ các file HDF5 chứa dữ liệu PPG và nhãn huyết áp.
        :param h5_files: Danh sách các file HDF5 chứa dữ liệu PPG.
        :param label: Loại nhãn cần lấy ('SBP' hoặc 'DBP').
        :type label: str
        :raises ValueError: Nếu label không phải là 'SBP' hoặc 'DBP'.
        :raises FileNotFoundError: Nếu không tìm thấy file HDF5 nào.
        """
        self.label = label  # 'SBP' or 'DBP'
        self.files = h5_files
        self.sample_map = []
        for file_idx, fname in enumerate(self.files):
            with h5py.File(fname, 'r') as f:
                n = f['PPG'].shape[0]
                self.sample_map.extend([(file_idx, i) for i in range(n)])

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.sample_map[idx]
        fname = self.files[file_idx]
        with h5py.File(fname, 'r') as f:
            ppg = f['PPG'][sample_idx]
            target = f[self.label][sample_idx]
        ppg = torch.tensor(ppg, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return ppg, target

train_files = sorted(glob.glob("ppg_split_files/train_part*.h5"))
val_files = sorted(glob.glob("ppg_split_files/val_part*.h5"))
test_files = sorted(glob.glob("ppg_split_files/test_part*.h5"))

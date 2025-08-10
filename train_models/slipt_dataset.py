import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import os
from setting import BASE_DIR
# ======================== CONFIG ========================
INPUT_FILE = os.path.join(BASE_DIR, 'ppg-dataset', 'MIMIC-III_ppg_dataset.h5')
OUT_DIR = os.path.join(BASE_DIR, 'ppg_split_files')  # Output folder for split .h5 files
BATCH_SIZE = 100_000             # Number of samples per output file
SEED = 42                        # Random seed for reproducibility
# ========================================================

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_indices_for_subjects(subj_idx_arr, subject_list):
    mask = np.isin(subj_idx_arr.flatten(), subject_list)
    return np.where(mask)[0]

def save_batches(ppg, label, subject_idx, indices, split_name, out_dir, batch_size):
    num_samples = len(indices)
    num_batches = int(np.ceil(num_samples / batch_size))
    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, num_samples)
        batch_idx = indices[start:end]

        with h5py.File(
                os.path.join(out_dir, f"{split_name}_part{b+1}.h5"), "w"
        ) as f:
            f.create_dataset("PPG", data=ppg[batch_idx])
            f.create_dataset("SBP", data=label[batch_idx, 0])
            f.create_dataset("DBP", data=label[batch_idx, 1])
            f.create_dataset("subject_idx", data=subject_idx[batch_idx])

        print(f"Saved: {split_name}_part{b+1}.h5 ({end-start} samples)")

def main():
    # Create output directory
    make_dir(OUT_DIR)

    # Đọc dữ liệu chỉ lấy chỉ số subject_idx vào RAM
    with h5py.File(INPUT_FILE, "r") as f:
        subject_idx = f['subject_idx'][:]

    # Lấy danh sách subject duy nhất và chia tập
    unique_subjects = np.unique(subject_idx)
    train_subj, temp_subj = train_test_split(
        unique_subjects, test_size=0.30, random_state=SEED
    )
    val_subj, test_subj = train_test_split(
        temp_subj, test_size=0.5, random_state=SEED
    )

    # Lấy index của từng split
    train_idx = get_indices_for_subjects(subject_idx, train_subj)
    val_idx = get_indices_for_subjects(subject_idx, val_subj)
    test_idx = get_indices_for_subjects(subject_idx, test_subj)

    print("Samples per split:", len(train_idx), len(val_idx), len(test_idx))

    # Đọc dữ liệu theo batch và lưu từng phần nhỏ
    with h5py.File(INPUT_FILE, "r") as f:
        ppg = f["ppg"]
        label = f["label"]

        print("Saving train set...")
        save_batches(ppg, label, subject_idx, train_idx, "train", OUT_DIR, BATCH_SIZE)
        print("Saving val set...")
        save_batches(ppg, label, subject_idx, val_idx, "val", OUT_DIR, BATCH_SIZE)
        print("Saving test set...")
        save_batches(ppg, label, subject_idx, test_idx, "test", OUT_DIR, BATCH_SIZE)

if __name__ == "__main__":
    main()
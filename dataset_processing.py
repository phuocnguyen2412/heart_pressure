import h5py

with h5py.File('MIMIC-III_ppg_dataset.h5', 'r') as f:
    print("Các group/dataset trong file:", list(f.keys()))
    # Đọc dữ liệu
    ppg = f['ppg'][:]            # shape (905400, 875)
    label = f['label'][:]        # shape (905400, 2)
    subject_idx = f['subject_idx'][:]  # shape (905400, 1)

print('PPG:', ppg.shape)
print('Label:', label.shape)
print('Subject_idx:', subject_idx.shape)

import matplotlib.pyplot as plt

# Lấy mẫu đầu tiên
sample_idx = 0
sample_ppg = ppg[sample_idx]
sample_bp = label[sample_idx]
subject = subject_idx[sample_idx][0]

plt.plot(sample_ppg)
plt.title(f'PPG Sample - Subject {subject} | SBP: {sample_bp[0]}, DBP: {sample_bp[1]}')
plt.xlabel('Time')
plt.ylabel('PPG Value')
plt.show()
# Hướng dẫn Calibration System

## 1. Chuẩn bị Video Input

### Bước 1: Copy video vào folder
```
data/calibration_videos/
```

### Bước 2: Đặt tên video theo format:
- video_1.mp4 (hoặc .MOV, .avi)
- video_2.mp4
- video_3.mp4
- ...
- video_9.mp4

## 2. Cấu trúc Data

### Train/Validation Set (ID 4-9):
- video_4 đến video_9
- Sử dụng để training calibration model

### Test Set (ID 1-3):
- video_1 đến video_3  
- Sử dụng để đánh giá hiệu quả calibration

## 3. Cách sử dụng

### Kiểm tra dữ liệu calibration:
```powershell
python calibration_data_loader.py
```

### Test model hiện tại trên calibration videos:
```powershell
python test_calibration.py
```

## 4. Output Files

### calibration_test_results.csv
Chứa kết quả comparison:
- ground_truth_sys/dia: Huyết áp chuẩn từ máy đo
- app_sys/dia: Kết quả cũ từ app
- predicted_sys/dia: Kết quả mới từ model hiện tại

## 5. Error Metrics

- **MAE (Mean Absolute Error)**: Trung bình sai số tuyệt đối
- **RMSE (Root Mean Square Error)**: Căn bậc hai của trung bình bình phương sai số  
- **Bias**: Sai số trung bình (có dấu)

## 6. Next Steps (Chưa implement)

1. **Linear Calibration**: Áp dụng linear regression để hiệu chỉnh
2. **Non-linear Calibration**: Sử dụng polynomial hoặc neural network
3. **Feature-based Calibration**: Hiệu chỉnh dựa trên đặc trung cá nhân

## 7. Folder Structure
```
data/
├── calibration_data.csv          # Ground truth data
├── calibration_videos/           # Input videos (cần copy video vào đây)
│   ├── video_1.mp4
│   ├── video_2.mp4
│   └── ...
└── calibration_test_results.csv  # Kết quả test (tự động tạo)
```

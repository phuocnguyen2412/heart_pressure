import os
import numpy as np
import pandas as pd
import json
from bp_extractor import BPExtractor
from setting import BASE_DIR

def optimize_heart_rate_parameters():
    """Tối ưu hóa tham số cho thuật toán tính nhịp tim."""
    # Đọc dữ liệu ground truth
    csv_path = os.path.join(BASE_DIR, "data", "nguyen_video", "truth.csv")
    df = pd.read_csv(csv_path)
    
    # Chuẩn bị đường dẫn video và ground truth
    video_paths = []
    ground_truth_hrs = []
    
    for _, row in df.iterrows():
        video_name = row["video_name"]
        video_path = os.path.join(BASE_DIR, "data", "nguyen_video", video_name)
        
        # Sử dụng trung bình của cả hai loại HR làm ground truth
        # hr_avg = (row["hr_instant_hr"] + row["hr_inpulse"]) / 2
        hr_avg = row["hr_instant_hr"]
        video_paths.append(video_path)
        ground_truth_hrs.append(hr_avg)
    
    # Khởi tạo extractor và tối ưu hóa tham số
    extractor = BPExtractor()
    best_params = extractor.optimize_hr_parameters(video_paths, ground_truth_hrs)
    
    # Lưu tham số tốt nhất vào file
    params_path = os.path.join(BASE_DIR, "hr_optimal_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Optimal parameters saved to {params_path}")
    
    # Đánh giá với tham số tốt nhất
    hr_preds = []
    for video_path in video_paths:
        hr = extractor.calculate_heart_rate_from_video(video_path, params=best_params)
        hr_preds.append(hr)
    
    # Tính MAE cho mỗi loại HR
    mae_instant = np.mean(np.abs(np.array(hr_preds) - np.array(df["hr_instant_hr"])))
    mae_inpulse = np.mean(np.abs(np.array(hr_preds) - np.array(df["hr_inpulse"])))
    
    print(f"Final MAE with optimal parameters:")
    print(f"  - hr_instant_hr: {mae_instant:.2f}")
    print(f"  - hr_inpulse: {mae_inpulse:.2f}")

if __name__ == "__main__":
    optimize_heart_rate_parameters()

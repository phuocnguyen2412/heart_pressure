from bp_extractor import BPExtractor
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pprint import pprint

from setting import BASE_DIR, SAMPLE_RATE
import matplotlib.pyplot as plt


STEP_FILE = os.path.join(BASE_DIR, "runs", "bp_eval", "last_step.txt")

def load_last_step(path: str) -> int:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return -1  # lần đầu -> sẽ thành 0 sau khi +1

def save_last_step(path: str, step: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(step))

def compute_metrics(predict_data, ground_truth):
    print(predict_data)
    print(ground_truth)
    predict_data = np.array(predict_data)
    ground_truth = np.array(ground_truth)
    
    max_error = max(abs(ground_truth - predict_data))
    min_error = min(abs(ground_truth - predict_data))
    mae = mean_absolute_error(ground_truth, predict_data)
    mse = mean_squared_error(ground_truth, predict_data)
    rmse = np.sqrt(mse)
    r2 = r2_score(ground_truth, predict_data)
    result = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "max_error": max_error, "min_error": min_error}
    return result


def evaluate_dataset(folder_path, csv_path, writer, dataset_name="v2", step=0):


    df = pd.read_csv(csv_path)
    hr_preds = []
    diastolic_preds = []
    systolic_preds = []

    for _, row in df.iterrows():
        video_path = os.path.join(folder_path, row["video_name"])
        print(video_path)
        predict_data = bp_inference_pipeline.predict_test_data(video_path)

        diastolic_preds.append(predict_data["diastolic"])
        systolic_preds.append(predict_data["systolic"])
        hr_preds.append(predict_data["hr_by_ppg"])

    # Tính metrics
    result_systolic = compute_metrics(df["sys"].tolist(), systolic_preds)
    result_diastolic = compute_metrics(df["dia"].tolist(), diastolic_preds)
    hr_instant_hr = compute_metrics(df["hr_instant_hr"].tolist(), hr_preds)
    hr_inpulse = compute_metrics(df["hr_inpulse"].tolist(), hr_preds)
    print(f"=== {dataset_name.upper()} hr_instant_hr ===")
    pprint(hr_instant_hr)
    print(f"=== {dataset_name.upper()} hr_inpulse ===")
    pprint(hr_inpulse)
    
    # print(f"=== {dataset_name.upper()} Systolic ===")
    # pprint(result_systolic)
    # print(f"=== {dataset_name.upper()} Diastolic ===")
    # pprint(result_diastolic)

    # Vẽ biểu đồ so sánh ground truth và predict cho từng loại
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(df["hr_instant_hr"].tolist(), label="hr_instant_hr Truth", marker='o')
    plt.plot(hr_preds, label="hr_instant_hr Predict", marker='x')
    plt.title(f"{dataset_name.upper()} - hr_instant_hr")
    plt.xlabel("Sample")
    plt.ylabel("HR (bpm)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(df["hr_inpulse"].tolist(), label="hr_inpulse Truth", marker='o')
    plt.plot(hr_preds, label="hr_inpulse Predict", marker='x')
    plt.title(f"{dataset_name.upper()} - hr_inpulse")
    plt.xlabel("Sample")
    plt.ylabel("HR (bpm)")
    plt.legend()
    plt.grid(True)
    # plt.plot(df["sys"].tolist(), label="Systolic Truth", marker='o')
    # plt.plot(systolic_preds, label="Systolic Predict", marker='x')
    # plt.title(f"{dataset_name.upper()} - Systolic")
    # plt.xlabel("Sample")
    # plt.ylabel("Systolic (mmHg)")
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(1, 3, 3)
    # plt.plot(df["dia"].tolist(), label="Diastolic Truth", marker='o')
    # plt.plot(diastolic_preds, label="Diastolic Predict", marker='x')
    # plt.title(f"{dataset_name.upper()} - Diastolic")
    # plt.xlabel("Sample")
    # plt.ylabel("Diastolic (mmHg)")
    # plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{writer}_evaluate.png")
    plt.show()

    # # Ghi log vào TensorBoard
    # writer.add_scalar(f"{dataset_name}/systolic_mae", result_systolic["mae"], step)
    # writer.add_scalar(f"{dataset_name}/diastolic_mae", result_diastolic["mae"], step)
    # writer.add_scalar(f"{dataset_name}/systolic_rmse", result_systolic["rmse"], step)
    # writer.add_scalar(f"{dataset_name}/diastolic_rmse", result_diastolic["rmse"], step)
    # writer.add_scalar(f"{dataset_name}/systolic_r2", result_systolic["r2"], step)
    # writer.add_scalar(f"{dataset_name}/diastolic_r2", result_diastolic["r2"], step)

    return result_systolic, result_diastolic

def evaluate_hr(csv_path):
    df = pd.read_csv(csv_path)
    hr_preds = []
    extractor = BPExtractor()  # Chỉ tạo một đối tượng extractor để tăng hiệu suất
    
    for _, row in df.iterrows():
        video_path = os.path.join(BASE_DIR, "data", "nguyen_video", row["video_name"])
        # Sử dụng hàm mới calculate_heart_rate_from_video
        hr = extractor.calculate_heart_rate_from_video(video_path)
        hr_preds.append(hr)
        # Tính metrics
    hr_instant_hr = compute_metrics(df["hr_instant_hr"].tolist(), hr_preds)
    hr_inpulse = compute_metrics(df["hr_inpulse"].tolist(), hr_preds)
    print(f"=== hr_instant_hr ===")
    pprint(hr_instant_hr)
    print(f"=== hr_inpulse ===")
    pprint(hr_inpulse)

    # Vẽ biểu đồ so sánh ground truth và predict cho từng loại
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(df["hr_instant_hr"].tolist(), label="hr_instant_hr Truth", marker='o')
    plt.plot(hr_preds, label="hr_instant_hr Predict", marker='x')
    plt.title(f"hr_instant_hr")
    plt.xlabel("Sample")
    plt.ylabel("HR (bpm)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(df["hr_inpulse"].tolist(), label="hr_inpulse Truth", marker='o')
    plt.plot(hr_preds, label="hr_inpulse Predict", marker='x')
    plt.title(f"hr_inpulse")
    plt.xlabel("Sample")
    plt.ylabel("HR (bpm)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"evaluate.png")
    plt.show()



if __name__ == "__main__":

    train_csv = os.path.join(BASE_DIR, "data", "nguyen_video", "truth.csv")
    video_folder = os.path.join(BASE_DIR, "data", "nguyen_video")
    evaluate_hr(train_csv)


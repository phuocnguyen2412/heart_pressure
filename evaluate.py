from main_pipeline import BloodPressureInferencePipeline
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pprint import pprint

from setting import BASE_DIR
from torch.utils.tensorboard import SummaryWriter

import mlflow
import tempfile

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
    mae = mean_absolute_error(ground_truth, predict_data)
    mse = mean_squared_error(ground_truth, predict_data)
    rmse = np.sqrt(mse)
    r2 = r2_score(ground_truth, predict_data)
    result = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
    return result


def evaluate_dataset(folder_path, csv_path, writer, dataset_name, step=0):
    df = pd.read_csv(csv_path)
    diastolic_preds = []
    systolic_preds = []

    for _, row in df.iterrows():
        video_path = os.path.join(folder_path, row["video"])
        print(video_path)
        predict_data = bp_inference_pipeline.predict_test_data(video_path)

        diastolic_preds.append(predict_data["diastolic"])
        systolic_preds.append(predict_data["systolic"])

    # Tính metrics
    result_systolic = compute_metrics(df["may_sys"].tolist(), systolic_preds)
    result_diastolic = compute_metrics(df["may_dia"].tolist(), diastolic_preds)

    print(f"=== {dataset_name.upper()} Systolic ===")
    pprint(result_systolic)
    print(f"=== {dataset_name.upper()} Diastolic ===")
    pprint(result_diastolic)

    # Ghi log vào TensorBoard
    writer.add_scalar(f"{dataset_name}/systolic_mae", result_systolic["mae"], step)
    writer.add_scalar(f"{dataset_name}/diastolic_mae", result_diastolic["mae"], step)
    writer.add_scalar(f"{dataset_name}/systolic_rmse", result_systolic["rmse"], step)
    writer.add_scalar(f"{dataset_name}/diastolic_rmse", result_diastolic["rmse"], step)
    writer.add_scalar(f"{dataset_name}/systolic_r2", result_systolic["r2"], step)
    writer.add_scalar(f"{dataset_name}/diastolic_r2", result_diastolic["r2"], step)

    return result_systolic, result_diastolic


if __name__ == "__main__":
    bp_inference_pipeline = BloodPressureInferencePipeline()
    writer = SummaryWriter(log_dir="runs/bp_eval")
    
    test_csv = os.path.join(BASE_DIR, "data", "test.csv")
    val_csv = os.path.join(BASE_DIR, "data", "val.csv")

    last_step = load_last_step(STEP_FILE)
    step = last_step + 1
    save_last_step(STEP_FILE, step)
    video_folder = os.path.join(BASE_DIR, "data", "video")
    # evaluate_dataset(video_folder, test_csv, writer, "test", step=step)
    evaluate_dataset(video_folder, val_csv, writer, "val", step=step)
    
    save_last_step(STEP_FILE, step)
    writer.close()

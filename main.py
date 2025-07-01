from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample

import numpy as np
import tempfile
import os

from matplotlib import pyplot as plt

from utils import extract_ppg_from_video

app = FastAPI()


@app.post("/upload_ppg")
async def upload_ppg(file: UploadFile = File(...)):
    # Lưu file tạm
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(await file.read())
        temp_path = temp_video.name

    # Xử lý video để trích xuất tín hiệu PPG

    ppg_signal = extract_ppg_from_video(temp_path)
    # cap = cv2.VideoCapture(temp_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # cap.release()
    # print("Extracted PPG Signal:", ppg_signal.shape)
    # ppg_signal_125hz = resample_ppg(ppg_signal, original_fps=fps, target_fps=125)
    # segments = split_segments(ppg_signal_125hz, segment_length=875)
    # print("Resampled PPG Signal:", segments.shape)

    # Tính toán huyết áp (giả lập)
    systolic, diastolic = estimate_blood_pressure(ppg_signal)

    # Xóa file tạm
    os.remove(temp_path)

    # Trả kết quả về FE
    return JSONResponse({"systolic": systolic, "diastolic": diastolic})


app.mount("/", StaticFiles(directory="static", html=True), name="static")

def split_segments(ppg_signal, segment_length=875):
    n_segments = len(ppg_signal) // segment_length
    segments = ppg_signal[:n_segments*segment_length].reshape((n_segments, segment_length))
    return segments

def resample_ppg(ppg_signal, original_fps, target_fps=125):
    n_points = int(len(ppg_signal) * target_fps / original_fps)
    return resample(ppg_signal, n_points)

def estimate_blood_pressure(ppg_signal):
    # Đây là bước áp dụng thuật toán ML/AI hoặc công thức tính toán
    # Ở đây placeholder: đưa ra giá trị giả lập
    systolic = 120 + np.random.randint(-10, 10)
    diastolic = 80 + np.random.randint(-5, 5)
    return systolic, diastolic
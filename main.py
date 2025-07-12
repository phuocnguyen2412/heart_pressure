from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import os

from pydantic import BaseModel

from predict_test import predict_test_data
from signal_extractor.pipeline import run_extract_signal

import subprocess
import os

def convert_to_mp4(input_path):
    base, _ = os.path.splitext(input_path)
    output_path = base + ".mp4"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode()}")
    return output_path
app = FastAPI()
class BloodPressureResult(BaseModel):
    systolic: float
    diastolic: float
    mean: float
    hr: float

@app.post("/upload_ppg", response_model=BloodPressureResult)
async def upload_ppg(file: UploadFile = File(...)):
    import time
    start_time = time.time()
    # Lưu file tạm
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(await file.read())
        temp_path = temp_video.name


    # # Đảm bảo convert sang mp4 nếu không phải mp4
    # if not temp_path.endswith(".mp4"):
    #     temp_path = convert_to_mp4(temp_path)

    ppg_signal = run_extract_signal(temp_path)
    output = predict_test_data(ppg_signal)

    # # Xóa file tạm
    # os.remove(temp_path)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    return BloodPressureResult(
        systolic=float(output['systolic']),
        diastolic=float(output['diastolic']),
        mean=float(output['mean']),
        hr= float(output['hr']),
    )

@app.get("/ping")
async def ping():
    return {"message": "pong"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")





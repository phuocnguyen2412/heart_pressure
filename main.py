from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample

import numpy as np
import tempfile
import os

from matplotlib import pyplot as plt

from utils import extract_ppg_from_video
from signal_extractor.pipeline import run_extract_signal
app = FastAPI()


@app.post("/upload_ppg")
async def upload_ppg(file: UploadFile = File(...)):
    # Lưu file tạm
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(await file.read())
        temp_path = temp_video.name

    ppg_signal = run_extract_signal(temp_path)
    output = predict_blood_pressure(ppg_signal)

    # Xóa file tạm
    os.remove(temp_path)

    # Trả kết quả về FE
    return JSONResponse({"systolic": output[0]})


app.mount("/", StaticFiles(directory="static", html=True), name="static")





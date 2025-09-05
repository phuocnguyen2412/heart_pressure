from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from pydantic import BaseModel

from main_pipeline import BloodPressureInferencePipeline
import os
import torch, torchvision
print(torch.__version__, torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
# def convert_to_mp4(input_path):
#     base, _ = os.path.splitext(input_path)
#     output_path = base + ".mp4"
#     cmd = [
#         "ffmpeg", "-y", "-i", input_path,
#         "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
#         output_path
#     ]
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if result.returncode != 0:
#         raise RuntimeError(f"FFmpeg failed: {result.stderr.decode()}")
#     return output_path

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class BloodPressureResult(BaseModel):
    systolic: float
    diastolic: float
    mean: float
    hr: float



@app.on_event("startup")
async def startup_event():  
    global bp_inference_pipeline
    bp_inference_pipeline = BloodPressureInferencePipeline()
    print("BP Inference Pipeline initialized")

@app.post("/upload_ppg", response_model=BloodPressureResult)
async def upload_ppg(file: UploadFile = File(...)):
    import time
    start_time = time.time()
    # Lưu file tạm
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(await file.read())
        temp_path = temp_video.name


    # Đảm bảo convert sang mp4 nếu không phải mp4
    if not temp_path.endswith(".mp4") or not temp_path.endswith(".MOV"):
        pass

    print(f"Processing video file: {temp_path}")
        
    output = bp_inference_pipeline.predict_test_data(temp_path)

    # Xóa file tạm
    os.remove(temp_path)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    return BloodPressureResult(
        systolic=float(output['systolic']),
        diastolic=float(output['diastolic']),
        mean=float(output['mean']),
        hr= float(output['hr_by_ppg']),
    )

@app.get("/ping")
async def ping():
    return {"message": "pong"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")





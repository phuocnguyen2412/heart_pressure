from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import uvicorn
from pydantic import BaseModel

from bp_extractor import BPExtractor
import os

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


@app.post("/upload_ppg")
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
        
    output = BPExtractor().calculate_heart_rate_from_video(temp_path)

    # Xóa file tạm
    os.remove(temp_path)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    return BloodPressureResult(hr=output, systolic=0, diastolic=0, mean=0)

@app.get("/ping")
async def ping():
    return {"message": "pong"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=True)
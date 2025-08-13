import pickle
import os
from setting import config, BASE_DIR

from models import MultiResUNet1D, UNetDS64
from processor import Processor
from bp_extractor import BPExtractor
import numpy as np
from scipy.signal import find_peaks

from utils import visualize_abp_waveform


class BloodPressureInferencePipeline:
    def __init__(self, extract_config=None):
        self.extract_config = extract_config or config
        # Load metadata
        self.meta = pickle.load(open(os.path.join(BASE_DIR, "data", "meta9.p"), "rb"))

        # Refine ABP waveform
        self.mdl2 = MultiResUNet1D(1024)
        self.mdl2.load_weights(os.path.join(BASE_DIR, "models", "RefinementNetwork.h5"))

        # Predict approximate ABP waveform
        self.mdl1 = UNetDS64(1024)
        self.mdl1.load_weights(
            os.path.join(BASE_DIR, "models", "ApproximateNetwork.h5")
        )

        self.processor = Processor()

        self.bp_extractor = BPExtractor(extract_config)

    def calculate_heart_rate(self, signal, fs=125):
        signal = np.array(signal).flatten()
        peaks, _ = find_peaks(signal, distance=0.4 * fs)
        return (len(peaks) / (len(signal) / fs)) * 60

    def extract_sbp_dbp(self, abp_signal, distance=30):
        abp_signal = np.array(abp_signal).flatten()
        sbp_idx, _ = find_peaks(abp_signal, distance=distance)
        dbp_idx, _ = find_peaks(-abp_signal, distance=distance)
        return abp_signal[sbp_idx], abp_signal[dbp_idx], sbp_idx, dbp_idx

    def predict_abp_from_ppg(self, ppg_signal):
        approx_abp = self.mdl1.predict(ppg_signal, verbose=1)
        refined_abp = self.mdl2.predict(approx_abp[0], verbose=1)
        return refined_abp

    def predict_test_data(self, video_path):

        ppg_signal = self.bp_extractor.extract_ppg_from_video(video_path)

        # Normalize and reshape PPG input
        min_ppg, max_ppg = self.meta["min_ppg"], self.meta["max_ppg"]
        min_abp, max_abp = self.meta["min_abp"], self.meta["max_abp"]
        ppg_signal = self.processor.clip_to_range(ppg_signal, min_ppg, max_ppg)
        ppg_norm = self.processor.min_max_scaler(ppg_signal, min_ppg, max_ppg)

        refined_abp = self.predict_abp_from_ppg(ppg_norm)

        # Denormalize ABP prediction
        abp_pred = self.processor.inverse_min_max_scaler(refined_abp, min_abp, max_abp)
        abp_pred = self.processor.butter_lowpass_filter(
            abp_pred.flatten(), fs=125, cutoff=5
        )

        # Save prediction
        visualize_abp_waveform(ppg_signal, abp_pred)

        # Estimate heart rate (weighted avg of PPG & ABP HR)
        hr = self.calculate_heart_rate(abp_pred)
        # # Tính distance tương ứng
        beat_interval_sec = 60.0 / hr
        distance = int(beat_interval_sec * 125 )  # Lấy 80% để tránh bỏ đỉnh gần
        # distance = 125
        print(f"Estimated heart rate: {hr:.2f} bpm, distance: {distance} samples")
        distance = max(20, distance)  # đảm bảo không quá nhỏ

        # Extract SBP, DBP
        sbp_vals, dbp_vals, sbp_idx, dbp_idx = self.extract_sbp_dbp(
            abp_pred, distance=distance
        )
        sbp, dbp = np.mean(sbp_vals), np.mean(dbp_vals)

        map_val = (2 * dbp + sbp) / 3

        # Plot ABP with annotations
        # self.plot_abp_with_sbp_dbp(abp_pred, sbp_idx, dbp_idx)
        true_hr = self.calculate_heart_rate(ppg_signal)
        result = {
            "hr": true_hr,
            "systolic": sbp, 
            "diastolic": dbp,
            "mean": map_val,
        }
        print(result)
        return result

if __name__ == "__main__":
    bp_inference_pipeline = BloodPressureInferencePipeline()
    data = bp_inference_pipeline.predict_test_data("data/video/Video_20250813_140108_336 - Nguyên Hà.mp4")
    print(data)
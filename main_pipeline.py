import pickle
import os
from setting import SAMPLE_RATE, TARGET_SAMPLES, config, BASE_DIR

from models import MultiResUNet1D, UNetDS64
from processor import Processor
from bp_extractor import BPExtractor
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

from utils import visualize_abp_waveform


class BloodPressureInferencePipeline:
    def __init__(self, extract_config=None):
        self.extract_config = extract_config or config
        # Load metadata
        self.meta = pickle.load(open(os.path.join(BASE_DIR, "data", "meta9.p"), "rb"))

        # Refine ABP waveform
        self.mdl2 = MultiResUNet1D(TARGET_SAMPLES)
        self.mdl2.load_weights(os.path.join(BASE_DIR, "models", "RefinementNetwork.h5"))

        # Predict approximate ABP waveform
        self.mdl1 = UNetDS64(TARGET_SAMPLES)
        self.mdl1.load_weights(
            os.path.join(BASE_DIR, "models", "ApproximateNetwork.h5")
        )

        self.processor = Processor()
        self.bp_extractor = BPExtractor(extract_config)

    def calculate_heart_rate(self, signal, fs=SAMPLE_RATE):
        signal = np.array(signal).flatten()

        def bandpass_filter(sig, fs, low=0.8, high=3.5, order=3):
            ny = 0.5 * fs
            b, a = butter(order, [low / ny, high / ny], btype="band")
            return filtfilt(b, a, sig)

        def preprocess_ppg(raw, fs):
            win = int(fs * 1.5)
            if win < 3:
                win = 3
            ma = np.convolve(raw, np.ones(win) / win, mode="same")
            detrended = raw - ma
            detrended = (detrended - np.mean(detrended)) / (np.std(detrended) + 1e-8)
            return detrended

        def compute_hr_time(sig_filt, fs, hr_min=40, hr_max=200, hr_hint=None):
            min_dist = int(fs * 60.0 / hr_max)
            # tăng tiêu chí để tránh nhiễu: prominence và width
            peaks, _ = find_peaks(sig_filt, distance=min_dist, prominence=0.2, width=4)
            if len(peaks) < 2:
                return np.nan, peaks
            ibi = np.diff(peaks) / fs
            med = np.median(ibi) if len(ibi) > 0 else np.nan
            good = (ibi > 0.3) & (ibi < 2.0)
            if np.isfinite(med):
                good = good & (np.abs(ibi - med) < 0.25 * med)
            if hr_hint is not None and not np.isnan(hr_hint) and hr_min <= hr_hint <= hr_max:
                expected_ibi = 60.0 / hr_hint
                band_ok = (ibi > 0.6 * expected_ibi) & (ibi < 1.6 * expected_ibi)
                good = good & band_ok
            ibi_f = ibi[good]
            if len(ibi_f) < 1:
                return np.nan, peaks
            hr = 60.0 / np.median(ibi_f)
            if hr < hr_min or hr > hr_max:
                return np.nan, peaks
            return hr, peaks

        def compute_hr_freq(sig_filt, fs, hr_min=40, hr_max=200):
            fmin, fmax = hr_min / 60.0, hr_max / 60.0
            f, pxx = welch(sig_filt, fs=fs, nperseg=min(len(sig_filt), 256))
            mask = (f >= fmin) & (f <= fmax)
            if not np.any(mask):
                return np.nan
            f_band = f[mask]
            p_band = pxx[mask]
            # tìm các đỉnh PSD và chọn đỉnh có tỉ lệ SNR tốt, tránh harmonic×2 và sub-harmonic
            prom = float(np.max(p_band)) * 0.10
            peak_idx, props = find_peaks(p_band, prominence=prom)
            if len(peak_idx) == 0:
                f_peak = f_band[np.argmax(p_band)]
            else:
                candidates = f_band[peak_idx]
                powers = p_band[peak_idx]
                order = np.argsort(-powers)
                # chọn đỉnh mạnh nhất trước
                f_peak = candidates[order[0]]
                p_main = powers[order[0]]
                # nếu có đỉnh cách ~2x (harmonic) thì giữ fundamental (đỉnh nhỏ hơn)
                for k in order[1:]:
                    f2 = candidates[k]
                    if 1.8*f_peak <= f2 <= 2.2*f_peak:
                        break
                # chống sub-harmonic: nếu HR từ đỉnh chính < 80 bpm và tồn tại đỉnh 84–114 bpm đủ mạnh
                hr_main = 60.0 * f_peak
                for k in order[1:]:
                    f_alt = candidates[k]
                    hr_alt = 60.0 * f_alt
                    # 84–114 bpm và đủ nổi bật so với đỉnh chính
                    if 84.0 <= hr_alt <= 114.0 and powers[k] >= 0.6 * p_main and hr_main < 80.0:
                        f_peak = f_alt
                        p_main = powers[k]
                        break
            hr = 60.0 * f_peak
            return hr

        def compute_hr_autocorr(sig_filt, fs, hr_min=40, hr_max=200):
            x = sig_filt - np.mean(sig_filt)
            ac = np.correlate(x, x, mode='full')[len(x)-1:]
            # bỏ lag=0, giới hạn cửa sổ lag tương ứng với HR range
            min_lag = int(fs * 60.0 / hr_max)
            max_lag = int(fs * 60.0 / hr_min)
            if max_lag <= min_lag + 1:
                return np.nan
            roi = ac[min_lag:max_lag]
            if len(roi) < 3:
                return np.nan
            pk, _ = find_peaks(roi)
            if len(pk) == 0:
                return np.nan
            best_lag = pk[np.argmax(roi[pk])]
            period = (best_lag + min_lag) / fs
            if period <= 0:
                return np.nan
            return 60.0 / period

        def estimate_hr(raw_ppg_window, fs):
            sig = preprocess_ppg(raw_ppg_window, fs)
            sig_filt = bandpass_filter(sig, fs)

            # Tính theo nhiều cửa sổ: 3.5s, 4.0s, 4.5s (overlap 50%) và chọn theo SNR
            window_secs_list = [3.5, 4.0, 4.5]
            hr_list, snr_list = [], []

            for wsec in window_secs_list:
                win = int(wsec * fs)
                step = max(1, win // 2)
                if len(sig_filt) < win:
                    windows = [(0, len(sig_filt))]
                else:
                    windows = [(i, min(i + win, len(sig_filt))) for i in range(0, len(sig_filt) - win + 1, step)]

                for s, e in windows:
                    seg = sig_filt[s:e]
                    hr_freq = compute_hr_freq(seg, fs)
                    hr_auto = compute_hr_autocorr(seg, fs)
                    hr_time, _ = compute_hr_time(seg, fs, hr_hint=hr_freq)
                    vals = [v for v in [hr_time, hr_freq, hr_auto] if not np.isnan(v)]
                    if len(vals) == 0:
                        continue
                    # đồng thuận và snap harmonic 0.5x/2x quanh median
                    med = float(np.median(vals))
                    candidates = [med]
                    for v in vals:
                        if 40 <= 0.5 * v <= 200:
                            candidates.append(0.5 * v)
                        if 40 <= 2.0 * v <= 200:
                            candidates.append(2.0 * v)
                    snap = min(candidates, key=lambda x: abs(x - med))
                    hr_list.append(snap)

                    # ước lượng SNR từ PSD tại tần số snap
                    f, pxx = welch(seg, fs=fs, nperseg=min(len(seg), 256))
                    f_snap = snap / 60.0
                    idx = int(np.argmin(np.abs(f - f_snap)))
                    signal_power = pxx[idx]
                    noise_power = np.median(pxx)
                    snr_list.append(float(signal_power / (noise_power + 1e-12)))

            if len(hr_list) == 0:
                return np.nan

            # Chọn top-k theo SNR rồi lấy median và làm mượt nhẹ
            k = max(3, int(0.3 * len(hr_list)))
            top_idx = np.argsort(snr_list)[-k:]
            top_vals = [hr_list[i] for i in top_idx]
            median_hr = float(np.median(top_vals))
            nearest3 = np.argsort([abs(h - median_hr) for h in top_vals])[:3]
            smooth = float(np.mean([top_vals[i] for i in nearest3]))
            smooth = max(40.0, min(200.0, smooth))
            return smooth

        return estimate_hr(signal, fs)

    def extract_sbp_dbp(self, abp_signal, distance=30):
        abp_signal = np.array(abp_signal).flatten()
        sbp_idx, _ = find_peaks(abp_signal, distance=distance)
         # Tìm DBP (diastolic valleys) - tìm điểm thấp nhất giữa các SBP
        dbp_values = []
        dbp_indices = []
        
        for i in range(len(sbp_idx) - 1):
            start_idx = sbp_idx[i]
            end_idx = sbp_idx[i + 1]
            segment = abp_signal[start_idx:end_idx]
            
            # Tìm điểm thấp nhất trong segment
            min_idx = np.argmin(segment) + start_idx
            dbp_values.append(abp_signal[min_idx])
            dbp_indices.append(min_idx)
        
        return abp_signal[sbp_idx], np.array(dbp_values), sbp_idx, np.array(dbp_indices)

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
        hr_by_abp = self.calculate_heart_rate(abp_pred, SAMPLE_RATE)
        # # Tính distance tương ứng
        beat_interval_sec = 60.0 / hr_by_abp
        distance = int(beat_interval_sec * 125 )  # Lấy 80% để tránh bỏ đỉnh gần
        # distance = 125
        print(f"Estimated heart rate: {hr_by_abp:.2f} bpm, distance: {distance} samples")
        distance = max(20, distance)  # đảm bảo không quá nhỏ

        # Extract SBP, DBP
        sbp_vals, dbp_vals, sbp_idx, dbp_idx = self.extract_sbp_dbp(
            abp_pred, distance=distance
        )
        sbp, dbp = np.median(sbp_vals), np.median(dbp_vals)

        map_val = (2 * dbp + sbp) / 3

        # Plot ABP with annotations
        # self.plot_abp_with_sbp_dbp(abp_pred, sbp_idx, dbp_idx)
        hr_by_ppg = self.calculate_heart_rate(ppg_signal, SAMPLE_RATE)
        result = {
            "hr_by_ppg": hr_by_ppg ,
            "hr_by_abp": hr_by_abp,
            "systolic": sbp, 
            "diastolic": dbp,
            "mean": map_val,
        }
        return result

if __name__ == "__main__":
    bp_inference_pipeline = BloodPressureInferencePipeline()
   
    video_path = os.path.join(BASE_DIR, "data", "Vid_Heart", "Video_20250904_211628_385.mp4")
    data = bp_inference_pipeline.predict_test_data(video_path)
    print(data)
    # ppg_signal = bp_inference_pipeline.bp_extractor.extract_ppg_from_video(video_path)
    # hr = bp_inference_pipeline.calculate_heart_rate(ppg_signal, SAMPLE_RATE, return_details=True)
    # print(hr)``
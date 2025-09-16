import traceback
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample, welch
import torch

from setting import BASE_DIR, SAMPLE_RATE, SEGMENT_SECONDS, TARGET_SAMPLES, config
from signal_extractor.signal_extract import SignalExtractor
from signal_extractor.signal_preprocessing import (
    SignalPreprocessor,
    visualize_signal,
)



class BPExtractor:
    def __init__(self, extract_config=None, load_optimal_params=True):
        """
        Initialize the BPExtractor class.
        
        Args:
            extract_config: Configuration for extraction
            load_optimal_params: Whether to load optimal HR parameters if available
        """
        self.extract_config = extract_config or config
        self.fps = SAMPLE_RATE
        self.hr_params = None
        
        # Tự động tải tham số tối ưu nếu có
        if load_optimal_params:
            self.load_optimal_hr_params()
            
    def load_optimal_hr_params(self, params_path=None):
        """Load optimal heart rate parameters from file."""
        import json
        
        if params_path is None:
            params_path = os.path.join(BASE_DIR, "hr_optimal_params.json")
            
        try:
            if os.path.exists(params_path):
                with open(params_path, "r") as f:
                    self.hr_params = json.load(f)
                print(f"Loaded optimal HR parameters from {params_path}")
                return True
        except Exception as e:
            print(f"Failed to load optimal HR parameters: {e}")
            
        return False

    def read_video(self, video_path):
        """
        Read the video and return the list of frames.
        """
        cap = cv2.VideoCapture(video_path)
        # self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video FPS: {self.fps}, Total frames: {total_frames}")
        duration_seconds = total_frames / self.fps
        print(f"Duration: {duration_seconds:.2f}s")

        center_frame = total_frames // 2
        half_seg = int((SEGMENT_SECONDS * self.fps) / 2)
        start_frame = max(center_frame - half_seg, 0)
        end_frame = min(center_frame + half_seg, total_frames)
        num_frames_to_read = end_frame - start_frame

        print(
            f"Cropping from frame {start_frame} to {end_frame} (~{num_frames_to_read} frames)"
        )

        # Đọc các frame cần thiết
        list_of_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(num_frames_to_read):
            ret, frame = cap.read()
            h, w, _ = frame.shape
            x1, x2 = int(w * 0.25), int(w * 0.75)
            y1, y2 = int(h * 0.25), int(h * 0.75)
            roi = frame[y1:y2, x1:x2]
            frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            if not ret:
                break
            list_of_frames.append(frame_rgb)
        cap.release()

        return list_of_frames
    
   
        
    def extract_signal(self, filename, video_path, output_folder):
        list_of_frames = self.read_video(video_path)

        se = SignalExtractor(int(self.fps))
        csv_fpath = os.path.join(output_folder, filename + ".csv")
        columns, extracted_s = [], []
        print("Extracting signal from", video_path, "with", len(list_of_frames))

        extractors = [
            {
                "functions": ["luma_component_mean"],
                "name": "luma_mean",
                "parameters": {"initial_skip_seconds": 0},
            },
            {
                "functions": ["red_channel_mean"],
                "name": "r_ch_mean",
                "parameters": {"initial_skip_seconds": 0},
            },
        ]

        for extractor in extractors:
            columns.append(extractor["name"])
            assert (
                len(extractor["functions"]) == 1
            ), "Only one extractor function is supported, check config.json"
            for fun_name in extractor["functions"]:
                fun = getattr(se, fun_name)
                f_output = fun(frames=list_of_frames, **extractor["parameters"])
                if len(f_output) != TARGET_SAMPLES:
                    f_output = resample(f_output, TARGET_SAMPLES)
                print(
                    "Extracted signal with function",
                    fun_name,
                    "with shape",
                    f_output.shape,
                )
            extracted_s.append(f_output.tolist())
        extracted_s = np.array(extracted_s) * -1
        assert (
            extracted_s.ndim == 2
        ), "Different functions resulted in different length of extracted signal"
        df = pd.DataFrame(extracted_s.T, columns=columns)
        df.to_csv(csv_fpath, sep=",", float_format="%.4f", index=False)

        # Visualize the signals
        n_extractors = len(extractors)
        csv_fname = filename + ".csv"
        pdf_fname = filename + ".pdf"
        df = pd.read_csv(
            os.path.join(config.output_folder, filename, csv_fname), index_col=False
        )
        df.iloc[30:].plot(
            kind="line",
            subplots=True,
            figsize=(16, 4 * n_extractors),
            layout=(n_extractors, 1),
            grid=True,
        )
        plt.savefig(
            os.path.join(config.output_folder, filename, pdf_fname), bbox_inches="tight"
        )
        plt.close()

    def process_single_signal_file(self, file_name, output_folder):
        """
        Xử lý 1 file tín hiệu (CSV), xuất CSV kết quả và PDF hình ảnh.
        """
        sp = SignalPreprocessor(SAMPLE_RATE)
        filepath = os.path.join(output_folder, file_name + ".csv")

        csv_fpath = os.path.join(output_folder, file_name + "_preprocessed.csv")
        img_fpath = os.path.join(output_folder, file_name + "_preprocessed.pdf")
        params = {
            "preprocessor": {
                "filter_chains": [
                    {
                        "flist": [
                            {
                                "name": "roll_avg",
                                "params": {"window_size_seconds": 1.01},
                            },
                            {"name": "sub", "params": {}},
                            {"name": "lpf", "params": {"filter_order": 2, "low": 4}},
                            {"name": "cut_start", "params": {"seconds": 0}},
                        ],
                        "name": "chain2",
                    },
                    {
                        "flist": [
                            {"name": "cut_start", "params": {"seconds": 0}},
                            {"name": "hpf", "params": {"cutoff": 0.5, "order": 1}},
                            {
                                "name": "bpf_bpm",
                                "params": {"mincut": 0.01, "multiplier": 3, "order": 1},
                            },
                        ],
                        "name": "dynamic_bpm",
                    },
                ],
                "sources": ["r_ch_mean"],
            }
        }

        try:

            to_plot, to_plot_names = [], []
            extracted_s = pd.read_csv(
                filepath, index_col=False, encoding_errors="ignore"
            )
            preprocessed, columns = [], []

            for source in params["preprocessor"]["sources"]:
                assert source in extracted_s.columns.values, "%s not in columns %s" % (
                    source,
                    filepath,
                )
                for filter_chain in params["preprocessor"]["filter_chains"]:
                    fun_list = filter_chain["flist"]
                    signal_at_step_j = [extracted_s[source].values]
                    name_at_step_j = [source]
                    for j, fun_dict in enumerate(fun_list):
                        # apply function
                        fun = getattr(sp, sp.shorter_names[fun_dict["name"]])
                        filtered_j = fun(
                            signal_at_step_j[-1],
                            prev_x=(
                                signal_at_step_j[-2]
                                if len(signal_at_step_j) > 1
                                else None
                            ),
                            **fun_dict["params"],
                        )
                        new_name = "%s>%s" % (name_at_step_j[-1], fun_dict["name"])

                        if len(filtered_j) == len(extracted_s[source].values):
                            signal_at_step_j.append(
                                np.real(filtered_j)
                            )  # discard imaginary part if any
                            name_at_step_j.append(new_name)
                        else:
                            to_plot.append(filtered_j)
                            to_plot_names.append(new_name)

                    preprocessed.extend(signal_at_step_j)
                    columns.extend(name_at_step_j)

            preprocessed = np.array(preprocessed)
            assert (
                preprocessed.ndim == 2
            ), "Different functions resulted in different length of preprocessed signal"
            df = pd.DataFrame(preprocessed.T, columns=columns)
            df.to_csv(csv_fpath, sep=",", float_format="%.8f", index=False)

            everything_to_plot = [a.tolist() for a in preprocessed] + to_plot
            everything_to_plot_labels = columns + to_plot_names

            visualize_signal(
                everything_to_plot,
                labels=everything_to_plot_labels,
                output_fname=img_fpath,
                source="r_ch_mean",
            )
            return df["r_ch_mean>roll_avg>sub>lpf>cut_start"].values
        except Exception as e:
            traceback.print_exc()
            print(e)

    def extract_ppg_from_video(self, video_path):
        video_name = os.path.basename(video_path)
        print("video_path:", video_path)
        output_folder = os.path.join(config.output_folder, video_name)
        os.makedirs(output_folder, exist_ok=True)

        self.extract_signal(video_name, video_path, output_folder)
        print("Signal extraction completed for", video_name)

        processed_signal = self.process_single_signal_file(video_name, output_folder)
        print("Signal preprocessing completed for", video_name)
        return processed_signal
        
    def calculate_heart_rate_from_video(self, video_path, params=None):
        """Extract PPG signal from video and calculate heart rate.
        
        Args:
            video_path: Path to video file
            params: Optional parameters to override defaults
        """
        signal = self.extract_ppg_from_video(video_path)
        return self.calculate_heart_rate(signal, params=params)
        
    def optimize_hr_parameters(self, video_paths, ground_truth_hrs):
        """Optimize parameters for heart rate calculation based on ground truth data.
        
        Args:
            video_paths: List of paths to video files
            ground_truth_hrs: List of ground truth heart rates
            
        Returns:
            Optimized parameters dictionary
        """
        from sklearn.model_selection import ParameterGrid
        from sklearn.metrics import mean_absolute_error
        import concurrent.futures
        
        # Định nghĩa không gian tìm kiếm tham số
        param_grid = {
            'filter': [
                {'low': 0.7, 'high': 3.0, 'order': 2},
                {'low': 0.8, 'high': 3.5, 'order': 3},
                {'low': 0.9, 'high': 4.0, 'order': 4}
            ],
            'peak_detection': [
                {'prominence': 0.2, 'width': 4, 'height': 0.05},
                {'prominence': 0.25, 'width': 5, 'height': 0.1},
                {'prominence': 0.3, 'width': 6, 'height': 0.15}
            ],
            'hr_range': [
                {'min': 40, 'max': 200},
                {'min': 50, 'max': 180},
                {'min': 60, 'max': 160}
            ]
        }
        
        # Tạo tất cả các tổ hợp tham số
        all_params = list(ParameterGrid(param_grid))
        print(f"Testing {len(all_params)} parameter combinations")
        
        # Các tín hiệu PPG được trích xuất trước để tránh xử lý lại
        ppg_signals = []
        for video_path in video_paths:
            signal = self.extract_ppg_from_video(video_path)
            ppg_signals.append(signal)
        
        best_mae = float('inf')
        best_params = None
        
        # Hàm đánh giá một bộ tham số
        def evaluate_params(params_idx):
            params = all_params[params_idx]
            predictions = []
            
            for signal in ppg_signals:
                hr = self.calculate_heart_rate(signal, params=params)
                predictions.append(hr)
                
            mae = mean_absolute_error(ground_truth_hrs, predictions)
            return params, mae
        
        # Xử lý song song để tăng tốc độ
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(evaluate_params, i) for i in range(len(all_params))]
            
            for future in concurrent.futures.as_completed(futures):
                params, mae = future.result()
                if mae < best_mae:
                    best_mae = mae
                    best_params = params
                    print(f"New best MAE: {best_mae:.2f} with params: {best_params}")
        
        print(f"Optimization complete. Best MAE: {best_mae:.2f}")
        print(f"Best parameters: {best_params}")
        
        return best_params
        
        
    def calculate_heart_rate(self, signal, fs=None, params=None):
        """Calculate heart rate from PPG signal.
        
        Args:
            signal: PPG signal array
            fs: Sampling frequency (defaults to SAMPLE_RATE if None)
            params: Dictionary of parameters to override defaults
        """
        if fs is None:
            fs = SAMPLE_RATE
            
        # Mặc định các tham số tối ưu
        default_params = {
            'filter': {
                'low': 0.8,    # Tần số cắt dưới (Hz)
                'high': 3.5,   # Tần số cắt trên (Hz)
                'order': 3     # Bậc của bộ lọc
            },
            'peak_detection': {
                'prominence': 0.25,  # Độ nổi bật tối thiểu của peak
                'width': 5,         # Độ rộng tối thiểu của peak
                'height': 0.1       # Chiều cao tối thiểu của peak
            },
            'hr_range': {
                'min': 40,     # Nhịp tim tối thiểu (bpm)
                'max': 200     # Nhịp tim tối đa (bpm)
            }
        }
        
        # Ghi đè các tham số mặc định nếu được cung cấp
        if params:
            # Cập nhật các tham số được cung cấp
            if 'filter' in params:
                default_params['filter'].update(params['filter'])
            if 'peak_detection' in params:
                default_params['peak_detection'].update(params['peak_detection'])
            if 'hr_range' in params:
                default_params['hr_range'].update(params['hr_range'])
        
        # Lưu trữ tham số để sử dụng trong các hàm con
        self.hr_params = default_params
            
        signal = np.array(signal).flatten()

        def bandpass_filter(sig, fs, low=0.8, high=3.5, order=3):
            # Cải thiện bộ lọc với tham số tốt hơn cho tín hiệu PPG
            ny = 0.5 * fs
            # Sử dụng bộ lọc Butterworth với dải tần số tối ưu cho PPG
            # Dải tần số 0.8-3.5Hz tương ứng với 48-210 bpm
            b, a = butter(order, [low / ny, high / ny], btype="band")
            # Áp dụng zero-phase filtering để tránh méo pha
            filtered = filtfilt(b, a, sig)
            return filtered

        def preprocess_ppg(raw, fs):
            # Tăng kích thước cửa sổ cho moving average để loại bỏ trend tốt hơn
            win = int(fs * 2.0)  # Tăng từ 1.5s lên 2.0s
            if win < 5:  # Tăng kích thước cửa sổ tối thiểu
                win = 5
                
            # Áp dụng moving average để loại bỏ trend chậm
            ma = np.convolve(raw, np.ones(win) / win, mode="same")
            
            # Loại bỏ trend
            detrended = raw - ma
            
            # Chuẩn hóa tín hiệu
            detrended = (detrended - np.mean(detrended)) / (np.std(detrended) + 1e-8)
            
            # Loại bỏ outliers (tín hiệu vượt quá 3 std)
            threshold = 3.0
            detrended[np.abs(detrended) > threshold] = np.sign(detrended[np.abs(detrended) > threshold]) * threshold
            
            return detrended

        def compute_hr_time(sig_filt, fs, hr_min=40, hr_max=200, hr_hint=None):
            # Tính khoảng cách tối thiểu giữa các peak dựa trên HR tối đa
            min_dist = int(fs * 60.0 / hr_max)
            
            # Cải thiện các tham số phát hiện peak
            # Tăng prominence để chỉ phát hiện các peak rõ ràng
            # Tăng width để loại bỏ các peak quá hẹp (thường là nhiễu)
            peaks, peak_props = find_peaks(sig_filt, distance=min_dist, prominence=0.25, width=5, height=0.1)
            
            if len(peaks) < 3:  # Yêu cầu ít nhất 3 peak để tính toán chính xác hơn
                return np.nan, peaks
                
            # Tính khoảng cách giữa các nhịp (inter-beat interval)
            ibi = np.diff(peaks) / fs
            med = np.median(ibi) if len(ibi) > 0 else np.nan
            
            # Lọc các khoảng cách không hợp lý (quá ngắn hoặc quá dài)
            # 0.3s ~ 200bpm, 2.0s ~ 30bpm
            good = (ibi > 0.3) & (ibi < 2.0)
            
            # Loại bỏ các outlier so với giá trị trung vị
            if np.isfinite(med):
                # Thu hẹp dải chấp nhận từ 25% xuống 20% để loại bỏ outlier tốt hơn
                good = good & (np.abs(ibi - med) < 0.2 * med)
                
            # Nếu có gợi ý HR, sử dụng nó để lọc thêm
            if hr_hint is not None and not np.isnan(hr_hint) and hr_min <= hr_hint <= hr_max:
                expected_ibi = 60.0 / hr_hint
                # Thu hẹp dải chấp nhận từ 60-160% xuống 70-140% để tăng độ chính xác
                band_ok = (ibi > 0.7 * expected_ibi) & (ibi < 1.4 * expected_ibi)
                good = good & band_ok
                
            # Chỉ giữ lại các khoảng cách hợp lý
            ibi_f = ibi[good]
            
            # Kiểm tra xem có đủ dữ liệu để tính HR không
            if len(ibi_f) < 2:  # Yêu cầu ít nhất 2 khoảng cách hợp lệ
                return np.nan, peaks
                
            # Tính HR từ khoảng cách trung vị
            hr = 60.0 / np.median(ibi_f)
            
            # Kiểm tra HR có nằm trong giới hạn hợp lý không
            if hr < hr_min or hr > hr_max:
                return np.nan, peaks
                
            return hr, peaks

        def compute_hr_freq(sig_filt, fs, hr_min=40, hr_max=200):
            # Chuyển giới hạn HR sang tần số (Hz)
            fmin, fmax = hr_min / 60.0, hr_max / 60.0
            
            # Cải thiện phân tích phổ Welch với cửa sổ Hanning và overlap 50%
            # Tăng nperseg để cải thiện độ phân giải tần số
            nperseg = min(len(sig_filt), 512)  # Tăng từ 256 lên 512
            noverlap = nperseg // 2  # 50% overlap
            f, pxx = welch(sig_filt, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
            
            # Lọc dải tần số liên quan đến HR
            mask = (f >= fmin) & (f <= fmax)
            if not np.any(mask):
                return np.nan
                
            f_band = f[mask]
            p_band = pxx[mask]
            
            # Tính SNR của phổ để đánh giá chất lượng tín hiệu
            noise_floor = np.median(p_band)
            snr = np.max(p_band) / (noise_floor + 1e-10)
            
            # Nếu SNR quá thấp, có thể tín hiệu không đáng tin cậy
            if snr < 3.0:  # Ngưỡng SNR tối thiểu
                # Thử tìm đỉnh rõ ràng hơn với prominence cao hơn
                prom = float(np.max(p_band)) * 0.15  # Tăng từ 0.10 lên 0.15
            else:
                # Nếu SNR tốt, có thể dùng ngưỡng thấp hơn để phát hiện đỉnh
                prom = float(np.max(p_band)) * 0.10
            
            # Tìm các đỉnh trong phổ công suất
            peak_idx, props = find_peaks(p_band, prominence=prom, width=1)
            
            if len(peak_idx) == 0:
                # Nếu không tìm thấy đỉnh nào, lấy giá trị lớn nhất
                f_peak = f_band[np.argmax(p_band)]
            else:
                # Phân tích các đỉnh tìm được
                candidates = f_band[peak_idx]
                powers = p_band[peak_idx]
                order = np.argsort(-powers)  # Sắp xếp theo công suất giảm dần
                
                # Chọn đỉnh mạnh nhất trước
                f_peak = candidates[order[0]]
                p_main = powers[order[0]]
                
                # Xử lý harmonic: nếu có đỉnh cách ~2x (harmonic) thì giữ fundamental (đỉnh nhỏ hơn)
                for k in order[1:]:
                    f2 = candidates[k]
                    if 1.8*f_peak <= f2 <= 2.2*f_peak:
                        # Nếu harmonic mạnh hơn fundamental gấp 1.5 lần, có thể fundamental là đúng
                        if powers[k] > 1.5 * p_main:
                            f_peak = f_peak / 2  # Giảm tần số xuống nửa
                            p_main = powers[k]
                        break
                        
                # Xử lý sub-harmonic: nếu HR từ đỉnh chính < 80 bpm và tồn tại đỉnh 84–114 bpm đủ mạnh
                hr_main = 60.0 * f_peak
                for k in order[1:]:
                    f_alt = candidates[k]
                    hr_alt = 60.0 * f_alt
                    
                    # Mở rộng dải HR được ưu tiên (80-120 bpm) và tăng ngưỡng power
                    if 80.0 <= hr_alt <= 120.0 and powers[k] >= 0.7 * p_main and hr_main < 80.0:
                        f_peak = f_alt
                        p_main = powers[k]
                        break
                        
            # Chuyển từ tần số (Hz) sang nhịp tim (bpm)
            hr = 60.0 * f_peak
            
            return hr

        def compute_hr_autocorr(sig_filt, fs, hr_min=40, hr_max=200):
            # Chuẩn hóa tín hiệu trước khi tính tự tương quan
            x = sig_filt - np.mean(sig_filt)
            x = x / (np.std(x) + 1e-8)  # Chuẩn hóa biên độ
            
            # Tính tự tương quan (chỉ lấy nửa sau của kết quả)
            ac = np.correlate(x, x, mode='full')[len(x)-1:]
            
            # Chuẩn hóa tự tương quan để giá trị tại lag=0 là 1.0
            ac = ac / ac[0] if ac[0] != 0 else ac
            
            # Bỏ qua lag=0 (tự tương quan với chính nó)
            ac = ac[1:]
            
            # Giới hạn cửa sổ lag tương ứng với HR range
            min_lag = max(1, int(fs * 60.0 / hr_max))
            max_lag = min(len(ac) - 1, int(fs * 60.0 / hr_min))
            
            if max_lag <= min_lag + 1:
                return np.nan
                
            # Lấy vùng quan tâm (region of interest)
            roi = ac[min_lag:max_lag]
            
            if len(roi) < 3:
                return np.nan
                
            # Áp dụng lọc trướt để làm mượt đường cong tự tương quan
            # Điều này giúp loại bỏ các đỉnh nhiễu nhỏ
            smooth_win = max(3, int(fs * 0.05))  # Cửa sổ lọc 50ms
            if smooth_win % 2 == 0:
                smooth_win += 1  # Đảm bảo cửa sổ là số lẻ
                
            if len(roi) > smooth_win:
                kernel = np.ones(smooth_win) / smooth_win
                roi_smooth = np.convolve(roi, kernel, mode='same')
            else:
                roi_smooth = roi
            
            # Tìm các đỉnh trong đường cong tự tương quan
            # Tăng prominence để chỉ phát hiện các đỉnh rõ ràng
            pk, pk_props = find_peaks(roi_smooth, prominence=0.1, width=2)
            
            if len(pk) == 0:
                return np.nan
                
            # Chọn đỉnh có biên độ lớn nhất
            best_lag = pk[np.argmax(roi_smooth[pk])]
            
            # Tính chu kỳ (giây)
            period = (best_lag + min_lag) / fs
            
            if period <= 0:
                return np.nan
                
            # Chuyển từ chu kỳ (giây) sang tần số nhịp tim (bpm)
            return 60.0 / period

        def estimate_hr(raw_ppg_window, fs):
            # Tiền xử lý tín hiệu PPG
            sig = preprocess_ppg(raw_ppg_window, fs)
            
            # Áp dụng bộ lọc dải thông để loại bỏ nhiễu tần số cao và thấp
            sig_filt = bandpass_filter(sig, fs)
            
            # Tăng số lượng cửa sổ và kích thước cửa sổ để cải thiện độ chính xác
            # Sử dụng cửa sổ từ 3s đến 5s để bắt được nhiều chu kỳ nhịp tim hơn
            window_secs_list = [3.0, 3.5, 4.0, 4.5, 5.0]
            hr_list, snr_list, method_list = [], [], []
            
            for wsec in window_secs_list:
                win = int(wsec * fs)
                step = max(1, win // 2)  # 50% overlap giữa các cửa sổ
                
                # Tạo các cửa sổ con từ tín hiệu
                if len(sig_filt) < win:
                    windows = [(0, len(sig_filt))]
                else:
                    windows = [(i, min(i + win, len(sig_filt))) for i in range(0, len(sig_filt) - win + 1, step)]
                
                for s, e in windows:
                    seg = sig_filt[s:e]
                    
                    # Áp dụng nhiều phương pháp để ước tính nhịp tim
                    hr_freq = compute_hr_freq(seg, fs)
                    hr_auto = compute_hr_autocorr(seg, fs)
                    hr_time, peaks = compute_hr_time(seg, fs, hr_hint=hr_freq)
                    
                    # Đánh giá chất lượng của tín hiệu dựa trên số lượng peak phát hiện được
                    peak_quality = 0
                    if peaks is not None and len(peaks) >= 3:
                        # Tính khoảng cách trung bình giữa các peak
                        ibi = np.diff(peaks)
                        # Độ ổn định của khoảng cách giữa các peak
                        cv = np.std(ibi) / np.mean(ibi) if np.mean(ibi) > 0 else float('inf')
                        peak_quality = 1.0 / (cv + 0.1)  # CV thấp = chất lượng cao
                    
                    # Chỉ sử dụng các giá trị hợp lệ
                    vals = []
                    methods = []
                    
                    if not np.isnan(hr_time):
                        vals.append(hr_time)
                        methods.append("time")
                        
                    if not np.isnan(hr_freq):
                        vals.append(hr_freq)
                        methods.append("freq")
                        
                    if not np.isnan(hr_auto):
                        vals.append(hr_auto)
                        methods.append("auto")
                    
                    if len(vals) == 0:
                        continue
                        
                    # Đồng thuận giữa các phương pháp và xử lý harmonic/subharmonic
                    med = float(np.median(vals))
                    candidates = [med]
                    
                    # Xem xét các khả năng harmonic và subharmonic
                    for v in vals:
                        # Subharmonic (nếu nhịp tim đo được gấp đôi thực tế)
                        if 40 <= 0.5 * v <= 200:
                            candidates.append(0.5 * v)
                        # Harmonic (nếu nhịp tim đo được bằng nửa thực tế)
                        if 40 <= 2.0 * v <= 200:
                            candidates.append(2.0 * v)
                    
                    # Chọn giá trị gần với trung vị nhất
                    snap = min(candidates, key=lambda x: abs(x - med))
                    hr_list.append(snap)
                    method_list.append(methods)
                    
                    # Ước tính SNR từ phổ công suất tại tần số tương ứng với nhịp tim
                    nperseg = min(len(seg), 512)  # Tăng kích thước cửa sổ phổ
                    f, pxx = welch(seg, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann')
                    f_snap = snap / 60.0  # Chuyển từ bpm sang Hz
                    
                    # Tìm chỉ số gần nhất trong mảng tần số
                    idx = int(np.argmin(np.abs(f - f_snap)))
                    
                    # Tính SNR: tỷ lệ giữa công suất tại tần số nhịp tim và mức nhiễu nền
                    signal_power = pxx[idx]
                    noise_floor = np.median(pxx)  # Ước tính mức nhiễu nền bằng trung vị
                    snr = float(signal_power / (noise_floor + 1e-12))
                    
                    # Kết hợp SNR với đánh giá chất lượng peak để có điểm chất lượng tổng hợp
                    quality_score = snr * (1 + peak_quality)
                    snr_list.append(quality_score)
            
            if len(hr_list) == 0:
                return np.nan
            
            # Phương pháp cải tiến để chọn kết quả cuối cùng:
            # 1. Chọn top-k kết quả có chất lượng cao nhất
            k = max(5, int(0.4 * len(hr_list)))  # Tăng số lượng mẫu để đảm bảo độ tin cậy
            top_idx = np.argsort(snr_list)[-k:]
            top_vals = [hr_list[i] for i in top_idx]
            top_methods = [method_list[i] for i in top_idx]
            
            # 2. Phân tích phân bố của các giá trị
            median_hr = float(np.median(top_vals))
            
            # 3. Tìm các giá trị gần với trung vị để loại bỏ outlier
            # Chỉ lấy các giá trị trong khoảng ±10% của trung vị
            close_vals = [v for v in top_vals if abs(v - median_hr) <= 0.1 * median_hr]
            
            if len(close_vals) >= 3:
                # Nếu có đủ giá trị gần trung vị, sử dụng chúng để tính trung bình
                smooth = float(np.mean(close_vals))
            else:
                # Nếu không đủ, lấy 3 giá trị gần trung vị nhất
                nearest3 = np.argsort([abs(h - median_hr) for h in top_vals])[:3]
                smooth = float(np.mean([top_vals[i] for i in nearest3]))
            
            # Đảm bảo kết quả nằm trong giới hạn hợp lý
            smooth = max(40.0, min(200.0, smooth))
            
            return smooth

        return estimate_hr(signal, SAMPLE_RATE) 
"""
Test Calibration Script
Script để test model hiện tại trên các video calibration và so sánh với ground truth
"""
import os
import pandas as pd
import numpy as np
from setting import BASE_DIR
from calibration_data_loader import CalibrationDataLoader
from signal_extractor.pipeline import run_extract_signal
from predict_test import predict_test_data

class CalibrationTester:
    def __init__(self):
        self.loader = CalibrationDataLoader()
        self.video_folder = os.path.join(BASE_DIR, 'data', 'calibration_videos')
        
    def test_single_video(self, video_name):
        """Test một video duy nhất"""
        video_path = os.path.join(self.video_folder, video_name)
        
        if not os.path.exists(video_path):
            print(f"Video không tồn tại: {video_path}")
            return None
            
        print(f"Đang xử lý video: {video_name}")
        
        try:
            # Extract PPG signal từ video
            ppg_signal = run_extract_signal(video_path)
            
            # Predict blood pressure
            result = predict_test_data(ppg_signal)
            
            return {
                'video': video_name,
                'predicted_sys': result['systolic'],
                'predicted_dia': result['diastolic'],
                'predicted_hr': result['hr'],
                'predicted_mean': result['mean']
            }
            
        except Exception as e:
            print(f"Lỗi khi xử lý {video_name}: {str(e)}")
            return None
    
    def test_all_videos(self, save_results=True):
        """Test tất cả video trong calibration set"""
        results = []
        
        # Lấy danh sách video cần test
        df = self.loader.get_all_data()
        
        for _, row in df.iterrows():
            video_name = row['video']
            video_extensions = ['.mp4', '.MOV', '.mov', '.avi']
            
            # Tìm video với các extension khác nhau
            video_found = False
            for ext in video_extensions:
                video_file = video_name + ext
                video_path = os.path.join(self.video_folder, video_file)
                
                if os.path.exists(video_path):
                    print(f"\nTìm thấy video: {video_file}")
                    result = self.test_single_video(video_file)
                    
                    if result:
                        # Thêm ground truth data
                        result.update({
                            'id': row['ID'],
                            'ground_truth_sys': row['may_sys'],
                            'ground_truth_dia': row['may_dia'],
                            'app_sys': row['app_sys'],
                            'app_dia': row['app_dia']
                        })
                        results.append(result)
                        video_found = True
                        break
            
            if not video_found:
                print(f"Không tìm thấy video cho: {video_name}")
        
        if save_results and results:
            # Lưu kết quả ra file CSV
            results_df = pd.DataFrame(results)
            output_path = os.path.join(BASE_DIR, 'data', 'calibration_test_results.csv')
            results_df.to_csv(output_path, index=False)
            print(f"\nĐã lưu kết quả ra: {output_path}")
            
            # Tính toán và hiển thị metrics
            self.analyze_results(results_df)
        
        return results
    
    def analyze_results(self, results_df):
        """Phân tích kết quả test"""
        print("\n=== PHÂN TÍCH KẾT QUẢ TEST ===")
        
        # So sánh app cũ vs ground truth
        sys_error_old = results_df['app_sys'] - results_df['ground_truth_sys']
        dia_error_old = results_df['app_dia'] - results_df['ground_truth_dia']
        
        # So sánh prediction mới vs ground truth
        sys_error_new = results_df['predicted_sys'] - results_df['ground_truth_sys']
        dia_error_new = results_df['predicted_dia'] - results_df['ground_truth_dia']
        
        print("\nApp cũ (từ CSV):")
        print(f"  Systolic - MAE: {np.mean(np.abs(sys_error_old)):.2f}, RMSE: {np.sqrt(np.mean(sys_error_old**2)):.2f}")
        print(f"  Diastolic - MAE: {np.mean(np.abs(dia_error_old)):.2f}, RMSE: {np.sqrt(np.mean(dia_error_old**2)):.2f}")
        
        print("\nPrediction mới (từ model hiện tại):")
        print(f"  Systolic - MAE: {np.mean(np.abs(sys_error_new)):.2f}, RMSE: {np.sqrt(np.mean(sys_error_new**2)):.2f}")
        print(f"  Diastolic - MAE: {np.mean(np.abs(dia_error_new)):.2f}, RMSE: {np.sqrt(np.mean(dia_error_new**2)):.2f}")
        
        # Chi tiết từng video
        print("\n=== CHI TIẾT TỪNG VIDEO ===")
        for _, row in results_df.iterrows():
            print(f"\n{row['video']} (ID: {row['id']}):")
            print(f"  Ground Truth: {row['ground_truth_sys']:.1f}/{row['ground_truth_dia']:.1f}")
            print(f"  App cũ: {row['app_sys']:.1f}/{row['app_dia']:.1f}")
            print(f"  Prediction mới: {row['predicted_sys']:.1f}/{row['predicted_dia']:.1f}")
            print(f"  Error app cũ: {row['app_sys']-row['ground_truth_sys']:.1f}/{row['app_dia']-row['ground_truth_dia']:.1f}")
            print(f"  Error mới: {row['predicted_sys']-row['ground_truth_sys']:.1f}/{row['predicted_dia']-row['ground_truth_dia']:.1f}")

def main():
    """Hàm chính để chạy test"""
    print("=== CALIBRATION TESTING TOOL ===")
    print("Chạy model hiện tại trên các video calibration")
    
    tester = CalibrationTester()
    
    # Kiểm tra folder video
    if not os.path.exists(tester.video_folder):
        print(f"Folder video không tồn tại: {tester.video_folder}")
        print("Vui lòng tạo folder và copy video vào trước khi chạy test!")
        return
    
    # Liệt kê video có sẵn
    videos = [f for f in os.listdir(tester.video_folder) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    print(f"\nTìm thấy {len(videos)} video trong folder:")
    for video in videos:
        print(f"  - {video}")
    
    if len(videos) == 0:
        print("\nKhông có video nào! Vui lòng copy video vào folder calibration_videos")
        print(f"Folder path: {tester.video_folder}")
        return
    
    # Chạy test
    print("\nBắt đầu test...")
    results = tester.test_all_videos()
    
    if results:
        print(f"\nHoàn thành test {len(results)} video!")
    else:
        print("\nKhông có kết quả nào được tạo ra!")

if __name__ == "__main__":
    main()

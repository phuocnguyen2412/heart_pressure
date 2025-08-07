"""
Test Model on Calibration Videos
Chạy toàn bộ video calibration qua model hiện tại và so sánh với ground truth
"""
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

from calibration_data_loader import CalibrationDataLoader
from predict_test import predict_test_data
from signal_extractor.pipeline import run_extract_signal
from setting import BASE_DIR

class ModelCalibrationTester:
    def __init__(self):
        self.data_loader = CalibrationDataLoader()
        self.ground_truth = self.data_loader.get_all_data()
        self.videos_folder = os.path.join(BASE_DIR, 'data', 'calibration_videos')
        self.results = []
        
    def find_video_file(self, video_name):
        """Tìm file video với các extension khác nhau"""
        possible_extensions = ['.mov', '.mp4', '.MOV', '.MP4']
        for ext in possible_extensions:
            video_path = os.path.join(self.videos_folder, video_name + ext)
            if os.path.exists(video_path):
                return video_path
        return None
    
    def process_single_video(self, video_id):
        """Xử lý một video và trả về kết quả"""
        video_name = f"video_{video_id}"
        video_path = self.find_video_file(video_name)
        
        if not video_path:
            print(f"❌ Không tìm thấy video: {video_name}")
            return None
            
        print(f"🔄 Đang xử lý {video_name}...")
        start_time = time.time()
        
        try:
            # Chạy pipeline giống như trong main.py
            ppg_signal = run_extract_signal(video_path)
            output = predict_test_data(ppg_signal)
            
            processing_time = time.time() - start_time
            
            result = {
                'video_id': video_id,
                'video_name': video_name,
                'video_path': video_path,
                'app_sys': output['systolic'],
                'app_dia': output['diastolic'],
                'app_mean': output['mean'],
                'app_hr': output['hr'],
                'processing_time': processing_time,
                'status': 'success'
            }
            
            print(f"✅ {video_name} - Systolic: {output['systolic']:.1f}, Diastolic: {output['diastolic']:.1f}, Time: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {video_name}: {str(e)}")
            return {
                'video_id': video_id,
                'video_name': video_name,
                'video_path': video_path,
                'app_sys': None,
                'app_dia': None,
                'app_mean': None,
                'app_hr': None,
                'processing_time': None,
                'status': f'error: {str(e)}',
                'error': str(e)
            }
    
    def run_all_videos(self):
        """Chạy tất cả videos và thu thập kết quả"""
        print("=== TESTING MODEL ON CALIBRATION VIDEOS ===")
        print(f"Videos folder: {self.videos_folder}")
        
        # Kiểm tra folder tồn tại
        if not os.path.exists(self.videos_folder):
            print(f"❌ Folder không tồn tại: {self.videos_folder}")
            return
        
        # List files trong folder
        files = os.listdir(self.videos_folder)
        print(f"Files in folder: {files}")
        
        total_start_time = time.time()
        self.results = []
        
        # Xử lý từng video theo ID
        for video_id in range(1, 10):  # Video 1-9
            result = self.process_single_video(video_id)
            if result:
                self.results.append(result)
        
        total_time = time.time() - total_start_time
        print(f"\n⏱️  Tổng thời gian xử lý: {total_time:.2f} seconds")
        print(f"📊 Đã xử lý: {len(self.results)} videos")
        
    def merge_with_ground_truth(self):
        """Merge kết quả với ground truth"""
        if not self.results:
            print("❌ Không có kết quả để merge")
            return None
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Merge với ground truth
        ground_truth_renamed = self.ground_truth.rename(columns={
            'video': 'video_name',
            'may_sys': 'ground_truth_sys',
            'may_dia': 'ground_truth_dia'
        })
        
        merged = pd.merge(
            results_df, 
            ground_truth_renamed[['video_name', 'ground_truth_sys', 'ground_truth_dia']], 
            on='video_name', 
            how='left'
        )
        
        return merged
    
    def calculate_detailed_errors(self, merged_df):
        """Tính toán lỗi chi tiết"""
        # Lọc những video xử lý thành công
        successful = merged_df[merged_df['status'] == 'success'].copy()
        
        if len(successful) == 0:
            print("❌ Không có video nào xử lý thành công")
            return None
        
        # Tính errors
        successful['sys_error'] = successful['app_sys'] - successful['ground_truth_sys']
        successful['dia_error'] = successful['app_dia'] - successful['ground_truth_dia']
        successful['sys_abs_error'] = np.abs(successful['sys_error'])
        successful['dia_abs_error'] = np.abs(successful['dia_error'])
        
        # Tính metrics
        metrics = {
            'total_videos': len(merged_df),
            'successful_videos': len(successful),
            'failed_videos': len(merged_df) - len(successful),
            'sys_mae': successful['sys_abs_error'].mean(),
            'sys_rmse': np.sqrt((successful['sys_error']**2).mean()),
            'sys_bias': successful['sys_error'].mean(),
            'dia_mae': successful['dia_abs_error'].mean(),
            'dia_rmse': np.sqrt((successful['dia_error']**2).mean()),
            'dia_bias': successful['dia_error'].mean(),
            'avg_processing_time': successful['processing_time'].mean()
        }
        
        return successful, metrics
    
    def print_detailed_report(self):
        """In báo cáo chi tiết"""
        merged_df = self.merge_with_ground_truth()
        if merged_df is None:
            return
        
        successful_df, metrics = self.calculate_detailed_errors(merged_df)
        if metrics is None:
            return
        
        print("\n" + "="*60)
        print("DETAILED COMPARISON REPORT")
        print("="*60)
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"  Total videos: {metrics['total_videos']}")
        print(f"  Successful: {metrics['successful_videos']}")
        print(f"  Failed: {metrics['failed_videos']}")
        print(f"  Avg processing time: {metrics['avg_processing_time']:.2f}s")
        
        # Per-video results
        print(f"\n📋 PER-VIDEO RESULTS:")
        print(f"{'ID':<3} {'Video':<10} {'Status':<10} {'GT_Sys':<7} {'App_Sys':<8} {'Sys_Err':<8} {'GT_Dia':<7} {'App_Dia':<8} {'Dia_Err':<8}")
        print("-" * 80)
        
        for _, row in merged_df.iterrows():
            if row['status'] == 'success':
                sys_err = row['app_sys'] - row['ground_truth_sys']
                dia_err = row['app_dia'] - row['ground_truth_dia']
                print(f"{row['video_id']:<3} {row['video_name']:<10} {'SUCCESS':<10} "
                      f"{row['ground_truth_sys']:<7.1f} {row['app_sys']:<8.1f} {sys_err:<8.1f} "
                      f"{row['ground_truth_dia']:<7.1f} {row['app_dia']:<8.1f} {dia_err:<8.1f}")
            else:
                print(f"{row['video_id']:<3} {row['video_name']:<10} {'FAILED':<10} {'N/A':<7} {'N/A':<8} {'N/A':<8} {'N/A':<7} {'N/A':<8} {'N/A':<8}")
        
        # Error metrics
        if metrics['successful_videos'] > 0:
            print(f"\n📈 ERROR METRICS (from {metrics['successful_videos']} successful videos):")
            print(f"  Systolic:")
            print(f"    MAE:  {metrics['sys_mae']:.2f} mmHg")
            print(f"    RMSE: {metrics['sys_rmse']:.2f} mmHg") 
            print(f"    Bias: {metrics['sys_bias']:.2f} mmHg")
            print(f"  Diastolic:")
            print(f"    MAE:  {metrics['dia_mae']:.2f} mmHg")
            print(f"    RMSE: {metrics['dia_rmse']:.2f} mmHg")
            print(f"    Bias: {metrics['dia_bias']:.2f} mmHg")
        
        # Failed videos
        failed_videos = merged_df[merged_df['status'] != 'success']
        if len(failed_videos) > 0:
            print(f"\n❌ FAILED VIDEOS:")
            for _, row in failed_videos.iterrows():
                print(f"  {row['video_name']}: {row['status']}")
        
        # So sánh với CSV data
        print(f"\n🔍 COMPARISON WITH CSV DATA:")
        csv_errors = self.data_loader.calculate_errors()
        if metrics['successful_videos'] > 0:
            print(f"  Model test vs CSV data:")
            print(f"    Systolic MAE: Model={metrics['sys_mae']:.2f} vs CSV={csv_errors['sys_mae']:.2f}")
            print(f"    Diastolic MAE: Model={metrics['dia_mae']:.2f} vs CSV={csv_errors['dia_mae']:.2f}")
            
            if abs(metrics['sys_mae'] - csv_errors['sys_mae']) < 1.0:
                print("  ✅ Model results MATCH CSV data (within 1 mmHg)")
            else:
                print("  ⚠️  Model results DIFFER from CSV data")
        
        return merged_df, metrics
    
    def save_results(self, output_file='calibration_test_results.csv'):
        """Lưu kết quả ra file CSV"""
        merged_df = self.merge_with_ground_truth()
        if merged_df is not None:
            output_path = os.path.join(BASE_DIR, output_file)
            merged_df.to_csv(output_path, index=False)
            print(f"\n💾 Kết quả đã được lưu tại: {output_path}")

def main():
    """Main function để chạy test"""
    tester = ModelCalibrationTester()
    
    # Chạy tất cả videos
    tester.run_all_videos()
    
    # In báo cáo chi tiết
    tester.print_detailed_report()
    
    # Lưu kết quả
    tester.save_results()

if __name__ == "__main__":
    main()

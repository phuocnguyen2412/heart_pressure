"""
Folder-Based Linear Calibration System
=====================================
Train trên folder VAL, test trên folder TEST
Mỗi folder chứa videos + ground_truth.csv
"""
import os
import pandas as pd
import numpy as np
import sys
from linear_calibration import LinearCalibrator
from main_pipeline import BloodPressureInferencePipeline
from setting import BASE_DIR

# Add signal_extractor to path
signal_extractor_path = os.path.join(BASE_DIR, 'signal_extractor')
if signal_extractor_path not in sys.path:
    sys.path.append(signal_extractor_path)

from signal_extractor.pipeline import run_extract_signal
from predict_test import predict_test_data


class FolderCalibrationSystem:
    """
    Hệ thống calibration dựa trên folder structure:
    - data/val/: Training data (videos + ground_truth.csv)
    - data/test/: Testing data (videos + ground_truth.csv)
    """
    
    def __init__(self):
        self.calibrator = LinearCalibrator()
        
        # Paths
        self.val_folder = os.path.join(BASE_DIR, 'data', 'val')
        self.test_folder = os.path.join(BASE_DIR, 'data', 'test')
        
        print(f"📁 VAL folder: {self.val_folder}")
        print(f"📁 TEST folder: {self.test_folder}")
    
    def run_model_on_videos_in_folder(self, folder_path, folder_name):
        """
        Chạy model trên tất cả videos trong folder và merge với ground truth
        """
        print(f"\n🎬 Running model on {folder_name} videos...")
        
        # Load ground truth CSV
        gt_csv_path = os.path.join(folder_path, 'ground_truth.csv')
        if not os.path.exists(gt_csv_path):
            raise FileNotFoundError(f"❌ Ground truth CSV not found: {gt_csv_path}")
        
        ground_truth_df = pd.read_csv(gt_csv_path)
        print(f"📊 Loaded ground truth: {len(ground_truth_df)} records")
        print(ground_truth_df)
        
        # Get all video files in folder
        video_files = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.mov', '.mp4', '.avi')):
                video_files.append(filename)
        
        print(f"🎥 Found {len(video_files)} videos: {video_files}")
          # Run model on each video and collect results
        results = []
        
        for _, row in ground_truth_df.iterrows():
            video_name_base = row['video']  # e.g., "video_1"
            
            # Find the actual video file with extension
            actual_video_file = None
            for video_file in video_files:
                # Check if video file starts with the base name
                if video_file.startswith(video_name_base + '.'):
                    actual_video_file = video_file
                    break
            
            if actual_video_file is None:
                print(f"⚠️ Video not found for: {video_name_base}")
                continue
                
            video_path = os.path.join(folder_path, actual_video_file)
            print(f"🔄 Processing {actual_video_file}...")
            
            try:
                bp_inference_pipeline = BloodPressureInferencePipeline()
                result = bp_inference_pipeline.predict_test_data(video_path)
                
                if result and 'systolic' in result and 'diastolic' in result:
                    # Extract predictions
                    app_sys = result['systolic']
                    app_dia = result['diastolic']
                    app_hr = result['hr']
                    app_mean = result['mean']
                    
                    # Create result record
                    result_record = {
                        'ID': row['ID'],
                        'video': actual_video_file,  # Use actual filename with extension
                        'gt_sys': row['may_sys'],  # Ground truth systolic
                        'gt_dia': row['may_dia'],  # Ground truth diastolic
                        'app_sys': app_sys,        # App prediction systolic
                        'app_dia': app_dia,        # App prediction diastolic
                        'app_hr': app_hr,
                        'app_mean': app_mean,
                        'status': 'success'
                    }
                    
                    results.append(result_record)
                    print(f"✅ {actual_video_file}: App={app_sys:.1f}/{app_dia:.1f}, GT={row['may_sys']}/{row['may_dia']}")
                    
                else:
                    print(f"❌ {actual_video_file}: Failed to get BP predictions")
                    
            except Exception as e:
                print(f"❌ {actual_video_file}: Error - {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        print(f"\n✅ {folder_name}: Successfully processed {len(results_df)} videos")
        
        return results_df
    
    def train_linear_calibration(self, val_results_df):
        """
        Train linear calibration từ validation results
        """
        print(f"\n🔧 Training Linear Calibration...")
        
        if len(val_results_df) == 0:
            raise ValueError("❌ No validation data for training!")
        
        # Extract training arrays
        app_sys_array = val_results_df['app_sys'].values
        gt_sys_array = val_results_df['gt_sys'].values
        app_dia_array = val_results_df['app_dia'].values
        gt_dia_array = val_results_df['gt_dia'].values
        
        print(f"   📊 Training samples: {len(app_sys_array)}")
        print(f"   📊 App Systolic range: {app_sys_array.min():.1f} - {app_sys_array.max():.1f} mmHg")
        print(f"   📊 GT Systolic range: {gt_sys_array.min():.1f} - {gt_sys_array.max():.1f} mmHg")
        print(f"   📊 App Diastolic range: {app_dia_array.min():.1f} - {app_dia_array.max():.1f} mmHg")
        print(f"   📊 GT Diastolic range: {gt_dia_array.min():.1f} - {gt_dia_array.max():.1f} mmHg")
          # Fit linear calibrator
        self.calibrator.fit(app_sys_array, gt_sys_array, app_dia_array, gt_dia_array)
        
        # Print calibration equations
        params = self.calibrator.calibration_params
        print("\n🔧 Linear Calibration Equations:")
        print(f"   Systolic: GT = {params['sys_slope']:.4f} * App + {params['sys_intercept']:.4f}")
        print(f"   Diastolic: GT = {params['dia_slope']:.4f} * App + {params['dia_intercept']:.4f}")
        
        return self.calibrator
    
    def test_calibration(self, test_results_df):
        """
        Test calibration trên test data và tính toán metrics chi tiết
        """
        print(f"\n🧪 Testing Calibration on Test Data...")
        
        if len(test_results_df) == 0:
            raise ValueError("❌ No test data for evaluation!")
        
        # Extract test arrays
        app_sys_test = test_results_df['app_sys'].values
        gt_sys_test = test_results_df['gt_sys'].values
        app_dia_test = test_results_df['app_dia'].values
        gt_dia_test = test_results_df['gt_dia'].values
          # Apply calibration
        calibrated_sys, calibrated_dia = self.calibrator.predict(app_sys_test, app_dia_test)
        
        # Calculate detailed metrics
        evaluation_results = self._calculate_detailed_metrics(
            app_sys_test, gt_sys_test, calibrated_sys,
            app_dia_test, gt_dia_test, calibrated_dia,
            test_results_df
        )
        
        return evaluation_results
    
    def _calculate_detailed_metrics(self, app_sys, gt_sys, cal_sys, app_dia, gt_dia, cal_dia, test_df):
        """
        Tính toán metrics chi tiết: before vs after calibration
        """
        # Errors before calibration
        before_sys_errors = app_sys - gt_sys
        before_dia_errors = app_dia - gt_dia
        
        # Errors after calibration
        after_sys_errors = cal_sys - gt_sys
        after_dia_errors = cal_dia - gt_dia
        
        # Overall metrics BEFORE calibration
        before_metrics = {
            'sys_mae': np.mean(np.abs(before_sys_errors)),
            'sys_rmse': np.sqrt(np.mean(before_sys_errors**2)),
            'sys_bias': np.mean(before_sys_errors),
            'dia_mae': np.mean(np.abs(before_dia_errors)),
            'dia_rmse': np.sqrt(np.mean(before_dia_errors**2)),
            'dia_bias': np.mean(before_dia_errors)
        }
        
        # Overall metrics AFTER calibration
        after_metrics = {
            'sys_mae': np.mean(np.abs(after_sys_errors)),
            'sys_rmse': np.sqrt(np.mean(after_sys_errors**2)),
            'sys_bias': np.mean(after_sys_errors),
            'dia_mae': np.mean(np.abs(after_dia_errors)),
            'dia_rmse': np.sqrt(np.mean(after_dia_errors**2)),
            'dia_bias': np.mean(after_dia_errors)
        }
        
        # Improvement metrics
        improvement = {
            'sys_mae': before_metrics['sys_mae'] - after_metrics['sys_mae'],
            'sys_rmse': before_metrics['sys_rmse'] - after_metrics['sys_rmse'],
            'sys_bias': abs(before_metrics['sys_bias']) - abs(after_metrics['sys_bias']),
            'dia_mae': before_metrics['dia_mae'] - after_metrics['dia_mae'],
            'dia_rmse': before_metrics['dia_rmse'] - after_metrics['dia_rmse'],
            'dia_bias': abs(before_metrics['dia_bias']) - abs(after_metrics['dia_bias'])
        }
        
        # Record-by-record details
        detailed_records = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            record_detail = {
                'ID': row['ID'],
                'video': row['video'],
                'gt_sys': gt_sys[i],
                'app_sys': app_sys[i],
                'cal_sys': cal_sys[i],
                'before_sys_error': before_sys_errors[i],
                'after_sys_error': after_sys_errors[i],
                'sys_improvement': abs(before_sys_errors[i]) - abs(after_sys_errors[i]),
                'gt_dia': gt_dia[i],
                'app_dia': app_dia[i],
                'cal_dia': cal_dia[i],
                'before_dia_error': before_dia_errors[i],
                'after_dia_error': after_dia_errors[i],
                'dia_improvement': abs(before_dia_errors[i]) - abs(after_dia_errors[i])
            }
            detailed_records.append(record_detail)
        
        return {
            'before_metrics': before_metrics,
            'after_metrics': after_metrics,
            'improvement': improvement,
            'detailed_records': detailed_records
        }
    
    def print_detailed_evaluation_results(self, evaluation_results):
        """
        In kết quả đánh giá chi tiết từng record và tổng quan
        """
        print("\n" + "="*120)
        print("📈 DETAILED CALIBRATION EVALUATION RESULTS")
        print("="*120)
        
        before = evaluation_results['before_metrics']
        after = evaluation_results['after_metrics']
        improvement = evaluation_results['improvement']
        records = evaluation_results['detailed_records']
        
        # Overall Summary
        print("\n📊 OVERALL PERFORMANCE SUMMARY:")
        print("-"*80)
        print("BEFORE CALIBRATION:")
        print(f"  Systolic  - MAE: {before['sys_mae']:.2f} mmHg, RMSE: {before['sys_rmse']:.2f} mmHg, Bias: {before['sys_bias']:+.2f} mmHg")
        print(f"  Diastolic - MAE: {before['dia_mae']:.2f} mmHg, RMSE: {before['dia_rmse']:.2f} mmHg, Bias: {before['dia_bias']:+.2f} mmHg")
        
        print("\nAFTER CALIBRATION:")
        print(f"  Systolic  - MAE: {after['sys_mae']:.2f} mmHg, RMSE: {after['sys_rmse']:.2f} mmHg, Bias: {after['sys_bias']:+.2f} mmHg")
        print(f"  Diastolic - MAE: {after['dia_mae']:.2f} mmHg, RMSE: {after['dia_rmse']:.2f} mmHg, Bias: {after['dia_bias']:+.2f} mmHg")
        
        print("\nIMPROVEMENT:")
        sys_mae_status = "✅ Better" if improvement['sys_mae'] > 0 else "❌ Worse"
        dia_mae_status = "✅ Better" if improvement['dia_mae'] > 0 else "❌ Worse"
        print(f"  Systolic  MAE: {improvement['sys_mae']:+.2f} mmHg ({sys_mae_status})")
        print(f"  Systolic  RMSE: {improvement['sys_rmse']:+.2f} mmHg")
        print(f"  Diastolic MAE: {improvement['dia_mae']:+.2f} mmHg ({dia_mae_status})")
        print(f"  Diastolic RMSE: {improvement['dia_rmse']:+.2f} mmHg")
        
        # Record-by-record detailed table
        print(f"\n📋 RECORD-BY-RECORD DETAILED COMPARISON:")
        print("-"*120)
        header = f"{'ID':<3} {'Video':<15} {'GT_Sys':<7} {'App_Sys':<8} {'Cal_Sys':<8} {'Before_Err':<11} {'After_Err':<10} {'GT_Dia':<7} {'App_Dia':<8} {'Cal_Dia':<8} {'Before_Err':<11} {'After_Err':<10}"
        print(header)
        print("-"*120)
        
        for record in records:
            row = (f"{record['ID']:<3} {record['video']:<15} "
                   f"{record['gt_sys']:<7.0f} {record['app_sys']:<8.1f} {record['cal_sys']:<8.1f} "
                   f"{record['before_sys_error']:+11.1f} {record['after_sys_error']:+10.1f} "
                   f"{record['gt_dia']:<7.0f} {record['app_dia']:<8.1f} {record['cal_dia']:<8.1f} "
                   f"{record['before_dia_error']:+11.1f} {record['after_dia_error']:+10.1f}")
            print(row)
        
        # Improvement summary for each record
        print(f"\n📈 IMPROVEMENT SUMMARY FOR EACH RECORD:")
        print("-"*80)
        print(f"{'ID':<3} {'Video':<15} {'Sys_Improvement':<20} {'Dia_Improvement':<20}")
        print("-"*80)
        
        for record in records:
            sys_improvement = record['sys_improvement']
            dia_improvement = record['dia_improvement']
            
            sys_status = "✅ Better" if sys_improvement > 0 else "❌ Worse"
            dia_status = "✅ Better" if dia_improvement > 0 else "❌ Worse"
            
            print(f"{record['ID']:<3} {record['video']:<15} "
                  f"{sys_improvement:+6.1f} mmHg {sys_status:<8} "
                  f"{dia_improvement:+6.1f} mmHg {dia_status}")
    
    def save_all_results(self, val_results_df, test_results_df, evaluation_results):
        """
        Lưu tất cả kết quả vào files
        """
        print(f"\n💾 Saving all results...")
        
        # Save validation results
        val_output_path = os.path.join(BASE_DIR, 'val_model_results.csv')
        val_results_df.to_csv(val_output_path, index=False)
        print(f"   📁 Validation results: {val_output_path}")
        
        # Save test results
        test_output_path = os.path.join(BASE_DIR, 'test_model_results.csv')
        test_results_df.to_csv(test_output_path, index=False)
        print(f"   📁 Test results: {test_output_path}")
        
        # Save detailed evaluation
        eval_df = pd.DataFrame(evaluation_results['detailed_records'])
        eval_output_path = os.path.join(BASE_DIR, 'calibration_detailed_evaluation.csv')
        eval_df.to_csv(eval_output_path, index=False)
        print(f"   📁 Detailed evaluation: {eval_output_path}")
        
        # Save calibration model
        self.calibrator.save()
        print(f"   📁 Calibration model: models/linear_calibration.pkl")
    
    def run_complete_pipeline(self):
        """
        Chạy toàn bộ pipeline: Train on VAL, Test on TEST
        """
        print("="*120)
        print("🎯 FOLDER-BASED LINEAR CALIBRATION PIPELINE")
        print("Train on VAL folder → Test on TEST folder")
        print("="*120)
        
        try:
            # Step 1: Run model on VAL data (for training calibration)
            val_results_df = self.run_model_on_videos_in_folder(self.val_folder, "VAL")
            
            if len(val_results_df) == 0:
                raise ValueError("❌ No validation results for training!")
            
            # Step 2: Train linear calibration
            self.train_linear_calibration(val_results_df)
            
            # Step 3: Run model on TEST data
            test_results_df = self.run_model_on_videos_in_folder(self.test_folder, "TEST")
            
            if len(test_results_df) == 0:
                raise ValueError("❌ No test results for evaluation!")
            
            # Step 4: Test calibration and get detailed metrics
            evaluation_results = self.test_calibration(test_results_df)
            
            # Step 5: Print detailed results
            self.print_detailed_evaluation_results(evaluation_results)
            
            # Step 6: Save all results
            self.save_all_results(val_results_df, test_results_df, evaluation_results)
            
            print(f"\n🎉 CALIBRATION PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"   📊 Training: {len(val_results_df)} videos from VAL folder")
            print(f"   📊 Testing: {len(test_results_df)} videos from TEST folder")
            
            # Summary of improvement
            improvement = evaluation_results['improvement']
            print(f"   📈 Systolic MAE improvement: {improvement['sys_mae']:+.2f} mmHg")
            print(f"   📈 Diastolic MAE improvement: {improvement['dia_mae']:+.2f} mmHg")
            print(f"   💾 All results saved, calibration model ready for production!")
            
            return val_results_df, test_results_df, evaluation_results
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


def main():
    """
    Main function để chạy folder-based calibration pipeline
    """
    print("🚀 Starting Folder-Based Linear Calibration System...")
    print("📁 Structure: data/val/ (training) → data/test/ (testing)")
    
    # Initialize system
    calibration_system = FolderCalibrationSystem()
    
    # Run complete pipeline
    val_results, test_results, evaluation = calibration_system.run_complete_pipeline()
    
    if val_results is not None and test_results is not None and evaluation is not None:
        print(f"\n✅ SUCCESS! Calibration system ready for production use.")
    else:
        print(f"\n❌ FAILED! Could not complete calibration pipeline.")


if __name__ == "__main__":
    main()

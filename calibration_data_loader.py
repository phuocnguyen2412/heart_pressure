"""
Calibration Data Handler - Updated Version
Xử lý dữ liệu hiệu chỉnh cho model đo huyết áp theo yêu cầu mới:
- Training: ID 1-6 (6 videos)  
- Test: ID 7-8 (2 videos)
- Removed: ID 4,9 (due to recording issues)
"""
import pandas as pd
import numpy as np
import os
from setting import BASE_DIR

class CalibrationDataLoader:
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(BASE_DIR, 'data', 'calibration_data.csv')
        self.csv_path = csv_path
        
    def load_calibration_data(self):
        """Load calibration data from CSV"""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"✅ Loaded calibration data: {len(df)} records")
            return df
        except Exception as e:
            print(f"❌ Error loading calibration data: {e}")
            return None
    
    def get_train_test_split(self):
        """
        Split data into training and test sets
        Training: ID 1-6 (6 videos) - excluding ID 4 which was removed
        Test: ID 7-8 (2 videos)
        Note: ID 4,9 were removed due to recording issues
        """
        df = self.load_calibration_data()
        if df is None:
            return None, None
        
        # Training set: ID 1,2,3,5,6 (note: ID 4 was removed)
        train_data = df[df['ID'].isin([1, 2, 3, 5, 6])].copy()
        
        # Test set: ID 7,8  
        test_data = df[df['ID'].isin([7, 8])].copy()
        
        print(f"📊 Training data: {len(train_data)} videos (ID {sorted(train_data['ID'].tolist())})")
        print(f"📊 Test data: {len(test_data)} videos (ID {sorted(test_data['ID'].tolist())})")
        
        return train_data, test_data
    
    def get_all_available_data(self):
        """Get all available calibration data (ID 1,2,3,5,6,7,8)"""
        return self.load_calibration_data()
    
    def calculate_baseline_errors(self, data=None):
        """Calculate baseline errors between app predictions and ground truth"""
        if data is None:
            data = self.load_calibration_data()
        
        if data is None or len(data) == 0:
            return None
            
        # Calculate errors
        sys_error = data['app_sys'] - data['may_sys']
        dia_error = data['app_dia'] - data['may_dia']
        
        metrics = {
            'sys_mae': np.mean(np.abs(sys_error)),
            'sys_rmse': np.sqrt(np.mean(sys_error**2)),
            'sys_bias': np.mean(sys_error),
            'dia_mae': np.mean(np.abs(dia_error)),
            'dia_rmse': np.sqrt(np.mean(dia_error**2)),
            'dia_bias': np.mean(dia_error),
            'count': len(data)
        }
        
        return metrics
    
    def analyze_baseline_errors(self):
        """Comprehensive analysis of baseline errors across all datasets"""
        train_data, test_data = self.get_train_test_split()
        all_data = self.get_all_available_data()
        
        if train_data is None or test_data is None:
            print("❌ Unable to load data for analysis")
            return None
        
        print("\n" + "="*60)
        print("📈 BASELINE ERROR ANALYSIS (Before Calibration)")
        print("="*60)
        
        # Training set analysis
        train_errors = self.calculate_baseline_errors(train_data)
        print(f"\n🔧 TRAINING SET (ID 1,2,3,5,6) - {train_errors['count']} samples:")
        print(f"   Systolic  - MAE: {train_errors['sys_mae']:.2f} mmHg, RMSE: {train_errors['sys_rmse']:.2f} mmHg, Bias: {train_errors['sys_bias']:.2f} mmHg")
        print(f"   Diastolic - MAE: {train_errors['dia_mae']:.2f} mmHg, RMSE: {train_errors['dia_rmse']:.2f} mmHg, Bias: {train_errors['dia_bias']:.2f} mmHg")
        
        # Test set analysis
        test_errors = self.calculate_baseline_errors(test_data)
        print(f"\n🧪 TEST SET (ID 7,8) - {test_errors['count']} samples:")
        print(f"   Systolic  - MAE: {test_errors['sys_mae']:.2f} mmHg, RMSE: {test_errors['sys_rmse']:.2f} mmHg, Bias: {test_errors['sys_bias']:.2f} mmHg")
        print(f"   Diastolic - MAE: {test_errors['dia_mae']:.2f} mmHg, RMSE: {test_errors['dia_rmse']:.2f} mmHg, Bias: {test_errors['dia_bias']:.2f} mmHg")
        
        # Overall analysis
        all_errors = self.calculate_baseline_errors(all_data)
        print(f"\n📊 OVERALL (ID 1,2,3,5,6,7,8) - {all_errors['count']} samples:")
        print(f"   Systolic  - MAE: {all_errors['sys_mae']:.2f} mmHg, RMSE: {all_errors['sys_rmse']:.2f} mmHg, Bias: {all_errors['sys_bias']:.2f} mmHg")
        print(f"   Diastolic - MAE: {all_errors['dia_mae']:.2f} mmHg, RMSE: {all_errors['dia_rmse']:.2f} mmHg, Bias: {all_errors['dia_bias']:.2f} mmHg")
        
        return {
            'train': train_errors,
            'test': test_errors, 
            'overall': all_errors
        }
    
    def print_detailed_comparison(self, data=None):
        """Print detailed comparison for each record"""
        if data is None:
            data = self.get_all_available_data()
        
        print("\n" + "="*100)
        print("📋 DETAILED RECORD-BY-RECORD COMPARISON")
        print("="*100)
        print(f"{'ID':<3} {'Video':<10} {'GT_Sys':<7} {'App_Sys':<8} {'Sys_Err':<8} {'GT_Dia':<7} {'App_Dia':<8} {'Dia_Err':<8}")
        print("-"*100)
        
        for _, row in data.iterrows():
            sys_err = row['app_sys'] - row['may_sys']
            dia_err = row['app_dia'] - row['may_dia']
            
            print(f"{row['ID']:<3} {row['video']:<10} {row['may_sys']:<7} {row['app_sys']:<8.1f} {sys_err:+8.1f} {row['may_dia']:<7} {row['app_dia']:<8.1f} {dia_err:+8.1f}")

if __name__ == "__main__":
    # Test the updated data loader
    print("🔍 Testing Updated Calibration Data Loader")
    loader = CalibrationDataLoader()
    
    # Analyze baseline errors
    baseline_analysis = loader.analyze_baseline_errors()
    
    # Print detailed comparison
    loader.print_detailed_comparison()
    
    print("\n✅ Calibration data loader test completed!")

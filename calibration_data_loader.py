"""
Calibration Data Handler
Xử lý dữ liệu hiệu chỉnh cho model đo huyết áp
"""
import pandas as pd
import numpy as np
import os
from setting import BASE_DIR

class CalibrationDataLoader:
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(BASE_DIR, 'data', 'calibration_data.csv')
        self.df = pd.read_csv(csv_path)
        
    def get_train_data(self):
        """Lấy dữ liệu train (ID 4-9)"""
        return self.df[self.df['ID'].between(4, 9)].copy()
    
    def get_test_data(self):
        """Lấy dữ liệu test (ID 1-3)"""
        return self.df[self.df['ID'].between(1, 3)].copy()
    
    def get_all_data(self):
        """Lấy toàn bộ dữ liệu"""
        return self.df.copy()
    
    def calculate_errors(self, data=None):
        """Tính toán lỗi giữa app và máy đo chuẩn"""
        if data is None:
            data = self.df
        
        sys_error = data['app_sys'] - data['may_sys']
        dia_error = data['app_dia'] - data['may_dia']
        
        return {
            'sys_mae': np.mean(np.abs(sys_error)),
            'sys_rmse': np.sqrt(np.mean(sys_error**2)),
            'sys_bias': np.mean(sys_error),
            'dia_mae': np.mean(np.abs(dia_error)),
            'dia_rmse': np.sqrt(np.mean(dia_error**2)),
            'dia_bias': np.mean(dia_error)
        }
    
    def print_error_summary(self):
        """In tóm tắt lỗi của toàn bộ dữ liệu"""
        train_data = self.get_train_data()
        test_data = self.get_test_data()
        all_data = self.get_all_data()
        
        print("=== CALIBRATION DATA SUMMARY ===")
        print(f"Train data (ID 4-9): {len(train_data)} samples")
        print(f"Test data (ID 1-3): {len(test_data)} samples")
        print(f"Total: {len(all_data)} samples")
        
        print("\n=== ERROR ANALYSIS (Before Calibration) ===")
        
        # Train set errors
        train_errors = self.calculate_errors(train_data)
        print("\nTrain Set Errors:")
        print(f"  Systolic - MAE: {train_errors['sys_mae']:.2f}, RMSE: {train_errors['sys_rmse']:.2f}, Bias: {train_errors['sys_bias']:.2f}")
        print(f"  Diastolic - MAE: {train_errors['dia_mae']:.2f}, RMSE: {train_errors['dia_rmse']:.2f}, Bias: {train_errors['dia_bias']:.2f}")
        
        # Test set errors
        test_errors = self.calculate_errors(test_data)
        print("\nTest Set Errors:")
        print(f"  Systolic - MAE: {test_errors['sys_mae']:.2f}, RMSE: {test_errors['sys_rmse']:.2f}, Bias: {test_errors['sys_bias']:.2f}")
        print(f"  Diastolic - MAE: {test_errors['dia_mae']:.2f}, RMSE: {test_errors['dia_rmse']:.2f}, Bias: {test_errors['dia_bias']:.2f}")
        
        # Overall errors
        all_errors = self.calculate_errors(all_data)
        print("\nOverall Errors:")
        print(f"  Systolic - MAE: {all_errors['sys_mae']:.2f}, RMSE: {all_errors['sys_rmse']:.2f}, Bias: {all_errors['sys_bias']:.2f}")
        print(f"  Diastolic - MAE: {all_errors['dia_mae']:.2f}, RMSE: {all_errors['dia_rmse']:.2f}, Bias: {all_errors['dia_bias']:.2f}")
        
        return train_errors, test_errors, all_errors

if __name__ == "__main__":
    # Test script
    loader = CalibrationDataLoader()
    loader.print_error_summary()

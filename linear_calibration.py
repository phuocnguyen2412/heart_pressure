"""
Linear Calibration Module
Hiệu chỉnh tuyến tính cho model đo huyết áp
"""
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from setting import BASE_DIR

class LinearCalibrator:
    """
    Linear Calibration cho Blood Pressure Prediction
    Formula: Calibrated_Value = a * Raw_Value + b
    """
    
    def __init__(self):
        self.sys_model = None  # Model cho systolic
        self.dia_model = None  # Model cho diastolic
        self.is_fitted = False
        self.calibration_params = {}
        
    def fit(self, raw_sys, true_sys, raw_dia, true_dia):
        """
        Fit linear models cho systolic và diastolic
        
        Args:
            raw_sys: Systolic values từ model gốc
            true_sys: Ground truth systolic values
            raw_dia: Diastolic values từ model gốc  
            true_dia: Ground truth diastolic values
        """
        # Reshape data cho sklearn
        raw_sys = np.array(raw_sys).reshape(-1, 1)
        true_sys = np.array(true_sys)
        raw_dia = np.array(raw_dia).reshape(-1, 1)
        true_dia = np.array(true_dia)
        
        # Fit systolic model
        self.sys_model = LinearRegression()
        self.sys_model.fit(raw_sys, true_sys)
        
        # Fit diastolic model
        self.dia_model = LinearRegression()
        self.dia_model.fit(raw_dia, true_dia)
        
        # Store parameters
        self.calibration_params = {
            'sys_slope': self.sys_model.coef_[0],
            'sys_intercept': self.sys_model.intercept_,
            'dia_slope': self.dia_model.coef_[0],
            'dia_intercept': self.dia_model.intercept_,
        }
        
        self.is_fitted = True
        
        print("🔧 Linear Calibration fitted successfully!")
        print(f"   Systolic: y = {self.calibration_params['sys_slope']:.4f} * x + {self.calibration_params['sys_intercept']:.4f}")
        print(f"   Diastolic: y = {self.calibration_params['dia_slope']:.4f} * x + {self.calibration_params['dia_intercept']:.4f}")
        
    def predict(self, raw_sys, raw_dia):
        """
        Apply calibration to raw predictions
        
        Args:
            raw_sys: Raw systolic predictions
            raw_dia: Raw diastolic predictions
            
        Returns:
            calibrated_sys, calibrated_dia
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction!")
            
        raw_sys = np.array(raw_sys).reshape(-1, 1)
        raw_dia = np.array(raw_dia).reshape(-1, 1)
        
        calibrated_sys = self.sys_model.predict(raw_sys)
        calibrated_dia = self.dia_model.predict(raw_dia)
        
        return calibrated_sys, calibrated_dia
    
    def calibrate_single(self, sys_value, dia_value):
        """
        Calibrate single prediction
        
        Args:
            sys_value: Single systolic value
            dia_value: Single diastolic value
            
        Returns:
            calibrated_sys, calibrated_dia
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction!")
            
        calibrated_sys = self.sys_model.predict([[sys_value]])[0]
        calibrated_dia = self.dia_model.predict([[dia_value]])[0]
        
        return calibrated_sys, calibrated_dia
    
    def evaluate(self, raw_sys, true_sys, raw_dia, true_dia):
        """
        Evaluate calibration performance
        
        Returns:
            dict with before/after metrics
        """
        # Before calibration
        before_sys_mae = mean_absolute_error(true_sys, raw_sys)
        before_sys_rmse = np.sqrt(mean_squared_error(true_sys, raw_sys))
        before_sys_bias = np.mean(np.array(raw_sys) - np.array(true_sys))
        
        before_dia_mae = mean_absolute_error(true_dia, raw_dia)
        before_dia_rmse = np.sqrt(mean_squared_error(true_dia, raw_dia))
        before_dia_bias = np.mean(np.array(raw_dia) - np.array(true_dia))
        
        # After calibration
        cal_sys, cal_dia = self.predict(raw_sys, raw_dia)
        
        after_sys_mae = mean_absolute_error(true_sys, cal_sys)
        after_sys_rmse = np.sqrt(mean_squared_error(true_sys, cal_sys))
        after_sys_bias = np.mean(cal_sys - np.array(true_sys))
        
        after_dia_mae = mean_absolute_error(true_dia, cal_dia)
        after_dia_rmse = np.sqrt(mean_squared_error(true_dia, cal_dia))
        after_dia_bias = np.mean(cal_dia - np.array(true_dia))
        
        return {
            'before': {
                'sys_mae': before_sys_mae,
                'sys_rmse': before_sys_rmse,
                'sys_bias': before_sys_bias,
                'dia_mae': before_dia_mae,
                'dia_rmse': before_dia_rmse,
                'dia_bias': before_dia_bias,
            },
            'after': {
                'sys_mae': after_sys_mae,
                'sys_rmse': after_sys_rmse,
                'sys_bias': after_sys_bias,
                'dia_mae': after_dia_mae,
                'dia_rmse': after_dia_rmse,
                'dia_bias': after_dia_bias,
            },
            'improvement': {
                'sys_mae': before_sys_mae - after_sys_mae,
                'sys_rmse': before_sys_rmse - after_sys_rmse,
                'dia_mae': before_dia_mae - after_dia_mae,
                'dia_rmse': before_dia_rmse - after_dia_rmse,
            }
        }
    
    def save(self, filepath=None):
        """Save calibration model"""
        if filepath is None:
            filepath = os.path.join(BASE_DIR, 'models', 'linear_calibration.pkl')
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'sys_model': self.sys_model,
            'dia_model': self.dia_model,
            'calibration_params': self.calibration_params,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"💾 Calibration model saved to: {filepath}")
    
    def load(self, filepath=None):
        """Load calibration model"""
        if filepath is None:
            filepath = os.path.join(BASE_DIR, 'models', 'linear_calibration.pkl')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration model not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.sys_model = save_data['sys_model']
        self.dia_model = save_data['dia_model']
        self.calibration_params = save_data['calibration_params']
        self.is_fitted = save_data['is_fitted']
        
        print(f"📂 Calibration model loaded from: {filepath}")
    
    def plot_calibration(self, raw_sys, true_sys, raw_dia, true_dia, save_path=None):
        """
        Plot before/after calibration comparison
        """
        cal_sys, cal_dia = self.predict(raw_sys, raw_dia)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Systolic before
        axes[0,0].scatter(true_sys, raw_sys, alpha=0.7, color='red')
        axes[0,0].plot([min(true_sys), max(true_sys)], [min(true_sys), max(true_sys)], 'k--', label='Perfect')
        axes[0,0].set_xlabel('Ground Truth Systolic')
        axes[0,0].set_ylabel('Raw Predicted Systolic')
        axes[0,0].set_title('Before Calibration - Systolic')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Systolic after
        axes[0,1].scatter(true_sys, cal_sys, alpha=0.7, color='blue')
        axes[0,1].plot([min(true_sys), max(true_sys)], [min(true_sys), max(true_sys)], 'k--', label='Perfect')
        axes[0,1].set_xlabel('Ground Truth Systolic')
        axes[0,1].set_ylabel('Calibrated Systolic')
        axes[0,1].set_title('After Calibration - Systolic')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Diastolic before
        axes[1,0].scatter(true_dia, raw_dia, alpha=0.7, color='red')
        axes[1,0].plot([min(true_dia), max(true_dia)], [min(true_dia), max(true_dia)], 'k--', label='Perfect')
        axes[1,0].set_xlabel('Ground Truth Diastolic')
        axes[1,0].set_ylabel('Raw Predicted Diastolic')
        axes[1,0].set_title('Before Calibration - Diastolic')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Diastolic after
        axes[1,1].scatter(true_dia, cal_dia, alpha=0.7, color='blue')
        axes[1,1].plot([min(true_dia), max(true_dia)], [min(true_dia), max(true_dia)], 'k--', label='Perfect')
        axes[1,1].set_xlabel('Ground Truth Diastolic')
        axes[1,1].set_ylabel('Calibrated Diastolic')
        axes[1,1].set_title('After Calibration - Diastolic')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(BASE_DIR, 'calibration_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Calibration plot saved to: {save_path}")        # plt.show()  # Commented for server environment

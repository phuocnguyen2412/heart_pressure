"""
Calibrated Prediction Wrapper
Wrapper để sử dụng model với calibration đã trained
"""
import os
from predict_test import predict_test_data
from linear_calibration import LinearCalibrator
from setting import BASE_DIR

class CalibratedPredictor:
    """
    Wrapper class để sử dụng model với linear calibration
    """
    
    def __init__(self, calibration_path=None):
        self.calibrator = LinearCalibrator()
        self.calibration_loaded = False
        
        # Try to load calibration
        if self.load_calibration(calibration_path):
            print("✅ Calibrated predictor ready!")
        else:
            print("⚠️  Using uncalibrated predictor")
    
    def load_calibration(self, calibration_path=None):
        """Load calibration model"""
        try:
            self.calibrator.load(calibration_path)
            self.calibration_loaded = True
            return True
        except Exception as e:
            print(f"⚠️  Could not load calibration: {e}")
            return False
    
    def predict(self, ppg_signal, apply_calibration=True):
        """
        Make prediction with optional calibration
        
        Args:
            ppg_signal: PPG signal data
            apply_calibration: Whether to apply calibration
            
        Returns:
            dict with prediction results
        """
        # Get raw prediction
        raw_output = predict_test_data(ppg_signal)
        
        # Apply calibration if available and requested
        if apply_calibration and self.calibration_loaded:
            try:
                cal_sys, cal_dia = self.calibrator.calibrate_single(
                    raw_output['systolic'], 
                    raw_output['diastolic']
                )
                
                # Create calibrated output
                calibrated_output = raw_output.copy()
                calibrated_output.update({
                    'systolic_raw': raw_output['systolic'],
                    'diastolic_raw': raw_output['diastolic'],
                    'systolic': cal_sys,
                    'diastolic': cal_dia,
                    'calibrated': True
                })
                
                return calibrated_output
                
            except Exception as e:
                print(f"⚠️  Calibration failed: {e}. Using raw prediction.")
                return raw_output
        else:
            # Return raw prediction
            raw_output['calibrated'] = False
            return raw_output


# Example usage functions
def predict_with_calibration(ppg_signal):
    """
    Simple function to get calibrated prediction
    """
    predictor = CalibratedPredictor()
    return predictor.predict(ppg_signal)

def predict_without_calibration(ppg_signal):
    """
    Simple function to get uncalibrated prediction
    """
    predictor = CalibratedPredictor()
    return predictor.predict(ppg_signal, apply_calibration=False)

# Quick test function
def test_calibrated_predictor():
    """Test the calibrated predictor"""
    print("🧪 Testing Calibrated Predictor...")
    
    # This would need actual PPG signal data to work
    # For now, just test the class initialization
    predictor = CalibratedPredictor()
    print(f"   Calibration loaded: {predictor.calibration_loaded}")
    
    if predictor.calibration_loaded:
        print("   ✅ Predictor ready for calibrated predictions")
    else:
        print("   ⚠️  Predictor will use raw predictions only")

if __name__ == "__main__":
    test_calibrated_predictor()

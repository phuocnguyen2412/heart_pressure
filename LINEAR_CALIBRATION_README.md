# Linear Calibration Module

## 📋 Tổng quan

Module Linear Calibration được thiết kế để hiệu chỉnh kết quả dự đoán huyết áp từ model AI, giảm sai số systematic bias và cải thiện độ chính xác tổng thể.

## 🎯 Mục tiêu

- **Fix systematic bias**: Model có xu hướng đo thấp (bias âm)
- **Cải thiện MAE/RMSE**: Giảm sai số trung bình
- **Easy integration**: Không thay đổi model gốc
- **Production ready**: Có thể deploy ngay

## 📁 Files chính

### 1. `linear_calibration.py`
- **LinearCalibrator**: Core calibration class
- **CalibrationTrainer**: Helper để train calibration
- Implements: `y_calibrated = a * y_raw + b`

### 2. `test_calibrated_model.py`  
- **CalibratedModelTester**: Test calibrated model
- So sánh before/after calibration
- Tạo detailed comparison report

### 3. `calibrated_predictor.py`
- **CalibratedPredictor**: Production wrapper
- Dễ dàng integrate vào existing code
- Auto-load calibration model

## 🚀 Cách sử dụng

### Bước 1: Train Calibration
```bash
python linear_calibration.py
```

**Output:**
- Trained calibration model
- Evaluation metrics  
- Calibration plots
- Saved model: `models/linear_calibration.pkl`

### Bước 2: Test Calibrated Model
```bash
python test_calibrated_model.py
```

**Output:**
- Comparison report (calibrated vs uncalibrated)
- Per-video results
- Overall improvement metrics

### Bước 3: Use in Production
```python
from calibrated_predictor import CalibratedPredictor

# Initialize
predictor = CalibratedPredictor()

# Make prediction với calibration
result = predictor.predict(ppg_signal)
print(f"Systolic: {result['systolic']:.1f} mmHg")
print(f"Diastolic: {result['diastolic']:.1f} mmHg")
```

## 📊 Expected Results

### Before Calibration:
- **Systolic**: MAE=12.42, RMSE=15.38, Bias=-1.65
- **Diastolic**: MAE=15.76, RMSE=18.37, Bias=-4.14

### After Calibration (Expected):
- **Systolic**: MAE~8-10, RMSE~10-12, Bias~0
- **Diastolic**: MAE~10-12, RMSE~12-15, Bias~0

### Improvement targets:
- **MAE improvement**: 2-5 mmHg
- **Bias reduction**: Near zero
- **Clinical grade**: Từ Grade C/D → Grade B

## 🔧 Integration với existing code

### Option 1: Modify main.py
```python
from calibrated_predictor import predict_with_calibration

# In upload_ppg endpoint
output = predict_with_calibration(ppg_signal)
```

### Option 2: Wrapper approach
```python
from calibrated_predictor import CalibratedPredictor

predictor = CalibratedPredictor()
output = predictor.predict(ppg_signal)
```

## 📈 Monitoring & Validation

### Key metrics to track:
1. **MAE/RMSE improvement**
2. **Bias reduction** 
3. **Clinical accuracy grade**
4. **Per-video error distribution**

### Validation approach:
- **Train set (ID 4-9)**: Fit calibration
- **Test set (ID 1-3)**: Validate performance
- **Cross-validation**: Ensure robustness

## 🎛️ Tuning & Advanced Options

### Linear Calibration Parameters:
- **Systolic**: `y = a_sys * x + b_sys`
- **Diastolic**: `y = a_dia * x + b_dia`

### Future Extensions:
1. **Polynomial Calibration**: Non-linear relationships
2. **Robust Calibration**: Handle outliers better
3. **Individual Calibration**: Per-person calibration
4. **Ensemble Calibration**: Multiple calibration methods

## 🚨 Troubleshooting

### Common Issues:

**1. Calibration model not found**
```
⚠️  Could not load calibration: [Errno 2] No such file or directory
```
**Solution**: Run `python linear_calibration.py` first

**2. No improvement after calibration**
```
❌ CALIBRATION INEFFECTIVE - Limited improvement
```
**Solution**: Check data quality, try polynomial calibration

**3. Calibration makes results worse**
```
🔶 CALIBRATION PARTIALLY SUCCESSFUL - Some improvement
```
**Solution**: Review training data, check for overfitting

## 📋 Testing Checklist

- [ ] Train calibration: `python linear_calibration.py`
- [ ] Test calibrated model: `python test_calibrated_model.py`  
- [ ] Verify improvement: MAE reduction > 2 mmHg
- [ ] Check bias reduction: |Bias| < 2 mmHg
- [ ] Visual inspection: Calibration plots
- [ ] Production test: Import và sử dụng CalibratedPredictor

## 📞 Support

Nếu gặp vấn đề:
1. Check error logs
2. Verify data trong `calibration_data.csv`
3. Ensure all videos processed successfully
4. Review calibration parameters

**Expected processing time**: ~1-2 minutes total cho toàn bộ pipeline.

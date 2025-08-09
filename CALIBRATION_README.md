# 🩺 Linear Calibration System for Blood Pressure Measurement

## 📋 Mục lục
1. [Tổng quan Hệ thống](#1-tổng-quan-hệ-thống)
2. [Cấu trúc Data và Chuẩn bị](#2-cấu-trúc-data-và-chuẩn-bị)
3. [Hướng dẫn Thực hành](#3-hướng-dẫn-thực-hành)
4. [Luồng hoạt động Linear Calibration](#4-luồng-hoạt-động-linear-calibration)
5. [Metrics đánh giá và Ý nghĩa](#5-metrics-đánh-giá-và-ý-nghĩa)
6. [Phân tích kết quả chi tiết](#6-phân-tích-kết-quả-chi-tiết)
7. [Best Practices](#7-best-practices)

---

## 1. 🎯 Tổng quan Hệ thống

### 🤔 Vấn đề cần giải quyết
Hệ thống AI đo huyết áp từ video PPG có thể có **systematic bias** - tức là luôn đo cao hoặc thấp hơn máy đo chuyên dụng một cách nhất quán.

**Ví dụ:**
```
Video 1: Model đo 130/75, Thực tế 142/106 → Sai số: -12/-31
Video 2: Model đo 125/77, Thực tế 122/86  → Sai số: +3/-9
Video 3: Model đo 132/73, Thực tế 136/75  → Sai số: -4/-2
```

### 🎯 Mục tiêu Linear Calibration
Tìm ra **quy luật toán học** để điều chỉnh predictions của model sao cho gần với ground truth hơn:

```
Systolic_calibrated = a × Systolic_raw + b
Diastolic_calibrated = c × Diastolic_raw + d
```

---

## 2. 📁 Cấu trúc Data và Chuẩn bị

### 📂 Folder Structure
```
data/
├── val/                    # Training data để học calibration
│   ├── ground_truth.csv    # Ground truth values
│   ├── video_1.mov         # Video files
│   ├── video_2.mov
│   ├── video_3.mov
│   ├── video_4.mov
│   └── video_5.mp4
├── test/                   # Test data để đánh giá
│   ├── ground_truth.csv
│   ├── video_6.mov
│   └── video_7.mp4
```

### 📊 Format Ground Truth CSV
```csv
ID,video,may_sys,may_dia
1,video_1,142,106
2,video_2,122,86
3,video_3,136,75
```

**Giải thích columns:**
- `ID`: Định danh unique cho mỗi video
- `video`: Tên file video (không có extension)
- `may_sys`: Ground truth systolic từ máy đo chuyên dụng
- `may_dia`: Ground truth diastolic từ máy đo chuyên dụng

### 🎬 Yêu cầu Videos
- **Format hỗ trợ**: `.mov`, `.mp4`, `.avi`
- **Nội dung**: PPG signal từ camera smartphone
- **Tên file**: Phải match với column `video` trong CSV (có thể khác extension)

---

## 3. 🚀 Hướng dẫn Thực hành

### Chuẩn bị Data
1. **Tạo folders**: `data/val/` (training) và `data/test/` (testing)
2. **Đặt videos**: `.mov`, `.mp4` files vào các folders
3. **Tạo CSV**: `ground_truth.csv` trong mỗi folder với format:
   ```csv
   ID,video,may_sys,may_dia
   1,video_1,142,106
   2,video_2,122,86
   ```

### Chạy Calibration
```powershell
cd d:\Heart\heart_pressure
python folder_calibration_clean.py
```

### Xem Kết quả
- `val_model_results.csv` - Kết quả training data
- `test_model_results.csv` - Kết quả test data  
- `calibration_detailed_evaluation.csv` - So sánh before/after
- **MAE < 10 mmHg**: Good performance
- **Improvement > 0**: Calibration helps

---

## 4. 🔄 Luồng hoạt động Linear Calibration

### Step 1: 🏃‍♂️ Chạy Model trên VAL Data
```python
# Chạy model AI trên mỗi video trong folder val/
for video in val_videos:
    ppg_signal = extract_ppg_signal(video)
    prediction = ai_model.predict(ppg_signal)
    # prediction = {'systolic': 130.4, 'diastolic': 69.6, 'hr': 72}
```

**Output:**
```
Video 1: App=132.8/73.4, GT=142/106
Video 2: App=121.6/76.6, GT=122/86
Video 3: App=130.4/69.6, GT=136/75
...
```

### Step 2: 📈 Train Linear Calibration
Sử dụng **Linear Regression** để tìm coefficients:

```python
# Systolic calibration
X_sys = [132.8, 121.6, 130.4, ...]  # App predictions
y_sys = [142, 122, 136, ...]        # Ground truth
→ Tìm a, b sao cho: GT_sys = a × App_sys + b

# Diastolic calibration  
X_dia = [73.4, 76.6, 69.6, ...]     # App predictions
y_dia = [106, 86, 75, ...]          # Ground truth
→ Tìm c, d sao cho: GT_dia = c × App_dia + d
```

**Kết quả:**
```
Systolic: GT = 0.4129 × App + 80.5871
Diastolic: GT = 1.0549 × App + 14.0646
```

### Step 3: 🧪 Test trên TEST Data
```python
# Chạy model trên test videos
for video in test_videos:
    raw_prediction = ai_model.predict(video)
    
    # Apply calibration
    calibrated_sys = 0.4129 × raw_prediction.systolic + 80.5871
    calibrated_dia = 1.0549 × raw_prediction.diastolic + 14.0646
```

### Step 4: 📊 So sánh Before vs After
```
Video 6:
- Raw:        130.4/69.6 vs GT 120/61  → Error: +10.4/+8.6
- Calibrated: 134.4/87.4 vs GT 120/61  → Error: +14.4/+26.4

Video 7:
- Raw:        123.1/66.1 vs GT 127/72  → Error: -3.9/+5.9
- Calibrated: 131.0/83.8 vs GT 127/72  → Error: +4.0/+11.8
```

---

## 5. 📏 Metrics đánh giá và Ý nghĩa

### 🎯 MAE (Mean Absolute Error)
**Định nghĩa:** Trung bình của giá trị tuyệt đối sai số

```python
MAE = (|error_1| + |error_2| + ... + |error_n|) / n
```

**Ví dụ tính toán:**
```
Errors: [+10.4, -3.9] mmHg
MAE = (|10.4| + |-3.9|) / 2 = (10.4 + 3.9) / 2 = 7.15 mmHg
```

**Ý nghĩa:**
- ✅ **MAE thấp** = Model chính xác hơn
- ❌ **MAE cao** = Model sai lệch nhiều
- 🎯 **Mục tiêu:** MAE < 10 mmHg cho BP measurement

**Thang đánh giá MAE cho Blood Pressure:**
```
MAE < 5 mmHg    → 🟢 Excellent (Xuất sắc)
MAE 5-10 mmHg   → 🟡 Good (Tốt)  
MAE 10-15 mmHg  → 🟠 Fair (Chấp nhận được)
MAE > 15 mmHg   → 🔴 Poor (Kém)
```

### 📊 RMSE (Root Mean Square Error)
**Định nghĩa:** Căn bậc hai của trung bình bình phương sai số

```python
RMSE = sqrt((error_1² + error_2² + ... + error_n²) / n)
```

**Ví dụ tính toán:**
```
Errors: [+10.4, -3.9] mmHg
RMSE = sqrt((10.4² + (-3.9)²) / 2) = sqrt((108.16 + 15.21) / 2) = sqrt(61.69) = 7.85 mmHg
```

**Ý nghĩa:**
- 📏 **RMSE > MAE**: Có outliers (sai số lớn bất thường)
- 📏 **RMSE ≈ MAE**: Sai số đều đặn
- 🎯 **Penalty outliers**: RMSE phạt nặng những sai số lớn

### ⚖️ Bias (Systematic Error)
**Định nghĩa:** Trung bình của sai số (có dấu)

```python
Bias = (error_1 + error_2 + ... + error_n) / n
```

**Ví dụ tính toán:**
```
Errors: [+10.4, -3.9] mmHg
Bias = (10.4 + (-3.9)) / 2 = 6.5 / 2 = +3.25 mmHg
```

**Ý nghĩa Bias:**
- ➕ **Bias > 0**: Model có xu hướng đo **cao hơn** thực tế
- ➖ **Bias < 0**: Model có xu hướng đo **thấp hơn** thực tế  
- 🎯 **Bias ≈ 0**: Model không có systematic bias

**Thang đánh giá Bias:**
```
|Bias| < 2 mmHg    → 🟢 Excellent
|Bias| 2-5 mmHg    → 🟡 Good
|Bias| 5-10 mmHg   → 🟠 Fair
|Bias| > 10 mmHg   → 🔴 Poor
```

### 📈 Improvement Metrics
**Cách tính:**
```python
Improvement = MAE_before - MAE_after
```

**Ý nghĩa:**
- ✅ **Improvement > 0**: Calibration làm model tốt hơn
- ❌ **Improvement < 0**: Calibration làm model tệ hơn

---

## 6. 🔍 Phân tích kết quả chi tiết

### 📊 Ví dụ Output thực tế
```
BEFORE CALIBRATION:
  Systolic  - MAE: 7.43 mmHg, RMSE: 8.02 mmHg, Bias: +3.02 mmHg
  Diastolic - MAE: 11.81 mmHg, RMSE: 12.25 mmHg, Bias: +11.81 mmHg

AFTER CALIBRATION:
  Systolic  - MAE: 9.32 mmHg, RMSE: 10.64 mmHg, Bias: +9.32 mmHg
  Diastolic - MAE: 30.17 mmHg, RMSE: 30.40 mmHg, Bias: +30.17 mmHg

IMPROVEMENT:
  Systolic  MAE: -1.90 mmHg (❌ Worse)
  Diastolic MAE: -18.37 mmHg (❌ Worse)
```

### 🔍 Phân tích Step by Step

#### 1. **Systolic Analysis:**
```
Before: MAE = 7.43 mmHg (🟡 Good)
After:  MAE = 9.32 mmHg (🟡 Good, nhưng worse)
→ Kết luận: Raw model đã tốt, calibration không cần thiết
```

#### 2. **Diastolic Analysis:**
```
Before: MAE = 11.81 mmHg (🟠 Fair)
After:  MAE = 30.17 mmHg (🔴 Poor!)
→ Kết luận: Calibration làm tệ hơn rất nhiều, có vấn đề nghiêm trọng
```

#### 3. **Bias Analysis:**
```
Systolic:
  Before: +3.02 mmHg → Model đo cao hơn 3 mmHg
  After:  +9.32 mmHg → Calibration tăng bias lên 9 mmHg

Diastolic:
  Before: +11.81 mmHg → Model đo cao hơn 12 mmHg  
  After:  +30.17 mmHg → Calibration làm bias tệ hơn x2.5!
```

### ⚠️ Warning Signs
**Khi nào Calibration thất bại:**
1. **MAE tăng** thay vì giảm
2. **RMSE >> MAE** → Có outliers lớn
3. **Bias tăng** thay vì gần 0
4. **Improvement âm** → Model worse after calibration

### 🎯 Success Criteria
**Calibration thành công khi:**
```
✅ MAE giảm ít nhất 2-3 mmHg
✅ RMSE giảm tương ứng
✅ |Bias| gần 0 hơn
✅ Improvement > 0 cho cả systolic và diastolic
```

---

## 7. 🛠️ Best Practices

### 📊 Data Requirements
```
Minimum: 20+ videos cho training
Recommended: 50+ videos cho training
Test set: Ít nhất 10-20% của total data
```

### 🎯 Quality Checks
1. **Data Consistency:**
   ```python
   # Check range hợp lý
   assert 80 <= systolic <= 200
   assert 40 <= diastolic <= 120
   assert systolic > diastolic
   ```

2. **Cross-Validation:**
   ```python
   # Chia data thành multiple folds
   for fold in range(5):
       train_calibration(fold_train_data)
       evaluate_on(fold_test_data)
   ```

### 🚨 Common Pitfalls
1. **Overfitting:** Training data quá ít
2. **Distribution Mismatch:** Test data khác biệt training data
3. **Outliers:** Video quality kém ảnh hưởng calibration
4. **Non-linear Relationships:** Linear model không đủ

### 💡 Recommendations
```
✅ DO:
- Collect diverse calibration data
- Validate on independent test set  
- Monitor both MAE and bias
- Use cross-validation

❌ DON'T:
- Train with < 20 videos
- Ignore systematic bias
- Apply calibration blindly
- Skip validation step
```

---

## 🎉 Kết luận

Linear Calibration là một **công cụ mạnh mẽ** để improve accuracy của AI blood pressure model, nhưng cần:

1. **Sufficient training data** (≥20 videos)
2. **Proper validation methodology** 
3. **Careful analysis of metrics** (MAE, RMSE, Bias)
4. **Domain knowledge** về blood pressure measurement

**Trong trường hợp cụ thể này:** Raw model đã khá tốt (MAE 7-12 mmHg), linear calibration không improve được do data ít và overfitting.

---
# ABP Multi-Phase Grid Search System

## 🎯 **Mục tiêu**
Tối ưu hóa parameters để cải thiện Blood Pressure prediction accuracy (Systolic/Diastolic R²) từ âm sang dương thông qua 3-phase systematic optimization.

---

## 📁 **Files được tạo**

### **1. `abp_multiphase_gridsearch.py`** - Main Grid Search System
**Công dụng**: Multi-phase optimization system với progressive parameter expansion

**Đặc điểm:**
- ✅ **3 Phases**: Phase1 (4 params) → Phase2 (6 params) → Phase3 (8+ params)
- ✅ **Default-first strategy**: Giá trị mặc định luôn được test đầu tiên làm baseline
- ✅ **Progressive optimization**: Optimal parameters từ phase trước được dùng cho phase sau
- ✅ **ABP-focused**: Tập trung vào Systolic/Diastolic, bỏ qua Heart Rate
- ✅ **params.yaml integration**: Automatic backup/restore preprocessing parameters
- ✅ **MLflow tracking**: Separate experiments cho mỗi phase
- ✅ **CSV exports**: Detailed results với timestamp cho mỗi phase

### **2. `test_abp_multiphase.py`** - Test System
**Công dụng**: Quick validation của multi-phase logic với subset nhỏ

**Đặc điểm:**
- ⚡ **Fast testing**: 2 videos, reduced parameter ranges
- 🔧 **Flexible**: Test individual phases hoặc full pipeline
- 📊 **Quick analysis**: Immediate feedback về improvements
- 🧹 **Clean testing**: Automatic cleanup

---

## 🔧 **Parameter Strategy**

### **📊 Default Values (Baseline)**
```python
default_values = {
    'butter_lowpass_cutoff': 5,        # Current pipeline default
    'window_size_seconds': 1.01,       # params.yaml default
    'lpf_cutoff': 4,                   # params.yaml default
    'distance_multiplier': 1.0,        # No modification (default)
    'hpf_cutoff': 0.5,                # params.yaml default
    'bpf_multiplier': 3,              # params.yaml default
}
```

### **🎯 Phase 1: Critical ABP Parameters**
```python
param_grid = {
    'butter_lowpass_cutoff': [5, 3, 4, 6, 7],           # Default first
    'distance_multiplier': [1.0, 0.7, 0.8, 0.9, 1.1, 1.2],  # Default first
    'window_size_seconds': [1.01, 0.8, 1.2, 1.5],      # Default first
    'lpf_cutoff': [4, 3, 5, 6],                         # Default first
}
```
**Total: 5×6×4×4 = 480 combinations**

### **🔶 Phase 2: Extended Parameters**
- Optimal parameters từ Phase 1 + 2 additional parameters
- Tự động cập nhật optimal values làm default

### **🔹 Phase 3: Advanced Fine-tuning** 
- Optimal parameters từ Phase 2 + advanced parameters
- Fine-tuning cho final optimization

---

## 🚀 **Workflow sử dụng**

### **Step 1: Quick Test**
```bash
cd d:\NEW2_BF\heart_pressure
python test_abp_multiphase.py
```
**Options:**
- Option 1: Phase 1 only (~15 phút)
- Option 2: Phase 1+2 (~30 phút)  
- Option 3: All phases (~45 phút)

### **Step 2: Full Optimization**
```bash
python abp_multiphase_gridsearch.py
```
**Estimated time:**
- Phase 1: ~6-8 giờ (480 combinations)
- Phase 2: ~2-3 giờ (optimal params + extensions)
- Phase 3: ~1-2 giờ (fine-tuning)
- **Total: ~10-13 giờ**

### **Step 3: Results Analysis**
**Auto-generated files:**
- `abp_gridsearch_phase1_results_TIMESTAMP.csv`
- `abp_gridsearch_phase2_results_TIMESTAMP.csv`
- `abp_gridsearch_phase3_results_TIMESTAMP.csv`

**MLflow UI:**
```bash
mlflow ui --backend-store-uri file:///d:/NEW2_BF/heart_pressure/mlruns
```

---

## 📊 **Key Features**

### **🎯 ABP-Focused Metrics**
```python
# Primary metrics
abp_r2_combined = (systolic_r2 + diastolic_r2) / 2
abp_mae_combined = (systolic_mae + diastolic_mae) / 2
abp_combined_score = abp_r2_combined - (abp_mae_combined / 100)
```

### **🔄 Progressive Optimization Flow**
```
Phase 1: Test critical parameters → Find optimal_1
    ↓
Phase 2: Use optimal_1 + extended parameters → Find optimal_2  
    ↓
Phase 3: Use optimal_2 + advanced parameters → Find optimal_3
```

### **✅ Default-First Strategy**
- **Experiment 1**: Luôn là default parameters (baseline)
- **Experiments 2+**: Variations around default values
- **Comparison**: Immediate assessment vs baseline

---

## 📈 **Expected Results**

### **Current Performance (Baseline)**
- Systolic R²: ~0.003 (practically no correlation)
- Diastolic R²: ~-1.067 (worse than random)
- Combined ABP R²: ~-0.532

### **Target Performance (After Optimization)**
- Systolic R²: > 0.3 (acceptable correlation)
- Diastolic R²: > 0.2 (positive correlation)  
- Combined ABP R²: > 0.25

### **Success Criteria**
- ✅ R² values turn from negative to positive
- ✅ MAE reduction of 10-20%
- ✅ Consistent performance across videos
- ✅ Reproducible optimal parameter set

---

## 🔍 **Parameter Analysis Focus**

### **🔥 Critical Impact (Phase 1)**
1. **`butter_lowpass_cutoff`**: Final ABP smoothing before peak detection
2. **`distance_multiplier`**: Direct impact on SBP/DBP peak detection accuracy
3. **`window_size_seconds`**: PPG preprocessing smoothing strength
4. **`lpf_cutoff`**: Signal preprocessing filter cutoff

### **🔶 Secondary Impact (Phase 2)**
5. **`hpf_cutoff`**: Baseline drift removal
6. **`bpf_multiplier`**: BPM-based filtering strength

### **🔹 Fine-tuning (Phase 3)**
7. **`lpf_order`**: Filter characteristics
8. **`bpf_mincut`**: BPM filter minimum threshold

---

## 🎯 **Optimization Strategy**

### **Why Multi-Phase?**
- **Manageable complexity**: 480 → 200 → 100 combinations instead of 10,000+
- **Progressive refinement**: Focus narrows to most promising regions
- **Time efficiency**: ~13 hours instead of ~weeks
- **Clear insights**: Understanding which parameters matter most

### **Why Default-First?**
- **Immediate baseline**: Clear reference point cho improvements
- **Early detection**: Quick identification nếu không có improvement
- **Fair comparison**: All variations compared against same baseline

---

## 💡 **Usage Tips**

1. **Always test first**: Run test script để verify implementation
2. **Monitor progress**: Check intermediate CSV files và MLflow UI
3. **Resource planning**: Ensure 10-13 hours of uninterrupted runtime
4. **Backup important**: Original params.yaml được auto-backup
5. **Results analysis**: Use MLflow UI để compare experiments visually

---

**Tóm lại**: System này tạo ra một scientific approach để systematically improve ABP prediction accuracy thông qua multi-phase parameter optimization, với default-first strategy để ensure clear baseline comparison và progressive refinement để achieve optimal performance.

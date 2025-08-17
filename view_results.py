import pandas as pd
import os
from setting import BASE_DIR

def view_grid_search_results():
    """Xem kết quả Grid Search"""
    
    # 1. Đọc results từ CSV
    results_file = os.path.join(BASE_DIR, "grid_search_phase1_results.csv")
    
    if not os.path.exists(results_file):
        print("Chưa có file kết quả! Hãy chạy grid search trước.")
        return
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} experiments from {results_file}")
    
    # 2. Hiển thị top 10 kết quả tốt nhất
    print("\n" + "="*80)
    print("TOP 10 BEST RESULTS (sorted by combined_score - lower is better)")
    print("="*80)
    
    # Sort by combined score (lower is better)
    df_sorted = df.sort_values('combined_score')
    top_10 = df_sorted.head(10)
    
    for idx, row in top_10.iterrows():
        print(f"\nRank {list(top_10.index).index(idx) + 1}:")
        print(f"  Parameters:")
        print(f"    butter_lowpass_cutoff: {row['butter_lowpass_cutoff']}")
        print(f"    distance_ratio: {row['distance_ratio']}")
        print(f"    segment_seconds: {row['segment_seconds']}")
        print(f"  Metrics:")
        print(f"    Systolic MAE: {row['systolic_mae']:.2f}, R²: {row['systolic_r2']:.3f}")
        print(f"    Diastolic MAE: {row['diastolic_mae']:.2f}, R²: {row['diastolic_r2']:.3f}")
        print(f"    Combined Score: {row['combined_score']:.3f}")
        print(f"    Videos Processed: {row['videos_processed']}")
    
    # 3. Parameter impact analysis
    print("\n" + "="*80)
    print("PARAMETER IMPACT ANALYSIS")
    print("="*80)
    
    params = ['butter_lowpass_cutoff', 'distance_ratio', 'segment_seconds']
    
    for param in params:
        print(f"\n--- {param} ---")
        param_groups = df.groupby(param)['combined_score'].agg(['mean', 'std', 'count'])
        
        for value, stats in param_groups.iterrows():
            print(f"  {value}: Score {stats['mean']:.3f} (±{stats['std']:.3f}) [{stats['count']} experiments]")
    
    # 4. Best vs Current baseline comparison
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    best_row = df_sorted.iloc[0]
    worst_row = df_sorted.iloc[-1]
    
    print(f"Best experiment:")
    print(f"  Systolic MAE: {best_row['systolic_mae']:.2f} (R² = {best_row['systolic_r2']:.3f})")
    print(f"  Diastolic MAE: {best_row['diastolic_mae']:.2f} (R² = {best_row['diastolic_r2']:.3f})")
    print(f"  Combined Score: {best_row['combined_score']:.3f}")
    
    print(f"\nWorst experiment:")
    print(f"  Systolic MAE: {worst_row['systolic_mae']:.2f} (R² = {worst_row['systolic_r2']:.3f})")
    print(f"  Diastolic MAE: {worst_row['diastolic_mae']:.2f} (R² = {worst_row['diastolic_r2']:.3f})")
    print(f"  Combined Score: {worst_row['combined_score']:.3f}")
    
    improvement = worst_row['combined_score'] - best_row['combined_score']
    print(f"\nImprovement: {improvement:.3f} points")
    
    # 5. R² positive analysis
    positive_r2_sys = len(df[df['systolic_r2'] > 0])
    positive_r2_dia = len(df[df['diastolic_r2'] > 0])
    
    print(f"\nR² Analysis:")
    print(f"  Experiments with positive systolic R²: {positive_r2_sys}/{len(df)} ({positive_r2_sys/len(df)*100:.1f}%)")
    print(f"  Experiments with positive diastolic R²: {positive_r2_dia}/{len(df)} ({positive_r2_dia/len(df)*100:.1f}%)")
    
    # 6. Recommend best parameters
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print(f"🏆 Best parameter combination:")
    print(f"   butter_lowpass_cutoff = {best_row['butter_lowpass_cutoff']}")
    print(f"   distance_ratio = {best_row['distance_ratio']}")
    print(f"   segment_seconds = {best_row['segment_seconds']}")
    
    print(f"\n📊 Expected performance with best parameters:")
    print(f"   Systolic: MAE = {best_row['systolic_mae']:.2f} mmHg, R² = {best_row['systolic_r2']:.3f}")
    print(f"   Diastolic: MAE = {best_row['diastolic_mae']:.2f} mmHg, R² = {best_row['diastolic_r2']:.3f}")
    
    return df_sorted

def view_mlflow_results():
    """Hướng dẫn xem kết quả trong MLflow"""
    
    print("\n" + "="*80)
    print("MLFLOW RESULTS VIEWING")
    print("="*80)
    
    print("Để xem kết quả chi tiết trong MLflow UI:")
    print("1. Mở terminal")
    print("2. Chuyển đến thư mục dự án:")
    print("   cd D:\\NEW2_BF\\heart_pressure")
    print("3. Khởi động MLflow UI:")
    print("   C:/ProgramData/miniconda3/Scripts/conda.exe run -n heart_pressure_clean --no-capture-output mlflow ui")
    print("4. Mở browser và vào: http://localhost:5000")
    print("5. Chọn experiment 'Grid_Search_Phase1'")
    print("6. So sánh các runs và xem charts")
    
    print(f"\nMLflow files location: {os.path.join(BASE_DIR, 'mlruns')}")

if __name__ == "__main__":
    print("Grid Search Phase 1 - Results Viewer")
    print("="*50)
    
    # View CSV results
    df = view_grid_search_results()
    
    # MLflow viewing instructions
    view_mlflow_results()

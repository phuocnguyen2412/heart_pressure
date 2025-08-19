#!/usr/bin/env python3
"""
View Best Parameters from ABP Grid Search Results
Hiển thị tham số tốt nhất từ các file CSV results
"""

import pandas as pd
import os
from setting import BASE_DIR


def view_best_parameters():
    """Xem tham số tốt nhất từ tất cả phases"""
    print("🏆 ABP GRID SEARCH - BEST PARAMETERS ANALYSIS")
    print("=" * 60)
    
    phase_files = [
        "abp_gridsearch_phase1_results.csv",
        "abp_gridsearch_phase2_results.csv", 
        "abp_gridsearch_phase3_results.csv"
    ]
    
    all_best_results = []
    
    for phase_file in phase_files:
        file_path = os.path.join(BASE_DIR, phase_file)
        
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {phase_file}")
            continue
            
        print(f"\n📊 ANALYZING: {phase_file}")
        print("-" * 50)
        
        try:
            df = pd.read_csv(file_path)
            
            if len(df) == 0:
                print("❌ No data in file")
                continue
            
            # Find best result (highest abp_combined_score)
            best_idx = df['abp_combined_score'].idxmax()
            best_result = df.loc[best_idx]
            
            # Find baseline result (first row - default parameters)
            baseline_result = df.loc[0]
            
            print(f"🥇 BEST RESULT:")
            print(f"   ABP Combined Score: {best_result['abp_combined_score']:.4f}")
            print(f"   Systolic R²: {best_result['systolic_r2']:.4f}")
            print(f"   Diastolic R²: {best_result['diastolic_r2']:.4f}")
            print(f"   Videos Processed: {best_result['videos_processed']}")
            
            # Extract parameter columns (exclude metrics)
            param_columns = [col for col in df.columns if col not in [
                'phase', 'systolic_mae', 'systolic_rmse', 'systolic_r2',
                'diastolic_mae', 'diastolic_rmse', 'diastolic_r2',
                'abp_r2_combined', 'abp_mae_combined', 'abp_combined_score',
                'videos_processed'
            ]]
            
            print(f"\n🎯 BEST PARAMETERS:")
            for param in param_columns:
                if param in best_result.index:
                    print(f"   {param}: {best_result[param]}")
            
            print(f"\n📊 BASELINE COMPARISON:")
            print(f"   Baseline Score: {baseline_result['abp_combined_score']:.4f}")
            improvement = best_result['abp_combined_score'] - baseline_result['abp_combined_score']
            print(f"   Improvement: {improvement:+.4f}")
            
            # Store for overall analysis
            best_result_info = {
                'phase': phase_file.replace('.csv', '').replace('abp_gridsearch_', ''),
                'score': best_result['abp_combined_score'],
                'systolic_r2': best_result['systolic_r2'],
                'diastolic_r2': best_result['diastolic_r2'],
                'parameters': {param: best_result[param] for param in param_columns if param in best_result.index}
            }
            all_best_results.append(best_result_info)
            
        except Exception as e:
            print(f"❌ Error reading {phase_file}: {e}")
    
    # Overall best across all phases
    if all_best_results:
        print(f"\n" + "=" * 60)
        print(f"🏆 OVERALL BEST RESULT ACROSS ALL PHASES")
        print("=" * 60)
        
        overall_best = max(all_best_results, key=lambda x: x['score'])
        
        print(f"🥇 Best Phase: {overall_best['phase']}")
        print(f"🏆 Best Score: {overall_best['score']:.4f}")
        print(f"📈 Systolic R²: {overall_best['systolic_r2']:.4f}")
        print(f"📈 Diastolic R²: {overall_best['diastolic_r2']:.4f}")
        
        print(f"\n🎯 FINAL OPTIMAL PARAMETERS:")
        for param, value in overall_best['parameters'].items():
            print(f"   {param}: {value}")
        
        print(f"\n💡 TO USE THESE PARAMETERS:")
        print(f"   1. Copy parameters above")
        print(f"   2. Update your pipeline configuration")
        print(f"   3. Test with validation data")
        
        return overall_best
    else:
        print("❌ No results found in any CSV files")
        return None


def compare_all_phases():
    """So sánh kết quả tốt nhất của tất cả phases"""
    print(f"\n📊 PHASE COMPARISON")
    print("=" * 60)
    
    phase_files = [
        "abp_gridsearch_phase1_results.csv",
        "abp_gridsearch_phase2_results.csv", 
        "abp_gridsearch_phase3_results.csv"
    ]
    
    comparison_data = []
    
    for phase_file in phase_files:
        file_path = os.path.join(BASE_DIR, phase_file)
        
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path)
            if len(df) == 0:
                continue
                
            best_result = df.loc[df['abp_combined_score'].idxmax()]
            baseline_result = df.loc[0]
            
            phase_name = phase_file.replace('.csv', '').replace('abp_gridsearch_', '')
            
            comparison_data.append({
                'Phase': phase_name,
                'Best_Score': best_result['abp_combined_score'],
                'Baseline_Score': baseline_result['abp_combined_score'],
                'Improvement': best_result['abp_combined_score'] - baseline_result['abp_combined_score'],
                'Systolic_R2': best_result['systolic_r2'],
                'Diastolic_R2': best_result['diastolic_r2'],
                'Total_Experiments': len(df)
            })
            
        except Exception as e:
            print(f"Error processing {phase_file}: {e}")
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison
        comparison_file = os.path.join(BASE_DIR, "abp_phases_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\n💾 Comparison saved to: {comparison_file}")


if __name__ == "__main__":
    # View best parameters
    best_overall = view_best_parameters()
    
    # Compare phases
    compare_all_phases()
    
    print(f"\n🎉 Analysis completed!")
    print(f"💡 Check the CSV files for detailed results")

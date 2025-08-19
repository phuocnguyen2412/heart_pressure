#!/usr/bin/env python3
"""
Quick ABP Test - Chạy trong 3 phút
Test minimal parameters với 1 video để validate logic
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abp_multiphase_gridsearch import ABPMultiPhaseGridSearch
import pandas as pd
import time


class QuickABPTest(ABPMultiPhaseGridSearch):
    def __init__(self):
        super().__init__()
        
        # Override với minimal parameter grids (chỉ 2-3 combinations mỗi phase)
        self.phase_configs = {
            'phase1': {
                'name': 'ABP_Quick_Test_Phase1',
                'description': 'Quick test với 2 parameters minimal',
                'param_grid': {
                    # Chỉ test 2 values: default và 1 variation
                    'butter_lowpass_cutoff': [5, 4],  # Default: 5, test: 4
                    'distance_multiplier': [1.0, 0.9],  # Default: 1.0, test: 0.9
                }
            },
            'phase2': {
                'name': 'ABP_Quick_Test_Phase2', 
                'description': 'Quick test phase2 với optimal từ phase1',
                'param_grid': {
                    # From Phase 1 (will be updated)
                    'butter_lowpass_cutoff': [],
                    'distance_multiplier': [],
                    
                    # Add 1 additional parameter
                    'window_size_seconds': [1.01, 1.2],  # Default: 1.01, test: 1.2
                }
            }
        }
        
        # Chỉ test 2 phases để tiết kiệm thời gian
        # Bỏ phase3 để test nhanh hơn
    
    def run_quick_test(self):
        """
        Quick test chỉ với 1 video và minimal parameters
        Target: 3 phút hoàn thành
        """
        print(f"\\n⚡ QUICK ABP TEST - 3 MINUTES TARGET")
        print(f"🎯 Goal: Validate multi-phase logic quickly")
        print(f"📹 Using: 1 video only")
        print(f"⏱️ Expected: ~3 minutes")
        
        start_time = time.time()
        
        # Load validation data và chỉ lấy 1 video đầu tiên
        df = pd.read_csv(self.val_csv)
        if len(df) > 1:
            df = df.head(3)  # Chỉ lấy 1 video
            print(f"📋 Using only 1 video: {df.iloc[0]['video']}")
        
        # Temporarily save single video dataset
        test_csv = self.val_csv.replace('.csv', '_quicktest.csv')
        df.to_csv(test_csv, index=False)
        original_val_csv = self.val_csv
        self.val_csv = test_csv
        
        try:
            all_results = {}
            
            # Phase 1: 2×2 = 4 combinations với 1 video = ~1.5 phút
            print(f"\\n🔥 QUICK PHASE 1 (4 combinations)")
            print(f"📊 Combinations: 2×2 = 4")
            print(f"⏱️ Expected: ~1.5 minutes")
            
            phase1_results = self.run_phase('phase1')
            all_results['phase1'] = phase1_results
            
            phase1_time = time.time() - start_time
            print(f"✅ Phase 1 completed in {phase1_time:.1f} seconds")
            
            if phase1_results and len(phase1_results) >= 2:  # Cần ít nhất 2 results để compare
                # Find optimal từ Phase 1
                optimal_phase1 = self.find_optimal_parameters(phase1_results)
                print(f"🏆 Phase 1 Optimal: {optimal_phase1}")
                
                # Update Phase 2 với optimal parameters
                self.update_phase_config_with_optimal('phase2', optimal_phase1)
                
                # Phase 2: optimal + 1 new param = ~1.5 phút
                print(f"\\n🔥 QUICK PHASE 2")
                print(f"⏱️ Expected: ~1.5 minutes")
                
                phase2_results = self.run_phase('phase2')
                all_results['phase2'] = phase2_results
                
                if phase2_results:
                    optimal_phase2 = self.find_optimal_parameters(phase2_results)
                    print(f"🏆 Phase 2 Optimal: {optimal_phase2}")
            
            # Quick Analysis
            total_time = time.time() - start_time
            print(f"\\n📊 QUICK TEST RESULTS")
            print(f"⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            
            for phase_key, results in all_results.items():
                if results and len(results) >= 1:
                    best_result = max(results, key=lambda x: x['abp_combined_score'])
                    baseline_result = results[0]
                    
                    print(f"\\n{phase_key.upper()}:")
                    print(f"  ✅ Experiments: {len(results)}")
                    print(f"  🏆 Best Score: {best_result['abp_combined_score']:.4f}")
                    print(f"  📊 Baseline Score: {baseline_result['abp_combined_score']:.4f}")
                    improvement = best_result['abp_combined_score'] - baseline_result['abp_combined_score']
                    print(f"  🚀 Improvement: {improvement:+.4f}")
                    print(f"  🎯 Best Params: {best_result['params']}")
            
            # Success validation
            success_criteria = [
                len(all_results) >= 1,  # At least 1 phase completed
                total_time <= 300,  # Under 5 minutes (3 min target + buffer)
                all(len(results) > 0 for results in all_results.values()),  # All phases have results
            ]
            
            if all(success_criteria):
                print(f"\\n✅ QUICK TEST SUCCESSFUL!")
                print(f"🎯 Multi-phase logic is working correctly")
                print(f"⚡ Completed in {total_time:.1f}s (target: 180s)")
                print(f"🚀 Ready for full optimization:")
                print(f"   python abp_multiphase_gridsearch.py")
                return True
            else:
                print(f"\\n⚠️ QUICK TEST HAD ISSUES")
                print(f"❌ Check implementation before full run")
                return False
                
        except Exception as e:
            print(f"\\n❌ QUICK TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Cleanup
            self.val_csv = original_val_csv
            if os.path.exists(test_csv):
                os.remove(test_csv)
            
            # Restore params.yaml if needed
            self.restore_params_yaml()


def main():
    """Run Quick ABP Test"""
    print("⚡ ABP Quick Test - 3 Minutes Target")
    print("=" * 40)
    
    # Initialize quick test
    quick_test = QuickABPTest()
    
    # Show test plan
    print(f"\\n📋 Test Plan:")
    print(f"  📹 Videos: 1 video only")
    print(f"  🔬 Phase 1: 4 combinations (2×2)")
    print(f"  🔬 Phase 2: ~4 combinations (optimal + variations)")
    print(f"  ⏱️ Target time: 3 minutes")
    print(f"  🎯 Goal: Validate multi-phase logic")
    
    # Confirm before running
    proceed = input(f"\\nProceed with quick test? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("Test cancelled.")
        return
    
    # Run quick test
    success = quick_test.run_quick_test()
    
    if success:
        print(f"\\n🎉 Quick test completed successfully!")
        print(f"💡 Tips for full run:")
        print(f"   - Ensure 10-13 hours of uninterrupted time")
        print(f"   - Monitor progress via CSV files")
        print(f"   - Check MLflow UI for detailed tracking")
    else:
        print(f"\\n🔧 Please fix issues before running full optimization")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test ABP Multi-Phase Grid Search với subset nhỏ
Quick validation trước khi chạy full optimization
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abp_multiphase_gridsearch import ABPMultiPhaseGridSearch
import pandas as pd


class ABPMultiPhaseTest(ABPMultiPhaseGridSearch):
    def __init__(self):
        super().__init__()
        
        # Override với test parameter grids nhỏ hơn
        self.phase_configs = {
            'phase1': {
                'name': 'ABP_Test_Phase1',
                'description': 'Test 4 tham số ABP với values nhỏ',
                'param_grid': {
                    # Chỉ test 2-3 values mỗi parameter (default first)
                    'butter_lowpass_cutoff': [5, 4, 6],  # Default: 5
                    'distance_multiplier': [1.0, 0.9, 1.1],  # Default: 1.0
                    'window_size_seconds': [1.01, 1.2],  # Default: 1.01
                    'lpf_cutoff': [4, 5],  # Default: 4
                }
            },
            'phase2': {
                'name': 'ABP_Test_Phase2',
                'description': 'Test phase2 với optimal từ phase1',
                'param_grid': {
                    # From Phase 1 (will be updated)
                    'butter_lowpass_cutoff': [],
                    'distance_multiplier': [],
                    'window_size_seconds': [],
                    'lpf_cutoff': [],
                    
                    # Additional parameters (smaller ranges)
                    'hpf_cutoff': [0.5, 0.7],  # Default: 0.5
                    'bpf_multiplier': [3, 2.5],  # Default: 3
                }
            },
            'phase3': {
                'name': 'ABP_Test_Phase3',
                'description': 'Test phase3 với optimal từ phase2',
                'param_grid': {
                    # From Phase 1&2 (will be updated)
                    'butter_lowpass_cutoff': [],
                    'distance_multiplier': [],
                    'window_size_seconds': [],
                    'lpf_cutoff': [],
                    'hpf_cutoff': [],
                    'bpf_multiplier': [],
                    
                    # Additional parameters
                    'lpf_order': [2, 3],  # Default: 2
                    'bpf_mincut': [0.01, 0.02],  # Default: 0.01
                }
            }
        }
    
    def run_test_with_limited_videos(self, max_videos=2, test_phases=['phase1']):
        """
        Test multi-phase grid search với limited videos và phases
        
        Args:
            max_videos: Maximum number of videos to process per experiment
            test_phases: List of phases to test ['phase1', 'phase2', 'phase3']
        """
        print(f"\\n🧪 ABP MULTI-PHASE GRID SEARCH TEST")
        print(f"📹 Testing with maximum {max_videos} videos per experiment")
        print(f"🔬 Testing phases: {test_phases}")
        
        # Load and limit validation data
        df = pd.read_csv(self.val_csv)
        if len(df) > max_videos:
            df = df.head(max_videos)
            print(f"📋 Limited to first {max_videos} videos for testing")
        
        # Temporarily save limited dataset
        test_csv = self.val_csv.replace('.csv', '_test.csv')
        df.to_csv(test_csv, index=False)
        original_val_csv = self.val_csv
        self.val_csv = test_csv
        
        try:
            all_results = {}
            
            # Test each requested phase
            for phase_key in test_phases:
                if phase_key not in self.phase_configs:
                    print(f"⚠️ Unknown phase: {phase_key}")
                    continue
                
                print(f"\\n🔥 Testing {phase_key.upper()}")
                
                # Calculate expected combinations
                phase_config = self.phase_configs[phase_key]
                param_grid = phase_config['param_grid']
                
                # Count non-empty parameter lists
                total_combinations = 1
                for param_name, values in param_grid.items():
                    if len(values) > 0:
                        total_combinations *= len(values)
                
                print(f"📊 Expected combinations: {total_combinations}")
                print(f"⏱️ Estimated time: ~{total_combinations * max_videos * 0.5:.1f} minutes")
                
                # Run phase
                if phase_key == 'phase1':
                    phase_results = self.run_phase('phase1')
                    all_results['phase1'] = phase_results
                    
                    if phase_results:
                        # Update phase2 with optimal parameters
                        optimal_phase1 = self.find_optimal_parameters(phase_results)
                        print(f"🏆 Test Phase 1 Optimal: {optimal_phase1}")
                        
                        if 'phase2' in test_phases:
                            self.update_phase_config_with_optimal('phase2', optimal_phase1)
                
                elif phase_key == 'phase2' and 'phase1' in all_results:
                    phase_results = self.run_phase('phase2')
                    all_results['phase2'] = phase_results
                    
                    if phase_results:
                        optimal_phase2 = self.find_optimal_parameters(phase_results)
                        print(f"🏆 Test Phase 2 Optimal: {optimal_phase2}")
                        
                        if 'phase3' in test_phases:
                            self.update_phase_config_with_optimal('phase3', optimal_phase2)
                
                elif phase_key == 'phase3' and 'phase2' in all_results:
                    phase_results = self.run_phase('phase3')
                    all_results['phase3'] = phase_results
                    
                    if phase_results:
                        optimal_phase3 = self.find_optimal_parameters(phase_results)
                        print(f"🏆 Test Phase 3 Optimal: {optimal_phase3}")
            
            # Test Analysis
            if all_results:
                print(f"\\n📊 TEST RESULTS SUMMARY")
                for phase_key, results in all_results.items():
                    if results:
                        best_result = max(results, key=lambda x: x['abp_combined_score'])
                        baseline_result = results[0]
                        
                        print(f"\\n{phase_key.upper()}:")
                        print(f"  ✅ Experiments completed: {len(results)}")
                        print(f"  🏆 Best ABP Score: {best_result['abp_combined_score']:.4f}")
                        print(f"  📊 Baseline ABP Score: {baseline_result['abp_combined_score']:.4f}")
                        print(f"  🚀 Improvement: {best_result['abp_combined_score'] - baseline_result['abp_combined_score']:+.4f}")
                        print(f"  🎯 Best Params: {best_result['params']}")
                
                print(f"\\n✅ ABP Multi-Phase Test completed successfully!")
                print(f"🚀 Ready to run full optimization:")
                print(f"   python abp_multiphase_gridsearch.py")
                
                return all_results
            else:
                print("❌ No test results generated")
                return None
                
        finally:
            # Restore original validation file path and cleanup
            self.val_csv = original_val_csv
            if os.path.exists(test_csv):
                os.remove(test_csv)


def main():
    """Run ABP Multi-Phase Grid Search test"""
    print("🧪 ABP Multi-Phase Grid Search - Quick Test")
    print("=" * 50)
    
    # Initialize test
    test_optimizer = ABPMultiPhaseTest()
    
    # Test options
    print("\\n🔧 Test Options:")
    print("1. Quick test (Phase 1 only, 2 videos)")
    print("2. Medium test (Phase 1+2, 2 videos)")
    print("3. Full test (All phases, 2 videos)")
    
    choice = input("\\nChoose test option (1-3): ").strip()
    
    if choice == "1":
        test_results = test_optimizer.run_test_with_limited_videos(
            max_videos=2, test_phases=['phase1']
        )
    elif choice == "2":
        test_results = test_optimizer.run_test_with_limited_videos(
            max_videos=2, test_phases=['phase1', 'phase2']
        )
    elif choice == "3":
        test_results = test_optimizer.run_test_with_limited_videos(
            max_videos=2, test_phases=['phase1', 'phase2', 'phase3']
        )
    else:
        print("Invalid choice, running quick test...")
        test_results = test_optimizer.run_test_with_limited_videos(
            max_videos=2, test_phases=['phase1']
        )
    
    if test_results:
        print(f"\\n🎯 Test successful! Multi-phase logic is working correctly.")
    else:
        print(f"\\n⚠️ Test had issues. Please check the implementation.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ABP Grid Search - Multi-Phase Optimization
Tập trung vào Blood Pressure prediction, bỏ qua Heart Rate parameters
Giá trị mặc định luôn được test đầu tiên để làm baseline
"""

import os
import pandas as pd
import numpy as np
from itertools import product
import mlflow
import mlflow.sklearn
from datetime import datetime
import json
import yaml
import shutil
from pprint import pprint

from main_pipeline import BloodPressureInferencePipeline
from setting import BASE_DIR, config
from evaluate import compute_metrics
import tempfile


class ABPMultiPhaseGridSearch:
    def __init__(self):
        """Initialize ABP-focused Grid Search với default values đầu tiên"""
        
        # ABP Default values (current pipeline settings)
        self.default_values = {
            'butter_lowpass_cutoff': 5,        # Current default in processor
            'window_size_seconds': 1.01,       # Current default in params.yaml
            'lpf_cutoff': 4,                   # Current default in params.yaml
            'distance_multiplier': 1.0,        # Current default (no modification)
            'hpf_cutoff': 0.5,                # Current default in params.yaml
            'bpf_multiplier': 3,              # Current default in params.yaml
        }
        
        # Phase definitions - Progressive parameter expansion
        self.phase_configs = {
            'phase1': {
                'name': 'ABP_Critical_Parameters_Phase1',
                'description': '4 tham số quan trọng nhất cho ABP prediction',
                'param_grid': {
                    # Butter lowpass cutoff - CRITICAL for final ABP quality
                    'butter_lowpass_cutoff': [5, 3, 4, 6, 7],  # Default first
                    
                    # Distance multiplier - CRITICAL for peak detection
                    'distance_multiplier': [1.0, 0.7, 0.8, 0.9, 1.1, 1.2],  # Default first
                    
                    # Window size - Important for preprocessing smoothing
                    'window_size_seconds': [1.01, 0.8, 1.2, 1.5],  # Default first
                    
                    # LPF cutoff - Important for signal preprocessing  
                    'lpf_cutoff': [4, 3, 5, 6],  # Default first
                }
            },
            'phase2': {
                'name': 'ABP_Extended_Parameters_Phase2', 
                'description': '6 tham số bao gồm phase1 + filtering parameters',
                'param_grid': {
                    # From Phase 1 (will be set to optimal values)
                    'butter_lowpass_cutoff': [],  # Will be filled from Phase 1 results
                    'distance_multiplier': [],
                    'window_size_seconds': [],
                    'lpf_cutoff': [],
                    
                    # Additional Phase 2 parameters
                    'hpf_cutoff': [0.5, 0.3, 0.7, 1.0],  # Default first
                    'bpf_multiplier': [3, 2, 2.5, 3.5, 4],  # Default first
                }
            },
            'phase3': {
                'name': 'ABP_Advanced_Parameters_Phase3',
                'description': '8+ tham số cho fine-tuning cuối cùng',
                'param_grid': {
                    # From Phase 1&2 (will be set to optimal values)
                    'butter_lowpass_cutoff': [],
                    'distance_multiplier': [],
                    'window_size_seconds': [],
                    'lpf_cutoff': [],
                    'hpf_cutoff': [],
                    'bpf_multiplier': [],
                    
                    # Additional Phase 3 parameters
                    'lpf_order': [2, 1, 3, 4],  # Default first
                    'bpf_mincut': [0.01, 0.005, 0.02, 0.05],  # Default first
                }
            }
        }
        
        # MLflow setup
        mlflow.set_tracking_uri("file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/"))
        
        # Dataset paths
        self.val_csv = os.path.join(BASE_DIR, "data", "val.csv")
        self.video_folder = os.path.join(BASE_DIR, "data", "video")
        
        # Results storage
        self.phase_results = {}
        
        # Backup params.yaml
        self.params_yaml_path = os.path.join(BASE_DIR, "signal_extractor", "params.yaml")
        self.params_backup_path = self.params_yaml_path + ".backup"
        
    def backup_params_yaml(self):
        """Backup original params.yaml"""
        if not os.path.exists(self.params_backup_path):
            shutil.copy(self.params_yaml_path, self.params_backup_path)
            print(f"✅ Backed up params.yaml")
    
    def restore_params_yaml(self):
        """Restore original params.yaml"""
        if os.path.exists(self.params_backup_path):
            shutil.copy(self.params_backup_path, self.params_yaml_path)
    
    def update_params_yaml(self, params):
        """Update params.yaml with preprocessing parameters"""
        try:
            with open(self.params_yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Update preprocessing parameters
            if 'window_size_seconds' in params:
                yaml_config['preprocessor']['filter_chains'][0]['flist'][0]['params']['window_size_seconds'] = params['window_size_seconds']
            
            if 'lpf_cutoff' in params:
                yaml_config['preprocessor']['filter_chains'][0]['flist'][2]['params']['low'] = params['lpf_cutoff']
                
            if 'lpf_order' in params:
                yaml_config['preprocessor']['filter_chains'][0]['flist'][2]['params']['filter_order'] = params['lpf_order']
            
            if 'hpf_cutoff' in params:
                yaml_config['preprocessor']['filter_chains'][1]['flist'][1]['params']['cutoff'] = params['hpf_cutoff']
            
            if 'bpf_multiplier' in params:
                yaml_config['preprocessor']['filter_chains'][1]['flist'][2]['params']['multiplier'] = params['bpf_multiplier']
                
            if 'bpf_mincut' in params:
                yaml_config['preprocessor']['filter_chains'][1]['flist'][2]['params']['mincut'] = params['bpf_mincut']
            
            with open(self.params_yaml_path, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False)
                
        except Exception as e:
            print(f"❌ Error updating params.yaml: {e}")
    
    def create_parameter_combinations(self, phase_config):
        """
        Tạo parameter combinations với default values đầu tiên
        """
        param_grid = phase_config['param_grid']
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        all_combinations = list(product(*param_values))
        
        # Tìm combination với all default values (should be first)
        default_combination = tuple(self.default_values.get(name, param_grid[name][0]) 
                                   for name in param_names)
        
        # Ensure default is first
        combinations = []
        if default_combination in all_combinations:
            combinations.append(default_combination)
            # Add remaining combinations
            combinations.extend([c for c in all_combinations if c != default_combination])
        else:
            # If exact default not found, add it first anyway
            combinations = [default_combination] + all_combinations
        
        # Convert to dict format
        param_combinations = []
        for combination in combinations:
            param_dict = dict(zip(param_names, combination))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def run_single_experiment(self, params, phase_name):
        """Chạy 1 experiment với ABP parameters"""
        print(f"\\n{'='*60}")
        print(f"Testing ABP parameters: {params}")
        print(f"{'='*60}")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("phase", phase_name)
            
            try:
                # Backup and update params.yaml
                self.backup_params_yaml()
                self.update_params_yaml(params)
                
                # Initialize pipeline
                pipeline = BloodPressureInferencePipeline(extract_config=config)
                
                # Load validation data
                df = pd.read_csv(self.val_csv)
                print(f"Processing {len(df)} validation videos...")
                
                predictions = []
                
                for idx, row in df.iterrows():
                    video_path = os.path.join(self.video_folder, row["video"])
                    print(f"[{idx+1}/{len(df)}] Processing: {row['video']}")
                    
                    if not os.path.exists(video_path):
                        print(f"⚠️ Video not found: {video_path}")
                        continue
                    
                    try:
                        # Predict with ABP parameters
                        predict_data = self.predict_with_abp_params(pipeline, video_path, params)
                        
                        predictions.append({
                            'diastolic': predict_data["diastolic"],
                            'systolic': predict_data["systolic"],
                            'hr': predict_data["hr"],
                            'mean': predict_data["mean"],
                            'true_diastolic': row["may_dia"],
                            'true_systolic': row["may_sys"],
                        })
                        
                    except Exception as e:
                        print(f"❌ Error processing {video_path}: {e}")
                        continue
                
                # Restore params.yaml
                self.restore_params_yaml()
                
                # Compute ABP metrics
                if len(predictions) > 0:
                    pred_df = pd.DataFrame(predictions)
                    
                    # Calculate ABP metrics (focus on systolic/diastolic)
                    systolic_true = pred_df["true_systolic"].tolist()
                    systolic_pred = pred_df["systolic"].tolist()
                    diastolic_true = pred_df["true_diastolic"].tolist()
                    diastolic_pred = pred_df["diastolic"].tolist()
                    
                    result_systolic = compute_metrics(systolic_true, systolic_pred)
                    result_diastolic = compute_metrics(diastolic_true, diastolic_pred)
                    
                    # Log ABP metrics to MLflow
                    mlflow.log_metric("systolic_mae", result_systolic["mae"])
                    mlflow.log_metric("systolic_rmse", result_systolic["rmse"])
                    mlflow.log_metric("systolic_r2", result_systolic["r2"])
                    mlflow.log_metric("diastolic_mae", result_diastolic["mae"])
                    mlflow.log_metric("diastolic_rmse", result_diastolic["rmse"])
                    mlflow.log_metric("diastolic_r2", result_diastolic["r2"])
                    mlflow.log_metric("videos_processed", len(predictions))
                    
                    # ABP Combined Score - Prioritize R² improvement
                    abp_r2_combined = (result_systolic["r2"] + result_diastolic["r2"]) / 2
                    abp_mae_combined = (result_systolic["mae"] + result_diastolic["mae"]) / 2
                    
                    # Combined score: Higher R² is better, Lower MAE is better
                    abp_combined_score = abp_r2_combined - (abp_mae_combined / 100)  # Scale MAE
                    
                    mlflow.log_metric("abp_r2_combined", abp_r2_combined)
                    mlflow.log_metric("abp_mae_combined", abp_mae_combined)
                    mlflow.log_metric("abp_combined_score", abp_combined_score)
                    
                    print(f"\\n=== ABP RESULTS ===")
                    print(f"Systolic - MAE: {result_systolic['mae']:.2f}, R²: {result_systolic['r2']:.4f}")
                    print(f"Diastolic - MAE: {result_diastolic['mae']:.2f}, R²: {result_diastolic['r2']:.4f}")
                    print(f"Combined ABP R²: {abp_r2_combined:.4f}")
                    print(f"Combined ABP Score: {abp_combined_score:.4f}")
                    print(f"Videos processed: {len(predictions)}")
                    
                    return {
                        'phase': phase_name,
                        'params': params,
                        'systolic_metrics': result_systolic,
                        'diastolic_metrics': result_diastolic,
                        'abp_r2_combined': abp_r2_combined,
                        'abp_mae_combined': abp_mae_combined,
                        'abp_combined_score': abp_combined_score,
                        'videos_processed': len(predictions)
                    }
                else:
                    print("❌ No videos processed successfully!")
                    mlflow.log_metric("videos_processed", 0)
                    return None
                    
            except Exception as e:
                print(f"❌ Experiment failed: {e}")
                mlflow.log_param("error", str(e))
                self.restore_params_yaml()
                return None
    
    def predict_with_abp_params(self, pipeline, video_path, params):
        """Predict với ABP parameters (params.yaml đã được update)"""
        # Extract PPG signal (với updated preprocessing parameters)
        ppg_signal = pipeline.bp_extractor.extract_ppg_from_video(video_path)
        
        # Normalize PPG
        min_ppg, max_ppg = pipeline.meta["min_ppg"], pipeline.meta["max_ppg"]
        min_abp, max_abp = pipeline.meta["min_abp"], pipeline.meta["max_abp"]
        ppg_signal = pipeline.processor.clip_to_range(ppg_signal, min_ppg, max_ppg)
        ppg_norm = pipeline.processor.min_max_scaler(ppg_signal, min_ppg, max_ppg)
        
        # Predict ABP
        refined_abp = pipeline.predict_abp_from_ppg(ppg_norm)
        
        # Denormalize ABP
        abp_pred = pipeline.processor.inverse_min_max_scaler(refined_abp, min_abp, max_abp)
        
        # Apply custom butter lowpass filter
        cutoff = params.get('butter_lowpass_cutoff', 5)
        abp_pred = pipeline.processor.butter_lowpass_filter(
            abp_pred.flatten(), fs=125, cutoff=cutoff
        )
        
        # Calculate heart rate for distance calculation
        hr = pipeline.calculate_heart_rate(abp_pred)
        beat_interval_sec = 60.0 / hr
        
        # Apply custom distance multiplier for peak detection
        distance_multiplier = params.get('distance_multiplier', 1.0)
        distance = int(beat_interval_sec * 125 * distance_multiplier)
        distance = max(20, distance)
        
        # Extract SBP, DBP với custom distance
        sbp_vals, dbp_vals, sbp_idx, dbp_idx = pipeline.extract_sbp_dbp(
            abp_pred, distance=distance
        )
        
        if len(sbp_vals) == 0 or len(dbp_vals) == 0:
            print("⚠️ No peaks detected, using fallback values")
            sbp, dbp = 120, 80
        else:
            sbp, dbp = np.mean(sbp_vals), np.mean(dbp_vals)
        
        return {
            "systolic": sbp,
            "diastolic": dbp,
            "hr": hr,
            "mean": (2 * dbp + sbp) / 3,
        }
    
    def run_phase(self, phase_key):
        """Chạy một phase của Grid Search"""
        phase_config = self.phase_configs[phase_key]
        print(f"\\n{'='*80}")
        print(f"🚀 STARTING {phase_key.upper()}: {phase_config['name']}")
        print(f"📋 Description: {phase_config['description']}")
        print(f"{'='*80}")
        
        # Set MLflow experiment
        mlflow.set_experiment(phase_config['name'])
        
        # Create parameter combinations (default first)
        param_combinations = self.create_parameter_combinations(phase_config)
        total_combinations = len(param_combinations)
        
        print(f"📊 Total combinations to test: {total_combinations}")
        print(f"🔄 Default parameters will be tested FIRST")
        
        # Show parameter grid        print(f"\\n📋 Parameter grid:")
        for param, values in phase_config['param_grid'].items():
            print(f"  {param}: {values}")
        
        results = []
        
        # Initialize CSV file with header (if not exists)
        self.initialize_csv_file(phase_key)
        
        # Run experiments
        for i, params in enumerate(param_combinations):
            print(f"\\n{'='*80}")
            print(f"🧪 EXPERIMENT {i+1}/{total_combinations} ({phase_key})")
            if i == 0:
                print("🔥 BASELINE (Default Parameters)")
            print(f"{'='*80}")
            
            result = self.run_single_experiment(params, phase_key)
            if result:
                results.append(result)
                
                # Append this single result to CSV immediately
                self.append_single_result_to_csv(phase_key, result)
                print(f"💾 Progress saved: {len(results)}/{total_combinations} experiments")
        
        print(f"\\n{'='*80}")
        print(f"✅ {phase_key.upper()} COMPLETED!")
        print(f"💾 Final CSV: abp_gridsearch_{phase_key}_results.csv ({len(results)} experiments)")
        print(f"{'='*80}")
        
        # Store phase results
        self.phase_results[phase_key] = results
        return results
    
    def initialize_csv_file(self, phase_key):
        """Initialize CSV file with header if it doesn't exist"""
        results_file = os.path.join(BASE_DIR, f"abp_gridsearch_{phase_key}_results.csv")
        
        if not os.path.exists(results_file):
            # Create empty CSV with header
            header_row = {
                'phase': '',
                'butter_lowpass_cutoff': '',
                'distance_multiplier': '',
                'window_size_seconds': '',
                'lpf_cutoff': '',
                'hpf_cutoff': '',
                'bpf_multiplier': '',
                'lpf_order': '',
                'bpf_mincut': '',
                'systolic_mae': '',
                'systolic_mse': '',
                'systolic_rmse': '',
                'systolic_r2': '',
                'diastolic_mae': '',
                'diastolic_mse': '',
                'diastolic_rmse': '',
                'diastolic_r2': '',
                'abp_r2_combined': '',
                'abp_mae_combined': '',
                'abp_combined_score': '',
                'videos_processed': ''
            }
            
            # Write header only
            df_header = pd.DataFrame([header_row])
            df_header.to_csv(results_file, index=False, header=True)
            # Remove the empty data row, keep only header
            df_empty = pd.DataFrame(columns=list(header_row.keys()))
            df_empty.to_csv(results_file, index=False)
            print(f"📄 Created CSV file: {results_file}")
    
    def append_single_result_to_csv(self, phase_key, result):
        """Append a single experiment result to CSV file"""
        results_file = os.path.join(BASE_DIR, f"abp_gridsearch_{phase_key}_results.csv")
        
        # Prepare single row
        row = {}
        row['phase'] = result['phase']
        
        # Add all possible parameters (fill with empty if not present)
        param_keys = ['butter_lowpass_cutoff', 'distance_multiplier', 'window_size_seconds', 
                     'lpf_cutoff', 'hpf_cutoff', 'bpf_multiplier', 'lpf_order', 'bpf_mincut']
        for key in param_keys:
            row[key] = result['params'].get(key, '')
        
        # Add metrics
        row.update({f"systolic_{k}": v for k, v in result['systolic_metrics'].items()})
        row.update({f"diastolic_{k}": v for k, v in result['diastolic_metrics'].items()})
        row['abp_r2_combined'] = result['abp_r2_combined']
        row['abp_mae_combined'] = result['abp_mae_combined']
        row['abp_combined_score'] = result['abp_combined_score']
        row['videos_processed'] = result['videos_processed']
        
        # Append to existing CSV
        df_new_row = pd.DataFrame([row])
        
        # Check if file exists and has data
        if os.path.exists(results_file):
            # Read existing data
            try:
                df_existing = pd.read_csv(results_file)
                # Append new row
                df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
            except pd.errors.EmptyDataError:
                # File exists but empty, just use new row
                df_combined = df_new_row
        else:
            # File doesn't exist, create with new row
            df_combined = df_new_row
        
        # Save back to CSV
        df_combined.to_csv(results_file, index=False)
        
        # Don't print every time to avoid spam, just return
        return results_file

    def save_phase_results(self, phase_key, results):
        """Save phase results to CSV (final results only)"""
        if not results:
            return
        
        # Simple filename without timestamp since we only save once per phase
        results_file = os.path.join(BASE_DIR, f"abp_gridsearch_{phase_key}_results.csv")
        
        rows = []
        for result in results:
            row = {}
            row['phase'] = result['phase']
            row.update(result['params'])
            row.update({f"systolic_{k}": v for k, v in result['systolic_metrics'].items()})
            row.update({f"diastolic_{k}": v for k, v in result['diastolic_metrics'].items()})
            row['abp_r2_combined'] = result['abp_r2_combined']
            row['abp_mae_combined'] = result['abp_mae_combined']
            row['abp_combined_score'] = result['abp_combined_score']
            row['videos_processed'] = result['videos_processed']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(results_file, index=False)
        print(f"💾 Final {phase_key} results saved to: {results_file}")
    
    def find_optimal_parameters(self, phase_results):
        """Tìm optimal parameters từ phase results"""
        if not phase_results:
            return None
        
        # Find best result based on ABP combined score
        best_result = max(phase_results, key=lambda x: x['abp_combined_score'])
        return best_result['params']
    
    def update_phase_config_with_optimal(self, phase_key, optimal_params):
        """Update phase config với optimal parameters từ phase trước"""
        phase_config = self.phase_configs[phase_key]
        
        for param_name, optimal_value in optimal_params.items():
            if param_name in phase_config['param_grid']:
                # Set optimal value as first value (default)
                current_values = phase_config['param_grid'][param_name]
                if len(current_values) == 0:  # Empty list, need to fill
                    # Add optimal value + some variations
                    if param_name == 'butter_lowpass_cutoff':
                        phase_config['param_grid'][param_name] = [optimal_value, optimal_value-1, optimal_value+1]
                    elif param_name == 'distance_multiplier':
                        phase_config['param_grid'][param_name] = [optimal_value, optimal_value-0.1, optimal_value+0.1]
                    elif param_name == 'window_size_seconds':
                        phase_config['param_grid'][param_name] = [optimal_value, optimal_value-0.2, optimal_value+0.2]
                    elif param_name == 'lpf_cutoff':
                        phase_config['param_grid'][param_name] = [optimal_value, optimal_value-1, optimal_value+1]
                    else:
                        phase_config['param_grid'][param_name] = [optimal_value]
                else:
                    # Ensure optimal value is first
                    if optimal_value in current_values:
                        current_values.remove(optimal_value)
                    phase_config['param_grid'][param_name] = [optimal_value] + current_values
    
    def run_multi_phase_optimization(self):
        """
        Chạy multi-phase optimization:
        Phase 1 -> Find optimal -> Phase 2 -> Find optimal -> Phase 3
        """
        print(f"\\n🎯 STARTING ABP MULTI-PHASE GRID SEARCH OPTIMIZATION")
        print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target: Improve Systolic/Diastolic R² from negative to positive")
        print(f"🚫 Excluding: Heart Rate optimization (focus on ABP only)")
        
        all_results = {}
        
        # Phase 1: Critical ABP Parameters
        print(f"\\n🔥 Phase 1: Critical ABP Parameters")
        phase1_results = self.run_phase('phase1')
        all_results['phase1'] = phase1_results
        
        if phase1_results:
            # Find optimal parameters from Phase 1
            optimal_phase1 = self.find_optimal_parameters(phase1_results)
            print(f"\\n🏆 Phase 1 Optimal Parameters: {optimal_phase1}")
            
            # Update Phase 2 config with optimal parameters
            self.update_phase_config_with_optimal('phase2', optimal_phase1)
            
            # Phase 2: Extended Parameters
            print(f"\\n🔥 Phase 2: Extended ABP Parameters")
            phase2_results = self.run_phase('phase2')
            all_results['phase2'] = phase2_results
            
            if phase2_results:
                # Find optimal parameters from Phase 2
                optimal_phase2 = self.find_optimal_parameters(phase2_results)
                print(f"\\n🏆 Phase 2 Optimal Parameters: {optimal_phase2}")
                
                # Update Phase 3 config with optimal parameters
                self.update_phase_config_with_optimal('phase3', optimal_phase2)
                
                # Phase 3: Advanced Fine-tuning
                print(f"\\n🔥 Phase 3: Advanced ABP Fine-tuning")
                phase3_results = self.run_phase('phase3')
                all_results['phase3'] = phase3_results
                
                if phase3_results:
                    optimal_phase3 = self.find_optimal_parameters(phase3_results)
                    print(f"\\n🏆 Phase 3 Final Optimal Parameters: {optimal_phase3}")
        
        # Final Analysis
        self.analyze_all_phases(all_results)
        
        return all_results
    
    def analyze_all_phases(self, all_results):
        """Phân tích kết quả từ tất cả phases"""
        print(f"\\n{'='*80}")
        print(f"📊 FINAL ABP OPTIMIZATION ANALYSIS")
        print(f"{'='*80}")
        
        for phase_key, results in all_results.items():
            if not results:
                continue
                
            print(f"\\n🔍 {phase_key.upper()} ANALYSIS:")
            
            # Find best result
            best_result = max(results, key=lambda x: x['abp_combined_score'])
            baseline_result = results[0]  # First result should be baseline
            
            print(f"\\n📈 Best Performance:")
            print(f"  Parameters: {best_result['params']}")
            print(f"  ABP Combined Score: {best_result['abp_combined_score']:.4f}")
            print(f"  Systolic R²: {best_result['systolic_metrics']['r2']:.4f}")
            print(f"  Diastolic R²: {best_result['diastolic_metrics']['r2']:.4f}")
            
            print(f"\\n📊 Baseline Performance:")
            print(f"  Parameters: {baseline_result['params']}")
            print(f"  ABP Combined Score: {baseline_result['abp_combined_score']:.4f}")
            print(f"  Systolic R²: {baseline_result['systolic_metrics']['r2']:.4f}")
            print(f"  Diastolic R²: {baseline_result['diastolic_metrics']['r2']:.4f}")
            
            # Improvement calculation
            improvement = best_result['abp_combined_score'] - baseline_result['abp_combined_score']
            print(f"\\n🚀 Improvement: {improvement:+.4f}")
            
        print(f"\\n✅ ABP Multi-Phase Optimization Completed!")
        print(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main execution function"""
    print("🎯 ABP Multi-Phase Grid Search Optimization")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = ABPMultiPhaseGridSearch()
    
    # Run multi-phase optimization
    all_results = optimizer.run_multi_phase_optimization()
    
    print("\\n🎉 ABP Grid Search optimization completed successfully!")
    return all_results


if __name__ == "__main__":
    main()

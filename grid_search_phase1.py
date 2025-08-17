import os
import pandas as pd
import numpy as np
from itertools import product
import mlflow
import mlflow.sklearn
from datetime import datetime
import json
from pprint import pprint

from main_pipeline import BloodPressureInferencePipeline
from setting import BASE_DIR, config
from evaluate import compute_metrics
import tempfile

class GridSearchPhase1:
    def __init__(self):
        # Phase 1: Tham số quan trọng nhất
        self.param_grid = {
            'butter_lowpass_cutoff': [3, 4, 5, 6],          # Processor filter cutoff
            'distance_ratio': [0.6, 0.8, 1.0],             # Peak detection distance ratio
            'segment_seconds': [6, 8, 10],                  # Video segment length
        }
        
        # Calculate total combinations
        self.total_combinations = np.prod([len(v) for v in self.param_grid.values()])
        print(f"Total combinations to test: {self.total_combinations}")
        
        # MLflow setup
        mlflow.set_tracking_uri("file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/"))
        mlflow.set_experiment("Grid_Search_Phase1")
        
        # Dataset paths
        self.val_csv = os.path.join(BASE_DIR, "data", "val.csv")
        self.video_folder = os.path.join(BASE_DIR, "data", "video")
        
    def create_custom_config(self, params):
        """Tạo config với tham số custom"""
        custom_config = config
        # Modify config với parameters
        if 'segment_seconds' in params:
            custom_config.segment_seconds = params['segment_seconds']
        return custom_config
        
    def run_single_experiment(self, params):
        """Chạy 1 experiment với tham số cụ thể"""
        print(f"\n{'='*60}")
        print(f"Testing parameters: {params}")
        print(f"{'='*60}")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            
            try:
                # Create custom config
                custom_config = self.create_custom_config(params)
                
                # Initialize pipeline với custom config
                pipeline = BloodPressureInferencePipeline(extract_config=custom_config)
                
                # Modify pipeline parameters
                if 'butter_lowpass_cutoff' in params:
                    pipeline.processor.default_cutoff = params['butter_lowpass_cutoff']
                
                # Load validation data
                df = pd.read_csv(self.val_csv)
                print(f"Processing {len(df)} validation videos...")
                
                diastolic_preds = []
                systolic_preds = []
                
                for idx, row in df.iterrows():
                    video_path = os.path.join(self.video_folder, row["video"])
                    print(f"[{idx+1}/{len(df)}] Processing: {row['video']}")
                    
                    try:
                        # Predict với custom parameters
                        predict_data = self.predict_with_params(pipeline, video_path, params)
                        
                        diastolic_preds.append(predict_data["diastolic"])
                        systolic_preds.append(predict_data["systolic"])
                        
                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
                        # Skip video nếu có lỗi
                        continue
                
                # Compute metrics
                if len(systolic_preds) > 0:
                    result_systolic = compute_metrics(df["may_sys"].tolist()[:len(systolic_preds)], systolic_preds)
                    result_diastolic = compute_metrics(df["may_dia"].tolist()[:len(diastolic_preds)], diastolic_preds)
                    
                    # Log metrics
                    mlflow.log_metric("systolic_mae", result_systolic["mae"])
                    mlflow.log_metric("systolic_rmse", result_systolic["rmse"])
                    mlflow.log_metric("systolic_r2", result_systolic["r2"])
                    mlflow.log_metric("diastolic_mae", result_diastolic["mae"])
                    mlflow.log_metric("diastolic_rmse", result_diastolic["rmse"])
                    mlflow.log_metric("diastolic_r2", result_diastolic["r2"])
                    mlflow.log_metric("videos_processed", len(systolic_preds))
                    
                    # Combined score (lower is better)
                    combined_score = (result_systolic["mae"] + result_diastolic["mae"]) - (result_systolic["r2"] + result_diastolic["r2"])
                    mlflow.log_metric("combined_score", combined_score)
                    
                    print(f"\n=== RESULTS ===")
                    print(f"Systolic - MAE: {result_systolic['mae']:.2f}, R²: {result_systolic['r2']:.3f}")
                    print(f"Diastolic - MAE: {result_diastolic['mae']:.2f}, R²: {result_diastolic['r2']:.3f}")
                    print(f"Combined Score: {combined_score:.3f}")
                    print(f"Videos processed: {len(systolic_preds)}")
                    
                    return {
                        'params': params,
                        'systolic_metrics': result_systolic,
                        'diastolic_metrics': result_diastolic,
                        'combined_score': combined_score,
                        'videos_processed': len(systolic_preds)
                    }
                else:
                    print("No videos processed successfully!")
                    mlflow.log_metric("videos_processed", 0)
                    return None
                    
            except Exception as e:
                print(f"Experiment failed: {e}")
                mlflow.log_param("error", str(e))
                return None
    
    def predict_with_params(self, pipeline, video_path, params):
        """Predict với custom parameters"""
        # Extract PPG
        ppg_signal = pipeline.bp_extractor.extract_ppg_from_video(video_path)
        
        # Normalize PPG
        min_ppg, max_ppg = pipeline.meta["min_ppg"], pipeline.meta["max_ppg"]
        min_abp, max_abp = pipeline.meta["min_abp"], pipeline.meta["max_abp"]
        ppg_signal = pipeline.processor.clip_to_range(ppg_signal, min_ppg, max_ppg)
        ppg_norm = pipeline.processor.min_max_scaler(ppg_signal, min_ppg, max_ppg)
        
        # Predict ABP
        refined_abp = pipeline.predict_abp_from_ppg(ppg_norm)
        
        # Denormalize với custom cutoff
        abp_pred = pipeline.processor.inverse_min_max_scaler(refined_abp, min_abp, max_abp)
        
        # Apply custom butter lowpass filter
        cutoff = params.get('butter_lowpass_cutoff', 5)
        abp_pred = pipeline.processor.butter_lowpass_filter(
            abp_pred.flatten(), fs=125, cutoff=cutoff
        )
        
        # Calculate heart rate
        hr = pipeline.calculate_heart_rate(abp_pred)
        beat_interval_sec = 60.0 / hr
        
        # Apply custom distance ratio
        distance_ratio = params.get('distance_ratio', 1.0)
        distance = int(beat_interval_sec * 125 * distance_ratio)
        distance = max(20, distance)
        
        # Extract SBP, DBP với custom distance
        sbp_vals, dbp_vals, sbp_idx, dbp_idx = pipeline.extract_sbp_dbp(
            abp_pred, distance=distance
        )
        sbp, dbp = np.mean(sbp_vals), np.mean(dbp_vals)
        
        return {
            "systolic": sbp,
            "diastolic": dbp,
            "hr": hr,
            "mean": (2 * dbp + sbp) / 3,
        }
    
    def run_grid_search(self):
        """Chạy full grid search"""
        print(f"Starting Grid Search Phase 1...")
        print(f"Parameter grid: {self.param_grid}")
        print(f"Total experiments: {self.total_combinations}")
        
        results = []
        
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        for i, combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, combination))
            
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {i+1}/{self.total_combinations}")
            print(f"{'='*80}")
            
            result = self.run_single_experiment(params)
            if result:
                results.append(result)
                
                # Save intermediate results
                self.save_results(results)
        
        print(f"\n{'='*80}")
        print("GRID SEARCH COMPLETED!")
        print(f"{'='*80}")
        
        return results
    
    def save_results(self, results):
        """Save results to CSV"""
        if not results:
            return
            
        results_file = os.path.join(BASE_DIR, "grid_search_phase1_results.csv")
        
        rows = []
        for result in results:
            row = {}
            row.update(result['params'])
            row.update({f"systolic_{k}": v for k, v in result['systolic_metrics'].items()})
            row.update({f"diastolic_{k}": v for k, v in result['diastolic_metrics'].items()})
            row['combined_score'] = result['combined_score']
            row['videos_processed'] = result['videos_processed']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
    
    def analyze_results(self, results):
        """Phân tích kết quả"""
        if not results:
            print("No results to analyze!")
            return
            
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        # Find best parameters
        best_result = min(results, key=lambda x: x['combined_score'])
        print(f"Best parameters: {best_result['params']}")
        print(f"Best combined score: {best_result['combined_score']:.3f}")
        print(f"Systolic R²: {best_result['systolic_metrics']['r2']:.3f}")
        print(f"Diastolic R²: {best_result['diastolic_metrics']['r2']:.3f}")
        
        # Parameter importance analysis
        print(f"\n--- Parameter Impact Analysis ---")
        for param_name in self.param_grid.keys():
            param_impact = {}
            for result in results:
                param_val = result['params'][param_name]
                if param_val not in param_impact:
                    param_impact[param_val] = []
                param_impact[param_val].append(result['combined_score'])
            
            print(f"\n{param_name}:")
            for val, scores in param_impact.items():
                avg_score = np.mean(scores)
                print(f"  {val}: {avg_score:.3f} (±{np.std(scores):.3f})")

if __name__ == "__main__":
    grid_search = GridSearchPhase1()
    results = grid_search.run_grid_search()
    grid_search.analyze_results(results)

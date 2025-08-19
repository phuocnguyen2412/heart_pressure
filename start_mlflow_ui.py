#!/usr/bin/env python3
"""
Start MLflow UI để xem kết quả Grid Search
"""

import os
import subprocess
from setting import BASE_DIR

def start_mlflow_ui():
    """Start MLflow UI"""
    mlruns_path = os.path.join(BASE_DIR, "mlruns")
    
    print("🚀 Starting MLflow UI...")
    print(f"📁 MLruns path: {mlruns_path}")
    print("🌐 URL: http://localhost:5000")
    print("🔍 Experiments to check:")
    print("   - ABP_Critical_Parameters_Phase1")
    print("   - ABP_Extended_Parameters_Phase2") 
    print("   - ABP_Advanced_Parameters_Phase3")
    print("\n💡 To stop: Press Ctrl+C")
    print("-" * 50)
    
    try:
        # Change to BASE_DIR and start MLflow UI
        os.chdir(BASE_DIR)
        subprocess.run(["mlflow", "ui", "--backend-store-uri", f"file:///{mlruns_path.replace(os.sep, '/')}", "--port", "5000"])
    except KeyboardInterrupt:
        print("\n✅ MLflow UI stopped")
    except Exception as e:
        print(f"❌ Error starting MLflow UI: {e}")
        print("💡 Make sure MLflow is installed: pip install mlflow")

if __name__ == "__main__":
    start_mlflow_ui()

#!/usr/bin/env python3
"""
Launch Professional US Stock Analyzer
"""

import subprocess
import sys
import os

def main():
    print("🚀 Launching Professional US Stock Analyzer...")
    print("📊 Real Data | Multiple Valuation Methods | AI-Optimized Models")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("professional_us_stock_analyzer.py"):
        print("❌ professional_us_stock_analyzer.py not found!")
        print("Please run this script from the directory containing the analyzer.")
        return
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "professional_us_stock_analyzer.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Professional US Stock Analyzer stopped.")
    except Exception as e:
        print(f"❌ Error launching analyzer: {e}")

if __name__ == "__main__":
    main()
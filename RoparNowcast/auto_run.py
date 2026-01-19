import time
import sys
import os
from datetime import datetime, timedelta

# Ensure we can import from the 'src' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing prediction logic
from src.nowcast_predict import run_prediction

def start_automation():
    print("🤖 [AUTO-PILOT] RoparNowcast System Activated")
    print("   -> Mode: Rapid Update Cycle (5 Minutes)")
    print("   -> Press 'Ctrl + C' to stop.")
    print("-" * 50)

    while True:
        try:
            start_time = datetime.now()
            print(f"\n⚡ Cycle Started: {start_time.strftime('%H:%M:%S')}")
            
            # 1. Run the Prediction
            run_prediction()
            
            # 2. Calculate Wait Time
            # We want to run exactly every 5 minutes (300 seconds)
            elapsed = (datetime.now() - start_time).total_seconds()
            wait_time = max(0, 300 - elapsed)
            
            next_run = datetime.now() + timedelta(seconds=wait_time)
            print(f"✅ Cycle Complete. Next run at: {next_run.strftime('%H:%M:%S')}")
            print(f"💤 Sleeping for {int(wait_time)} seconds...")
            
            time.sleep(wait_time)

        except KeyboardInterrupt:
            print("\n🛑 System stopped by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Critical Error: {e}")
            print("   -> Retrying in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    start_automation()
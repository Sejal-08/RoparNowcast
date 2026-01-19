import os
import sys

# Ensure project root is in path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("⚡ ROPAR NOWCAST: SYSTEM INITIALIZATION")
    print("=====================================")
    
    # 1. Train Models
    print("\n[1/2] 🧠 Training Models from Scratch...")
    print("      (This builds fresh XGBoost models from your data/clean_training_data.csv)")
    try:
        from src.nowcast_train import run_training
        run_training()
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Generate First Forecast
    print("\n[2/2] 🚀 Generating First Forecast...")
    print("      (Fetching live data and running inference)")
    try:
        from src.nowcast_predict import run_prediction
        run_prediction()
    except Exception as e:
        print(f"❌ Prediction Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ INITIALIZATION COMPLETE!")
    print("=====================================")
    print("To view the dashboard, run this command in a new terminal:")
    print("   streamlit run dashboard.py")
    print("\nTo keep the forecast updating automatically, run:")
    print("   python auto_run.py")

if __name__ == "__main__":
    main()

import requests
import json
from datetime import datetime, timedelta

# CONFIG
SOURCE_API_URL = "https://gtk47vexob.execute-api.us-east-1.amazonaws.com/campusdata"
DEVICE_ID = "1"

def debug_api():
    # 1. Fetch Yesterday's Data
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=1)
    s_str, e_str = start_dt.strftime("%d-%m-%Y"), end_dt.strftime("%d-%m-%Y")
    
    print(f"🔎 DEBUGGING: Fetching {s_str} to {e_str}...")

    params = {"deviceid": DEVICE_ID, "startdate": s_str, "enddate": e_str}
    resp = requests.get(SOURCE_API_URL, params=params)
    
    # 2. Check Raw Response
    print(f"✅ Status Code: {resp.status_code}")
    
    try:
        data = resp.json()
    except:
        print("❌ CRITICAL: Response is not JSON!")
        print(resp.text[:500])
        return

    # 3. Check Data Structure
    if isinstance(data, str):
        print("⚠️ NOTE: Data was stringified JSON. Parsing now...")
        data = json.loads(data)
        
    print(f"📊 Data Type: {type(data)}")
    
    if isinstance(data, list):
        print(f"📉 Rows Found: {len(data)}")
        if len(data) > 0:
            print("\n👀 FIRST ROW SAMPLE:")
            print(json.dumps(data[0], indent=4))
        else:
            print("⚠️ The list is empty. No data for these dates.")
            
    elif isinstance(data, dict):
        print("⚠️ Data is a DICTIONARY, not a list. Keys found:", data.keys())
        # Often APIs return {"body": [...]} or {"data": [...]}
        if "body" in data:
            print("💡 Found 'body' key. Parsing inner content...")
            inner_data = json.loads(data["body"]) if isinstance(data["body"], str) else data["body"]
            print(f"📉 Rows in Body: {len(inner_data)}")
            if len(inner_data) > 0:
                print(json.dumps(inner_data[0], indent=4))

if __name__ == "__main__":
    debug_api()
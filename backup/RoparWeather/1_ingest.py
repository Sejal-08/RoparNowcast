import requests
import boto3
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
import time

# ================================
# CONFIG
# ================================
SOURCE_API_URL = "https://gtk47vexob.execute-api.us-east-1.amazonaws.com/campusdata"
DEVICE_ID = "1"
DYNAMO_TABLE = "RoparWeather_History"
REGION = "us-east-1"
MONTHS_BACK = 6

def get_weekly_chunks(days_back=180):
    """Generates (start, end) date tuples for 7-day chunks."""
    end_date = datetime.now()
    current_start = end_date - timedelta(days=days_back)
    
    chunks = []
    while current_start < end_date:
        chunk_end = min(current_start + timedelta(days=7), end_date)
        chunks.append((
            current_start.strftime("%d-%m-%Y"),
            chunk_end.strftime("%d-%m-%Y")
        ))
        current_start = chunk_end + timedelta(days=1) # Move to next day
    return chunks

def bulk_ingest():
    dynamodb = boto3.resource('dynamodb', region_name=REGION)
    table = dynamodb.Table(DYNAMO_TABLE)
    chunks = get_weekly_chunks(days_back=30 * MONTHS_BACK)
    
    print(f"🔄 Starting Bulk Ingest: {len(chunks)} weeks to process...")

    total_records = 0
    
    for start_str, end_str in chunks:
        print(f"   📡 Fetching {start_str} to {end_str}...", end=" ")
        
        try:
            # 1. Fetch from API
            params = {"deviceid": DEVICE_ID, "startdate": start_str, "enddate": end_str}
            resp = requests.get(SOURCE_API_URL, params=params)
            
            # Handle Response Formats
            raw_data = resp.json()
            if isinstance(raw_data, str):
                import json
                raw_data = json.loads(raw_data)
                
            items = raw_data.get('items', [])
            
            if not items:
                print("⚠️ No data.")
                continue

            # 2. Upload to DynamoDB (Batch)
            with table.batch_writer() as batch:
                added = 0
                for row in items:
                    try:
                        # Clean Key Fields
                        ts = pd.to_datetime(row['TimeStamp']).strftime("%Y-%m-%dT%H:%M:%S")
                        
                        # Prepare Item (Convert all to string -> Decimal for safety)
                        item = {
                            'device_id': str(DEVICE_ID),
                            'timestamp': ts,
                            'temp': Decimal(str(row.get('CurrentTemperature', 0))),
                            'humidity': Decimal(str(row.get('CurrentHumidity', 0))),
                            'pressure': Decimal(str(row.get('AtmPressure', 0))),
                            'rain': Decimal(str(row.get('RainfallHourly', 0))),
                            'wind_speed': Decimal(str(row.get('WindSpeed', 0))),
                            'light': Decimal(str(row.get('LightIntensity', 0))),
                            'wind_dir': Decimal(str(row.get('WindDirection', 0)))
                        }
                        
                        batch.put_item(Item=item)
                        added += 1
                    except:
                        continue
            
            print(f"✅ Uploaded {added} records.")
            total_records += added
            
            # Sleep slightly to be kind to the API
            time.sleep(1)

        except Exception as e:
            print(f"❌ Failed: {e}")

    print(f"\n🎉 DONE! Total records uploaded: {total_records}")

if __name__ == "__main__":
    bulk_ingest()
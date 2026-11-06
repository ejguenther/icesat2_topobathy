import requests
import json
import time
import os

# --- Configurations ---
PLANET_API_KEY = "" #deleted key for now
out_dir = 'lovers_key_time_series'

session = requests.Session()
session.auth = (PLANET_API_KEY, "")

# Define your Area of Interest (AOI) - 2 sq km polygon
geojson_geometry = {
  "type": "Polygon",
  "coordinates": [
    [
      [-81.89526551915193, 26.38581406503307],  # 1. Bottom-Left
      [-81.86803167978749, 26.38581406503307],  # 2. Bottom-Right
      [-81.86803167978749, 26.40506008304391],  # 3. Top-Right
      [-81.89526551915193, 26.40506008304391],  # 4. Top-Left
      [-81.89526551915193, 26.38581406503307]   # 5. Bottom-Left (Closes the loop)
    ]
  ]
}

# --- 1. SEARCH: Build the Filters ---

geometry_filter = {
  "type": "GeometryFilter",
  "field_name": "geometry",
  "config": geojson_geometry
}

# Expanded Window: Full 3 years (2022 to 2024)
date_filter = {
  "type": "DateRangeFilter",
  "field_name": "acquired",
  "config": {
    "gte": "2022-01-01T00:00:00.000Z",
    "lte": "2024-12-31T23:59:59.000Z"
  }
}

cloud_cover_filter = {
  "type": "RangeFilter",
  "field_name": "cloud_cover",
  "config": {
    "lte": 0.15
  }
}

combined_filter = {
  "type": "AndFilter",
  "config": [geometry_filter, date_filter, cloud_cover_filter]
}

search_request = {
  "item_types": ["PSScene"],
  "filter": combined_filter
}

# --- Execute Search with Pagination Handling ---
print("Searching for imagery from 2022 to 2024...")
response = session.post('https://api.planet.com/data/v1/quick-search', json=search_request)
search_results = response.json()

# Grab initial page of IDs
item_ids = [feature['id'] for feature in search_results.get('features', [])]

# UPGRADE 1: Follow "_next" links using GET until all pages are exhausted
next_url = search_results.get('_links', {}).get('_next')
while next_url:
    print(f"Fetching next page of results... (Found {len(item_ids)} images so far)")
    response = session.get(next_url)
    search_results = response.json()
    item_ids.extend([feature['id'] for feature in search_results.get('features', [])])
    next_url = search_results.get('_links', {}).get('_next')

print(f"\nTotal images found matching your criteria: {len(item_ids)}")

if len(item_ids) == 0:
    print("No images found. Exiting.")
    exit()

# --- 2. ORDER: Handle Batching to Avoid Payload Overload ---
# UPGRADE 2: Split total list into safe chunks of 400 items per order
chunk_size = 400
item_id_chunks = [item_ids[i:i + chunk_size] for i in range(0, len(item_ids), chunk_size)]
order_ids = []

print(f"Splitting search results into {len(item_id_chunks)} order batch(es).")

for idx, chunk in enumerate(item_id_chunks):
    order_payload = {
        "name": f"Time_Series_Batch_{idx+1}",
        "products": [
            {
                "item_ids": chunk,
                "item_type": "PSScene",
                "product_bundle": "visual"  # <-- SWITCHED TO VISUAL: Presentation-ready RGB files
            }
        ],
        "tools": [
            {
                "clip": {
                    "aoi": geojson_geometry
                }
            }
        ]
    }

    print(f"Placing order for Batch {idx+1}/{len(item_id_chunks)}...")
    order_response = session.post('https://api.planet.com/compute/ops/orders/v2', json=order_payload)
    order_info = order_response.json()
    
    if 'id' in order_info:
        order_ids.append(order_info['id'])
    else:
        print(f"Error placing order for batch {idx+1}: {order_info}")

# --- 3. DOWNLOAD: Monitor and Download All Batches ---
print(f"\nMonitoring {len(order_ids)} orders for completion...")

for order_id in order_ids:
    order_url = f"https://api.planet.com/compute/ops/orders/v2/{order_id}"
    
    while True:
        print(f"Checking status for Order {order_id}...")
        check_response = session.get(order_url)
        order_status = check_response.json()
        status = order_status.get('state')
        
        if status == 'success':
            print(f"Order {order_id} is ready! Downloading assets...")
            results = order_status['_links']['results']
            
            for result in results:
                download_url = result['location']
                filename = result['name']
                file_path = os.path.join(out_dir, filename)
                
                # Dynamic directory patch to fix previous FileNotFoundError
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                print(f"Downloading {filename}...")
                r = session.get(download_url, allow_redirects=True)
                with open(file_path, 'wb') as f:
                    f.write(r.content)
            break
        elif status in ['failed', 'partial']:
            print(f"Order {order_id} failed or partially failed. Status: {status}")
            break
        else:
            print(f"Order status is '{status}'. Waiting 30 seconds...")
            time.sleep(30)

print("\nAll downloads completed successfully! Your data is ready in 'lovers_key_time_series'.")

import os
import cv2
import numpy as np
from PIL import Image

# --- Configurations ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
input_dir = os.path.join(DATA_DIR, 'lovers_key_time_series', 'images')
output_video = os.path.join(DATA_DIR, 'lovers_key_timelapse.mp4')

# Target dimensions (Requirement 1)
TARGET_WIDTH = 912
TARGET_HEIGHT = 718

# Video speed (Frames Per Second)
# 3 or 4 FPS is usually a good sweet spot for a smooth presentation
FPS = 3 

# --- Step 1: Gather and Sort Images ---
# Get all visual clip files and sort them alphabetically 
# Because Planet filenames start with the timestamp, sorting alphabetically sorts them chronologically
files = [f for f in os.listdir(input_dir) if f.endswith('_Visual_clip.tif')]
files.sort()

if not files:
    print("No matching Planet images found in the directory.")
    exit()

# Initialize OpenCV Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, FPS, (TARGET_WIDTH, TARGET_HEIGHT))

# --- Step 2: Initialize Persistent Canvas (Requirement 2) ---
# This serves as our background. Partial images will stack on top of this,
# allowing the prior image's data to show through transparent sections.
canvas = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0, 255))

processed_count = 0

print(f"Processing {len(files)} potential files...")

for file in files:
    file_path = os.path.join(input_dir, file)
    
    try:
        # Open image using Pillow to seamlessly handle Alpha channels (transparency)
        img = Image.open(file_path)
        
        # REQUIREMENT 1: Filter out irregular sizes
        if img.width != TARGET_WIDTH or img.height != TARGET_HEIGHT:
            print(f"Skipping {file}: Irregular dimensions ({img.width}x{img.height})")
            continue
            
        # Ensure image is in RGBA mode so we can access its transparency mask
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        # REQUIREMENT 2 (Option B): Layer on top of prior image
        # The third argument 'img' acts as the transparency mask
        canvas.paste(img, (0, 0), img)
        
        # Convert our composite canvas to standard RGB and then into a NumPy array for OpenCV
        frame_rgb = canvas.convert('RGB')
        frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
        
        # REQUIREMENT 3: Extract and parse date from filename
        # Example: '20220102_154214_..._clip.tif' -> '20220102'
        date_raw = file.split('_')[0] 
        formatted_date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        
        # --- Burn Date into Bottom Right Corner ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_color = (255, 255, 255)  # White text
        shadow_color = (0, 0, 0)      # Black shadow for readability against bright water/sand
        
        # Calculate text size to position it dynamically in the bottom-right corner
        text_size, _ = cv2.getTextSize(formatted_date, font, font_scale, thickness)
        text_w, text_h = text_size
        
        margin_x = 25
        margin_y = 25
        text_x = TARGET_WIDTH - text_w - margin_x
        text_y = TARGET_HEIGHT - margin_y
        
        # Draw a subtle drop shadow behind text
        cv2.putText(frame_bgr, formatted_date, (text_x + 2, text_y + 2), font, font_scale, shadow_color, thickness + 1)
        # Draw main text
        cv2.putText(frame_bgr, formatted_date, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Write frame to the video file
        video_writer.write(frame_bgr)
        processed_count += 1
        print(f"Successfully added frame: {formatted_date}")
        
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Clean up resources
video_writer.release()
print(f"\nTime-lapse complete! {processed_count} frames written to '{output_video}'.")
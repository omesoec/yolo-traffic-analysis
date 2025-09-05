import os
import sys
import json
import numpy as np
import cv2
from ultralytics import YOLO,solutions
from collections import defaultdict
import matplotlib.pyplot as plt

# --- USER CONFIGURATION ---
VIDEO_PATH = "../testvideos/tohuu.mov"
CONFIG_FILE_PATH = "../configs/regions.json"

# --- Automatic Output Path Generation ---
video_base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_VIDEO_PATH = f"../testvideos/{video_base_name}_output_calibration.avi"

# --- Processing and Model Configuration ---
TIME_LIMIT_SECONDS = 20
MODEL_PATH = "../train-scripts/testrun/yolov8m_traffic_default/weights/best.pt"


# --- HOMOGRAPHY CALCULATION FUNCTION ---
def calculate_homography_from_dimensions(image_points_ordered, real_world_width, real_world_height):
    if len(image_points_ordered) != 4:
        raise ValueError("Exactly four corresponding points are required.")
    pixel_points = np.float32(image_points_ordered).reshape(-1, 1, 2)
    w, h = real_world_width, real_world_height
    world_points_defined = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    homography_matrix, _ = cv2.findHomography(pixel_points, world_points_defined)
    return homography_matrix


def load_regions_config(config_path, video_filename):
    print(f"Loading configurations from: {config_path}")
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)
    with open(config_path, 'r') as f:
        all_configs = json.load(f)
    if video_filename not in all_configs:
        print(f"Error: No configuration found for '{video_filename}' in '{config_path}'")
        print(f"Available configurations are for: {list(all_configs.keys())}")
        sys.exit(1)
    print(f"Successfully loaded configuration for '{video_filename}'.")
    return all_configs[video_filename]


# --- SCRIPT START ---
video_key = os.path.basename(VIDEO_PATH)
config_data = load_regions_config(CONFIG_FILE_PATH, video_key)
REGIONS_CONFIG = config_data.get("regions", [])
HOMOGRAPHY_CONFIG = config_data.get("homography", {})

if not REGIONS_CONFIG:
    print("Error: 'regions' key not found in the configuration for this video.")
    sys.exit(1)

H_MATRIX = None
if HOMOGRAPHY_CONFIG:
    try:
        H_MATRIX = calculate_homography_from_dimensions(
            HOMOGRAPHY_CONFIG["pixel_points_ordered"],
            HOMOGRAPHY_CONFIG["real_width_m"],
            HOMOGRAPHY_CONFIG["real_height_m"]
        )
        print("Homography matrix calculated successfully.")
    except (KeyError, ValueError) as e:
        print(f"Error processing homography configuration: {e}")
        H_MATRIX = None
else:
    print("Warning: Homography configuration not found. Will not perform coordinate transformation.")

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Error reading video file: {VIDEO_PATH}"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_limit = int(TIME_LIMIT_SECONDS * fps) if TIME_LIMIT_SECONDS > 0 and fps > 0 else -1
print(f"Frame limit set to: {frame_limit if frame_limit != -1 else 'No limit'}")

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

counters = []
for region_config in REGIONS_CONFIG:
    counter = solutions.ObjectCounter(
        show=False,
        region=region_config["points"],
        model=model,
        classes=region_config.get("classes") or None,
        tracker="botsort.yaml",
    )
    counters.append(counter)
print(f"Initialized {len(counters)} region counters.")

trajectories = defaultdict(list)
current_frame_count = 0

print("Processing video...")
while cap.isOpened():
    if frame_limit != -1 and current_frame_count >= frame_limit:
        print(f"Time limit of {TIME_LIMIT_SECONDS} seconds reached. Stopping.")
        break
    success, frame = cap.read()
    if not success:
        print("Video processing complete.")
        break
    current_frame_count += 1
    annotated_frame = frame.copy()
    
    for counter in counters:
        counter(annotated_frame)

        if H_MATRIX is not None and counter.tracks:
            tracks = counter.tracks
            if tracks[0].boxes.id is not None:
                track_ids = tracks[0].boxes.id.int().cpu().tolist()
                boxes = tracks[0].boxes.xyxy.cpu().tolist()

                for track_id, box in zip(track_ids, boxes):
                    x_pixel = (box[0] + box[2]) / 2
                    y_pixel = box[3]
                    
                    pixel_point = np.array([[[x_pixel, y_pixel]]], dtype='float32')
                    world_point = cv2.perspectiveTransform(pixel_point, H_MATRIX)
                    
                    if world_point is not None:
                        world_x, world_y = world_point[0][0]
                        if not trajectories[track_id] or trajectories[track_id][-1] != (world_x, world_y):
                            trajectories[track_id].append((world_x, world_y))

    cv2.imshow("YOLOv8 Detection", annotated_frame)
    video_writer.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"Processing complete. Output video saved to: {OUTPUT_VIDEO_PATH}")

# --- NEW: Generate Final Trajectory Plot with Zones ---
if H_MATRIX is not None and trajectories:
    print("Generating final trajectory plot with zones...")
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # Step 1: Plot the detection zones
    print("Transforming and plotting detection zones...")
    for region_config in REGIONS_CONFIG:
        # Get pixel points and convert to a NumPy array for transformation
        pixel_points = np.array(region_config["points"], dtype='float32')
        pixel_points_reshaped = pixel_points.reshape(-1, 1, 2)
        
        # Transform the polygon's corners to world coordinates
        world_points = cv2.perspectiveTransform(pixel_points_reshaped, H_MATRIX)
        
        if world_points is not None:
            # Reshape back for plotting and close the polygon
            world_points = world_points.reshape(-1, 2)
            closed_polygon_points = np.vstack([world_points, world_points[0]])
            
            # Extract x and y coordinates for plotting
            x_coords = closed_polygon_points[:, 0]
            y_coords = closed_polygon_points[:, 1]
            
            # Plot the zone with a dashed line
            plt.plot(x_coords, y_coords, linestyle='--', label=region_config.get("name", "Zone"))

    # Step 2: Plot the vehicle trajectories
    print("Plotting vehicle trajectories...")
    for track_id, path in trajectories.items():
        if len(path) > 1:
            # Unzip the list of tuples into two lists: x and y coordinates
            x_coords, y_coords = zip(*path)
            # Plot the trajectory with a solid line and small markers
            plt.plot(x_coords, y_coords, marker='o', linestyle='-', markersize=2)

    # Step 3: Finalize the plot
    plt.title("Vehicle Trajectories and Detection Zones in Real-World Coordinates")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    
    # Invert y-axis to match the top-down perspective of the video
    # (where Y=0 is at the "bottom" of the homography region)
    # ax.invert_yaxis()
    ax.invert_xaxis()
    plt.grid(True)
    plt.legend() # Display the labels for the zones
    plt.show()
    
else:
    print("No trajectories were recorded or homography was not calculated. Skipping plot.")
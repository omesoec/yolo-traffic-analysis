import cv2
import os
import sys

# --- USER CONFIGURATION ---

# 1. Path to the video you want to process
VIDEO_PATH = "../testvideos/ncthach2.mov"

# 2. Name for the output image file
OUTPUT_IMAGE_NAME = "first_frame2.jpg"

# --- END OF CONFIGURATION ---


def save_first_frame(video_path, output_path):
    """
    Reads a video file and saves its very first frame as an image.

    Args:
        video_path (str): The full path to the video file.
        output_path (str): The path to save the output JPEG image.
    """
    # --- 1. Open the video file ---
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        sys.exit(1) # Exit the script with an error

    # --- 2. Read the first frame ---
    success, frame = cap.read()

    # Release the video file immediately since we're done with it
    cap.release()

    # --- 3. Save the frame ---
    if success:
        # If the frame was read successfully, save it
        cv2.imwrite(output_path, frame)
        print(f"Successfully extracted the first frame.")
        print(f"Image saved as: {output_path}")
    else:
        # If we couldn't even read the first frame, the video might be empty or corrupt
        print("Error: Could not read the first frame from the video.")
        sys.exit(1)


# This part makes the script runnable from the command line
if __name__ == "__main__":
    # Get the directory of the script to save the image in the same place
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(script_dir, OUTPUT_IMAGE_NAME)
    
    save_first_frame(VIDEO_PATH, output_file_path)
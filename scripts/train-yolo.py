import os
import shutil
import torch
from ultralytics import YOLO

# =================================================================================================
# ---  USER CONFIGURATION  ---
# =================================================================================================
# SET TRAINING MODE
#    - If True, enables aggressive augmentations like mosaic, mixup, and copy-paste.
#    - If False, uses the default YOLOv8 augmentations.
ENABLE_AGGRESSIVE_AUGMENTATION = True
# 1. BASE MODEL FOR TRANSFER LEARNING
#    Choose the pre-trained model to start from. You selected 'yolov8l.pt' for high accuracy.
#    Options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
BASE_MODEL = 'testrun/yolov8m_augmented/weights/last.pt'

# 2. DATASET CONFIGURATION FILE
#    Path to your dataset.yaml file. This file tells YOLO where your data is and what the classes are.
DATASET_CONFIG = '../datasets_merged/dataset.yaml'

# 3. TRAINING HYPERPARAMETERS
#    - epochs: Number of times to loop through the entire dataset. 100 is a good starting point.
#    - imgsz: The image size the model will be trained on. 640 is standard for YOLOv8.
#             Larger sizes (e.g., 1280) can improve accuracy for small objects but require more VRAM.
EPOCHS = 100
IMAGE_SIZE = 900
BATCH_SIZE = 11 # Adjust based on your GPU memory. 16 is a good starting point for most GPUs.
# 4. OUTPUT CONFIGURATION
#    - project: The name of the main output directory for all experimental runs.
#    - experiment_name: A unique, descriptive name for this specific training run. This will also be
#                       used for your final, cleaned model file name.
#    - final_model_dir: The clean, top-level directory where your best model will be saved.
PROJECT_NAME = 'testrun'
EXPERIMENT_NAME = 'yolov8m_augmented' # Descriptive name for the run
FINAL_MODEL_DIR = '../models'

# =================================================================================================
# ---  SCRIPT LOGIC  ---
# =================================================================================================

def train_model():
    """
    Main function to run the YOLOv8 training process.
    """
    print("--- Starting YOLOv8 Training ---")
    
    # --- Device Check ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("WARNING: Training on CPU is very slow. A CUDA-enabled GPU is highly recommended.")

    # --- Initialize Model ---
    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # --- Define Augmentation Parameters ---
    # Create a dictionary to hold our training arguments.
    train_args = {
        'resume': True,
        'data': DATASET_CONFIG,
        'epochs': EPOCHS,
        'imgsz': IMAGE_SIZE,
        'batch': BATCH_SIZE,
        'project': PROJECT_NAME,
        'name': EXPERIMENT_NAME,
        'exist_ok': True
    }

    # If aggressive augmentation is enabled, add the specific parameters to the dictionary.
    if ENABLE_AGGRESSIVE_AUGMENTATION:
        print("\n*** AGGRESSIVE AUGMENTATION ENABLED ***")
        augmentation_params = {
            'mosaic': 1.0,
            'mixup': 0.5,
            'copy_paste': 0.5,
            # 'hsv_h': 0.025,
            # 'hsv_s': 0.8,
            # 'hsv_v': 0.5,
            'degrees': 10.0,
            'translate': 0.2,
            'scale': 0.6,
            'shear': 5.0,
            'perspective': 0.001
        }
        train_args.update(augmentation_params)
    else:
        print("\n*** Using Default YOLOv8 Augmentations ***")

    # --- Start Training ---
    print(f"Starting training run '{EXPERIMENT_NAME}' for {EPOCHS} epochs...")
    # The '**train_args' syntax unpacks the dictionary into keyword arguments for the function.
    results = model.train(**train_args)
    
    print("--- Training Complete ---")

    # --- Save the Best Model Cleanly ---
    print("--- Saving Final Model ---")
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(results.save_dir, 'weights/best.pt')
    final_model_name = f"{EXPERIMENT_NAME}.pt"
    final_model_path = os.path.join(FINAL_MODEL_DIR, final_model_name)

    if os.path.exists(best_model_path):
        shutil.copyfile(best_model_path, final_model_path)
        print(f"Successfully copied best model to: {final_model_path}")
    else:
        print(f"ERROR: Could not find best model at '{best_model_path}'.")
        
    print("--- Process Finished ---")


# This ensures the script runs only when executed directly (not when imported)
if __name__ == '__main__':
    train_model()
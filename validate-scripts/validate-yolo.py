from ultralytics import YOLO

# =============================================================================
# ---  CONFIGURATION  ---
# =============================================================================

# 1. PATH TO THE MODEL YOU WANT TO VALIDATE
#    This should be the 'best.pt' file from your training run.

PROJECT_NAME = 'valrun'
EXPERIMENT_NAME = 'yolov8m' # Descriptive name for the run

MODEL_PATH = 'testrun/yolov8m_traffic_default/weights/best.pt'  # Or '../runs/detect/.../weights/best.pt'
# 2. PATH TO YOUR DATASET CONFIGURATION FILE
DATASET_CONFIG = '../datasets/dataset.yaml'

# 3. IMAGE SIZE
#    This should match the image size you used for training.
IMAGE_SIZE = 640  # Or 1280, etc.

# =============================================================================

def validate_model():
    """
    Loads a trained YOLOv8 model and evaluates its performance on the validation dataset.
    """
    print(f"--- Starting validation for model: {MODEL_PATH} ---")

    # Load the trained model
    model = YOLO(MODEL_PATH)

    # Run the validation
    # The 'val' method returns a metrics object with all the performance data.
    metrics = model.val(
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        data=DATASET_CONFIG,
        imgsz=IMAGE_SIZE,
        split='val'  # Specify that you want to run on the validation set
    )

    print("\n--- Validation Complete ---")
    print("Overall Box Metrics:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")  # Main metric
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP75:    {metrics.box.map75:.4f}")
    
    print("\nPrecision and Recall:")
    print(f"  Precision: {metrics.box.p[0]:.4f}") # Precision for the primary class (or overall)
    print(f"  Recall:    {metrics.box.r[0]:.4f}") # Recall for the primary class (or overall)
    
    print(f"\nFor a detailed breakdown, check the new validation folder created in 'runs/detect/'")


if __name__ == '__main__':
    validate_model()
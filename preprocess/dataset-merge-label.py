import os
import shutil

# --- CONFIGURATION ---
# The original dataset directories
TRAIN_DIR_ORIGINAL = '../datasets/train'
VAL_DIR_ORIGINAL = '../datasets/val'

# The new directories where the modified dataset will be saved
TRAIN_DIR_MERGED = '../datasets_merged/train'
VAL_DIR_MERGED = '../datasets_merged/val'
OUTPUT_DATA_YAML = '../datasets_merged/data.yaml'

# --- CLASS MAPPING ---
# Define the new class structure.
# We are mapping original class IDs to new class IDs.
#
# Original IDs:
# 0: Motorcycle, 1: Car, 2: Bus, 3: Truck, 4: Transporter, 5: Container, 6: Big_Transporter
#
# New IDs:
# 0: Motorcycle, 1: Car, 2: Large_Vehicle
CLASS_MAPPING = {
    0: 0,  # Motorcycle -> Motorcycle
    1: 1,  # Car -> Car
    2: 2,  # Bus -> Large_Vehicle
    3: 2,  # Truck -> Large_Vehicle
    4: 1,  # Transporter -> Car
    5: 2,  # Container -> Large_Vehicle
    6: 2,  # Big_Transporter -> Large_Vehicle
}

# Define the names for the NEW `data.yaml` file
NEW_CLASS_NAMES = {
    0: 'Motorcycle',
    1: 'Car',
    2: 'Large_Vehicle'
}

# --- SCRIPT LOGIC ---
def process_directory(original_dir, merged_dir):
    """
    Processes a directory by copying images and converting label files.
    """
    if not os.path.exists(original_dir):
        print(f"ERROR: Original directory not found: {original_dir}")
        return
    
    # Create the new directory
    os.makedirs(merged_dir, exist_ok=True)
    print(f"Processing directory: {original_dir} -> {merged_dir}")

    total_files = len(os.listdir(original_dir))
    processed_files = 0

    # Iterate over all files in the original directory
    for filename in os.listdir(original_dir):
        original_filepath = os.path.join(original_dir, filename)
        merged_filepath = os.path.join(merged_dir, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # If it's an image, just copy it over
            shutil.copy(original_filepath, merged_filepath)
        
        elif filename.lower().endswith('.txt'):
            # If it's a label file, read it, convert it, and write the new version
            with open(original_filepath, 'r') as f_in:
                new_lines = []
                for line in f_in:
                    if line.strip():
                        parts = line.strip().split()
                        original_class_id = int(parts[0])
                        
                        # Check if the class ID is in our mapping
                        if original_class_id in CLASS_MAPPING:
                            new_class_id = CLASS_MAPPING[original_class_id]
                            # Reconstruct the line with the new class ID
                            new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                            new_lines.append(new_line)
                
                # Write the new label file in the merged directory
                with open(merged_filepath, 'w') as f_out:
                    f_out.writelines(new_lines)
        
        processed_files += 1
        if processed_files % 200 == 0:
            print(f"  - Processed {processed_files}/{total_files} files...")

    print(f"Finished processing {original_dir}.")


def create_yaml_file():
    """Creates the data.yaml file for the new merged dataset."""
    content = f"""
# Path to the training images directory
train: train

# Path to the validation images directory
val: val

# Number of classes
nc: {len(NEW_CLASS_NAMES)}

# Class names in the correct order
names:
"""
    for i in range(len(NEW_CLASS_NAMES)):
        content += f"  {i}: {NEW_CLASS_NAMES[i]}\n"
        
    with open(OUTPUT_DATA_YAML, 'w') as f:
        f.write(content)
    print(f"\nCreated new data configuration file at: {OUTPUT_DATA_YAML}")


if __name__ == '__main__':
    # Process both the training and validation sets
    process_directory(TRAIN_DIR_ORIGINAL, TRAIN_DIR_MERGED)
    process_directory(VAL_DIR_ORIGINAL, VAL_DIR_MERGED)
    
    # Create the final .yaml file
    create_yaml_file()
    
    print("\nDataset class merging complete!")
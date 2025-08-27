# ==============================================================================
#           YOLO Dataset Analysis Script: Unified Labels-Only
#
# Description:
#   This script performs an extremely fast analysis on the COMBINED train and
#   validation sets by reading ONLY the .txt label files. It generates a
#   single set of plots representing the entire dataset.
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
TRAIN_DIR = '../datasets/train'
VAL_DIR = '../datasets/val'
REPORT_DIR = '../report/label_unified'
CLASS_NAMES = [
    'Motorcycle', 'Car', 'Bus', 'Truck', 'Transporter',
    'Container', 'Big_Transporter'
]

# --- 2. CORE PARSING FUNCTION ---
def parse_labels_only(data_dir, class_names):
    """Parses all .txt label files in a directory, robustly handling blank lines."""
    data = []
    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found at '{data_dir}'.")
        return pd.DataFrame()

    label_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    print(f"Found {len(label_files)} label files in '{data_dir}'. Processing...")

    for label_file in label_files:
        image_name = os.path.splitext(label_file)[0] + '.jpg'
        with open(os.path.join(data_dir, label_file), 'r') as f:
            lines = f.readlines()
            if not lines:
                data.append([image_name, -1, 'no_object'])
            else:
                lines_processed = 0
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line:
                        parts = stripped_line.split()
                        if len(parts) > 0:
                            class_id = int(parts[0])
                            data.append([image_name, class_id, class_names[class_id]])
                            lines_processed += 1
                if lines_processed == 0:
                    data.append([image_name, -1, 'no_object'])

    return pd.DataFrame(data, columns=['image_name', 'class_id', 'class_name'])

# --- 3. PLOTTING FUNCTIONS ---
def plot_class_distribution(df, title, filename):
    """Plots the total number of bounding boxes for each class."""
    print(f"  - Generating plot: {title}")
    plt.figure(figsize=(12, 7))
    class_counts = df[df['class_name'] != 'no_object']['class_name'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis', order=class_counts.index)
    plt.title(title, fontsize=16)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Number of Instances (Bounding Boxes)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_bboxes_per_image(df, title, filename):
    """Plots the distribution of how many objects appear per image."""
    print(f"  - Generating plot: {title}")
    bboxes_per_image = df[df['class_name'] != 'no_object'].groupby('image_name').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(bboxes_per_image, bins=max(1, bboxes_per_image.max()), discrete=True)
    plt.title(title, fontsize=16)
    plt.xlabel('Number of Bounding Boxes per Image', fontsize=12)
    plt.ylabel('Frequency (Number of Images)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

# --- 4. MAIN EXECUTION ---
def main():
    """Main function to orchestrate the unified labels-only analysis."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Unified label analysis reports will be saved to '{REPORT_DIR}'")

    # --- Parse Data from both sets ---
    print("\n--- Parsing Training Labels ---")
    train_df = parse_labels_only(TRAIN_DIR, CLASS_NAMES)
    print("\n--- Parsing Validation Labels ---")
    val_df = parse_labels_only(VAL_DIR, CLASS_NAMES)

    # --- Combine DataFrames ---
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    print("\n--- Train and Validation sets have been combined. ---")

    if combined_df.empty:
        print("\nCritical Error: No data was parsed. Exiting.")
        return

    # --- Print Summary ---
    print("\n" + "="*50)
    print("      UNIFIED LABELS-ONLY DATASET OVERVIEW")
    print("="*50)
    print(f"Total Unique Images: {combined_df['image_name'].nunique()}")
    print(f"Total Bounding Boxes: {len(combined_df[combined_df['class_name'] != 'no_object'])}")
    print("="*50)

    # --- Generate Plots on Combined Data ---
    print("\n--- Generating Plots for the Full Dataset ---")
    plot_class_distribution(combined_df, 'Class Distribution (Full Dataset)', '1_class_distribution.png')
    plot_bboxes_per_image(combined_df, 'Objects per Image (Full Dataset)', '2_bboxes_per_image.png')

    print("\n✅ Unified labels-only analysis complete!")

if __name__ == '__main__':
    main()
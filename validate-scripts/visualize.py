import pandas as pd
import matplotlib.pyplot as plt

# Paths to your YOLO results
model_paths = {
    "YOLO-v8s": "../train-scripts/testrun/yolov8s_traffic_default/results.csv",
    "YOLO-v8m": "../train-scripts/testrun/yolov8m_traffic_default/results.csv",
    # "YOLO-Model-3": "runs/train/exp3/results.csv"
}

results = {name: pd.read_csv(path) for name, path in model_paths.items()}

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- (1) Validation Loss + mAP ---
ax1 = axes[0]
ax2 = ax1.twinx()

for i, (name, df) in enumerate(results.items()):
    val_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
    ax1.plot(df["epoch"], val_loss, label=f"{name} (val loss)", linestyle="--", color=f"C{i}")
     # plot mAP
    ax2.plot(df["epoch"], df["metrics/mAP50-95(B)"]*100, label=f"{name} (mAP@0.5:0.95)")

ax1.set_ylabel("Validation Loss", color="tab:red")
ax2.set_ylabel("mAP@0.5:0.95", color="tab:blue")
ax1.set_xlabel("Epoch")
ax1.set_title("mAP and Loss per Epoch")
ax1.grid(True)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# --- (2) Precision & Recall ---
ax3 = axes[1]
for i, (name, df) in enumerate(results.items()):
    ax3.plot(df["epoch"], df["metrics/precision(B)"], label=f"{name} (Precision)", color=f"C{i}")
    ax3.plot(df["epoch"], df["metrics/recall(B)"], linestyle="--", label=f"{name} (Recall)", color=f"C{i}")

ax3.set_title("Precision and Recall vs Epochs")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Score")
ax3.legend()
ax3.grid(True)

plt.suptitle("YOLO Model Comparison", fontsize=16, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('validation-plot.png')
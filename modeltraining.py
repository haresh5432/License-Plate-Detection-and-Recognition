# fine_tune_yolo.py
from ultralytics import YOLO
import os

# -------------------------
# 1. Set paths
# -------------------------
dataset_path = "/Users/hareshshokeen/Desktop/CV/roboflow_dataset"  # Path to your unzipped Roboflow dataset
yaml_file = "/Users/hareshshokeen/Desktop/CV/License Plate Recognition.v11i.yolov8/data.yaml"

# -------------------------
# 2. Load a pretrained YOLOv8 model
# -------------------------
# Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
model = YOLO("/Users/hareshshokeen/Desktop/CV/runs/detect/number_plate_quick/weights/best.pt")

# -------------------------
# 3. Train YOLO on your dataset
# -------------------------
model.train(
    data=yaml_file,
    epochs=3,          # just 3 epochs
    imgsz=320,         # smaller images → faster
    batch=4,           # small batch
    name="number_plate_quick",  
    workers=2,         # CPU-friendly
    save_period=1      # save every epoch
)

# -------------------------
# 4. Trained model location
# -------------------------
print("Training complete!")
print("Your trained model is saved at:")
print("runs/detect/number_plate_model/weights/best.pt")

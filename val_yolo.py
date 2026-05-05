import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("other_models/yolo11m-best.pt")
    # Train the model
    results = model.val(data=r"E:\resources\datasets\tea-buds-database\C\data.yaml",  imgsz=320, batch=32,
                          device=0, seed=2025, plots=False)


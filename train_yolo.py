import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("other_models/yolo26n.pt")
    # Train the model
    results = model.train(data=r"E:\resources\datasets\tea-buds-database\D\data.yaml", epochs=100, imgsz=320, batch=32,
                          device=0, seed=2025, plots=True, cos_lr=True)


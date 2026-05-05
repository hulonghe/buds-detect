from ultralytics import YOLO

if __name__ == '__main__':
    model_names = [
        # "yolo26n",
        # "yolo26s",
        # "yolo26m",
        # "yolo26l",
        # "yolo12n",
        # "yolo12s",
        # "yolo12m",
        # "yolo12l",
        # "yolo11n",
        # "yolo11s",
        # "yolo11m",
        # "yolo11l",
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        # "yolov5nu",
        # "yolov5su",
        # "yolov5mu",
        # "yolov5lu",
    ]
    data_names = [
        'A-old',
        # 'A-Crop',
        'B',
        'C',
    ]
    data_root = r"E:\resources\datasets\tea-buds-database"

    for model_name in model_names:
        for data_name in data_names:
            # Load a model
            model = YOLO(f"other_models/{model_name}.pt")
            # Train the model
            results = model.train(data=f"{data_root}/{data_name}/data.yaml",
                                  epochs=100, imgsz=320, batch=32,
                                  device=0, seed=2025, plots=False,warmup_epochs=3,
                                  project="zt-tea", name=f'{data_name}_{model_name}',
                                  exist_ok=True)

from pathlib import Path
from ultralytics import YOLO


def train_model(model_name: str, run_name: str, epochs: int = 40):
    # Base project directory (this file's folder)
    project_dir = Path(__file__).resolve().parent

    # Paths
    data_yaml = project_dir / "vehicle.yaml"
    project_runs = project_dir / "runs"        # where logs & weights will be stored

    print("=== YOLO Vehicle Training ===")
    print(f"Project dir : {project_dir}")
    print(f"Data yaml   : {data_yaml}")
    print(f"Runs folder : {project_runs}")
    print(f"Run name    : {run_name}")
    print(f"Model       : {model_name}")
    print("=============================")

    # Load model
    model = YOLO(model_name)

    # Train
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        device=0,           # 0 = first GPU
        project=str(project_runs),
        name=run_name,      # subfolder under runs/
        batch=16,
        workers=4,
        exist_ok=True
    )

    print("\nTraining complete!")
    print(f"Results saved in: {project_runs / run_name}")


def main():
    # 1) YOLOv8n (already trained, keep for reference)
    # Uncomment if you need to retrain
    # train_model(
    #     model_name="yolov8n.pt",
    #     run_name="train_vehicle_v8n_e40",
    #     epochs=40,
    # )

    # 2) YOLOv11n (new experiment)
    train_model(
        model_name="yolo11n.pt",
        run_name="train_vehicle_v11n_e40",  # distinct run name
        epochs=40,
    )


if __name__ == "__main__":
    main()
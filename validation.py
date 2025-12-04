from pathlib import Path
from ultralytics import YOLO


def main():
    project_dir = Path(__file__).resolve().parent
    runs_dir = project_dir / "runs"
    train_run_name = "train_vehicle"

    best_weights = runs_dir / train_run_name / "weights" / "best.pt"
    data_yaml = project_dir / "vehicle.yaml"

    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found at: {best_weights}")

    print("=== YOLO Vehicle Validation ===")
    print(f"Using weights: {best_weights}")
    print(f"Data yaml   : {data_yaml}")
    print("===============================")

    model = YOLO(str(best_weights))

    model.val(
        data=str(data_yaml),
        device="cpu",  # IMPORTANT: works even if no GPU
        project=str(runs_dir),
        name="val_vehicle",
        exist_ok=True,
    )

    print("\nValidation complete!")
    print(f"Results saved in: {runs_dir / 'val_vehicle'}")


if __name__ == "__main__":
    main()

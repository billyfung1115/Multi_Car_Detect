from pathlib import Path
from ultralytics import YOLO


def eval_model(weights_path: Path, label: str, data_yaml: Path):
    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml),
        imgsz=640,
        batch=16,
        device="cpu",     # IMPORTANT: CPU so it works without CUDA
        verbose=False,
    )

    print(f"\n=== {label} ===")
    for k, v in metrics.results_dict.items():
        try:
            v_float = float(v)
            print(f"{k:25s}: {v_float:.4f}")
        except Exception:
            print(f"{k:25s}: {v}")

    return metrics.results_dict


def main():
    project_dir = Path(__file__).resolve().parent

    # Existing trained weights (already produced on your machine)
    v8n_weights = project_dir / "runs" / "train_vehicle" / "weights" / "best.pt"
    v11n_weights = project_dir / "runs" / "train_vehicle_v11n_e40" / "weights" / "best.pt"
    data_yaml = project_dir / "vehicle.yaml"

    assert v8n_weights.exists(), f"v8n weights not found: {v8n_weights}"
    assert v11n_weights.exists(), f"v11n weights not found: {v11n_weights}"
    assert data_yaml.exists(), f"vehicle.yaml not found: {data_yaml}"

    print("Project dir:", project_dir)
    print("YOLOv8n weights :", v8n_weights)
    print("YOLOv11n weights:", v11n_weights)
    print("Data yaml       :", data_yaml)

    res_v8n = eval_model(v8n_weights, "YOLOv8n (40 epochs)", data_yaml)
    res_v11n = eval_model(v11n_weights, "YOLOv11n (40 epochs)", data_yaml)

    print("\nAvailable metric keys:", list(res_v8n.keys()))

    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    print("\n\n=== SUMMARY TABLE (key metrics) ===")
    header = ["Metric", "YOLOv8n", "YOLOv11n"]
    print(f"{header[0]:25s} {header[1]:>10s} {header[2]:>10s}")
    for k in keys:
        v8 = float(res_v8n.get(k, float("nan")))
        v11 = float(res_v11n.get(k, float("nan")))
        print(f"{k:25s} {v8:10.4f} {v11:10.4f}")


if __name__ == "__main__":
    main()

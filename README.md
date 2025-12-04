# Multi_Car_Detect

Project to train and run YOLO-based vehicle detection (Ultralytics). This repository contains training, validation, inference, and comparison utilities for YOLOv8/YOLOv11 experiments.

---

## Contents
- `train.py` - Train a YOLO model on the `vehicle.yaml` dataset.
- `validation.py` - Run validation using a trained `best.pt` from a given run.
- `infer_vehicle.py` - Run inference (image, video, webcam) using the YOLOv8-trained weights (`runs/train_vehicle/weights/best.pt`). Saves annotated outputs to `predictions/`.
- `infer_vehicle_v11.py` - Same as above but for the YOLOv11n-trained run (`runs/train_vehicle_v11n_e40/weights/best.pt`). Saves to `predictions_v11/`.
- `compare_modes.py` - Evaluate two different trained runs and print a side-by-side summary of selected metrics.
- `main.py` - Small placeholder script.
- `vehicle.yaml` - Dataset configuration used by Ultralytics training/validation.
- `requirements.txt` - Project dependencies and installation guidance.
- `runs/` - Output folder created by training/validation with logs and weights.
- `predictions/`, `predictions_v11/` - Annotated outputs from inference.

---

## Quick setup (Windows PowerShell)

1. Create a virtual environment (if you don't have one):

```powershell
cd "C:\Users\Billy Fung\Desktop\Multi_Car_Detect"
python -m venv .venv
```

2. Activate the venv (PowerShell v5.1):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
& ".\.venv\Scripts\Activate.ps1"
```

3. Install dependencies. The `requirements.txt` contains guidance for `torch` which often needs CUDA-specific wheels. A minimal install (pip will choose CPU torch if unspecified):

```powershell
python -m pip install -r requirements.txt
```

If you need GPU-enabled PyTorch, use the official PyTorch installation instructions and wheel index. Example (replace version/CUDA to match your environment):

```powershell
# Example CPU-only wheels (explicit):
# pip install torch==2.2.2+cpu torchvision==0.18.3+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Example CUDA 12.1 (replace with your CUDA version):
# pip install torch==2.2.2+cu121 torchvision==0.18.3+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

---

## How each script is used

- `train.py`
	- Purpose: Train a model using Ultralytics YOLO.
	- Default in `main()` trains `yolo11n.pt` for 40 epochs and writes results under `runs/train_vehicle_v11n_e40`.
	- Example:

```powershell
python train.py
```

- `validation.py`
	- Purpose: Run validation using a specific `best.pt` from `runs/train_vehicle/weights/best.pt`.
	- Example:

```powershell
python validation.py
```

- `infer_vehicle.py` and `infer_vehicle_v11.py`
	- Purpose: Run inference on a single image, a video file, or webcam. Edit the `main()` function to set `image_path` or `video_path` (examples are commented inline).
	- Output: annotated images/videos saved to `predictions/` (v8) or `predictions_v11/` (v11).
	- Example (set `image_path` inside file then run):

```powershell
python infer_vehicle.py
# or
python infer_vehicle_v11.py
```

	- While a video or webcam is running, press `q` to quit the live display loop.

- `compare_modes.py`
	- Purpose: Load two `best.pt` files (YOLOv8 run and YOLOv11 run), run `.val()` on each using the `vehicle.yaml` dataset, and print metrics and a short summary table.
	- Requires the two runs to exist:
		- `runs/train_vehicle/weights/best.pt`
		- `runs/train_vehicle_v11n_e40/weights/best.pt`
	- Example:

```powershell
python compare_modes.py
```

---

## Outputs and where to find them
- Training and validation results: `runs/<run_name>/` (weights under `runs/<run_name>/weights/`)
- Annotated inference outputs: `predictions/` and `predictions_v11/`

---

## Notes & tips
- Edit `infer_vehicle.py` / `infer_vehicle_v11.py` `main()` to conveniently point to test images or videos.
- `train.py` uses `device=0` (first GPU). Change `device` to `"cpu"` or another index if needed.
- Metric key names printed by `compare_modes.py` depend on the installed `ultralytics` version; you may need to adjust the `keys` list in that file if keys differ.

---

If you'd like, I can:
- Make the inference scripts accept command-line arguments for source and device.
- Pin versions for `ultralytics` and `opencv-python` in `requirements.txt`.
- Create a small wrapper to run any script with friendly CLI options.

Feel free to tell me which of the above you want next.

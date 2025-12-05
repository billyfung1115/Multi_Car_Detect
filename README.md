# Multi_Car_Detect

Project to train and run YOLO-based vehicle detection (Ultralytics). This repository contains training, validation, inference, and comparison utilities for YOLOv8/YOLOv11 experiments.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Getting Started in Google Colab](#getting-started-in-google-colab)
- [Quick Setup (Windows PowerShell)](#quick-setup-windows-powershell)
- [Quick Setup (Linuxmacos)](#quick-setup-linuxmacos)
- [Requirements](#requirements)
- [Usage](#usage)
- [Environment Check](#environment-check)
- [Outputs](#outputs)
- [Notes](#notes)

---

## Project Overview

Multi_Car_Detect is a collection of scripts to train and run YOLO-based vehicle detection using the Ultralytics library. The repository contains training, validation, inference, and comparison utilities for experiments using YOLOv8 and YOLOv11 models.

**Remark:**
The original training was performed using GPU hardware for efficiency. However, due to compute limitations in free Colab environments, code defaults and examples have been updated to use CPU by default. If you have access to a GPU (locally or in a Colab Pro session), you can modify scripts to use GPU by setting device=0 or device="cuda" as appropriate.
---

## Folder Structure

After cloning, your folder structure should look like:


Multi_Car_Detect/
  ├── Multi_Car_Detect/
  │     ├── train.py
  │     ├── validation.py
  │     ├── infer_vehicle.py
  │     ├── infer_vehicle_v11.py
  │     ├── compare_modes.py
  │     ├── main.py
  │     ├── vehicle.yaml
  │     ├── requirements.txt
  │     ├── README.md
  │     └── ...
  ├── .gitignore
  ├── requirements.txt
  ├── README.md
  └── ...

> *Note:* Most scripts and requirements are in the inner Multi_Car_Detect directory.

---

## Getting Started in Google Colab

You can run this project directly in [Google Colab](https://colab.research.google.com/):

python
# Clone the repo
!git clone https://github.com/billyfung1115/Multi_Car_Detect.git

# Install dependencies
!pip install -r Multi_Car_Detect/Multi_Car_Detect/requirements.txt

# (Optional) Display README in Colab
from IPython.display import Markdown, display
with open('Multi_Car_Detect/Multi_Car_Detect/README.md', 'r', encoding='utf-8') as f:
    display(Markdown(f.read()))


*Tip:*  
Select a GPU runtime in Colab (Runtime > Change runtime type > GPU) for best performance.

---

## Quick Setup (Windows PowerShell)

1. *Create and activate a virtual environment:*

   powershell
   cd "C:\Users\Billy Fung\Desktop\Multi_Car_Detect\Multi_Car_Detect"
   python -m venv .venv
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
   & ".\.venv\Scripts\Activate.ps1"
   

2. *Install dependencies:*  
   The requirements.txt contains guidance for torch which often needs CUDA-specific wheels.  
   Minimal install (pip will choose CPU torch if unspecified):

   powershell
   python -m pip install -r requirements.txt
   

   If you need GPU-enabled PyTorch, use the official PyTorch installation instructions and wheel index. Example:

   powershell
   # Example CPU-only wheels (explicit):
   # pip install torch==2.2.2+cpu torchvision==0.18.3+cpu -f https://download.pytorch.org/whl/torch_stable.html

   # CUDA 13.0:
   # pip install torch==2.2.2+cu130 torchvision==0.18.3+cu130 -f https://download.pytorch.org/whl/torch_stable.html
   

---

## Quick Setup (LinuxmacOS)

bash
cd Multi_Car_Detect/Multi_Car_Detect
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


For GPU-enabled PyTorch, follow the [official install guide](https://pytorch.org/get-started/locally/) and use the appropriate CUDA wheels.

---

## Requirements

- Python 3.11+ (virtual environment recommended)
- See requirements.txt for pinned package versions and wheel guidance.

---

## Usage

All scripts are in the inner Multi_Car_Detect folder. Make sure you are in that directory before running the following commands.

- *Train a model:*

  bash
  python train.py
  
  - Default: trains yolo11n.pt for 40 epochs, saves to runs/train_vehicle_v11n_e40/.

- *Run validation:*

  bash
  python validation.py
  
  - Requires trained weights in runs/<run_name>/weights/best.pt.

- *Run inference:*

  bash
  python infer_vehicle.py
  python infer_vehicle_v11.py
  
  - Edit the main() function in each script to set image_path or video_path.
  - Annotated outputs are saved to predictions/ (YOLOv8) or predictions_v11/ (YOLOv11).

- *Compare two training runs:*

  bash
  python compare_modes.py
  
  - Requires:
    - runs/train_vehicle/weights/best.pt
    - runs/train_vehicle_v11n_e40/weights/best.pt
  - Prints side-by-side summary metrics.

---

## Environment Check

The repository contains scripts/env_info.py to display versions for Python, torch, torchvision, torchaudio, ultralytics, and OpenCV, plus CUDA availability and device info.

bash
python scripts/env_info.py


---

## Outputs

- *Training/validation logs and weights:* runs/<run_name>/
- *Annotated inference outputs:* predictions/, predictions_v11/

---

## Notes

- *Device selection:* Adjust device arguments in scripts as needed (e.g., device=0 for GPU, device="cpu" for CPU).
- *Metric keys/fields:* Ultralytics output format may vary between versions; update compare_modes.py if needed.
- *Colab tips:* Restart runtime if you experience issues with file deletion or environment.

---

For questions or contributions, feel free to open an issue or pull request!

---
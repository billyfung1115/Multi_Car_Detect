import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" 

import cv2
from pathlib import Path
from ultralytics import YOLO


# CONFIGURATION
MAX_DISPLAY_WIDTH = 960  # max width of the OpenCV window
WINDOW_NAME_IMAGE = "YOLOv8 - Image"
WINDOW_NAME_VIDEO = "YOLOv8 - Video"
WINDOW_NAME_WEBCAM = "YOLOv8 - Webcam"
CONF_THRESH = 0.25
IMG_SIZE = 640


def resize_for_display(img, max_width=MAX_DISPLAY_WIDTH):
    """Resize image to a max width (for display only)."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_model():
    project_dir = Path(__file__).resolve().parent
    weights_path = project_dir / "runs" / "train_vehicle" / "weights" / "best.pt"

    if not weights_path.exists():
        raise FileNotFoundError(f"Best weights not found at: {weights_path}")

    print(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))
    return model, project_dir


def infer_image(model, project_dir, source_path):
    """Run inference on a single image, show it, and save annotated copy."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path}")

    print(f"\nRunning inference on image: {source_path}")

    results = model(
        str(source_path),
        device="cpu",          # <- changed from 0 to "cpu"
        conf=CONF_THRESH,
        imgsz=IMG_SIZE
    )

    out_dir = project_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_path.stem}_pred.jpg"

    cv2.namedWindow(WINDOW_NAME_IMAGE, cv2.WINDOW_NORMAL)

    for r in results:
        im = r.plot()  # full-size annotated image (BGR)

        # SAVE (full resolution)
        cv2.imwrite(str(out_path), im)
        print(f"Saved annotated image to: {out_path}")

        # DISPLAY (resized)
        display_im = resize_for_display(im).copy()
        cv2.imshow(WINDOW_NAME_IMAGE, display_im)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def infer_video(model, project_dir, source_path):
    """Run inference on a video, show it, and save annotated video."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    print(f"\nRunning inference on video: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")

    # Prepare writer for saving output video
    out_dir = project_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_path.stem}_pred.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    print(f"Output video will be saved to: {out_path}")

    cv2.namedWindow(WINDOW_NAME_VIDEO, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            device="cpu",      # <- changed from 0 to "cpu"
            conf=CONF_THRESH,
            imgsz=IMG_SIZE,
            verbose=False
        )

        annotated_frame = results[0].plot()  # original size

        # SAVE frame (original size)
        writer.write(annotated_frame)

        # DISPLAY (resized)
        display_frame = resize_for_display(annotated_frame).copy()
        cv2.imshow(WINDOW_NAME_VIDEO, display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopping (q pressed).")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Video inference finished. Saved to: {out_path}")


def infer_webcam(model, project_dir, cam_index=0):
    """Optional: live webcam inference (no saving by default)."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {cam_index}")

    print("Running webcam inference. Press 'q' to quit.")
    cv2.namedWindow(WINDOW_NAME_WEBCAM, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            device="cpu",      # <- changed from 0 to "cpu"
            conf=CONF_THRESH,
            imgsz=IMG_SIZE,
            verbose=False
        )
        annotated_frame = results[0].plot()
        display_frame = resize_for_display(annotated_frame).copy()

        cv2.imshow(WINDOW_NAME_WEBCAM, display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam inference finished.")


def main():
    model, project_dir = load_model()

    # 1) Single image (your demo image in the repo)
    image_path = "demo image/car.jpg"
    infer_image(model, project_dir, image_path)

    # 2) Video file (optional – update path and uncomment)
    # video_path = "demo video/traffic.mp4"
    # infer_video(model, project_dir, video_path)

    # 3) Webcam (optional)
    # infer_webcam(model, project_dir, cam_index=0)

    print("\nEdit infer_vehicle.py -> main() to change image/video/webcam options.")


if __name__ == "__main__":
    main()

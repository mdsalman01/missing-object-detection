"""Run YOLO on a single image, print detections, and save an annotated copy."""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect objects in a single image with YOLO.")
    parser.add_argument("image", type=str, help="Path to the image to run detection on")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO weights file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--out", type=str, default="debug/detect_one_out.jpg", help="Annotated output path")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(args.image, conf=args.conf, verbose=False)
    result = results[0]

    print(f"{args.image}: {len(result.boxes)} detections (conf >= {args.conf})")
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        print(f"  {class_name:15s} conf={confidence:.2f}  box=({x1}, {y1}, {x2}, {y2})")

    annotated = result.plot()  # BGR numpy array with boxes/labels drawn
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image to {out_path}")


if __name__ == "__main__":
    main()

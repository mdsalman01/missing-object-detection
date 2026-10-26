"""Print basic metadata (resolution, fps, frame count, duration) for one or more videos."""
import argparse
from pathlib import Path

import cv2


def print_info(video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"{video_path.name}: FAILED TO OPEN")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0

    cap.release()

    print(f"{video_path.name}:")
    print(f"  resolution   : {width}x{height}")
    print(f"  fps          : {fps:.2f}")
    print(f"  frame_count  : {frame_count}")
    print(f"  duration (s) : {duration:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print video metadata.")
    parser.add_argument(
        "videos",
        nargs="*",
        default=None,
        help="Paths to video files. Defaults to everything in ./videos",
    )
    args = parser.parse_args()

    if args.videos:
        paths = [Path(p) for p in args.videos]
    else:
        paths = sorted(Path("videos").glob("*.mp4"))

    if not paths:
        print("No videos found.")
        return

    for path in paths:
        print_info(path)


if __name__ == "__main__":
    main()

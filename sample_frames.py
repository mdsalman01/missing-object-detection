"""Evenly sample N frames per video, resize to a common width, save as JPGs for inspection."""
import argparse
from pathlib import Path

import cv2


def sample_video(video_path: Path, out_dir: Path, n_frames: int, target_width: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"{video_path.name}: FAILED TO OPEN")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print(f"{video_path.name}: no frames found")
        cap.release()
        return

    # Evenly spaced indices across the whole clip, including both ends.
    indices = [round(i * (frame_count - 1) / (n_frames - 1)) for i in range(n_frames)]

    video_out_dir = out_dir / video_path.stem
    video_out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for order, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"{video_path.name}: could not read frame {frame_idx}")
            continue

        h, w = frame.shape[:2]
        scale = target_width / w
        resized = cv2.resize(frame, (target_width, round(h * scale)))

        out_path = video_out_dir / f"frame_{order:02d}_src{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_path), resized)
        saved += 1

    cap.release()
    print(f"{video_path.name}: saved {saved}/{n_frames} frames to {video_out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evenly sample frames from videos.")
    parser.add_argument("videos", nargs="*", default=None, help="Video paths. Defaults to ./videos/*.mp4")
    parser.add_argument("--n-frames", type=int, default=12, help="Frames to sample per video")
    parser.add_argument("--width", type=int, default=960, help="Target width for resized output")
    parser.add_argument("--out-dir", type=str, default="debug", help="Output directory")
    args = parser.parse_args()

    paths = [Path(p) for p in args.videos] if args.videos else sorted(Path("videos").glob("*.mp4"))
    if not paths:
        print("No videos found.")
        return

    out_dir = Path(args.out_dir)
    for path in paths:
        sample_video(path, out_dir, args.n_frames, args.width)


if __name__ == "__main__":
    main()

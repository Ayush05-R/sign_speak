"""
Capture images from webcam to build a labeled dataset.
Prompts for label and captures a fixed number of images per label.
"""
import argparse
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def capture_for_label(
    cap: cv2.VideoCapture,
    label: str,
    out_dir: str,
    num_images: int,
    delay_ms: int,
    flip: bool,
) -> None:
    ensure_dir(out_dir)
    captured = 0
    last_capture = 0.0
    print(f"Capturing {num_images} images for label: {label}")
    while captured < num_images:
        ok, frame = cap.read()
        if not ok:
            print("Webcam read failed.")
            break
        if flip:
            frame = cv2.flip(frame, 1)

        now = time.time()
        ready = (now - last_capture) * 1000.0 >= delay_ms
        if ready:
            filename = f"{label}_{int(now * 1000)}_{captured}.jpg"
            cv2.imwrite(os.path.join(out_dir, filename), frame)
            last_capture = now
            captured += 1

        cv2.putText(
            frame,
            f"Label: {label}  {captured}/{num_images}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Dataset Capture", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            print("Capture aborted by user.")
            return

    print(f"Done: {captured} images saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Capture images to build a dataset.")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "data", "raw", "static"))
    parser.add_argument("--num-images", type=int, default=200, help="Images per label")
    parser.add_argument("--delay-ms", type=int, default=150, help="Delay between captures")
    parser.add_argument("--flip", action="store_true", help="Mirror the webcam feed")
    parser.add_argument("--label", default=None, help="Capture a single label and exit")
    parser.add_argument("--labels-file", default=None, help="Path to a file with one label per line")
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    try:
        if args.label:
            label = args.label.strip()
            if label:
                label_dir = os.path.join(args.output_dir, label)
                capture_for_label(cap, label, label_dir, args.num_images, args.delay_ms, args.flip)
            return

        if args.labels_file:
            if not os.path.isfile(args.labels_file):
                print("Labels file not found:", args.labels_file)
                return
            with open(args.labels_file, "r") as f:
                labels = [line.strip() for line in f if line.strip()]
            for label in labels:
                label_dir = os.path.join(args.output_dir, label)
                capture_for_label(cap, label, label_dir, args.num_images, args.delay_ms, args.flip)
            return

        while True:
            label = input("Enter label (or 'quit'): ").strip()
            if not label:
                continue
            if label.lower() == "quit":
                break
            label_dir = os.path.join(args.output_dir, label)
            capture_for_label(cap, label, label_dir, args.num_images, args.delay_ms, args.flip)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

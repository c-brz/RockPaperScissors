import os
import sys
import cv2
import numpy as np
import mediapipe as mp


def extract_frame_images(video_dir, destination_dir, gesture_class=None, save=False):
    """Process all videos in a directory structure"""
    # Expected directory structure: video_dir/gesture_name/video_files

    gesture_dirs = (
        [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
        if gesture_class is None
        else [gesture_class]
    )
    print(f"Found gesture directories: {gesture_dirs}")

    for gesture_name in gesture_dirs:
        gesture_path = os.path.join(video_dir, gesture_name, "videos")
        video_files = [
            f for f in os.listdir(gesture_path) if f.endswith((".mp4", ".avi", ".mov"))
        ]
        print(f"Processing {len(video_files)} videos for gesture: {gesture_name}")
        for video_file in video_files:
            video_path = os.path.join(gesture_path, video_file)
            video_file_name = video_file.split(".")[0]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file {video_path}")
                continue

            N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_dir = os.path.join(
                    destination_dir, gesture_name, "images", video_file_name
                )
                os.makedirs(image_dir, exist_ok=True)
                # frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                image_file = os.path.join(
                    image_dir, f"frame_{str(frame_idx).zfill(3)}.png"
                )
                if save:
                    cv2.imwrite(image_file, rgb_frame)
                frame_idx += 1
            cap.release()


if __name__ == "__main__":

    gesture = sys.argv[1] if len(sys.argv) > 1 else None
    extract_frame_images(
        video_dir="../../RockPaperScissors/my_rps_dataset/data/align",
        destination_dir="../../RockPaperScissors/my_rps_dataset/data/align",
        gesture_class=gesture,
        save=True,
    )

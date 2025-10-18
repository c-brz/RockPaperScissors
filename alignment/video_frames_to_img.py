import os
import sys
import cv2
import json
import numpy as np
import mediapipe as mp


def extract_frame_images(
    video_dir,
    destination_dir,
    gesture_class=None,
    save=False,
    save_to_colorspace="RGB",
    overwrite=False,
):
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

            new_images_dir = os.path.join(
                destination_dir, gesture_name, "images", video_file_name
            )
            if (
                not os.path.isdir(new_images_dir)  # dir does not exist
                or (
                    os.path.isdir(new_images_dir)
                    and len(os.listdir(new_images_dir)) == 0
                )  # dir exists but is empty
                or overwrite  # exists but we overwrite
            ):
                print("🤩 Creating images for video:", video_file_name)
            else:
                print("     Skipping existing video:", video_file_name)
                continue  # skip already processed videos

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
                    image_dir, f"frame_{str(frame_idx).zfill(4)}.png"
                )
                if save:
                    # cv2.imwrite(image_file, rgb_frame)
                    if save_to_colorspace == "RGB":
                        cv2.imwrite(
                            image_file, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        )
                    else:
                        cv2.imwrite(image_file, rgb_frame)
                frame_idx += 1
            cap.release()


def create_empty_metadata_file(destination_dir, gesture_class=None):
    gesture_dirs = (
        [
            d
            for d in os.listdir(destination_dir)
            if os.path.isdir(os.path.join(destination_dir, d))
        ]
        if gesture_class is None
        else [gesture_class]
    )
    for gesture_name in gesture_dirs:

        image_paths = os.path.join(destination_dir, gesture_name, "images")

        for image_dir in os.listdir(image_paths):

            if gesture_name not in image_dir:
                continue  # skip .DS_STORE files??
            metadata_path = os.path.join(image_paths, image_dir, "metadata.json")

            metadata = {
                "label": gesture_name,
                "total_frames": 0,
                "start_frame_idx": 0,
                "action_frame_idx": 0,
                "end_frame_idx": 0,
            }

            if os.path.isfile(metadata_path):
                print(f"Metadata file already exists at {metadata_path}, skipping...")
                continue
            else:
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)
                print(f"Created metadata file at {metadata_path}")


if __name__ == "__main__":

    # Check how many command line arguments are provided
    # Valid arguments:
    # 1. gesture class (optional): e.g., "rock", "paper", "scissors" else checks all directories
    # 2. --overwrite flag (optional): whether to overwrite existing images
    # 3. --save flag: whether to save extracted images

    if len(sys.argv) > 4:
        print(
            "Usage: python video_frames_to_img.py [gesture_class] [--overwrite] [--save]"
        )
        sys.exit(1)
    elif len(sys.argv) == 4:
        gesture = sys.argv[1]
        overwrite = sys.argv[2] == "--overwrite"
        save = sys.argv[3] == "--save"
    elif len(sys.argv) == 3:
        if (sys.argv[1] == "--overwrite" and sys.argv[2] == "--save") or (
            sys.argv[2] == "--overwrite" and sys.argv[1] == "--save"
        ):
            gesture = None
            overwrite = True
            save = True
        elif sys.argv[2] == "--overwrite":
            gesture = sys.argv[1]
            overwrite = True
            save = False
        elif sys.argv[2] == "--save":
            gesture = sys.argv[1]
            overwrite = False
            save = True
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--overwrite":
            gesture = None
            overwrite = True
            save = False
        elif sys.argv[1] == "--save":
            gesture = None
            overwrite = False
            save = True
        else:
            gesture = sys.argv[1]
            overwrite = False
            save = False
    else:
        gesture = None
        overwrite = False
        save = False

    extract_frame_images(
        video_dir="../../RockPaperScissors/my_rps_dataset/data/align2",
        destination_dir="../../RockPaperScissors/my_rps_dataset/data/align2",
        gesture_class=gesture,
        save=save,
        overwrite=overwrite,
    )

    create_empty_metadata_file(
        destination_dir="../../RockPaperScissors/my_rps_dataset/data/align2",
        gesture_class=gesture,
    )

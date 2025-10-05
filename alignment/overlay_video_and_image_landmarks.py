import os
import sys
import cv2
import json
import numpy as np

sys.path.append("../")
from modules.utils import (
    load_landmarks,
    get_video_metadata,
    get_cropped_video_indices,
    load_splits_cropped,
    ALIGN_DIR,
)


def overlay_video_and_image_landmarks(
    video_landmarks_path, video_frames_path, gesture_class: str = None
):

    if gesture_class is not None:
        video_frames_dir = [gesture_class]
    else:
        video_frames_dir = os.listdir(video_frames_path)

    for gesture_dir in video_frames_dir:
        gesture_path = os.path.join(video_frames_path, gesture_dir)

        print(f"Processing gesture directory: {gesture_path}")
        for img_dir in os.listdir(os.path.join(gesture_path, "images")):
            if img_dir.startswith("."):
                continue
            # Load landmarks from corresponding video
            image_frames_dir = os.path.join(gesture_path, "images", img_dir)
            image_frames = [
                f for f in os.listdir(image_frames_dir) if f.endswith(".png")
            ]
            video_landmarks = load_landmarks(
                os.path.join(video_landmarks_path, f"{img_dir}_landmarks.npz")
            )
            print(
                f"    Processing {img_dir} {video_landmarks.shape} ({len(image_frames)} frames)"
            )

            for frame_file in image_frames:
                # Get video landmarks for given frame
                image_path = os.path.join(gesture_path, "images", img_dir, frame_file)
                frame_idx = frame_file.split(".")[0].split("_")[-1]
                frame_file_idx = int(frame_idx)

                image_landmarks = video_landmarks[frame_file_idx]
                image = cv2.imread(image_path)
                if image_landmarks is None:
                    print(f"Error loading image file: {image_path}")
                    continue

                image_landmarks_3d = image_landmarks.reshape(-1, 3)
                new_img_dir = os.path.join(image_frames_dir, "vid_landmarks")
                os.makedirs(new_img_dir, exist_ok=True)

                # Visialize 3D landmarks on image and save as image
                viz_img = image.copy()
                for landmark in image_landmarks_3d:
                    x, y, z = landmark
                    # Convert world coordinates to pixel coordinates
                    x = int(x * image.shape[1])
                    y = int(y * image.shape[0])
                    z = z
                    # check if nan
                    if np.isnan(x) or np.isnan(y) or np.isnan(z):
                        continue
                    # print(f"Landmark: x={x}, y={y}, z={z}")
                    cv2.circle(viz_img, (int(x), int(y)), 3, (0, 0, 255), -1)
                # cv2.imshow("Image with 3D Landmarks", viz_img)
                # cv2.imwrite(f"{new_img_dir}/{frame_file}", viz_img)
                print(f"Saved image with landmarks to {new_img_dir}/{frame_file}")


if __name__ == "__main__":

    LANDMARKS_DIR_VID = (
        "/Users/christina/code/RockPaperScissors/my_rps_dataset/landmarks"
    )
    ALIGN_DIR = "../../RockPaperScissors/my_rps_dataset/data/align/"

    # Read gesture_class arg from command line
    if sys.argv and len(sys.argv) > 1:
        gesture_class_arg = sys.argv[1]
        print(f"Processing only gesture class: {gesture_class_arg}")
    else:
        gesture_class_arg = None
        print("Processing all gesture classes")

    overlay_video_and_image_landmarks(
        LANDMARKS_DIR_VID, ALIGN_DIR, gesture_class=gesture_class_arg
    )

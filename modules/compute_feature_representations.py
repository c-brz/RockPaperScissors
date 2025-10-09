import os
import numpy as np
import pandas as pd
from get_frames import get_video_csv, get_frame_landmarks


def normalize_landmarks(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize landmarks relative to wrist position at each frame."""

    # print("Normalizing landmarks with transform()")

    if landmarks_df.empty:
        return landmarks_df

    landmarks_df[["x_world_norm", "y_world_norm", "z_world_norm"]] = (
        landmarks_df.groupby("frame_idx")[
            ["x_world_m", "y_world_m", "z_world_m"]
        ].transform(lambda x: x - x.iloc[0])
    )

    return landmarks_df


def scale_landmarks(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """Scale landmarks of each frame by the hand size (assume landmarks are normalized wrt wrist)."""

    if landmarks_df.empty:
        return landmarks_df

    def scale_func(group):
        coords = ["x_world_norm", "y_world_norm", "z_world_norm"]
        result = group.copy()

        for coord in coords:
            coord_max = group[coord].max()
            coord_min = group[coord].min()
            hand_size = coord_max - coord_min
            if hand_size == 0:
                hand_size = 1.0
            result[coord.replace("norm", "scaled")] = (
                group[coord] - coord_min
            ) / hand_size

        return result

    landmarks_df = landmarks_df.groupby("frame_idx", group_keys=False).apply(scale_func)

    return landmarks_df

    return landmarks_df


if __name__ == "__main__":
    video_base_dir = (
        "/Users/christina/code/RockPaperScissors/my_rps_dataset/data/align2"
    )
    # video_name = "paper_2"
    csv_name = "img_landmarks"

    gesture_dirs = [f for f in os.listdir(video_base_dir) if not f.startswith(".")]

    for gesture_dir in gesture_dirs:
        gesture_path = os.path.join(video_base_dir, gesture_dir)
        video_names = [f for f in os.listdir(gesture_path) if not f.startswith(".")]

        vid_dirs = [
            f
            for f in os.listdir(os.path.join(gesture_path, "images"))
            if not f.startswith(".")
        ]

        for video_name in vid_dirs:
            video_base_dir2 = os.path.join(gesture_path, "images", video_name)

            df = get_video_csv(video_base_dir, video_name, csv_name)
            df_norm = normalize_landmarks(df)
            df_scaled = scale_landmarks(df_norm)
            df_scaled.to_csv(
                os.path.join(
                    video_base_dir,
                    gesture_dir,
                    "images",
                    video_name,
                    "img_landmarks_norm_v3.csv",
                ),
                index=False,
            )

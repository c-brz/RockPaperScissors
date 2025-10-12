import os
import numpy as np
import pandas as pd
from get_frames import get_video_csv, get_frame_landmarks
from sklearn import preprocessing


def normalize_landmarks(
    landmarks_df: pd.DataFrame,
    group_col: str = "frame_idx",
    coord_cols: list = ["x_world_m", "y_world_m", "z_world_m"],
) -> pd.DataFrame:
    """Normalize landmarks relative to wrist position at each frame."""

    if landmarks_df.empty:
        return landmarks_df

    landmarks_df[["x_world_norm", "y_world_norm", "z_world_norm"]] = (
        landmarks_df.groupby(group_col)[coord_cols].transform(lambda x: x - x.iloc[0])
    )

    return landmarks_df


def scale_landmarks_standard(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """Scale landmarks ([0, 1]) of each frame by the hand size."""

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


def scale_landmarks(landmarks_df, group_col, coord_cols):
    """Scale landmarks ([-1, 1]) of each frame by the hand size."""

    def scale_func(grouped_landmarks, coord_cols, feature_range=(-1, 1)):

        if grouped_landmarks.empty:
            return grouped_landmarks

        res = grouped_landmarks.copy()
        new_cols = [i.replace("_norm", "_scaled") for i in coord_cols]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
        res.loc[:, new_cols] = min_max_scaler.fit_transform(
            grouped_landmarks[coord_cols].values
        )
        return res

    result = landmarks_df.groupby(group_col, group_keys=False).apply(
        scale_func, coord_cols
    )
    return result


if __name__ == "__main__":

    # Get video_base_dir, csv_name, gesture_dirs, save_as, save_data (bool) from command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_base_dir", type=str, default=video_base_dir)
    parser.add_argument("--csv_name", type=str, default=csv_name)
    parser.add_argument("--save_as", type=str, default=save_as)
    args = parser.parse_args()

    video_base_dir = args.video_base_dir
    csv_name = args.csv_name
    gesture_dirs = args.gesture_dirs
    save_as = args.save_as

    # video_base_dir = ("/Users/christina/code/RockPaperScissors/my_rps_dataset/data/align2")
    # csv_name = "img_landmarks"
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

            df_norm = normalize_landmarks(
                df,
                group_col="frame_idx",
                coord_cols=["x_world_m", "y_world_m", "z_world_m"],
            )

            df_scaled = scale_landmarks(
                df_norm,
                group_col="frame_idx",
                coord_cols=["x_world_norm", "y_world_norm", "z_world_norm"],
            )

            # If save_as is provided, save the normalized and scaled landmarks
            if save_as:
                df_scaled.to_csv(
                    os.path.join(
                        video_base_dir,
                        gesture_dir,
                        "images",
                        video_name,
                        f"{save_as}.csv",
                    ),
                    index=False,
                )

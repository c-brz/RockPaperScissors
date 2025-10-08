import numpy as np
import pandas as pd
import csv
import cv2
import os


def get_video_csv(video_base_dir, video_name, csv_name):
    """Load the CSV file with landmark info for a given video."""
    # video_name = os.path.splitext(os.path.basename(video_base_dir))[0]
    gesture = video_name.split("_")[0]
    csv_path = os.path.join(
        video_base_dir, gesture, "images", video_name, f"{csv_name}.csv"
    )
    print(f"Reading CSV file at: {csv_path}")
    df = pd.read_csv(csv_path)

    return df


def get_frame_landmarks(vid_df, frame_id=None, return_world_lm=False) -> pd.DataFrame:
    """Get landmarks for a specific frame from the video dataframe. If frame_id is None, return all frames."""

    if frame_id is None:
        frame_data = vid_df
    else:
        frame_data = vid_df[vid_df["frame_idx"] == frame_id]
        if not return_world_lm:
            frame_data = frame_data[
                ["x_world_m", "y_world_m", "z_world_m"]
            ]  # (n_landmarks, 3)
    return frame_data


if __name__ == "__main__":
    # . /Users/christina/code/RockPaperScissors/my_rps_dataset/data/align2
    #   /paper
    #       /images
    #           /paper_2
    #               /img_landmarks.csv

    video_base_dir = (
        "/Users/christina/code/RockPaperScissors/my_rps_dataset/data/align2"
    )
    video_name = "paper_49"
    csv_name = "img_landmarks"

    df = get_video_csv(video_base_dir, video_name, csv_name)
    frame_landmarks = get_frame_landmarks(df, frame_id=20, return_world_lm=True)

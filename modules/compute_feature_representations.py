import os
import numpy as np
import pandas as pd
from get_frames import get_video_csv, get_frame_landmarks
from sklearn import preprocessing
import argparse
from enum import Enum


class Landmarks(Enum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


def normalize_landmarks(
    landmarks_df: pd.DataFrame,
    group_col: str = "frame_idx",
    coord_cols: list = ["x_world_m", "y_world_m", "z_world_m"],
    convert_to_mm: bool = True,
) -> pd.DataFrame:
    """Normalize landmarks relative to wrist position at each frame."""

    if landmarks_df.empty:
        return landmarks_df

    landmarks_df[["x_world_norm", "y_world_norm", "z_world_norm"]] = (
        landmarks_df.groupby(group_col)[coord_cols].transform(lambda x: x - x.iloc[0])
    )
    if convert_to_mm:
        landmarks_df[["x_world_norm", "y_world_norm", "z_world_norm"]] *= 1000.0
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


def compute_global_scaling_factor(
    video_base_dir: str,
    csv_name: str,
    cols_to_use: list,
    normalize_first: bool = True,
) -> float:

    def compute_hand_size(landmarks: pd.DataFrame, cols_to_use: list) -> float:
        """Compute hand size as the distance between wrist and middle finger MCP."""
        wrist = landmarks[landmarks.landmark_idx == Landmarks.WRIST.value][
            cols_to_use
        ].values[0]
        middle_finger_mcp = landmarks[
            landmarks.landmark_idx == Landmarks.MIDDLE_FINGER_MCP.value
        ][cols_to_use].values[0]
        hand_size = np.linalg.norm(np.array(wrist) - np.array(middle_finger_mcp)).item()

        return hand_size

    ALL_SCALE_FACTORS = []
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
            df = get_video_csv(video_base_dir, video_name, csv_name)
            df = normalize_landmarks(df, convert_to_mm=True) if normalize_first else df
            res = df.groupby("frame_idx").apply(compute_hand_size, cols_to_use).values
            ALL_SCALE_FACTORS.append(res)

    global_scale_factor = np.mean(
        [item for sublist in ALL_SCALE_FACTORS for item in sublist]
    )
    print(
        f"\n\n===== Global scale factor: {global_scale_factor} (out of {len(ALL_SCALE_FACTORS)} samples) =====\n\n"
    )
    return global_scale_factor


def scale_landmarks_fixed(landmarks_df, group_col, coord_cols):
    """Scale landmarks of each frame by the hand size using fixed landmarks (wrist and middle finger mcp)."""

    def scale_func(grouped_landmarks, coord_cols, feature_range=(-1, 1)):

        if grouped_landmarks.empty:
            return grouped_landmarks

        res = grouped_landmarks.copy()
        new_cols = [
            i.replace("_norm", "_scaled") for i in coord_cols
        ]  # TODO: make this more general

        hand_size = np.linalg.norm(
            res[res.landmark_idx == Landmarks.PINKY_TIP.value][coord_cols].values
            - res[res.landmark_idx == Landmarks.WRIST.value][coord_cols].values
        )

        if hand_size == 0:
            hand_size = 1.0

        res.loc[:, new_cols] = grouped_landmarks[coord_cols].values / hand_size

        return res

    result = landmarks_df.groupby(group_col, group_keys=False).apply(
        scale_func, coord_cols
    )
    return result


def rotate_landmarks(landmarks_df, group_col, coords_col) -> pd.DataFrame:

    if landmarks_df.empty:
        return landmarks_df

    def rotate_func(grp1, coords_col, base_lm=[0, 5, 17]):

        result = grp1.copy()

        wrist = grp1[grp1.landmark_idx == base_lm[0]][coords_col].values[0]
        index_mcp = grp1[grp1.landmark_idx == base_lm[1]][coords_col].values[0]
        pinky_mcp = grp1[grp1.landmark_idx == base_lm[2]][coords_col].values[0]

        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        def normalize_vec(v):
            return v / np.linalg.norm(v)

        norm_v1_v2 = np.cross(v1, v2)
        norm_v1_v2 = normalize_vec(norm_v1_v2)

        new_x_axis = normalize_vec(v1)
        new_z_axis = normalize_vec(norm_v1_v2)
        new_y_axis = np.cross(new_z_axis, new_x_axis)

        R = np.stack([new_x_axis, new_y_axis, new_z_axis], axis=1)
        assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix is not valid"
        rotated_coords = np.dot(R.T, ((grp1[coords_col].values)).T).T
        rotated_coords = (rotated_coords + 1) / 2
        rotated_coords = rotated_coords - rotated_coords[0]
        result.loc[:, ["x_world_rotated", "y_world_rotated", "z_world_rotated"]] = (
            rotated_coords
        )

        return result

    result = landmarks_df.groupby(group_col, group_keys=False).apply(
        rotate_func, coords_col=coords_col
    )

    return result


def _angle_between_points(a, b, c):
    """Calculate the angle (in degrees) at point b given points a, b, c.
    --------
    a, b, c: (x, y) or (x, y, z) coordinates of the points.
    Returns: angle in degrees.
    """

    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def _get_line_segments():
    finger_landmarks = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }
    WRIST = 0

    line_segments = []
    for finger_lm in finger_landmarks.values():
        line_segments.append((WRIST, finger_lm[0]))
        for i in range(len(finger_lm) - 1):
            start_lm = finger_lm[i]
            end_lm = finger_lm[i + 1]
            line_segments.append((start_lm, end_lm))
    return line_segments


def compute_angles_between_landmarks(landmarks_df, group_col, coords_col):

    if landmarks_df.empty:
        return []

    def angle_func(group, coords_col):
        # res = group.copy()
        line_segments = _get_line_segments()
        angles = []
        for seg_idx, seg in enumerate(line_segments):
            if seg_idx == len(line_segments) - 1:
                break
            seg1 = seg
            seg2 = line_segments[seg_idx + 1]
            coords_1 = group[group.landmark_idx == seg1[0]][coords_col].values[0]
            coords_2 = group[group.landmark_idx == seg1[1]][coords_col].values[0]
            coords_3 = group[group.landmark_idx == seg2[1]][coords_col].values[0]
            angle = _angle_between_points(coords_1, coords_2, coords_3)
            print(
                f"Angle between {seg1} and {seg2}: {angle} -> {coords_1}, {coords_2}, {coords_3}"
            )
            angles.append(angle)
        return angles

    result = landmarks_df.groupby(group_col, group_keys=False).apply(
        angle_func, coords_col=coords_col
    )
    result.name = "angles"
    return result


if __name__ == "__main__":

    # Get video_base_dir, csv_name, gesture_dirs, save_as, save_data (bool) from command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_base_dir", type=str)
    parser.add_argument("--csv_name", type=str)
    parser.add_argument("--save_as", type=str)
    parser.add_argument(
        "--compute_coords",
        action="store_true",
        help="Compute normalized/scaled/rotated coordinates",
    )
    parser.add_argument(
        "--compute_angles", action="store_true", help="Compute angles between landmarks"
    )

    args = parser.parse_args()

    # Ensure only one computation type is specified
    if args.compute_coords and args.compute_angles:
        parser.error("Cannot specify both --compute_coords and --compute_angles")
    if not args.compute_coords and not args.compute_angles:
        parser.error("Must specify either --compute_coords or --compute_angles")

    video_base_dir = args.video_base_dir
    csv_name = args.csv_name
    save_as = args.save_as

    GLOBAL_SCALE_FACTOR = compute_global_scaling_factor(
        video_base_dir,
        csv_name,
        cols_to_use=["x_world_norm", "y_world_norm", "z_world_norm"],
        normalize_first=True,
    )
    print(f"Using GLOBAL_SCALE_FACTOR={GLOBAL_SCALE_FACTOR}")

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
                convert_to_mm=True,
            )

            # df_scaled = scale_landmarks(
            #     df_norm,
            #     group_col="frame_idx",
            #     coord_cols=["x_world_norm", "y_world_norm", "z_world_norm"],
            # )

            df_scaled = scale_landmarks(
                df_norm,
                group_col="frame_idx",
                coord_cols=["x_world_norm", "y_world_norm", "z_world_norm"],
            )

            df_scaled2 = df_norm.copy()
            df_scaled2[["x_world_scaled", "y_world_scaled", "z_world_scaled"]] = (
                df_norm[["x_world_norm", "y_world_norm", "z_world_norm"]]
                / GLOBAL_SCALE_FACTOR
            )

            df_rotated = rotate_landmarks(
                df_scaled2,
                group_col="frame_idx",
                coords_col=[
                    "x_world_scaled",
                    "y_world_scaled",
                    "z_world_scaled",
                ],
            )

            df_rotated[["x_world_scaled", "y_world_scaled", "z_world_scaled"]] = (
                df_scaled[["x_world_scaled", "y_world_scaled", "z_world_scaled"]]
            )
            # df_rotated = get_video_csv(video_base_dir, video_name, csv_name)

            if args.compute_angles:
                print("Computing angles...")
                final_df = compute_angles_between_landmarks(
                    df_rotated,
                    "frame_idx",
                    ["x_world_scaled", "y_world_scaled", "z_world_scaled"],
                )
            elif args.compute_coords:
                final_df = df_rotated.copy()

            # If save_as is provided, save the normalized and scaled landmarks
            if save_as:
                final_df.to_csv(
                    os.path.join(
                        video_base_dir,
                        gesture_dir,
                        "images",
                        video_name,
                        f"{save_as}.csv",
                    ),
                    index=False if args.compute_coords else True,
                )

                print(
                    f"🟢 Saving {gesture_dir}/{video_name}, final_df shape={final_df.shape}"
                )

    # Example usage: python compute_feature_representations.py --video_base_dir "/Users/christina/code/RockPaperScissors/my_rps_dataset/data/align2" --csv_name "img_landmarks" --save_as "img_landmarks_norm_v8" --compute_coords

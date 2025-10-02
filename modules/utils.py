import numpy as np
import json
import os

# Config
SPLIT_FILE = "../../my_rps_dataset/splits_noval.json"
DATA_DIR_LIST = [
    "../../my_rps_dataset/features_coords",
    "../../my_rps_dataset/features_angles",
]
ALIGN_DIR = "RockPaperScissors/my_rps_dataset/data/align/"

OBS_RATIOS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load_landmarks(path):
    """Load npz containing landmarks."""
    data = np.load(path)
    return data["landmarks"]  # shape (T, K, 2 or 3)


def frame_to_vector(lm_frame):
    """Flatten keypoints from (K,3) -> (3K,)."""
    return lm_frame.reshape(-1)


def sequence_to_feature_matrix(lm_seq, use_vel=False):
    """Convert sequence of landmarks (T,K,D) into matrix (T, D*K)."""

    # Check if lm_seq is actually 3D
    if len(lm_seq.shape) == 2:
        return lm_seq

    T, K, D = lm_seq.shape
    X = np.stack([frame_to_vector(lm_seq[t]) for t in range(T)], axis=0)  # (T, K*D)
    if use_vel:
        V = np.zeros_like(X)
        V[1:] = X[1:] - X[:-1]
        X = np.concatenate([X, V], axis=1)  # (T, 2*K*D)
    return X


# Dataset loading
def load_splits(split_file, data_dir):
    with open(split_file, "r") as f:
        splits = json.load(f)
    datasets = {}

    for split_name, file_list in splits.items():
        samples = []
        for fname in file_list:
            path = os.path.join(data_dir, fname)
            label = fname.split("_")[0]  # class from filename prefix
            lm = load_landmarks(path)
            samples.append({"id": fname, "label": label, "landmarks": lm})
        datasets[split_name] = samples
    return datasets


def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(y_true == y_pred))


def get_overlapping_chunks(features, chunk_size=5, overlap=1):
    """Get overlapping chunks of the features (x[0:chunk_size], x[1:chunk_size+1]) along the first dimension"""
    step = chunk_size - overlap
    chunks = []
    idx = []
    # Assume features is (T, D, 3)
    n_frames = features.shape[0]
    for start in range(0, n_frames - chunk_size + 1, overlap):
        chunk = features[start : start + chunk_size]
        # print((start, start + chunk_size), overlap)
        chunks.append(chunk)
        idx.append((start, start + chunk_size))
    chunks.append(features[start:])
    idx.append((start + chunk_size, n_frames))
    return chunks, idx


def get_video_metadata(video_dir):
    gesture_dirs = [
        d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))
    ]
    all_metadata = []
    for gesture_name in gesture_dirs:
        images_path = os.path.join(video_dir, gesture_name, "images")
        # Safely check if images_path contains any files
        if not os.path.exists(images_path):
            print(f"{images_path} does not exist")
            continue
        image_dirs = [f for f in os.listdir(images_path)]
        metadata_files = [
            os.path.join(images_path, d, "metadata.json")
            for d in image_dirs
            if (
                d.startswith(f"{gesture_name}")
                and os.path.exists(os.path.join(images_path, d, "metadata.json"))
            )
        ]

        # metadata_path = os.path.join(video_dir, gesture_name, "images/metadata.json")
        # print(f"Reading {metadata_path}")
        print(f"{gesture_name}: {len(image_dirs)} image files")
        print(metadata_files)
        all_metadata.extend(metadata_files)
    return all_metadata


def get_cropped_video_indices(video_metadata: dict):
    start, onset, end, n_frames = (
        video_metadata["start_frame_idx"],
        video_metadata["action_frame_idx"],
        video_metadata["end_frame_idx"],
        video_metadata["total_frames"],
    )
    return start, onset, end, n_frames


def load_splits_cropped(gesture_dir, data_dir):

    datasets = {}
    gestures = [
        d
        for d in os.listdir(gesture_dir)
        if os.path.isdir(os.path.join(gesture_dir, d))
    ]
    for gesture in gestures:
        samples = []

        images_path = os.path.join(gesture_dir, gesture, "images")
        video_dirs = [
            d
            for d in os.listdir(images_path)
            if os.path.isdir(os.path.join(images_path, d))
        ]

        for vid_name in video_dirs:
            landmarks_path = os.path.join(data_dir, f"{vid_name}_landmarks.npz")
            lm = load_landmarks(landmarks_path)
            metadata_file = os.path.join(images_path, vid_name, "metadata.json")
            # print(f"Loading metadata from {metadata_file}")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                start, onset, end, n_frames = get_cropped_video_indices(data)
                lm = lm[start : end + 1]
                samples.append({"id": f"{vid_name}", "label": gesture, "landmarks": lm})
        datasets[gesture] = samples
    return datasets

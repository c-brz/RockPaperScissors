import os
import sys
import json
import numpy as np


# Read existing metadata files and update end_frame_idx
def get_metadata_files(destination_dir, gesture_class=None):
    gesture_dirs = (
        [
            d
            for d in os.listdir(destination_dir)
            if os.path.isdir(os.path.join(destination_dir, d))
        ]
        if gesture_class is None
        else [gesture_class]
    )

    all_meta = []

    for gesture_dir in gesture_dirs:
        img_dirs = os.path.join(destination_dir, gesture_dir, "images")

        for img_dir in os.listdir(img_dirs):
            metadata_path = os.path.join(img_dirs, img_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                gesture_duration = (
                    metadata["end_frame_idx"] - metadata["action_frame_idx"] + 1
                )
                # print((f"found {metadata['total_frames']} frames"))
                all_meta.append(
                    {"id": img_dir, "duration": gesture_duration, **metadata}
                )

    return all_meta


def get_max_duration(destination_dir, gesture_class=None):
    all_meta = get_metadata_files(destination_dir, gesture_class)
    if not all_meta:
        return 0
    # Find sample with maximum duration and return its metadata
    max_meta = max(all_meta, key=lambda x: x["duration"])

    return max_meta


def upsample_video_frames(destination_dir, gesture_class=None):
    all_meta = get_metadata_files(destination_dir, gesture_class)
    if not all_meta:
        return

    max_duration_sample = get_max_duration(destination_dir, gesture_class)
    max_duration = max_duration_sample["duration"]
    print(f"Max duration sample: {max_duration_sample}")

    for meta in all_meta:
        gesture_duration = meta["duration"]
        if gesture_duration >= max_duration:
            continue  # no need to upsample

        # Update metadata file
        meta["updated"] = True
        meta["end_frame_idx_old"] = meta["end_frame_idx"]
        meta["end_frame_idx"] = min(
            meta["action_frame_idx"] + max_duration - 1, meta["total_frames"] - 1
        )

        # save updated metadata
        gesture_name = meta["label"]
        img_dir = os.path.join(destination_dir, gesture_name, "images", meta["id"])
        metadata_path = os.path.join(img_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=4)
        print(f"Updated metadata file at {metadata_path}")

    return


if __name__ == "__main__":

    align_dir = "../../RockPaperScissors/my_rps_dataset/data/align"
    # upsample_video_frames(align_dir, gesture_class=None)
    all_meta = get_metadata_files(align_dir, gesture_class=None)
    print(f"Found {len(all_meta)} metadata files")

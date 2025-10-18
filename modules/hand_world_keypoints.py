import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple, Iterable

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

HAND_LANDMARK_NAMES = [
    "WRIST",  # 0
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",  # 1-4
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",  # 5-8
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",  # 9-12
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",  # 13-16
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",  # 17-20
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch MediaPipe Hands on images: export 3D world landmarks (m) and 2D (pixels & normalized)."
    )
    # Inputs
    p.add_argument(
        "inputs",
        nargs="+",
        help="File(s), directory(ies), or glob(s). Example: imgs/ or imgs/*.jpg hand1.png",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="If an input is a directory, search recursively for images.",
    )
    p.add_argument(
        "--patterns",
        default="*.jpg,*.jpeg,*.png",
        help="Comma-separated glob patterns used inside directories (default: *.jpg,*.jpeg,*.png)",
    )

    # MediaPipe settings
    p.add_argument("--max_hands", type=int, default=1)
    p.add_argument("--det_conf", type=float, default=0.5)
    p.add_argument("--complexity", type=int, choices=[0, 1], default=1)

    # Outputs
    p.add_argument(
        "--save_overlay",
        action="store_true",
        help="Write an overlay PNG next to each image (or to --out_dir if set).",
    )
    p.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Append all landmarks to this CSV (created if missing).",
    )
    p.add_argument(
        "--save_json",
        type=str,
        default="",
        help="If set, write per-image JSON files to this folder.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="If set, write overlays to this directory instead of next to the image.",
    )

    return p.parse_args()


def iter_image_files(
    inputs: List[str], recursive: bool, patterns: str
) -> Iterable[Path]:
    pats = [p.strip() for p in patterns.split(",") if p.strip()]
    seen = set()
    for item in inputs:
        p = Path(item)
        if any(ch in item for ch in "*?[]"):  # glob pattern in CLI
            for q in sorted(Path().glob(item)):
                if q.is_file() and q.suffix.lower() in IMG_EXTS and q not in seen:
                    seen.add(q)
                    yield q
            continue
        if p.is_file():
            if p.suffix.lower() in IMG_EXTS and p not in seen:
                seen.add(p)
                yield p
        elif p.is_dir():
            if recursive:
                for pat in pats:
                    for q in sorted(p.rglob(pat)):
                        if (
                            q.is_file()
                            and q.suffix.lower() in IMG_EXTS
                            and q not in seen
                        ):
                            seen.add(q)
                            yield q
            else:
                for pat in pats:
                    for q in sorted(p.glob(pat)):
                        if (
                            q.is_file()
                            and q.suffix.lower() in IMG_EXTS
                            and q not in seen
                        ):
                            seen.add(q)
                            yield q
        else:
            # Non-existing path ignored
            continue


def group_files_by_directory(files: List[Path]) -> dict:
    """Group files by their parent directory."""
    groups = {}
    for file_path in files:
        dir_path = file_path.parent
        if dir_path not in groups:
            groups[dir_path] = []
        groups[dir_path].append(file_path)
    return groups


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_csv(path: str):
    file_path = Path(path)
    is_new = not file_path.exists()
    ensure_parent(file_path)
    f = open(file_path, "w", newline="", encoding="utf-8")  # Changed from "a" to "w"
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_path",
            "frame_idx",
            "image_width",
            "image_height",
            "hand_idx",
            "handedness",
            "landmark_idx",
            "landmark_name",
            "x_world_m",
            "y_world_m",
            "z_world_m",
            "x_px",
            "y_px",
            "x_norm",
            "y_norm",
        ],
    )
    writer.writeheader()  # Always write header since we're creating new files
    return writer, f


def get_csv_path_for_directory(base_csv_path: str, directory: Path) -> Path:
    """Generate a CSV path for a specific directory, saved in that directory."""
    if not base_csv_path:
        return None

    base_path = Path(base_csv_path)

    # Save the CSV in the same directory as the images
    csv_name = f"{base_path.stem}.csv"
    return directory / csv_name


def json_out_path(json_dir: str, img_path: Path) -> Path:
    out_dir = Path(json_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{img_path.stem}_hands.json"


def overlay_out_path(out_dir: str, img_path: Path) -> Path:
    if out_dir:
        out = Path(out_dir) / f"{img_path.stem}_overlay.png"
    else:
        out = img_path.with_name(f"{img_path.stem}_overlay.png")
    ensure_parent(out)
    return out


def main():
    args = parse_args()

    # Collect images
    files = list(iter_image_files(args.inputs, args.recursive, args.patterns))
    if not files:
        print("No images found.")
        return

    # Group files by directory if using CSV output
    if args.save_csv:
        file_groups = group_files_by_directory(files)
        print(f"Found {len(file_groups)} directories with images:")
        for directory, file_list in file_groups.items():
            print(f"  {directory}: {len(file_list)} files")
    else:
        # If not using CSV, treat all files as one group
        file_groups = {Path("."): files}

    # MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=args.max_hands,
        model_complexity=args.complexity,
        min_detection_confidence=args.det_conf,
    )

    try:
        # Process each directory group
        for directory, files_in_dir in file_groups.items():
            csv_writer, csv_file = (None, None)

            # Set up CSV for this directory if needed
            if args.save_csv:
                csv_path = get_csv_path_for_directory(args.save_csv, directory)
                if csv_path:
                    csv_writer, csv_file = prepare_csv(str(csv_path))
                    print(f"\nProcessing directory: {directory}")
                    print(f"CSV output: {csv_path}")

            # Process files in this directory
            for img_path in files_in_dir:
                # Check if the filename contains "overlay" to avoid reprocessing overlays
                if "overlay" in img_path.name:
                    continue
                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"Skipping (failed to read): {img_path}")
                    continue
                H, W = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                res = hands.process(rgb)

                handedness_labels: List[str] = []
                if res.multi_handedness:
                    for hnd in res.multi_handedness:
                        handedness_labels.append(
                            hnd.classification[0].label
                        )  # "Left" / "Right"

                world_sets = res.multi_hand_world_landmarks or []
                pix_sets = res.multi_hand_landmarks or []

                # Console summary
                print(f"\nImage: {img_path}")
                if not world_sets and not pix_sets:
                    print(world_sets)
                    print(pix_sets)
                    print("  No hands detected.")
                    continue

                # JSON payload (per image)
                img_payload = {
                    "image_path": str(img_path),
                    "image_size": {"width": W, "height": H},
                    "hands": [],
                }

                # Build overlay if requested
                overlay = bgr.copy() if args.save_overlay and pix_sets else None
                if overlay is not None:
                    print("     Creating overlay image.")

                # Iterate hands by index
                num_hands = max(len(world_sets), len(pix_sets))
                for i in range(num_hands):
                    label = (
                        handedness_labels[i]
                        if i < len(handedness_labels)
                        else "Unknown"
                    )
                    wlms = world_sets[i] if i < len(world_sets) else None
                    plms = pix_sets[i] if i < len(pix_sets) else None

                    # Collect lists
                    world_list = []
                    pixel_list = []
                    norm_list = []

                    for j in range(21):
                        # World (meters)
                        if wlms is not None and j < len(wlms.landmark):
                            wm = wlms.landmark[j]
                            wx, wy, wz = wm.x, wm.y, wm.z
                        else:
                            wx = wy = wz = None

                        # 2D normalized (0-1) and pixels
                        if plms is not None and j < len(plms.landmark):
                            pm = plms.landmark[j]
                            nx, ny = float(pm.x), float(pm.y)
                            x_px = int(round(nx * W))
                            y_px = int(round(ny * H))
                        else:
                            nx = ny = None
                            x_px = y_px = None

                        world_list.append(
                            {
                                "index": j,
                                "name": HAND_LANDMARK_NAMES[j],
                                "x": wx,
                                "y": wy,
                                "z": wz,
                            }
                        )
                        norm_list.append(
                            {
                                "index": j,
                                "name": HAND_LANDMARK_NAMES[j],
                                "x": nx,
                                "y": ny,
                            }
                        )
                        pixel_list.append(
                            {
                                "index": j,
                                "name": HAND_LANDMARK_NAMES[j],
                                "x": x_px,
                                "y": y_px,
                            }
                        )

                        # CSV row
                        if csv_writer is not None:
                            csv_writer.writerow(
                                {
                                    "image_path": str(img_path),
                                    "frame_idx": int(
                                        os.path.basename(
                                            img_path
                                        )  # e.g. frame_0009.png
                                        .strip()
                                        .split(".")[0]
                                        .split("_")[-1]
                                    ),
                                    "image_width": W,
                                    "image_height": H,
                                    "hand_idx": i,
                                    "handedness": label,
                                    "landmark_idx": j,
                                    "landmark_name": HAND_LANDMARK_NAMES[j],
                                    "x_world_m": "" if wx is None else f"{wx:.6f}",
                                    "y_world_m": "" if wy is None else f"{wy:.6f}",
                                    "z_world_m": "" if wz is None else f"{wz:.6f}",
                                    "x_px": "" if x_px is None else x_px,
                                    "y_px": "" if y_px is None else y_px,
                                    "x_norm": "" if nx is None else f"{nx:.6f}",
                                    "y_norm": "" if ny is None else f"{ny:.6f}",
                                }
                            )

                    # Add to JSON payload
                    img_payload["hands"].append(
                        {
                            "index": i,
                            "handedness": label,
                            "world_landmarks_m": world_list,
                            "norm_landmarks": norm_list,
                            "pixel_landmarks": pixel_list,
                        }
                    )

                    # Draw overlay
                    if overlay is not None and plms is not None:
                        mp_drawing.draw_landmarks(
                            overlay,
                            plms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

                # Save JSON
                if args.save_json:
                    out_json = json_out_path(args.save_json, img_path)
                    ensure_parent(out_json)
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(img_payload, f, indent=2)
                    print(f"  Wrote JSON: {out_json}")

                # Save overlay
                if overlay is not None:
                    out_img = overlay_out_path(args.out_dir, img_path)
                    cv2.imwrite(str(out_img), overlay)
                    print(f"  Saved overlay: {out_img}")

            # Close CSV file for this directory
            if csv_file is not None:
                csv_file.close()
                print(f"Closed CSV for directory: {directory}")

    finally:
        hands.close()


if __name__ == "__main__":
    # Example usage: python hand_world_keypoints.py "../../RockPaperScissors/my_rps_dataset/data/align2" --recursive --save_overlay --save_csv img_landmarks
    main()

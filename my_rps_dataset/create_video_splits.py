import os
import json
import numpy as np
from glob import glob
import sys

DATA_DIR = "features/"
OUT_FILE = "./splits.json"
RANDOM_SEED = 42
RATIOS_TTV = (0.7, 0.2, 0.1)  # train, test, val
RATIOS_TT = (0.7, 0.3)  # train, test
np.random.seed(RANDOM_SEED)
use_val = True

if __name__ == "__main__":

    # Read DATA_DIR and OUT_FILE from command line if provided or use default otherwieså
    # Flags: --data_dir <path> --out_file <path> --use_val <True/False> --help for usage
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Usage: python create_video_splits.py [--data_dir <path>] [--out_file <path>] [--use_val <True/False>]"
        )
        sys.exit(0)
    N_ARGS = len(sys.argv)
    if N_ARGS > 1:
        for i in range(1, N_ARGS, 2):
            if sys.argv[i] == "--data_dir":
                DATA_DIR = sys.argv[i + 1]
            elif sys.argv[i] == "--out_file":
                OUT_FILE = sys.argv[i + 1]
            elif sys.argv[i] == "--use_val":
                use_val = sys.argv[i + 1].lower() == "true"
                RATIOS = RATIOS_TTV if use_val else RATIOS_TT
        print(f"Using DATA_DIR: {DATA_DIR}, OUT_FILE: {OUT_FILE}, use_val: {use_val}")
    else:
        print(f"Using default DATA_DIR: {DATA_DIR} and OUT_FILE: {OUT_FILE}")
        RATIOS = RATIOS_TTV if use_val else RATIOS_TT

    files = sorted(glob(os.path.join(DATA_DIR, "*.npz")))

    classes = {}
    for f in files:
        label = os.path.basename(f).split("_")[0]  # "rock", "paper", "scissors"
        classes.setdefault(label, []).append(os.path.basename(f))

    splits = {"train": [], "val": [], "test": []}

    # Do stratified splitting per class
    for label, flist in classes.items():
        n = len(flist)
        np.random.shuffle(flist)
        n_train = int(RATIOS[0] * n)
        train_files = flist[:n_train]
        if use_val:
            RATIOS = RATIOS_TTV
            n_val = int(RATIOS[1] * n)
            # test = rest
            val_files = flist[n_train : n_train + n_val]
            test_files = flist[n_train + n_val :]
            splits["val"].extend(val_files)
        else:
            RATIOS = RATIOS_TT
            n_test = int(RATIOS[1] * n)
            test_files = flist[n_train : n_train + n_test]
        splits["train"].extend(train_files)
        splits["test"].extend(test_files)

    # Save JSON
    with open(OUT_FILE, "w") as fp:
        json.dump(splits, fp, indent=2)

    print(f"Saved {OUT_FILE}")
    for k, v in splits.items():
        print(
            f"{k}: {len(v)} samples ({', '.join(sorted(set(f.split('__')[0] for f in v)))})"
        )

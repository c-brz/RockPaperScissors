import os
import torch
import numpy as np
from torch.utils.data import Dataset
from .hand_visualizations import extract_hand_landmarks  # adjust path as needed

LABELS: dict[str, int] = {'rock': 0, 'paper': 1, 'scissors': 2}

class RPSDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        feature = extract_hand_landmarks(self.image_paths[idx])
        if feature is None:
            feature = np.zeros(63)  # fallback for missing landmarks
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

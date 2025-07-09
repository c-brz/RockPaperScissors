# %% [markdown]
# Rock-Paper-Scissors Project: using https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/data to train a simple fully connected nn.
# 
# More info: https://docs.google.com/document/d/1bluoVcS2wuhsjR2EdAjWaUpyXfjgfQ6otWEb3xfnWF8/edit?usp=sharing 

# %%
#import os
# Redirect C-level stderr to /dev/null
#devnull = os.open(os.devnull, os.O_WRONLY)
#os.dup2(devnull, 2)

# %%
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_LOG"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# %%
import warnings
warnings.filterwarnings('ignore')
import contextlib

# %% [markdown]
# ## 1. Load data

# %%
import os
import sys
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#from modules.dataset import RPSDataset
DATASET_PATH = "/Users/christina/.cache/kagglehub/datasets/drgfreeman/rockpaperscissors/versions/2"

# --- Setup path so imports work ---
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.dataset import *
from modules.hand_visualizations import extract_hand_landmarks

# %% [markdown]
# ## 2. Model Definition

# %%

class RPSModel(nn.Module):
    def __init__(self, activation_func: str = "relu"):
        super().__init__()
        
        activation = self.get_activation(activation_func)

        self.net = nn.Sequential(
            nn.Linear(63, 128),
            activation,
            nn.Linear(128, 64),
            activation,
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

    def get_activation(self, name):
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

# %%

# Load dataset and split into train/test sets
def load_dataset(dataset_dir):
    image_paths, labels = [], []

    for label_name in LABELS:
        class_dir = os.path.join(dataset_dir, label_name)
        for img in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img))
            labels.append(LABELS[label_name])

    return image_paths, labels

def split_dataset(image_paths, labels, test_size=0.2, random_state=42):
    return train_test_split(image_paths, labels, test_size=test_size, random_state=random_state)

# Training
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0.0, 0

    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()

    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc


# Evaluation
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
    return correct / len(loader.dataset)


# Visualization
def visualize_prediction(model, image_path):
    feature = extract_hand_landmarks(image_path)
    if feature is not None:
        x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        pred = model(x).argmax(1).item()
        label = list(LABELS.keys())[pred]

        image = plt.imread(image_path)
        plt.imshow(image)
        plt.title(f"Predicted: {label}")
        plt.axis('off')
        plt.show()
    else:
        print("No hand landmarks detected.")



def main():
    # dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    # train_paths, test_paths, train_labels, test_labels = load_dataset(dataset_dir)

    dataset_dir = DATASET_PATH

    print(f"Using dataset directory: {dataset_dir}")
    image_paths, labels = load_dataset(dataset_dir)
    train_paths, test_paths, train_labels, test_labels = split_dataset(image_paths, labels)

    print(f"Train set size: {len(train_paths)}")
    train_ds = RPSDataset(train_paths, train_labels)
    print(f"Test set size: {len(test_paths)}")
    test_ds = RPSDataset(test_paths, test_labels)

    print("Creating DataLoaders...")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    print("Creating model, criterion, and optimizer...")
    model = RPSModel(activation_func="relu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...\n")
    for epoch in range(10):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch + 1}: Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | Test Acc={test_acc:.4f}")

    # Visualize one test prediction
    print("\nPrediction example:")
    visualize_prediction(model, test_paths[0])

# %%
if __name__ == "__main__":
    main()

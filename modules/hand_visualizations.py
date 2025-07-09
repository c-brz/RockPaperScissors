import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import mediapipe as mp
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class HandVisualRepresentations:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def raw_rgb_representation(self, image):
        """Raw RGB pixels - baseline representation"""
        # Resize to standard size
        resized = cv2.resize(image, (224, 224))
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def hand_segmentation_mask(self, image):
        """Create hand segmentation mask using color-based thresholding"""
        # Convert to HSV for better skin color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define skin color range (adjust based on your dataset)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find largest contour (assumed to be hand)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        return mask.astype(np.float32) / 255.0
    
    def improved_hand_segmentation(self, image):
        """Improved segmentation using K-means clustering"""
        # Reshape image for K-means
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3  # Assuming background, hand, and shadows/variations
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)
        
        # Find the cluster that represents the hand (usually the one with mid-range intensity)
        cluster_means = [np.mean(centers[i]) for i in range(k)]
        hand_cluster = np.argsort(cluster_means)[1]  # Middle intensity cluster
        
        # Create binary mask for hand cluster
        mask = (labels.flatten() == hand_cluster).astype(np.uint8) * 255
        mask = mask.reshape(image.shape[:2])
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask.astype(np.float32) / 255.0
    
    def edge_representation(self, image):
        """Extract edge maps of the hand"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Optional: Dilate edges to make them more prominent
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges.astype(np.float32) / 255.0
    
    def hand_landmarks_representation(self, image):
        """Extract MediaPipe hand landmarks"""
        results = self.hands.process(image)
        
        landmarks_array = np.zeros((21, 2))  # 21 landmarks, x,y coordinates
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # First hand
            
            for i, landmark in enumerate(hand_landmarks.landmark):
                landmarks_array[i, 0] = landmark.x * image.shape[1]  # x coordinate
                landmarks_array[i, 1] = landmark.y * image.shape[0]  # y coordinate
        
        return landmarks_array
    
    def create_landmark_image(self, image, landmarks):
        """Create visual representation of landmarks"""
        landmark_image = np.zeros_like(image)
        
        if np.any(landmarks):  # If landmarks were detected
            # Draw landmarks as circles
            for landmark in landmarks:
                if landmark[0] > 0 and landmark[1] > 0:  # Valid landmark
                    cv2.circle(landmark_image, 
                             (int(landmark[0]), int(landmark[1])), 
                             5, (255, 255, 255), -1)
            
            # Draw connections between landmarks
            connections = [
                # Thumb
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Index finger
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Middle finger
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Ring finger
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Pinky
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            
            for connection in connections:
                start_point = landmarks[connection[0]]
                end_point = landmarks[connection[1]]
                if (start_point[0] > 0 and start_point[1] > 0 and 
                    end_point[0] > 0 and end_point[1] > 0):
                    cv2.line(landmark_image,
                           (int(start_point[0]), int(start_point[1])),
                           (int(end_point[0]), int(end_point[1])),
                           (255, 255, 255), 2)
        
        return landmark_image
    
    def optical_flow_representation(self, prev_frame, curr_frame):
        """Compute dense optical flow between two frames"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None)
        
        # Alternative: Dense optical flow
        flow_dense = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Convert flow to HSV for visualization
        h, w = flow_dense.shape[:2]
        fx, fy = flow_dense[:,:,0], flow_dense[:,:,1]
        
        # Convert to polar coordinates
        mag, ang = cv2.cartToPolar(fx, fy)
        
        # Create HSV image
        hsv_flow = np.zeros((h, w, 3), dtype=np.uint8)
        hsv_flow[:,:,0] = ang * 180 / np.pi / 2  # Hue represents direction
        hsv_flow[:,:,1] = 255  # Full saturation
        hsv_flow[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude
        
        # Convert HSV to RGB
        flow_rgb = cv2.cvtColor(hsv_flow, cv2.COLOR_HSV2RGB)
        
        return flow_rgb.astype(np.float32) / 255.0, flow_dense
    
    def temporal_difference(self, prev_frame, curr_frame):
        """Compute temporal difference between frames"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Threshold to remove noise
        _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        return diff_thresh.astype(np.float32) / 255.0
    
    def process_rps_image(self, image_path):
        """Process a single RPS image and extract all representations"""
        image = self.load_image(image_path)
        
        representations = {}
        
        # Raw RGB
        representations['raw_rgb'] = self.raw_rgb_representation(image)
        
        # Segmentation masks
        representations['seg_mask'] = self.hand_segmentation_mask(image)
        representations['seg_mask_improved'] = self.improved_hand_segmentation(image)
        
        # Edge representation
        representations['edges'] = self.edge_representation(image)
        
        # Hand landmarks
        landmarks = self.hand_landmarks_representation(image)
        representations['landmarks'] = landmarks
        representations['landmark_image'] = self.create_landmark_image(image, landmarks)
        
        return representations, image

# Dataset exploration and setup
def explore_rps_dataset(dataset_path):
    """Explore the RPS dataset structure"""
    dataset_path = Path(dataset_path)
    
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset exists: {dataset_path.exists()}")
    
    if not dataset_path.exists():
        print("Dataset not found!")
        return None, None, None
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(dataset_path.rglob(ext)))
    
    print(f"Total images found: {len(all_images)}")
    
    # Organize by class
    class_images = {}
    labels = []
    image_paths = []
    
    for img_path in all_images:
        # Extract class from filename or parent directory
        filename = img_path.stem.lower()
        parent_dir = img_path.parent.name.lower()
        
        # Determine class
        if 'rock' in filename or 'rock' in parent_dir:
            class_name = 'rock'
        elif 'paper' in filename or 'paper' in parent_dir:
            class_name = 'paper'
        elif 'scissors' in filename or 'scissors' in parent_dir:
            class_name = 'scissors'
        else:
            # Try to infer from directory structure
            continue
        
        if class_name not in class_images:
            class_images[class_name] = []
        
        class_images[class_name].append(img_path)
        image_paths.append(img_path)
        labels.append(class_name)
    
    # Print class distribution
    print("\nClass distribution:")
    for class_name, images in class_images.items():
        print(f"{class_name}: {len(images)} images")
    
    # Show sample paths
    print("\nSample image paths:")
    for class_name, images in class_images.items():
        if images:
            print(f"{class_name}: {images[0]}")
    
    return image_paths, labels, class_images

# Example usage and visualization
def visualize_representations(image_path):
    """Visualize all representations for a single image"""
    processor = HandVisualRepresentations()
    representations, original_image = processor.process_rps_image(image_path)
    
    # Create subplot for visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Hand Representations for {Path(image_path).name}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    # Raw RGB (resized)
    axes[0, 1].imshow(representations['raw_rgb'])
    axes[0, 1].set_title('Processed RGB')
    axes[0, 1].axis('off')
    
    # Basic segmentation
    axes[0, 2].imshow(representations['seg_mask'], cmap='gray')
    axes[0, 2].set_title('Basic Segmentation')
    axes[0, 2].axis('off')
    
    # Improved segmentation
    axes[0, 3].imshow(representations['seg_mask_improved'], cmap='gray')
    axes[0, 3].set_title('K-means Segmentation')
    axes[0, 3].axis('off')
    
    # Edge representation
    axes[1, 0].imshow(representations['edges'], cmap='gray')
    axes[1, 0].set_title('Edge Map')
    axes[1, 0].axis('off')
    
    # Landmarks visualization
    axes[1, 1].imshow(representations['landmark_image'])
    axes[1, 1].set_title('Hand Landmarks')
    axes[1, 1].axis('off')
    
    # Show landmarks on original
    landmark_overlay = original_image.copy()
    landmarks = representations['landmarks']
    if np.any(landmarks):
        for landmark in landmarks:
            if landmark[0] > 0 and landmark[1] > 0:
                cv2.circle(landmark_overlay, (int(landmark[0]), int(landmark[1])), 3, (255, 0, 0), -1)
    
    axes[1, 2].imshow(landmark_overlay)
    axes[1, 2].set_title('Landmarks on Original')
    axes[1, 2].axis('off')
    
    # Print landmark statistics
    if np.any(landmarks):
        axes[1, 3].text(0.1, 0.5, f'Landmarks detected: {np.sum(np.any(landmarks, axis=1))}/21\n\n'
                                   f'Hand span (pixels):\n'
                                   f'Width: {np.max(landmarks[:, 0]) - np.min(landmarks[:, 0]):.1f}\n'
                                   f'Height: {np.max(landmarks[:, 1]) - np.min(landmarks[:, 1]):.1f}',
                        transform=axes[1, 3].transAxes, fontsize=10, verticalalignment='center')
    else:
        axes[1, 3].text(0.1, 0.5, 'No landmarks detected', 
                        transform=axes[1, 3].transAxes, fontsize=10, verticalalignment='center')
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return representations

def visualize_multiple_samples(class_images, num_samples=3):
    """Visualize representations for multiple samples from each class"""
    processor = HandVisualRepresentations()
    
    fig, axes = plt.subplots(len(class_images), num_samples, figsize=(15, 12))
    fig.suptitle('Sample Images from Each Class', fontsize=16)
    
    for class_idx, (class_name, images) in enumerate(class_images.items()):
        for sample_idx in range(min(num_samples, len(images))):
            img_path = images[sample_idx]
            image = processor.load_image(img_path)
            
            axes[class_idx, sample_idx].imshow(image)
            axes[class_idx, sample_idx].set_title(f'{class_name.capitalize()} - Sample {sample_idx + 1}')
            axes[class_idx, sample_idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# PyTorch dataset class for different representations
class RPSRepresentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, representation_type='raw_rgb', transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.representation_type = representation_type
        self.transform = transform
        self.processor = HandVisualRepresentations()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        representations, _ = self.processor.process_rps_image(image_path)
        
        if self.representation_type == 'raw_rgb':
            data = representations['raw_rgb']
        elif self.representation_type == 'seg_mask':
            data = representations['seg_mask']
            data = np.stack([data, data, data], axis=-1)  # Convert to 3-channel
        elif self.representation_type == 'edges':
            data = representations['edges']
            data = np.stack([data, data, data], axis=-1)  # Convert to 3-channel
        elif self.representation_type == 'landmarks':
            data = representations['landmarks'].flatten()  # Flatten to 1D array
        else:
            raise ValueError(f"Unknown representation type: {self.representation_type}")
        
        if self.transform:
            if self.representation_type != 'landmarks':
                data = self.transform(data)
            
        return data, label

mp_hands = mp.solutions.hands

def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    return None
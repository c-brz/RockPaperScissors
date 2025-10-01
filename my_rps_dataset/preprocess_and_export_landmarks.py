import os
import cv2
import numpy as np
import mediapipe as mp


class RPSLandmarkExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks_from_video(self, video_path, save_data=False):
        """Extract hand landmarks from video"""
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []
        no_hand_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # if results.multi_hand_landmarks:
            if results.multi_hand_world_landmarks:
                # Get first hand landmarks
                hand_landmarks = results.multi_hand_world_landmarks[0]

                # Extract x, y, z coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_sequence.append(landmarks)

            else:
                no_hand_frames += 1
                landmarks_sequence.append([np.nan] * 63)  # 21 landmarks * 3 coordinates

        print(f"Total frames with no hand detected: {no_hand_frames}")
        cap.release()

        return np.array(landmarks_sequence)

    def load_landmarks_from_npz(self, npz_path):
        """Load landmarks from a .npz file"""
        # print(f"Loading landmarks from {npz_path}")
        data = np.load(npz_path)
        return data["landmarks"], data["label"]

    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position. Assume landmarks shape is (T, 63)"""

        if len(landmarks) == 0:
            return landmarks

        normalized = landmarks.copy()  # (T, 63)
        # print(f"normalize_landmarks(): {landmarks.shape}")

        for i in range(len(landmarks)):
            if np.sum(landmarks[i]) == 0:  # Skip zero frames
                continue

            # Reshape to get individual landmarks
            frame_landmarks = landmarks[i].reshape(21, 3)

            # Get wrist position (landmark 0)
            wrist = frame_landmarks[0]
            # print(f"Wrist position at frame {i}: {wrist}")

            # Subtract wrist position from all landmarks
            frame_landmarks = frame_landmarks - wrist
            normalized[i] = frame_landmarks.flatten()

        return normalized

    def scale_landmarks(self, landmarks):
        """Scale landmarks to a fixed size. Assume landmarks shape is (T, 63)"""
        if len(landmarks) == 0:
            return landmarks

        normalized = landmarks.copy()

        for i in range(len(landmarks)):
            if np.sum(landmarks[i]) == 0:
                continue

            # Reshape to get individual landmarks
            frame_landmarks = landmarks[i]

            # Get wrist (landmark 0) and middle finger tip (landmark 12) positions
            wrist = frame_landmarks[0]
            middle_finger_tip = frame_landmarks[12]

            # Calculate hand size (distance from wrist to middle finger tip)
            hand_size = np.abs(middle_finger_tip - wrist)

            # Normalize by hand size (avoid division by zero)
            if hand_size > 0:
                frame_landmarks = frame_landmarks / hand_size
            else:
                print(f"Hand size is zero at frame {i}. Skipping normalization.")

            normalized[i] = frame_landmarks.flatten()

        return normalized

    def make_rotation_invariant(self, landmarks):
        """
        Align hand landmarks to a palm-based local coordinate system.
        Assumes input is already translation & scale normalized.

        Args:
            landmarks: (21,3) numpy array of hand landmarks
        Returns:
            aligned: (21,3) numpy array in palm-based coordinate frame
        """

        all_landmarks = landmarks.reshape(-1, 21, 3)
        new_landmarks = landmarks.copy()

        for i in range(len(all_landmarks)):
            landmarks = all_landmarks[i]

            # Skip frames with NaNs
            if np.isnan(landmarks).any():
                continue

            index_mcp = landmarks[5]  # index base joint
            pinky_mcp = landmarks[17]  # pinky base joint

            # Check for NaNs in key points
            if np.isnan(index_mcp).any() or np.isnan(pinky_mcp).any():
                continue

            # Palm axes
            if np.linalg.norm(index_mcp) == 0 or np.linalg.norm(pinky_mcp) == 0:
                continue

            x_axis = index_mcp / np.linalg.norm(index_mcp)
            y_axis = pinky_mcp / np.linalg.norm(pinky_mcp)
            z_axis = np.cross(x_axis, y_axis)
            if np.linalg.norm(z_axis) == 0:
                continue
            z_axis /= np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)  # re-orthogonalize

            # Rotation matrix
            R = np.stack([x_axis, y_axis, z_axis], axis=1)
            aligned = landmarks @ R
            new_landmarks[i] = aligned.flatten()

        return new_landmarks

    def compute_angle_between_joints(self, landmarks):

        # triplets = [(2, 3, 4), # thumb
        #             (6, 7, 8), # index
        #             (10, 11, 12), # middle
        #             (14, 15, 16), # ring
        #             (18, 19, 20)] # pinky

        triplets = [
            (0, 2, 4),  # thumb
            (0, 5, 8),  # index
            (0, 9, 12),  # middle
            (0, 13, 16),  # ring
            (0, 17, 20),
        ]  # pinky

        def angle_between_points(a, b, c):
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

        all_landmarks = landmarks.reshape(-1, 21, 3).copy()
        angles = []

        for i in range(len(all_landmarks)):
            landmarks = all_landmarks[i]

            if np.sum(landmarks) == 0:
                angles.extend([0.0] * len(triplets))
                continue

            for triplet in triplets:
                a, b, c = (
                    landmarks[triplet[0]],
                    landmarks[triplet[1]],
                    landmarks[triplet[2]],
                )
                angle = angle_between_points(a, b, c)
                angles.append(angle)

            # print(f"Frame {i}: Angles: {angles[-len(triplets):]}")
        angles_np = np.array(angles).reshape(-1, len(triplets))
        # print(f"Angles ({angles_np.shape})")

        return angles_np

    def compute_angle_between_joints_from_wrist(self, landmarks):
        """
        Computes the angles (in degrees) between the wrist and key joints for each finger across multiple frames of hand landmarks.
        For each frame, calculates the angle at the MCP joint for each finger, using the wrist, MCP, and fingertip coordinates.

        Parameters
        ----------
        landmarks : np.ndarray
            A numpy array of shape (num_frames, 21, 3) or (num_frames * 21 * 3,) containing the 3D coordinates of hand landmarks
            for multiple frames.

        Returns
        -------
        np.ndarray
            A numpy array of shape (num_frames, 5) containing the computed angles (in degrees) for each finger per frame.
        """

        # Angle between wrist (0), MCP (5, 9, 13, 17), and TIP (8, 12, 16, 20) for each finger

        triplets = [
            (0, 2, 4),  # thumb
            (0, 5, 8),  # index
            (0, 9, 12),  # middle
            (0, 13, 16),  # ring
            (0, 17, 20),
        ]  # pinky

        def angle_between_points(a, b, c):
            """Calculate the angle (in degrees) at point b given points a, b, c.
            --------
            a, b, c: (x, y) or (x, y, z) coordinates of the points.
            Returns: angle in degrees.
            """

            ba = np.array(b) - np.array(a)
            bc = np.array(c) - np.array(a)

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

            return np.degrees(angle)

        all_landmarks = landmarks.reshape(-1, 21, 3).copy()
        angles = []

        for i in range(len(all_landmarks)):
            landmarks = all_landmarks[i]

            if np.sum(landmarks) == 0:
                angles.extend([0.0] * len(triplets))
                continue

            for triplet in triplets:
                a, b, c = (
                    landmarks[triplet[0]],
                    landmarks[triplet[1]],
                    landmarks[triplet[2]],
                )
                angle = angle_between_points(a, b, c)
                angles.append(angle)

        angles_np = np.array(angles).reshape(-1, len(triplets))
        # print(f"Angles ({angles_np.shape})")

        return angles_np

    def extract_landmarks_raw(self, video_dir, destination_dir):
        """Process all videos in a directory structure"""

        all_sequences = []
        all_labels = []

        # Expected directory structure: video_dir/gesture_name/video_files
        gesture_dirs = [
            d
            for d in os.listdir(video_dir)
            if os.path.isdir(os.path.join(video_dir, d))
        ]

        for gesture_name in gesture_dirs:

            gesture_path = os.path.join(video_dir, gesture_name)
            video_files = [
                f
                for f in os.listdir(gesture_path)
                if f.endswith((".mp4", ".avi", ".mov"))
            ]

            print(f"Processing {len(video_files)} videos for gesture: {gesture_name}")

            for video_file in video_files:

                video_path = os.path.join(gesture_path, video_file)
                video_file_name = video_file.split(".")[0]
                landmarks = self.extract_landmarks_from_video(video_path)

                if len(landmarks) > 0:
                    normalized_landmarks_3d = landmarks.copy()
                    print(
                        f"Saving landmarks for {video_file_name} with shape {normalized_landmarks_3d.shape}"
                    )
                else:
                    print(f"No landmarks found for {video_file_name}")
                    normalized_landmarks_3d = np.array([np.nan] * 63).reshape(-1, 21, 3)

                np.savez(
                    os.path.join(destination_dir, f"{video_file_name}_landmarks.npz"),
                    landmarks=normalized_landmarks_3d,
                    label=gesture_name,
                )

        return np.array(all_sequences), np.array(all_labels)

    def extract_features_coords(
        self,
        video_dir,
        destination_dir,
        normalize_data: bool = False,
        scale_data: bool = False,
        save_data: bool = False,
    ):
        """Process all class videos in the given directory"""

        all_sequences = []
        all_labels = []
        # List all files in video_dir
        gesture_dirs = [d for d in os.listdir(video_dir)]
        # print(f"Found gesture directories: {gesture_dirs}")

        for gesture_name in gesture_dirs:
            landmarks, label = self.load_landmarks_from_npz(
                os.path.join(video_dir, gesture_name)
            )
            # print(f"Loaded landmarks shape: {landmarks}")

            if len(landmarks) > 0:
                # Normalize landmarks
                normalized_landmarks = self.normalize_landmarks(landmarks)
                scaled_landmarks = self.scale_landmarks(normalized_landmarks)
                processed_landmarks = self.make_rotation_invariant(scaled_landmarks)

            # Save to .npz files
            normalized_landmarks_3d = processed_landmarks.reshape(-1, 21, 3)

            if save_data:
                print(
                    f"Saving landmarks for {gesture_name} with shape {normalized_landmarks_3d.shape}"
                )
                np.savez(
                    os.path.join(destination_dir, f"{gesture_name}"),
                    landmarks=normalized_landmarks_3d,
                    label=gesture_name,
                )

        return normalized_landmarks_3d, gesture_name

    def extract_features_angles(
        self, video_dir, destination_dir, save_data: bool = False, method="from_mcp"
    ):
        """Process all class videos in the given directory"""

        all_sequences = []
        all_labels = []
        # List all files in video_dir
        gesture_dirs = [d for d in os.listdir(video_dir)]

        for gesture_name in gesture_dirs:
            landmarks, label = self.load_landmarks_from_npz(
                os.path.join(video_dir, gesture_name)
            )
            # print(f"Loaded landmarks shape: {landmarks.shape}")

            if len(landmarks) > 0:
                if method == "from_mcp":
                    res = self.compute_angle_between_joints(landmarks)
                elif method == "from_wrist":
                    res = self.compute_angle_between_joints_from_wrist(landmarks)
                normalized_landmarks_3d = res.reshape(-1, 5)
            else:
                normalized_landmarks_3d = np.array([np.nan] * 5).reshape(-1, 5)
                print(f"No landmarks found for {gesture_name}")

            if save_data:
                print(
                    f"Saving landmarks for {gesture_name} with shape {normalized_landmarks_3d.shape}"
                )
                np.savez(
                    os.path.join(destination_dir, f"{gesture_name}"),
                    landmarks=normalized_landmarks_3d,
                    label=gesture_name,
                )

        return normalized_landmarks_3d, gesture_name


if __name__ == "__main__":

    feature_extractor = RPSLandmarkExtractor()
    landmarks = feature_extractor.extract_landmarks_raw(
        "/Users/christina/code/RockPaperScissors/my_rps_dataset/data", "./landmarks"
    )

    feature_extractor = RPSLandmarkExtractor()
    land, gest = feature_extractor.extract_features_coords(
        "./landmarks", "./features_coords", save_data=True
    )

    feature_extractor = RPSLandmarkExtractor()
    land, gest = feature_extractor.extract_features_angles(
        "./landmarks", "./features_angles", save_data=True, method="from_mcp"
    )

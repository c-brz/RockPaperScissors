#from torch._dynamo.polyfills.pytree import __name
import cv2
import os
import time
import argparse

# Find the next available video number
def get_next_video_number(directory, gesture_label):
    existing_files = os.listdir(directory)
    video_numbers = []
    
    for filename in existing_files:
        if filename.startswith(f"{gesture_label}_") and filename.endswith(".avi"):
            try:
                # Extract number from filename like "rock_5.avi"
                number_part = filename.replace(f"{gesture_label}_", "").replace(".avi", "")
                video_numbers.append(int(number_part))
            except ValueError:
                continue
    
    return max(video_numbers) + 1 if video_numbers else 0

if __name__ == "__main__":
    
    # Parse command line argument for gesture label
    parser = argparse.ArgumentParser(description="Record Rock-Paper-Scissors gesture videos.")
    parser.add_argument("gesture_label", type=str, help="Gesture label: rock, paper, or scissors")
    args = parser.parse_args()
    gesture_label = args.gesture_label.lower()
        
    save_dir = f"data/{gesture_label}"
    os.makedirs(save_dir, exist_ok=True)

    # Get starting number for new videos
    start_number = get_next_video_number(save_dir, gesture_label)

    num_videos = 15
    video_length = 2  # seconds
    fps = 15
    frame_size = (400, 400)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")

    print(f"Starting video recording from number {start_number}")

    for i in range(num_videos):
        video_number = start_number + i
        print(f"Get ready! Recording '{gesture_label}' video {video_number} ({i+1}/{num_videos})")
        time.sleep(2)
        
        out = cv2.VideoWriter(f"{save_dir}/{gesture_label}_{video_number}.avi",
                            cv2.VideoWriter.fourcc(*'MJPG'),
                            fps,
                            frame_size)

        start_time = time.time()
        while time.time() - start_time < video_length:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, frame_size)
            out.write(frame)
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        print(f"Saved {gesture_label}_{video_number}.avi")

    cap.release()
    cv2.destroyAllWindows()
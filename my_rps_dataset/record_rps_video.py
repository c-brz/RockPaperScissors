import cv2
import os
import time
import argparse

# Parse command line argument for gesture label
parser = argparse.ArgumentParser(description="Record Rock-Paper-Scissors gesture videos.")
parser.add_argument("gesture_label", type=str, help="Gesture label: rock, paper, or scissors")
args = parser.parse_args()
gesture_label = args.gesture_label.lower()

# Parameters
#gesture_label = "rock"  # change to "paper" or "scissors"
save_dir = f"data/{gesture_label}"
os.makedirs(save_dir, exist_ok=True)

num_videos = 2
video_length = 2  # seconds
fps = 10
frame_size = (400, 400)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam")

for i in range(num_videos):
    print(f"Get ready! Recording '{gesture_label}' video {i+1}/{num_videos}")
    time.sleep(2)
    
    out = cv2.VideoWriter(f"{save_dir}/{gesture_label}_{i}.avi",
                          cv2.VideoWriter.fourcc(*'MJPG'),
                          fps,
                          frame_size)

    #out = cv2.VideoWriter(f"{save_dir}/{gesture_label}_{i}.mp4",
    #                    cv2.VideoWriter.fourcc(*'MP4V'),
    #                    fps,
    #                    frame_size)
    
    start_time = time.time()
    while time.time() - start_time < video_length:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw a centered square that matches the frame size
        #box_size = frame_size[0]  # Assuming square frames
        #x1 = 0
        #y1 = 0
        #x2 = x1 + box_size
        #y2 = y1 + box_size
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green border, thickness=2
        # Ensure frame size matches VideoWriter
        
        frame = cv2.resize(frame, frame_size)
        out.write(frame)
        cv2.imshow('Recording...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    print(f"Saved {gesture_label}_{i}.mp4")

cap.release()
cv2.destroyAllWindows()

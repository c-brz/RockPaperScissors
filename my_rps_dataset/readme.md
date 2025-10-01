# Rock-Paper-Scissors Gesture Video Recorder

This script records short webcam videos of hand gestures for the Rock-Paper-Scissors dataset.

## Usage

1. **Activate your Python environment and install requirements:**
    ```sh
    pip install opencv-python
    ```

2. **Run the script from the command line:**
    ```sh
    python record_rps_video.py <gesture_label>
    ```
    Replace `<gesture_label>` with `rock`, `paper`, or `scissors`.  
    Example:
    ```sh
    python record_rps_video.py rock
    ```

3. **What it does:**
    - Records videos (2 seconds each) of your gesture using your webcam.
    - Saves videos as `.avi` files using the MJPG codec in `data/<gesture_label>/`.
    - Shows a live webcam preview during recording.

4. **Notes:**
    - Press `q` to stop recording early.
    - Tested on macOS Sonoma 14.1

## Output

- Videos are saved as:  
  `data/<gesture_label>/<gesture_label>_0.avi`, `data/<gesture_label>/<gesture_label>_1.avi`, etc.

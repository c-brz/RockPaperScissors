# Rock-Paper-Scissors Hand Gesture Recognition

## Project Structure
 
### `my_rps_dataset/`
Core dataset management and preprocessing tools:
- `record_rps_video.py` - Record webcam videos of hand gestures 
- `preprocess_and_export_landmarks.py` - Extract MediaPipe hand landmarks from videos
- `create_video_splits.py` - Generate train/test/validation dataset splits
- `features/` - Exported landmark coordinates in .npz format
- `features_angles/` - Hand joint angle features derived from landmarks
- `features_coords/` - Processed 3D coordinate features
- `landmarks/` - Raw landmark data extracted from videos

### `main/experiments_dtw/`
DTW-based classification experiments:
- `dtw.ipynb` - Dynamic Time Warping classification experiments
- `check.ipynb` - Data validation and analysis

### `alignment/`
Video processing and data alignment tools:
- `video_frames_to_img.py` - Extract individual frames from videos
- `data_check.ipynb` - Validate landmark data at cropped video frames

## Getting Started

1. Set up the environment:
   ```bash
   python -m venv rps_venv
   source rps_venv/bin/activate  # On Windows: rps_venv\Scripts\activate
   pip install -r my_rps_dataset/requirements.txt
   ```

2. Record gesture videos:
   ```bash
   cd my_rps_dataset
   python record_rps_video.py rock
   python record_rps_video.py paper
   python record_rps_video.py scissors
   ```

3. Extract landmarks and features:
   ```bash
   python preprocess_and_export_landmarks.py
   ```

4. Create dataset splits:
   ```bash
   python create_video_splits.py --data_dir features/ --out_file splits.json
   ```

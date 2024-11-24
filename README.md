# Video Reticle Tracker

This Python script allows users to add a reticle to each frame of a video, with its position interpolated between user-specified keyframes. The script is designed for tracking objects in videos and detecting nearby "stars" or bright spots to fine-tune the reticle's position.

---

## Features

- **Interactive Object Selection:** Users can click on an object in specific keyframes to define its position.
- **Position Interpolation:** Automatically interpolates positions for frames between the keyframes.
- **Star Detection:** Adjusts reticle position by detecting the closest bright spot near the interpolated position.
- **Customizable Reticle:** Configure reticle color, size, and gap parameters.
- **Keyframe Saving:** Saves keyframes as TIFF images for further analysis or reference.
- **CSV Support:** Saves and loads keyframe positions from a CSV file.

---

## Requirements

- Python 3.x
- OpenCV
- NumPy

You can install the required libraries using pip:

```bash
pip install opencv-python numpy
```

---

## Usage

### 1. Configure Input and Output

Modify the following global parameters in the script to suit your needs:

- `video_input_path`: Path to the input video.
- `video_output_path`: Path to save the output video.
- `n_positions`: Number of keyframes for manual position selection.
- `reticle_color`, `reticle_thickness`, `reticle_length`, `gap_size`: Customize the reticle's appearance.


### 2. Select Keyframe Positions

- The script will prompt you to click on the object in the specified keyframes.
- Use the displayed window to click the object and press any key to confirm the selection.

### 3. Output

- The processed video with the reticle overlay will be saved to the specified output path.
- Keyframe images will be saved in the `key_frames` directory.
- Keyframe positions will be saved to a CSV file (`positions.csv`).




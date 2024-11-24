import cv2
import numpy as np
import os
import csv

# === Global Parameters ===
video_input_path = 'video.mp4'  # Path to your input video
video_output_path = 'output_video_with_reticle.mp4'  # Path to save the output video

# Number of key frames where positions are specified
n_positions = 5  # Set the number of positions (including first and last frames)

# Reticle parameters
reticle_color = (0, 255, 0)  # Green color in BGR
reticle_thickness = 1        # Thickness of the reticle lines
reticle_length = 20          # Length of each line in the reticle
gap_size = 5                 # Gap size at the center of the reticle

# Output directory for saving key frames
output_frames_dir = 'key_frames'  # Directory to save the key frames

# Positions CSV file
positions_csv_path = 'positions.csv'

# Variable to store clicked position
clicked_position = None  # Initialize clicked_position here

# Parameters for star detection
search_window_size = 20  # Size of the square window around the interpolated position
max_star_radius = 5      # Maximum radius of stars to detect

# ==========================

def main():
    global clicked_position  # Declare clicked_position as global in this function
    # Open the video file
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # Generate equally spaced frame indices for key positions
    frame_indices = np.linspace(0, total_frames - 1, n_positions, dtype=int)

    # Initialize key_positions list
    key_positions = []

    # Check if positions.csv exists
    if os.path.exists(positions_csv_path):
        print(f"Loading positions from {positions_csv_path}")
        key_positions = load_positions_from_csv(positions_csv_path)
    else:
        print(f"{positions_csv_path} not found. Please input the object's positions at the following frames:")
        positions = []
        for idx in frame_indices:
            # Set the video to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {idx}.")
                return

            # Display the frame and let the user click on the position
            window_name = f'Frame {idx}'
            cv2.imshow(window_name, frame)
            print(f"Click on the object's position in the displayed window and press any key.")
            clicked_position = None  # Reset clicked_position before each frame
            cv2.setMouseCallback(window_name, click_event)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if clicked_position is not None:
                positions.append((idx, clicked_position))
                print(f"Position at frame {idx}: {clicked_position}")
            else:
                print("No position clicked. Exiting.")
                return

            # Save the frame without reticle
            save_frame(frame, idx)

        # Save positions to CSV
        save_positions_to_csv(positions, positions_csv_path)
        key_positions.extend(positions)

    # Sort key_positions by frame index
    key_positions.sort(key=lambda x: x[0])

    # Interpolate positions for all frames
    positions = interpolate_positions(key_positions, total_frames)

    # Reset video capture to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the interpolated position of the object
        x, y = positions[frame_idx]

        # Detect the closest star to the interpolated position and adjust the position
        adjusted_position = detect_star_near_position(frame, (int(x), int(y)))

        # Draw the reticle on the frame at the adjusted position
        draw_reticle(frame, adjusted_position)

        # Write the frame to the output video
        out.write(frame)

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    print("Processing complete. Output saved to:", video_output_path)

def interpolate_positions(key_positions, total_frames):
    """
    Interpolates positions between the key positions for all frames.
    Returns a list of (x, y) positions for each frame.
    """
    positions = []
    key_frames = [kp[0] for kp in key_positions]
    key_x = [kp[1][0] for kp in key_positions]
    key_y = [kp[1][1] for kp in key_positions]

    all_frames = np.arange(total_frames)
    x_positions = np.interp(all_frames, key_frames, key_x)
    y_positions = np.interp(all_frames, key_frames, key_y)

    positions = list(zip(x_positions, y_positions))
    return positions

def detect_star_near_position(frame, position):
    """
    Detects the brightest spot (star) near the given position in the frame.
    Returns the adjusted position centered on the detected star.
    """
    x, y = position
    h, w = frame.shape[:2]

    # Define the search window around the position
    x_start = max(x - search_window_size // 2, 0)
    x_end = min(x + search_window_size // 2, w)
    y_start = max(y - search_window_size // 2, 0)
    y_end = min(y + search_window_size // 2, h)

    # Extract the region of interest (ROI)
    roi = frame[y_start:y_end, x_start:x_end]

    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)

    # Threshold to create a binary image
    _, thresh = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours (potential stars)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_distance = float('inf')
    closest_star_center = position  # Default to the interpolated position
    for cnt in contours:
        # Calculate moments to find the center of the contour
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        # Calculate the distance to the interpolated position within ROI coordinates
        distance = np.hypot(cX - search_window_size // 2, cY - search_window_size // 2)
        if distance < min_distance:
            min_distance = distance
            # Adjust center coordinates to frame coordinates
            closest_star_center = (x_start + cX, y_start + cY)

    return closest_star_center

def draw_reticle(frame, position):
    """
    Draws a thin green cross without the center at the given position.
    """
    x, y = position
    # Draw horizontal lines
    cv2.line(frame,
             (x - reticle_length, y),
             (x - gap_size, y),
             reticle_color,
             reticle_thickness)
    cv2.line(frame,
             (x + gap_size, y),
             (x + reticle_length, y),
             reticle_color,
             reticle_thickness)
    # Draw vertical lines
    cv2.line(frame,
             (x, y - reticle_length),
             (x, y - gap_size),
             reticle_color,
             reticle_thickness)
    cv2.line(frame,
             (x, y + gap_size),
             (x, y + reticle_length),
             reticle_color,
             reticle_thickness)

def save_frame(frame, frame_idx):
    """
    Saves the given frame as a TIFF image with the frame index in the filename.
    """
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)
    filename = f"frame_{frame_idx}.tif"
    filepath = os.path.join(output_frames_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"Frame {frame_idx} saved to {filepath}")

def click_event(event, x, y, flags, params):
    global clicked_position  # Declare clicked_position as global in this function
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_position = (x, y)
        print(f"Clicked position: {clicked_position}")

def save_positions_to_csv(positions, csv_path):
    """
    Saves the positions to a CSV file.
    """
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_idx', 'x', 'y'])
        for idx, pos in positions:
            writer.writerow([idx, pos[0], pos[1]])
    print(f"Positions saved to {csv_path}")

def load_positions_from_csv(csv_path):
    """
    Loads positions from a CSV file.
    Returns a list of (frame_idx, (x, y)) tuples.
    """
    positions = []
    with open(csv_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            idx = int(row['frame_idx'])
            x = float(row['x'])
            y = float(row['y'])
            positions.append((idx, (x, y)))
    return positions

if __name__ == "__main__":
    main()

Warning: This small Python script for video-to-3D animation is highly experimental and extremely buggy. No bug fixes are planned. Proceed at your own risk.
This Python script implements a basic pipeline for vision-based human pose estimation and rudimentary 3D visualization. Inspired by applications of computer vision, this project explores the foundational steps of extracting pose information from video and creating a simple 3D representation.

The script leverages the powerful YOLOv11-pose model to detect individuals in video frames and extract 2D keypoint data.  It then takes pose data from two simulated "camera views" (processed from the same video currently) and generates a simplified 3D visualization of the human pose.

**Key Functionalities:**

*   **2D Pose Estimation (`create_pose_data`):**
    *   Utilizes the pre-trained YOLOv11-pose model (`yolov11n-pose.pt`) for human pose detection.
    *   Processes video files using OpenCV.
    *   Detects individuals and extracts 17 keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
    *   Saves extracted 2D pose data (keypoint coordinates, confidence scores, bounding boxes) in JSON format per frame.
    *   Includes a `camera_id` in the JSON output to simulate multi-camera setup (currently using the same video for "camera A" and "camera B").
    *   Option to display annotated video output with pose overlays for debugging (`show_video=True`).

*   **Rudimentary 3D Visualization (`create_3d_video`):**
    *   Reads 2D pose data from two JSON files (representing "camera A" and "camera B" views).
    *   Performs a simplified 3D reconstruction based on the 2D keypoints.
    *   Generates 3D scatter plots of the reconstructed pose for each frame using Matplotlib.
    *   Visualizes skeletal connections between keypoints in 3D.
    *   Saves each 3D plot as a PNG image frame in a specified output directory.
    *   **Important Limitation:** The 3D reconstruction is highly simplified and not geometrically accurate. It is primarily for basic visualization and not for precise 3D measurement.  Depth estimation is approximated, assuming parallel cameras and relying on Y-coordinate position in one view.

**Methodology - In Detail (Similar to Video Explanation):**

1.  **2D Pose Detection with YOLOv11-pose:**
    *   **Model Choice:** Employs the `yolov11n-pose.pt` model from Ultralytics YOLOv11 for its balance of speed and accuracy in pose estimation.  This pre-trained model is ready to use without requiring custom training data.
    *   **Video Input:**  The `create_pose_data` function takes a video file path as input. OpenCV handles video reading frame by frame.
    *   **YOLOv11 Inference:** For each frame, the YOLOv11-pose model is run to detect people and their poses.
    *   **Keypoint Extraction:** The script extracts the (x, y) coordinates and confidence of the 17 detected keypoints.
    *   **JSON Data Storage:**  The processed data for each frame (frame number, camera ID, detections including bounding boxes and keypoints) is structured and saved as a JSON file.  This structured format is designed for easy further processing, particularly for the 3D visualization step.

2.  **Simplified 3D Visualization:**
    *   **Two "Camera Views":** The `create_3d_video` function expects two JSON files as input.  Currently, in the example usage, these are generated from the *same* video processed twice (as "camera A" and "camera B").  In a real multi-camera setup, these would come from different cameras viewing the same scene.
    *   **Loading JSON Data:** The script loads the 2D pose data from both JSON files, ensuring they have the same number of frames for synchronization.
    *   **Rudimentary 3D Calculation (`Pose3D.from2D`):**  This is the core of the simplified 3D approach.  It makes strong assumptions to estimate 3D positions:
        *   **Parallel Cameras:** Assumes the two "cameras" are positioned parallel to each other.
        *   **Depth from Y-coordinate (Approximation):**  Estimates the depth (Z-coordinate) based on the vertical position (Y-coordinate) of keypoints in "camera A" view.  This is a very rough approximation and not geometrically accurate triangulation.
        *   **Centering:** Centers the X and Y coordinates based on average keypoint positions in each view.
    *   **3D Plotting:** Matplotlib's `mplot3d` is used to create 3D scatter plots.  Lines are drawn between keypoints based on predefined `POSE_ANCHORS` to visualize the human skeleton in 3D.
    *   **Saving Frames:** Each 3D plot is saved as a PNG image frame. These frames could be combined into a video to create a 3D animation, although this script currently only generates individual frames.

**Code Structure:**

The code is organized into several key components:

*   **`create_pose_data(video_path, output_json_path, camera_id, show_video=False)`:** Function for 2D pose estimation and JSON data saving.
*   **`create_3d_video(json_path_a, json_path_b, output_frames_dir)`:** Function for rudimentary 3D visualization and frame saving.
*   **`Pose` Dataclass:** Represents 2D pose data (x, y coordinates, confidence).
*   **`Pose3D` Dataclass:** Represents 3D pose data (x, y, z coordinates).  Includes the `from2D` method for simplified 3D reconstruction.
*   **`Point`, `Point3D`, `Detection` Dataclasses:** Helper dataclasses for structuring point and detection data.
*   **`load_json(json_path)`:** Utility function to load JSON data from a file.
*   **`if __name__ == '__main__':` block:**  Example usage demonstrating how to run the `create_pose_data` and `create_3d_video` functions on a sample video (`test.mp4`).

**Requirements:**

*   Python 3.x
*   Ultralytics YOLOv11 (`pip install ultralytics`)
*   OpenCV (`pip install opencv-python`)
*   JSON (built-in to Python)
*   NumPy (`pip install numpy`)
*   Matplotlib (`pip install matplotlib`)
*   Dataclasses (built-in to Python >= 3.7, `pip install dataclasses` for older versions)
*   Typing (built-in to Python >= 3.5, `pip install typing` for older versions)

**How to Use:**

1.  **Install Requirements:**  `pip install -r requirements.txt` (if you create a `requirements.txt` file listing the dependencies, or install them individually as listed above).
2.  **Prepare Video:** Place your video file (e.g., `test.mp4`) in the same directory as the script.
3.  **Run the Script:** Execute the Python script.  It will:
    *   Process `test.mp4` twice to simulate "camera A" and "camera B" views, saving 2D pose data to `output/pose_data_a.json` and `output/pose_data_b.json`.
    *   Generate 3D visualization frames in the `output/frames` directory.
4.  **Customize:**
    *   **Input Video:** Change `video_path` in the `if __name__ == '__main__':` block to process a different video.
    *   **Output Paths:** Modify `output_dir`, `output_json_path_a`, `output_json_path_b`, `output_frames_dir` to customize output file locations.
    *   **`show_video=True`:**  Enable `show_video=True` in `create_pose_data` to view the annotated video output during 2D pose estimation.
    *   **Camera IDs:**  Experiment with different `camera_id` values if you are simulating or have actual multi-camera data.
    *   **3D Plotting Parameters:** Adjust `VIEW_X_MIN`, `VIEW_X_MAX`, `VIEW_Y_MIN`, `VIEW_Y_MAX`, `VIEW_Z_MIN`, `VIEW_Z_MAX`, and `ax.view_init()` in `create_3d_video` to customize the 3D plot's appearance.

**Limitations and Future Work:**

*   **Simplified 3D Reconstruction:** The current 3D visualization is very basic and not accurate.  Future improvements should focus on implementing more robust 3D reconstruction techniques like:
    *   Stereo vision and triangulation with calibrated cameras.
    *   Structure from Motion (SfM) approaches.
*   **Simulated Multi-Camera:** The script currently simulates two camera views from the same video.  Real-world applications would require actual multi-camera setups and calibration.
*   **Single Person 3D:** The 3D visualization currently focuses on the first detected person.  Extending to multi-person 3D pose estimation and handling occlusions is needed.
*   **No Temporal Tracking:**  The script processes each frame independently.  Implementing temporal tracking would improve pose estimation consistency and enable motion analysis over time.
*   **Performance:**  Optimization for real-time performance could be explored for live applications.

**Inspiration and References:**

*   **[Computer Vision for Football Analysis in Python with Yolov8 & OpenCV - YouTube](https://www.youtube.com/watch?v=yJWAtr3kvPU)** -  Inspired the application of computer vision for sports analysis.
*   **[Ultralytics YOLOv8 Documentation - Pose Estimation](https://docs.ultralytics.com/tasks/pose/)** - YOLOv8-pose model documentation.

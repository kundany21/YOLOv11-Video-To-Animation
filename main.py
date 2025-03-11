import cv2
import json
from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib

def create_pose_data(video_path, output_json_path, camera_id, show_video=False):
    """
    Processes a video, extracts pose and bounding box data using YOLO,
    and saves the data to a JSON file.

    Args:
        video_path (str): Path to the input video file.
        output_json_path (str): Path to save the output JSON file.
        camera_id (int):  An ID to identify the camera (e.g., 0 or 1).
                           This is important for later 3D reconstruction.
        show_video (bool): Whether to display the annotated video (for debugging).
    """

    model = YOLO("yolo11n-pose.pt")  

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}, Dimensions: {frame_width}x{frame_height}")


    results_list = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break   

        frame_number += 1
        print(f"Processing frame: {frame_number}")

        results = model(frame, verbose=False)[0]

        frame_data = {
            "frame_number": frame_number,
            "camera_id": camera_id,  
            "detections": []
        }

        if results.boxes is not None:
            for i in range(len(results.boxes.data)):
                box = results.boxes.xyxy[i].cpu().numpy().tolist()
                confidence = results.boxes.conf[i].cpu().item()

                if results.keypoints is not None:
                    keypoints = results.keypoints.xy[i].cpu().numpy().tolist()
                    keypoints_normalized = results.keypoints.xyn[i].cpu().numpy().tolist()
                else:
                    keypoints = []
                    keypoints_normalized = []

                frame_data["detections"].append({
                    "box": box,
                    "confidence": confidence,
                    "keypoints": keypoints,
                    "keypoints_normalized": keypoints_normalized
                })
        results_list.append(frame_data)

        if show_video:
            annotated_frame = results.plot()  
            cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_video:
        cv2.destroyAllWindows()

    with open(output_json_path, "w") as f:
        json.dump(results_list, f, indent=4)

    print(f"Pose data saved to: {output_json_path}")



def create_3d_video(json_path_a, json_path_b, output_frames_dir):
    """
    Creates individual 3D visualization frames from two JSON files containing pose data.
    Saves each frame as a PNG file in the specified directory.
    """
    matplotlib.use('Agg')  
    import matplotlib.pyplot as plt

    
    os.makedirs(output_frames_dir, exist_ok=True)

    
    VIEW_X_MIN = -200
    VIEW_X_MAX = 200
    VIEW_Y_MIN = -200
    VIEW_Y_MAX = 200
    VIEW_Z_MIN = 0
    VIEW_Z_MAX = 500
    
    POSE_ANCHORS = [
       [0, 1], [0, 2], [1, 3], [2, 4],
       [5, 6], [5, 7], [6, 8], [7, 9],
       [8, 10], [5, 11], [6, 12], [11, 12],
       [11, 13], [12, 14], [13, 15], [14, 16]
    ]

    
    data_a = load_json(json_path_a)
    data_b = load_json(json_path_b)

    if len(data_a) != len(data_b):
        raise ValueError("JSON files must have the same number of frames.")

    
    for i in range(len(data_a)):
        frame_a = data_a[i]
        frame_b = data_b[i]
        print(f"Processing frame {i}")

        if frame_a["detections"] and frame_b["detections"]:
            pose_a = Pose.load(frame_a["detections"][0]["keypoints"])
            pose_b = Pose.load(frame_b["detections"][0]["keypoints"])
            pose3d = Pose3D.from2D(pose_a, pose_b, view_height=VIEW_Z_MAX)

            
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')

            ax.scatter(pose3d.x, pose3d.y, pose3d.z, c='r', marker='o')
            for bone in POSE_ANCHORS:
                ax.plot([pose3d.x[bone[0]], pose3d.x[bone[1]]],
                       [pose3d.y[bone[0]], pose3d.y[bone[1]]],
                       [pose3d.z[bone[0]], pose3d.z[bone[1]]], color='blue')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim([VIEW_X_MIN, VIEW_X_MAX])
            ax.set_ylim([VIEW_Y_MIN, VIEW_Y_MAX])
            ax.set_zlim([VIEW_Z_MIN, VIEW_Z_MAX])
            ax.view_init(elev=20., azim=-35)

            
            frame_path = os.path.join(output_frames_dir, f'frame_{i:04d}.png')
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)  
        else:
            print(f"Skipping frame {i} due to missing detections.")

    print(f"Frames saved to: {output_frames_dir}")




@dataclass
class Pose:
    x: np.ndarray
    y: np.ndarray
    confidence: np.ndarray

    @classmethod
    def load(cls, data: List[float]) -> 'Pose':
        x, y, confidence = [], [], []
        for i in range(17):
            x.append(data[i][0])  
            y.append(data[i][1])  
            confidence.append(data[i][2] if len(data[i]) > 2 else 1.0)  
        return cls(
            x=np.array(x),
            y=np.array(y),
            confidence=np.array(confidence)
        )

@dataclass
class Pose3D:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @classmethod
    def from2D(cls, pose_a: Pose, pose_b: Pose, view_height: float = 500) -> 'Pose3D':
        
        valid_points = (pose_a.x != 0) & (pose_a.y != 0) & (pose_b.x != 0)
        
        if np.any(valid_points):
            x_center = np.mean(pose_a.x[valid_points])
            y_center = np.mean(pose_b.x[valid_points])
            
            
            max_y = np.max(pose_a.y[valid_points])
        else:
            x_center = 0
            y_center = 0
            max_y = view_height
        
        
        x = np.where(pose_a.x != 0, pose_a.x - x_center, 0)
        y = np.where(pose_b.x != 0, pose_b.x - y_center, 0)
        
        
        z = np.where(pose_a.y != 0, max_y - pose_a.y, 0)
        
        return cls(x=x, y=y, z=z)

@dataclass
class Point:
    x: float
    y: float

    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

@dataclass
class Point3D:
    x: float
    y: float
    z: float

    @classmethod
    def from2D(cls, point_a: Point, point_b: Point) -> 'Point3D':
        
        return cls(
            x=point_a.x,
            y=point_b.x,
            z=point_a.y  
        )



@dataclass
class Detection:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_id: int

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Point:
        return Point(
            x=(self.x_min + self.x_max) / 2,
            y=(self.y_min + self.y_max) / 2
        )

    @classmethod
    def load(cls, data: List[float]) -> 'Detection':
        return cls(
            x_min=float(data[0]),
            y_min=float(data[1]),
            x_max=float(data[2]),
            y_max=float(data[3]),
            confidence=float(data[4]),
            class_id=int(data[5])
        )

    @classmethod
    def filter(cls, detections: List['Detection'], class_id: int) -> Optional['Detection']:
        filtered_detections = [
            detection
            for detection
            in detections
            if detection.class_id == class_id
        ]
        return filtered_detections[0] if len(filtered_detections) == 1 else None

def load_json(json_path: str) -> List:
    """Load JSON data from a file."""
    with open(json_path, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "test.mp4")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    output_json_path_a = os.path.join(output_dir, "pose_data_a.json")
    output_json_path_b = os.path.join(output_dir, "pose_data_b.json")
    output_frames_dir = os.path.join(output_dir, "frames")  

    
    create_pose_data(video_path, output_json_path_a, camera_id=0, show_video=True)
    create_pose_data(video_path, output_json_path_b, camera_id=1, show_video=True)

    
    create_3d_video(output_json_path_a, output_json_path_b, output_frames_dir)

   
    
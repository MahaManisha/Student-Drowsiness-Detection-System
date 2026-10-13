"""
MediaPipe FaceMesh Detection & Tracking Module.
Handles multi-face detection, selects the primary (largest) face,
and extracts normalized 3D and pixel 2D facial landmarks.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.logger import logger


class FaceMeshDetector:
    """Wrapper for MediaPipe FaceMesh with multi-face tracking and landmark extraction."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 5,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "face_landmarker.task"
        )
        if not os.path.exists(model_path):
            logger.info("Downloading FaceLandmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

            
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False)
            
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process frame and return landmark data for the largest face detected.

        Returns:
            dict containing:
                - face_found (bool): True if at least one face detected
                - multiple_faces (bool): True if more than 1 face detected
                - face_count (int): Total faces detected
                - landmarks_norm (np.ndarray): Shape (N, 3) normalized landmarks (0.0 to 1.0)
                - landmarks_pixel (np.ndarray): Shape (N, 3) pixel coordinate landmarks
                - bbox (tuple): (x_min, y_min, width, height) of primary face
        """
        result = {
            "face_found": False,
            "multiple_faces": False,
            "face_count": 0,
            "landmarks_norm": None,
            "landmarks_pixel": None,
            "bbox": None,
        }

        if frame is None or frame.size == 0:
            return result

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            if self.running_mode == vision.RunningMode.VIDEO:
                current_time_ms = int(time.time() * 1000)
                if current_time_ms <= self.last_timestamp_ms:
                    current_time_ms = self.last_timestamp_ms + 1
                self.last_timestamp_ms = current_time_ms
                mesh_results = self.detector.detect_for_video(mp_image, current_time_ms)
            else:
                mesh_results = self.detector.detect(mp_image)
                
            if not mesh_results.face_landmarks:
                return result

            face_count = len(mesh_results.face_landmarks)
            result["face_count"] = face_count
            result["multiple_faces"] = face_count > 1

            # Select largest face by bounding box area
            max_area = -1
            best_bbox = None
            best_norm_landmarks = None
            best_pixel_landmarks = None

            for face_landmarks in mesh_results.face_landmarks:
                # Convert landmarks to numpy array
                norm_coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in face_landmarks],
                    dtype=np.float64,
                )
                pixel_coords = np.array(
                    [[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks],
                    dtype=np.float64,
                )

                x_min, y_min = np.min(pixel_coords[:, :2], axis=0)
                x_max, y_max = np.max(pixel_coords[:, :2], axis=0)
                box_w = max(0, x_max - x_min)
                box_h = max(0, y_max - y_min)
                area = box_w * box_h

                if area > max_area:
                    max_area = area
                    best_bbox = (int(x_min), int(y_min), int(box_w), int(box_h))
                    best_norm_landmarks = norm_coords
                    best_pixel_landmarks = pixel_coords

            if best_norm_landmarks is not None:
                result["face_found"] = True
                result["landmarks_norm"] = best_norm_landmarks
                result["landmarks_pixel"] = best_pixel_landmarks
                result["bbox"] = best_bbox

        except Exception as e:
            logger.error(f"FaceMesh processing error: {e}")

        return result

    def close(self):
        """Release FaceMesh resources."""
        self.detector.close()

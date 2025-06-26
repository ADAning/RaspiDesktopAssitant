"""
For the Raspberry Pi desktop assistant's visual signal acquisition, capturing video input from the Raspberry Pi camera (CSI camera or USB camera).
"""

import cv2
import base64


class Camera:
    def __init__(self):
        camera_indexes = self.find_available_cameras()
        if len(camera_indexes) == 0:
            raise RuntimeError(
                "No cameras found. Please connect a camera and try again."
            )
        self.capture = cv2.VideoCapture(camera_indexes[0])

    def find_available_cameras(max_tested=5):
        available_cameras = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def capture_frame(self):
        """Capture a single frame from the camera."""
        ret, frame = self.capture.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera.")
        return frame
    
    def capture_frame_base64(self):
        """Capture a single frame and return it as a base64 encoded string."""
        frame = self.capture_frame()
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
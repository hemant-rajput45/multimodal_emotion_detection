"""
Facial Modality Preprocessor
Handles video frame extraction, face detection, cropping, normalization, and augmentation
as described in Section V-C of the research paper.
"""

import os
import cv2
import numpy as np
from PIL import Image

class FacialPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.face_cascade = None
        try:
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                if os.path.isfile(cascade_path):
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        self.face_cascade = cascade
        except Exception:
            self.face_cascade = None

    def detect_and_crop_face(self, image_input):
        """
        Detect face bounding box and crop with padding.
        Input: image as numpy array (BGR or RGB) or file path.
        Returns: Cropped face image in RGB (target_size)
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image from {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("Unsupported image input type")

        if len(image.shape) == 2:
            gray = image
            color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            color = image

        faces = []
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )
            except Exception:
                faces = []

        if len(faces) > 0:
            # Pick largest detected face
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            # Add small margin/padding around face
            pad_w = int(0.1 * w)
            pad_h = int(0.1 * h)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(color.shape[1], x + w + pad_w)
            y2 = min(color.shape[0], y + h + pad_h)
            cropped = color[y1:y2, x1:x2]
        else:
            # Fallback to center-crop if no face is detected
            h, w = color.shape[:2]
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            cropped = color[start_y:start_y + min_dim, start_x:start_x + min_dim]

        # Resize to target size (224x224)
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB
        rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb_face

    def transform(self, image_input):
        """
        Full pipeline: detect, crop, normalize, and return normalized float tensor [3, H, W]
        """
        rgb_face = self.detect_and_crop_face(image_input)
        # Normalization (Standard ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        norm_img = rgb_face.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (norm_img - mean) / std
        # Transpose from (H, W, C) to (C, H, W)
        chw_img = np.transpose(norm_img, (2, 0, 1))
        return chw_img

    def extract_frames_from_video(self, video_path, num_frames=5):
        """
        Extracts representative facial frames from a video file.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(self.detect_and_crop_face(frame))
        cap.release()
        return frames

"""Lightweight video diagnostics; warnings are guidance, not reconstruction scores."""
from pathlib import Path

import cv2
import numpy as np


def analyze_video(path: Path, samples: int = 18) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not decode the uploaded video")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = round(count / fps, 1) if fps > 0 else 0
        if not width or not height:
            raise ValueError("Video resolution is unavailable")
        brightness, sharpness = [], []
        # Seek through representative frames, never load the entire video into RAM.
        indices = np.linspace(0, max(count - 1, 0), min(samples, max(1, count))).astype(int)
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if gray.shape[1] > 480:
                gray = cv2.resize(gray, (480, max(1, int(480 * gray.shape[0] / gray.shape[1]))))
            brightness.append(float(np.mean(gray)))
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if not brightness:
            raise ValueError("Video contains no readable frames")
        warnings = []
        if min(width, height) < 480:
            warnings.append("Low resolution. Record at 720p or higher for more detail.")
        if duration and duration < 6:
            warnings.append("Short clip. Record a slower, complete view of the object.")
        if np.median(brightness) < 45:
            warnings.append("Image looks dark. Use brighter, even lighting.")
        if np.median(brightness) > 225:
            warnings.append("Image looks overexposed. Reduce strong lighting.")
        if np.median(sharpness) < 35:
            warnings.append("Many sampled frames appear soft. Move more slowly and keep the object in focus.")
        return {
            "width": width, "height": height, "fps": round(fps, 2),
            "duration_seconds": duration, "frames_sampled": len(brightness),
            "mean_brightness": round(float(np.mean(brightness)), 1),
            "median_sharpness": round(float(np.median(sharpness)), 1),
            "warnings": warnings,
        }
    finally:
        cap.release()

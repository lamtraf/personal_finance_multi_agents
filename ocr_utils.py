import cv2
import numpy as np
import os
import tempfile
import logging

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str) -> str:
    """
    Preprocess the image to improve OCR accuracy with advanced techniques:
    - Convert to grayscale
    - Denoise (Gaussian + bilateral)
    - Enhance contrast (CLAHE)
    - Remove background (adaptive thresholding)
    - Deskew (correct rotation)
    - Crop text region
    - Sharpen text
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logger.info("Converted image to grayscale")

        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        logger.info("Applied Gaussian and bilateral denoising")

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        logger.info("Enhanced contrast using CLAHE")

        thresh = cv2.adaptiveThreshold(
            contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        logger.info("Applied adaptive thresholding")

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        rotated = cv2.rotate(thresh, angle, reshape=True)
        rotated = np.clip(rotated, 0, 255).astype(np.uint8)
        logger.info(f"Deskewed image by {angle} degrees")

        contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(np.concatenate(contours))
            cropped = rotated[y:y+h, x:x+w]
        else:
            cropped = rotated
        logger.info("Cropped text region")

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(cropped, -1, kernel)
        logger.info("Applied sharpening filter")

        temp_file = os.path.join(tempfile.gettempdir(), "temp_preprocessed_image.png")
        cv2.imwrite(temp_file, sharpened)
        logger.info(f"Saved preprocessed image to: {temp_file}")
        return temp_file
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return image_path
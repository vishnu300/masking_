import os
import shutil
import cv2
import re
import keras_ocr
import numpy as np
from io import BytesIO

def preprocess_image(image_blob):
    """
    Preprocess the Aadhaar image for OCR by converting it to grayscale.
    Args:
        image_blob: Byte data of the image.
    Returns:
        Preprocessed image as a NumPy array.
    """
    # Convert the BLOB to a NumPy array
    image_data = np.frombuffer(image_blob.read(), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Resize image for better OCR accuracy
    resized_image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_LINEAR)

    return resized_image

def detect_masking_with_kerasocr(image_blob):
    """
    Detect if the Aadhaar image has the first 8 digits masked using Keras OCR.
    Args:
        image_blob: Byte data of the Aadhaar image.
    Returns:
        str: "Masked", "Not Masked", "VID Detected", or "No Aadhaar Number Detected".
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image_blob)

    # Initialize Keras OCR Pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Perform OCR detection
    prediction_groups = pipeline.recognize([preprocessed_image])

    # Extract detected text
    detected_text = ' '.join([text for text, _ in prediction_groups[0]])

    # Aadhaar number pattern (XXXX XXXX XXXX)
    aadhar_pattern = r'\b\d{4} \d{4} \d{4}\b'
    # VID pattern (16 digits)
    vid_pattern = r'\b\d{4} \d{4} \d{4} \d{4}\b'

    # Check for Aadhaar numbers
    aadhar_match = re.search(aadhar_pattern, detected_text)

    # Exclude if it's a VID
    vid_match = re.search(vid_pattern, detected_text)

    if aadhar_match and not vid_match:
        aadhar_number = aadhar_match.group(0)
        # Check if the first 8 digits are masked or obscured
        if re.fullmatch(r'[Xx]{4} [Xx]{4} \d{4}', aadhar_number) or \
           re.fullmatch(r'\d{4} [Xx]{4} [Xx]{4}', aadhar_number):
            return "Masked"
        else:
            return "Not Masked"
    elif vid_match:
        return "VID Detected"
    else:
        return "No Aadhaar Number Detected"
    
img_ = r"C:\Users\DELL\VISHNU_AM\Projects\python_code_pj\WhatsApp Image 2025-01-04 at 19.47.10_ca6537ec.jpg"
with open(img_, "rb") as image_file:
    aadhaar_detection_result = detect_masking_with_kerasocr(image_file)

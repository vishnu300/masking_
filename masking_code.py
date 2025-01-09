import cv2
import numpy as np
import re
import pytesseract
from pytesseract import Output

def preprocess_image(image_blob):
    # Convert the BLOB to a NumPy array
    image_data = np.frombuffer(image_blob.read(), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding for better contrast
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Perform morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    return processed_image

def detect_aadhar_with_tesseract(image_blob):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_blob)

    # Perform OCR with Tesseract
    custom_config = r'--oem 3 --psm 6'
    ocr_result = pytesseract.image_to_string(preprocessed_image, config=custom_config)

    # Aadhaar number pattern (XXXX XXXX XXXX)
    aadhar_pattern = r'\b\d{4} \d{4} \d{4}\b'
    # VID pattern (16 digits)
    vid_pattern = r'\b\d{4} \d{4} \d{4} \d{4}\b'

    # Check for Aadhaar numbers
    aadhar_match = re.search(aadhar_pattern, ocr_result)

    # Exclude if it's a VID
    vid_match = re.search(vid_pattern, ocr_result)

    if aadhar_match and not vid_match:
        aadhar_number = aadhar_match.group(0)
        # Check if the first 8 digits are masked
        if re.fullmatch(r'[Xx]{4} [Xx]{4} \d{4}', aadhar_number):
            return "Masked"
        else:
            return "Not Masked"
    elif vid_match:
        return "VID Detected"
    else:
        return "No Aadhaar Number Detected"
    
    
def inpaint_masked_regions(image):
    """
    Detect and inpaint masked regions to enhance OCR accuracy.
    Args:
        image: Grayscale image with masked regions.
    Returns:
        Inpainted image.
    """
    # Detect masked regions using contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Inpaint the masked regions
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_image

# Pass a file-like object

img_ = r"C:\Users\DELL\VISHNU_AM\Projects\python_code_pj\WhatsApp Image 2025-01-04 at 19.47.10_ca6537ec.jpg"
with open(img_, "rb") as image_file:
    aadhaar_detection_result = detect_aadhar_with_tesseract(image_file)


# # Replace 'path/to/your/image.jpg' with the actual path to your Aadhaar card image
# aadhaar_detection_result = detect_aadhar_with_tesseract(r"C:\Users\DELL\VISHNU_AM\Projects\python_code_pj\WhatsApp Image 2025-01-04 at 19.47.10_ca6537ec.jpg")
# print(aadhaar_detection_result)
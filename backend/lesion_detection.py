"""
Lesion Detection Module using OpenCV
Detects and highlights abnormal regions in CT brain scans
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def preprocess_ct_image(image_path, target_size=(224, 224)):
    """Preprocess CT image for model input"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image, target_size)
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image_normalized)
    
    return image, image_resized, image_tensor


def detect_lesion_ct(image_path, output_path=None):
    """
    Detect lesion in CT scan using multiple methods:
    - Thresholding for abnormal intensity regions
    - Contour detection for potential lesions
    - Edge detection for boundaries
    
    Returns:
        - annotated_image: Image with lesion highlighted
        - lesion_bbox: Bounding box coordinates (x, y, w, h) or None
        - lesion_center: Center point of detected lesion or None
    """
    # Read original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create a copy for annotation
    annotated_image = original_image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to detect abnormal regions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply Otsu's thresholding for better separation
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine thresholds
    combined_thresh = cv2.bitwise_and(thresh, otsu_thresh)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
    combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to remove noise
    min_area = 100  # Minimum area for a valid lesion
    max_area = original_image.shape[0] * original_image.shape[1] * 0.5
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
    
    lesion_bbox = None
    lesion_center = None
    
    if valid_contours:
        # Get the largest contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        lesion_bbox = (x, y, w, h)
        
        # Get center point
        lesion_center = (x + w // 2, y + h // 2)
        
        # Draw rectangle on annotated image
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        # Draw circle at center
        cv2.circle(annotated_image, lesion_center, 20, (0, 0, 255), -1)
        
        # Add text label
        cv2.putText(annotated_image, "LESION DETECTED", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # If no contours found, try alternative method using intensity analysis
        # Look for hyperdense (bright) or hypodense (dark) regions
        
        # Calculate mean and std
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Detect outliers (potential lesions)
        threshold_high = mean_val + 1.5 * std_val
        threshold_low = mean_val - 1.5 * std_val
        
        # Create mask for abnormal regions
        abnormal_mask = ((gray > threshold_high) | (gray < threshold_low)).astype(np.uint8) * 255
        
        # Find contours in abnormal regions
        contours, _ = cv2.findContours(abnormal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                lesion_bbox = (x, y, w, h)
                lesion_center = (x + w // 2, y + h // 2)
                
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.circle(annotated_image, lesion_center, 20, (0, 0, 255), -1)
                cv2.putText(annotated_image, "LESION DETECTED", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                break
    
    # Save the annotated image if output path provided
    if output_path:
        cv2.imwrite(output_path, annotated_image)
    
    return annotated_image, lesion_bbox, lesion_center


def highlight_lesion_region(image_path, stroke_type, output_path):
    """
    Highlight lesion region based on stroke type
    
    Args:
        image_path: Path to input CT image
        stroke_type: Type of stroke ('Hemorrhagic Stroke' or 'Ischemic Stroke')
        output_path: Path to save highlighted image
    
    Returns:
        Path to saved highlighted image
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create copy for highlighting
    highlighted = image.copy()
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Detect lesion regions
    annotated, bbox, center = detect_lesion_ct(image_path, None)
    
    if bbox and center:
        x, y, w, h = bbox
        cx, cy = center
        
        # Draw enhanced highlighting
        # Outer glow effect
        for thickness in [15, 10, 5]:
            color = (0, 0, 255)  # Red for stroke
            cv2.rectangle(highlighted, (x - 5, y - 5), (x + w + 5, y + h + 5), color, thickness)
        
        # Center marker
        cv2.circle(highlighted, (cx, cy), 15, (0, 255, 255), -1)
        cv2.circle(highlighted, (cx, cy), 25, (0, 0, 255), 2)
        
        # Add label
        stroke_label = "HEMORRHAGIC" if stroke_type == "Hemorrhagic Stroke" else "ISCHEMIC"
        cv2.putText(highlighted, f"{stroke_label} STROKE", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(highlighted, f"Location: ({cx}, {cy})", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # If no specific lesion found, highlight the center region as suspected area
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Draw a circular region of interest
        radius = min(w, h) // 4
        cv2.circle(highlighted, (cx, cy), radius, (0, 0, 255), 5)
        cv2.circle(highlighted, (cx, cy), 10, (0, 255, 255), -1)
        
        # Add label
        stroke_label = "HEMORRHAGIC" if stroke_type == "Hemorrhagic Stroke" else "ISCHEMIC"
        cv2.putText(highlighted, f"SUSPECTED {stroke_label} STROKE", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    # Save the highlighted image
    cv2.imwrite(output_path, highlighted)
    
    return output_path


def generate_lesion_heatmap(image_path, output_path):
    """
    Generate a heatmap overlay showing potential lesion regions
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply various filters to detect abnormalities
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create heatmap
    heatmap = np.zeros_like(gray).astype(np.float32)
    
    # Add contributions from different methods
    heatmap += (edges / 255.0) * 0.3
    heatmap += (255 - thresh) / 255.0 * 0.4
    
    # Apply colormap
    heatmap_normalized = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # Blend with original
    blended = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    cv2.imwrite(output_path, blended)
    
    return output_path


if __name__ == "__main__":
    # Test the lesion detection
    print("Lesion detection module loaded successfully!")
    
    # Create a test image if needed
    test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
    cv2.imwrite("test_ct.png", test_image)
    
    print("Test image created. Run detect_lesion_ct on a real CT scan for lesion detection.")


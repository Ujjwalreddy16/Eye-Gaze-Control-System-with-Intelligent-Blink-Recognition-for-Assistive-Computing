# enhanced_utils.py
# Enhanced utility functions with TRUE GAZE TRACKING support

import numpy as np
import cv2
from collections import deque
from enhanced_config import *

# Global buffers for EAR smoothing
left_ear_history = deque(maxlen=EAR_BUFFER_SIZE)
right_ear_history = deque(maxlen=EAR_BUFFER_SIZE)

def get_eye_aspect_ratio(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR) with improved accuracy
    
    Args:
        eye_points: List of 6 (x, y) tuples representing eye landmarks
        
    Returns:
        EAR value as float
    """
    if len(eye_points) != 6:
        return 0.0
    
    try:
        # Convert to numpy array
        points = np.array(eye_points)
        
        # Calculate vertical distances
        A = np.linalg.norm(points[1] - points[5])  # |p2 - p6|
        B = np.linalg.norm(points[2] - points[4])  # |p3 - p5|
        
        # Calculate horizontal distance
        C = np.linalg.norm(points[0] - points[3])  # |p1 - p4|
        
        # Avoid division by zero
        if C == 0:
            return 0.0
            
        # EAR calculation
        ear = (A + B) / (2.0 * C)
        
        # Clamp to reasonable range
        return max(0.0, min(1.0, ear))
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Error calculating EAR: {e}")
        return 0.0

def get_smoothed_ear(eye_points, side="left"):
    """
    Calculate smoothed EAR using historical data
    
    Args:
        eye_points: Eye landmark points
        side: "left" or "right" eye
        
    Returns:
        Smoothed EAR value
    """
    current_ear = get_eye_aspect_ratio(eye_points)
    
    if side == "left":
        left_ear_history.append(current_ear)
        history = list(left_ear_history)
    else:
        right_ear_history.append(current_ear)
        history = list(right_ear_history)
    
    if len(history) < 3:
        return current_ear
    
    # Use weighted average with more weight on recent values
    weights = np.array([0.1, 0.2, 0.3, 0.4] if len(history) >= 4 else [0.3, 0.7])
    weights = weights[:len(history)]
    weights = weights / weights.sum()  # Normalize
    
    return np.average(history[-len(weights):], weights=weights)

def detect_pupil_darkest_point(eye_region):
    """
    Detect pupil using darkest point method
    
    Args:
        eye_region: Grayscale eye region image
        
    Returns:
        tuple: (x, y) pupil position, quality score
    """
    if eye_region.size == 0:
        return (0, 0), 0.0
    
    try:
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(eye_region, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
        
        # Find minimum (darkest) location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        # Calculate quality based on contrast
        contrast = (max_val - min_val) / 255.0
        quality = min(1.0, contrast * 2)
        
        return min_loc, quality
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Darkest point detection error: {e}")
        h, w = eye_region.shape
        return (w // 2, h // 2), 0.0

def detect_pupil_hough_circles(eye_region):
    """
    Detect pupil using Hough circle detection
    
    Args:
        eye_region: Grayscale eye region image
        
    Returns:
        tuple: (x, y) pupil position, quality score
    """
    if eye_region.size == 0 or not USE_HOUGH_CIRCLES:
        return (0, 0), 0.0
    
    try:
        h, w = eye_region.shape
        
        # Apply median blur
        blurred = cv2.medianBlur(eye_region, 5)
        
        # Calculate radius range
        min_radius = int(min(h, w) * MIN_PUPIL_RADIUS_PERCENT)
        max_radius = int(min(h, w) * MAX_PUPIL_RADIUS_PERCENT)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=HOUGH_MIN_DIST,
            param1=HOUGH_PARAM1,
            param2=HOUGH_PARAM2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None and len(circles[0]) > 0:
            # Use the first detected circle
            x, y, r = circles[0][0]
            
            # Calculate quality based on radius
            ideal_radius = (min_radius + max_radius) / 2
            radius_diff = abs(r - ideal_radius) / ideal_radius
            quality = max(0.1, 1.0 - radius_diff)
            
            return (int(x), int(y)), quality
        
        return (w // 2, h // 2), 0.0
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Hough circles detection error: {e}")
        h, w = eye_region.shape
        return (w // 2, h // 2), 0.0

def detect_pupil_contours(eye_region):
    """
    Detect pupil using contour detection
    
    Args:
        eye_region: Grayscale eye region image
        
    Returns:
        tuple: (x, y) pupil position, quality score
    """
    if eye_region.size == 0 or not USE_CONTOUR_DETECTION:
        return (0, 0), 0.0
    
    try:
        h, w = eye_region.shape
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(eye_region, (5, 5), 0)
        
        # Threshold
        if ADAPTIVE_THRESHOLD:
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            _, thresh = cv2.threshold(
                blurred, PUPIL_THRESHOLD_MIN, 255, cv2.THRESH_BINARY_INV
            )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (w // 2, h // 2), 0.0
        
        # Find best contour
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < MIN_CONTOUR_AREA:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < CIRCULARITY_THRESHOLD:
                continue
            
            # Get center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Distance from center
            center_dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            center_score = 1.0 - (center_dist / max_dist)
            
            # Combined score
            score = circularity * 0.7 + center_score * 0.3
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), best_score
        
        return (w // 2, h // 2), 0.0
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Contour detection error: {e}")
        h, w = eye_region.shape
        return (w // 2, h // 2), 0.0

def get_enhanced_pupil_position(eye_region):
    """
    Enhanced pupil detection using multiple methods
    
    Args:
        eye_region: Grayscale eye region image
        
    Returns:
        tuple: ((cx, cy), quality_score)
    """
    if eye_region.size == 0:
        return (0, 0), 0.0
    
    h, w = eye_region.shape
    
    # Method 1: Darkest point
    dark_pos, dark_quality = detect_pupil_darkest_point(eye_region)
    
    # Method 2: Hough circles
    hough_pos, hough_quality = detect_pupil_hough_circles(eye_region)
    
    # Method 3: Contours
    contour_pos, contour_quality = detect_pupil_contours(eye_region)
    
    # Combine methods with weighted average
    total_weight = (PUPIL_WEIGHT_DARKEST * dark_quality + 
                   PUPIL_WEIGHT_HOUGH * hough_quality + 
                   PUPIL_WEIGHT_CONTOUR * contour_quality)
    
    if total_weight > 0:
        final_x = (dark_pos[0] * PUPIL_WEIGHT_DARKEST * dark_quality +
                  hough_pos[0] * PUPIL_WEIGHT_HOUGH * hough_quality +
                  contour_pos[0] * PUPIL_WEIGHT_CONTOUR * contour_quality) / total_weight
        
        final_y = (dark_pos[1] * PUPIL_WEIGHT_DARKEST * dark_quality +
                  hough_pos[1] * PUPIL_WEIGHT_HOUGH * hough_quality +
                  contour_pos[1] * PUPIL_WEIGHT_CONTOUR * contour_quality) / total_weight
        
        # Average quality
        avg_quality = total_weight / (PUPIL_WEIGHT_DARKEST + PUPIL_WEIGHT_HOUGH + PUPIL_WEIGHT_CONTOUR)
        
        return (int(final_x), int(final_y)), avg_quality
    else:
        # Fallback to center
        return (w // 2, h // 2), 0.1

def smooth_position(prev, current, alpha):
    """
    Smooth position using exponential moving average
    
    Args:
        prev: Previous position
        current: Current position  
        alpha: Smoothing factor (0-1)
        
    Returns:
        Smoothed position as integer
    """
    if not (0 <= alpha <= 1):
        alpha = max(0, min(1, alpha))
    
    smoothed = prev * (1 - alpha) + current * alpha
    return int(smoothed)

def apply_median_filter(positions):
    """
    Apply median filter to a list of positions
    
    Args:
        positions: List of (x, y) tuples
        
    Returns:
        Median filtered (x, y) position
    """
    if not positions:
        return (0, 0)
    
    if len(positions) == 1:
        return positions[0]
    
    x_values = [p[0] for p in positions]
    y_values = [p[1] for p in positions]
    
    median_x = np.median(x_values)
    median_y = np.median(y_values)
    
    return (int(median_x), int(median_y))

def calculate_eye_center(eye_points):
    """
    Calculate the center of an eye region
    
    Args:
        eye_points: List of eye landmark points
        
    Returns:
        (center_x, center_y) tuple
    """
    if not eye_points:
        return (0, 0)
    
    points = np.array(eye_points)
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    
    return (int(center_x), int(center_y))

def validate_eye_landmarks(eye_points, frame_shape):
    """
    Validate that eye landmarks are reasonable
    
    Args:
        eye_points: Eye landmark points
        frame_shape: Shape of the frame (height, width)
        
    Returns:
        Boolean indicating if landmarks are valid
    """
    if len(eye_points) != 6:
        return False
    
    height, width = frame_shape[:2]
    
    for point in eye_points:
        x, y = point
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
    
    # Check if points form a reasonable eye shape
    points = np.array(eye_points)
    eye_width = np.max(points[:, 0]) - np.min(points[:, 0])
    eye_height = np.max(points[:, 1]) - np.min(points[:, 1])
    
    if eye_width <= 0 or eye_height <= 0:
        return False
        
    # Eye should have reasonable aspect ratio (2:1 to 4:1)
    aspect_ratio = eye_width / eye_height
    return 1.5 <= aspect_ratio <= 4.5

def normalize_pupil_in_eye(pupil_pos, eye_landmarks, eye_bbox):
    """
    Normalize pupil position within eye region (0-1 range)
    This is the KEY function for true gaze tracking!
    
    Args:
        pupil_pos: (x, y) pupil position in frame coordinates
        eye_landmarks: List of 6 eye landmark points
        eye_bbox: (x, y, w, h) bounding box of eye region
        
    Returns:
        (norm_x, norm_y) normalized position (0.0-1.0)
    """
    if not eye_landmarks or len(eye_landmarks) != 6:
        return 0.5, 0.5
    
    eye_points = np.array(eye_landmarks)
    
    # Get eye boundaries from landmarks
    eye_left = np.min(eye_points[:, 0])
    eye_right = np.max(eye_points[:, 0])
    eye_top = np.min(eye_points[:, 1])
    eye_bottom = np.max(eye_points[:, 1])
    
    eye_width = eye_right - eye_left
    eye_height = eye_bottom - eye_top
    
    if eye_width == 0 or eye_height == 0:
        return 0.5, 0.5
    
    # Normalize pupil position relative to eye boundaries
    norm_x = (pupil_pos[0] - eye_left) / eye_width
    norm_y = (pupil_pos[1] - eye_top) / eye_height
    
    # Clamp to valid range
    norm_x = max(0.0, min(1.0, norm_x))
    norm_y = max(0.0, min(1.0, norm_y))
    
    return norm_x, norm_y

def map_gaze_to_screen(norm_x, norm_y, screen_width, screen_height):
    """
    Map normalized gaze coordinates to screen coordinates
    
    Args:
        norm_x, norm_y: Normalized gaze position (0-1)
        screen_width, screen_height: Screen dimensions
        
    Returns:
        (screen_x, screen_y) screen coordinates
    """
    # Invert X axis (looking left = cursor left)
    norm_x = 1 - norm_x
    
    # Expand the usable range
    norm_x = (norm_x - GAZE_RANGE_X_MIN) / (GAZE_RANGE_X_MAX - GAZE_RANGE_X_MIN)
    norm_y = (norm_y - GAZE_RANGE_Y_MIN) / (GAZE_RANGE_Y_MAX - GAZE_RANGE_Y_MIN)
    
    # Clamp to valid range
    norm_x = max(0.0, min(1.0, norm_x))
    norm_y = max(0.0, min(1.0, norm_y))
    
    # Apply sensitivity curve
    norm_x = np.power(norm_x, GAZE_SENSITIVITY_CURVE)
    norm_y = np.power(norm_y, GAZE_SENSITIVITY_CURVE)
    
    # Map to screen with margins
    screen_x = int(norm_x * (screen_width - 2 * SCREEN_BORDER_MARGIN) + SCREEN_BORDER_MARGIN)
    screen_y = int(norm_y * (screen_height - 2 * SCREEN_BORDER_MARGIN) + SCREEN_BORDER_MARGIN)
    
    # Final clamp
    screen_x = max(0, min(screen_width - 1, screen_x))
    screen_y = max(0, min(screen_height - 1, screen_y))
    
    return screen_x, screen_y
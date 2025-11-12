# true_gaze_tracker.py
# True gaze tracking based on pupil position within the eye - FIXED VERSION

import numpy as np
import cv2
import pyautogui
import time
from collections import deque
from enhanced_config import *

class TrueGazeTracker:
    def __init__(self, predictor, detector, screen_width, screen_height):
        self.detector = detector
        self.predictor = predictor
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Calibration data for mapping pupil position to screen coordinates
        self.calibration_points = {}  # {screen_pos: (pupil_x, pupil_y)}
        self.is_calibrated = False
        
        # Eye region reference points (to detect head movement vs eye movement)
        self.left_eye_region_center = None
        self.right_eye_region_center = None
        self.eye_region_history = deque(maxlen=10)
        
        # Pupil position history for smoothing
        self.pupil_history = deque(maxlen=PUPIL_HISTORY_SIZE)
        
        # Cursor smoothing
        self.prev_cursor_x = screen_width // 2
        self.prev_cursor_y = screen_height // 2
        
        # Gaze range calibration (min/max pupil positions)
        self.pupil_range = {
            'left': {'x_min': 0.2, 'x_max': 0.8, 'y_min': 0.3, 'y_max': 0.7},
            'right': {'x_min': 0.2, 'x_max': 0.8, 'y_min': 0.3, 'y_max': 0.7}
        }
        
        # Baseline tracking
        self.baseline_samples = []
        self.baseline_established = False
        
        print("True Gaze Tracker initialized - Pupil tracking mode")
        print("IMPORTANT: Keep your head relatively still for accurate tracking")
    
    def normalize_pupil_position(self, pupil_pos_local, eye_region_shape):
        """
        FIXED: Normalize pupil position using LOCAL eye region coordinates
        This is the KEY fix - we work in eye-space, not frame-space
        
        Args:
            pupil_pos_local: (x, y) pupil position in LOCAL eye region coordinates
            eye_region_shape: (height, width) of the eye region
            
        Returns:
            (norm_x, norm_y): Normalized position within eye (0.0-1.0)
        """
        h, w = eye_region_shape
        
        if w == 0 or h == 0:
            return 0.5, 0.5
        
        # FIXED: Simply normalize within the eye region dimensions
        # pupil_pos_local is already in the eye region's coordinate system
        norm_x = pupil_pos_local[0] / w
        norm_y = pupil_pos_local[1] / h
        
        # Clamp to valid range
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        return norm_x, norm_y
    
    def detect_pupil_in_eye(self, eye_region, method='combined'):
        """
        Advanced pupil detection focusing on iris/pupil center
        Uses multiple methods for robustness
        """
        if eye_region.size == 0:
            return None
        
        h, w = eye_region.shape
        
        try:
            # Method 1: Darkest region (pupil is typically darkest)
            blurred = cv2.GaussianBlur(eye_region, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
            darkest_point = min_loc
            
            # Method 2: Circular Hough Transform for iris/pupil
            circles = None
            if USE_HOUGH_CIRCLES:
                try:
                    circles = cv2.HoughCircles(
                        blurred, cv2.HOUGH_GRADIENT, 
                        dp=HOUGH_DP, 
                        minDist=HOUGH_MIN_DIST,
                        param1=HOUGH_PARAM1, 
                        param2=HOUGH_PARAM2, 
                        minRadius=int(min(h, w) * MIN_PUPIL_RADIUS_PERCENT), 
                        maxRadius=int(min(h, w) * MAX_PUPIL_RADIUS_PERCENT)
                    )
                except:
                    circles = None
            
            if circles is not None and len(circles[0]) > 0:
                # Use the first detected circle (usually the pupil/iris)
                x, y, r = circles[0][0]
                hough_point = (int(x), int(y))
            else:
                hough_point = darkest_point
            
            # Method 3: Contour-based detection
            contour_point = darkest_point
            if USE_CONTOUR_DETECTION:
                try:
                    if ADAPTIVE_THRESHOLD:
                        thresh = cv2.adaptiveThreshold(
                            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, 11, 2
                        )
                    else:
                        _, thresh = cv2.threshold(
                            blurred, PUPIL_THRESHOLD_MIN, 255, cv2.THRESH_BINARY_INV
                        )
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Find the most circular contour near the center
                        best_contour = None
                        best_score = 0
                        
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area < MIN_CONTOUR_AREA:
                                continue
                            
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter == 0:
                                continue
                            
                            # Circularity score
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            
                            if circularity < CIRCULARITY_THRESHOLD:
                                continue
                            
                            # Distance from center score
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                center_dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                                center_score = 1 - (center_dist / (np.sqrt(w**2 + h**2) / 2))
                            else:
                                continue
                            
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
                                contour_point = (cx, cy)
                except:
                    pass
            
            # Combine methods with weighted average
            # Weight Hough more if it found something good
            if circles is not None:
                final_x = int(hough_point[0] * PUPIL_WEIGHT_HOUGH + 
                            darkest_point[0] * PUPIL_WEIGHT_DARKEST + 
                            contour_point[0] * PUPIL_WEIGHT_CONTOUR)
                final_y = int(hough_point[1] * PUPIL_WEIGHT_HOUGH + 
                            darkest_point[1] * PUPIL_WEIGHT_DARKEST + 
                            contour_point[1] * PUPIL_WEIGHT_CONTOUR)
            else:
                final_x = int(darkest_point[0] * 0.6 + contour_point[0] * 0.4)
                final_y = int(darkest_point[1] * 0.6 + contour_point[1] * 0.4)
            
            # Ensure pupil is within bounds
            final_x = max(0, min(w - 1, final_x))
            final_y = max(0, min(h - 1, final_y))
            
            return (final_x, final_y)
            
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Pupil detection error: {e}")
            return (w // 2, h // 2)
    
    def extract_eye_region(self, gray, eye_landmarks):
        """Extract eye region with proper bounds"""
        eye_points = np.array(eye_landmarks)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(eye_points)
        
        # Add padding
        padding = EYE_REGION_PADDING
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
        eye_region = gray[y:y+h, x:x+w]
        
        return eye_region, (x, y, w, h)
    
    def process_frame(self, gray, frame):
        """
        FIXED: Process frame with true gaze tracking
        Main changes: Use LOCAL pupil coordinates for normalization
        """
        try:
            # Detect faces
            faces = self.detector(gray)
            if not faces:
                cv2.putText(frame, "No face detected - Position yourself in frame", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame, 0, 0, 0
            
            face = faces[0]
            landmarks = self.predictor(gray, face)
            
            # Extract eye landmarks
            left_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            
            # Calculate EAR for blink detection
            left_ear = self.calculate_ear(left_eye_landmarks)
            right_ear = self.calculate_ear(right_eye_landmarks)
            
            # Extract eye regions
            left_region, left_bbox = self.extract_eye_region(gray, left_eye_landmarks)
            right_region, right_bbox = self.extract_eye_region(gray, right_eye_landmarks)
            
            # Detect pupils in each eye (returns LOCAL coordinates)
            left_pupil_local = self.detect_pupil_in_eye(left_region)
            right_pupil_local = self.detect_pupil_in_eye(right_region)
            
            if left_pupil_local is None or right_pupil_local is None:
                return frame, left_ear, right_ear, 0
            
            # FIXED: Normalize using LOCAL coordinates and eye region shape
            # This is the KEY - we normalize within the eye region, not the frame
            left_norm_x, left_norm_y = self.normalize_pupil_position(
                left_pupil_local, left_region.shape
            )
            right_norm_x, right_norm_y = self.normalize_pupil_position(
                right_pupil_local, right_region.shape
            )
            
            # For visualization, convert to global frame coordinates
            left_pupil_global = (left_bbox[0] + left_pupil_local[0], 
                                left_bbox[1] + left_pupil_local[1])
            right_pupil_global = (right_bbox[0] + right_pupil_local[0],
                                 right_bbox[1] + right_pupil_local[1])
            
            # Average both eyes for more stable tracking
            avg_norm_x = (left_norm_x + right_norm_x) / 2
            avg_norm_y = (left_norm_y + right_norm_y) / 2
            
            # Apply calibration range mapping
            # This maps the natural eye movement range to full screen
            gaze_x = self.map_with_calibration(avg_norm_x, 'x')
            gaze_y = self.map_with_calibration(avg_norm_y, 'y')
            
            # Smooth the gaze position
            self.pupil_history.append((gaze_x, gaze_y))
            if USE_MEDIAN_FILTER and len(self.pupil_history) >= 3:
                # Use median filter for better noise rejection
                x_values = [p[0] for p in self.pupil_history]
                y_values = [p[1] for p in self.pupil_history]
                smoothed_x = np.median(x_values)
                smoothed_y = np.median(y_values)
            else:
                smoothed_x, smoothed_y = gaze_x, gaze_y
            
            # Map to screen coordinates
            cursor_x, cursor_y = self.map_to_screen(smoothed_x, smoothed_y)
            
            # Move cursor
            self.move_cursor_smooth(cursor_x, cursor_y)
            
            # Visualization
            if SHOW_DEBUG:
                self.draw_gaze_info(frame, left_eye_landmarks, right_eye_landmarks,
                                  left_pupil_global, right_pupil_global,
                                  left_norm_x, left_norm_y, right_norm_x, right_norm_y,
                                  cursor_x, cursor_y, left_ear, right_ear)
            
            return frame, left_ear, right_ear, 1.0
            
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Frame processing error: {e}")
            return frame, 0, 0, 0
    
    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        if len(eye_points) != 6:
            return 0.3
        
        points = np.array(eye_points)
        A = np.linalg.norm(points[1] - points[5])
        B = np.linalg.norm(points[2] - points[4])
        C = np.linalg.norm(points[0] - points[3])
        
        if C == 0:
            return 0.3
        
        return (A + B) / (2.0 * C)
    
    def map_with_calibration(self, value, axis):
        """
        FIXED: Map normalized pupil position to screen coordinate with calibration
        Improved range expansion and sensitivity
        """
        if axis == 'x':
            # Invert X axis (looking left moves cursor left)
            value = 1 - value
            # Expand range using config values
            value = (value - GAZE_RANGE_X_MIN) / (GAZE_RANGE_X_MAX - GAZE_RANGE_X_MIN)
        else:
            # Expand range using config values
            value = (value - GAZE_RANGE_Y_MIN) / (GAZE_RANGE_Y_MAX - GAZE_RANGE_Y_MIN)
        
        # Clamp to valid range
        value = max(0.0, min(1.0, value))
        
        # Apply sensitivity curve
        value = np.power(value, GAZE_SENSITIVITY_CURVE)
        
        # Apply cursor sensitivity multiplier
        # Center-biased: easier control near center
        center_offset = value - 0.5
        value = 0.5 + (center_offset * CURSOR_SENSITIVITY)
        
        # Final clamp
        value = max(0.0, min(1.0, value))
        
        return value
    
    def map_to_screen(self, gaze_x, gaze_y):
        """Map gaze coordinates to screen coordinates"""
        # Apply screen mapping with margins
        screen_x = int(gaze_x * (self.screen_width - 2 * SCREEN_BORDER_MARGIN) + SCREEN_BORDER_MARGIN)
        screen_y = int(gaze_y * (self.screen_height - 2 * SCREEN_BORDER_MARGIN) + SCREEN_BORDER_MARGIN)
        
        # Clamp to screen bounds
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        return screen_x, screen_y
    
    def move_cursor_smooth(self, target_x, target_y):
        """
        FIXED: Move cursor with improved smoothing and dead zone
        """
        try:
            # Apply smoothing
            alpha = SMOOTHING_ALPHA
            smooth_x = int(self.prev_cursor_x * (1 - alpha) + target_x * alpha)
            smooth_y = int(self.prev_cursor_y * (1 - alpha) + target_y * alpha)
            
            # Calculate movement distance
            distance = np.sqrt((smooth_x - self.prev_cursor_x)**2 + 
                             (smooth_y - self.prev_cursor_y)**2)
            
            # Only move if distance exceeds minimum threshold
            if distance >= MIN_CURSOR_MOVEMENT:
                # Check if we're in dead zone (center of screen)
                if not self._in_dead_zone(smooth_x, smooth_y):
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                    self.prev_cursor_x, self.prev_cursor_y = smooth_x, smooth_y
                
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Cursor movement error: {e}")
    
    def _in_dead_zone(self, x, y):
        """Check if position is in dead zone to reduce jitter"""
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        
        dead_zone_x = self.screen_width * DEAD_ZONE_SIZE
        dead_zone_y = self.screen_height * DEAD_ZONE_SIZE
        
        # Check if near center AND movement is small
        near_center = (abs(x - center_x) < dead_zone_x and 
                      abs(y - center_y) < dead_zone_y)
        
        small_movement = (abs(x - self.prev_cursor_x) < MIN_CURSOR_MOVEMENT * 2 and
                         abs(y - self.prev_cursor_y) < MIN_CURSOR_MOVEMENT * 2)
        
        return near_center and small_movement
    
    def draw_gaze_info(self, frame, left_eye, right_eye, left_pupil, right_pupil,
                      left_nx, left_ny, right_nx, right_ny, cursor_x, cursor_y, left_ear, right_ear):
        """Draw detailed gaze tracking information"""
        # Draw eye regions
        for point in left_eye:
            cv2.circle(frame, point, 2, DEBUG_COLOR_LEFT_EYE, -1)
        for point in right_eye:
            cv2.circle(frame, point, 2, DEBUG_COLOR_RIGHT_EYE, -1)
        
        # Draw eye bounding boxes
        if SHOW_EYE_REGIONS:
            left_rect = cv2.boundingRect(np.array(left_eye))
            right_rect = cv2.boundingRect(np.array(right_eye))
            cv2.rectangle(frame, left_rect, DEBUG_COLOR_LEFT_EYE, 1)
            cv2.rectangle(frame, right_rect, DEBUG_COLOR_RIGHT_EYE, 1)
        
        # Draw pupils
        if SHOW_PUPIL_DETECTION:
            cv2.circle(frame, left_pupil, 3, DEBUG_COLOR_PUPIL, -1)
            cv2.circle(frame, right_pupil, 3, DEBUG_COLOR_PUPIL, -1)
            
            # Draw crosshair on pupils
            cv2.drawMarker(frame, left_pupil, DEBUG_COLOR_GAZE, cv2.MARKER_CROSS, 10, 2)
            cv2.drawMarker(frame, right_pupil, DEBUG_COLOR_GAZE, cv2.MARKER_CROSS, 10, 2)
        
        # Info text
        y = 30
        cv2.putText(frame, f"Left Pupil: ({left_nx:.2f}, {left_ny:.2f})", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (0, 255, 0), DEBUG_THICKNESS)
        y += 20
        cv2.putText(frame, f"Right Pupil: ({right_nx:.2f}, {right_ny:.2f})", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (0, 255, 0), DEBUG_THICKNESS)
        y += 20
        cv2.putText(frame, f"Cursor: ({cursor_x}, {cursor_y})", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (255, 255, 0), DEBUG_THICKNESS)
        y += 20
        cv2.putText(frame, f"EAR: L={left_ear:.3f} R={right_ear:.3f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (255, 255, 255), DEBUG_THICKNESS)
        y += 25
        
        # Status message
        status_color = (0, 255, 255)
        cv2.putText(frame, "Keep HEAD still - Move EYES only", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, status_color, DEBUG_THICKNESS)
        
        # Show tracking mode
        y = frame.shape[0] - 10
        cv2.putText(frame, "Mode: True Gaze Tracking (Pupil-based)", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (0, 255, 255), DEBUG_THICKNESS)
    
    def toggle_debug(self):
        """Toggle debug mode"""
        global SHOW_DEBUG
        SHOW_DEBUG = not SHOW_DEBUG
        print(f"Debug mode: {'ON' if SHOW_DEBUG else 'OFF'}")
    
    def start_calibration(self):
        """Start calibration"""
        print("Calibration starting...")
        self.is_calibrated = False
        self.calibration_points = {}
    
    def stop_calibration(self):
        """Stop calibration"""
        if len(self.calibration_points) >= 5:
            self.is_calibrated = True
            print("Calibration completed!")
        else:
            print("Insufficient calibration data")
    
    def reset_calibration(self):
        """Reset calibration"""
        self.calibration_points = {}
        self.is_calibrated = False
        self.pupil_history.clear()
        print("Calibration reset!")
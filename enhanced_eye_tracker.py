# enhanced_eye_tracker.py
# Enhanced eye tracker with calibration and improved accuracy

import numpy as np
import cv2
import pyautogui
import time
from collections import deque
from enhanced_utils import *
from enhanced_config import *

class EnhancedEyeTracker:
    def __init__(self, predictor, detector, screen_width, screen_height):
        self.detector = detector
        self.predictor = predictor
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Smoothing and filtering
        self.position_buffer = deque(maxlen=5)
        self.prev_cursor_x = screen_width // 2
        self.prev_cursor_y = screen_height // 2
        
        # Calibration data
        self.calibration_data = []
        self.is_calibrated = False
        self.calibration_matrix = None
        
        # Tracking quality metrics
        self.face_detection_confidence = 0
        self.eye_region_quality = 0
        self.tracking_stability = 0
        
        # Performance optimization
        self.frame_skip_counter = 0
        
        # Debug and calibration modes
        self.debug_mode = SHOW_DEBUG
        self.calibration_active = False
        self.calibration_points = []
        self.calibration_data = []
        
    def process_frame(self, gray, frame):
        """Enhanced frame processing with quality assessment"""
        self.frame_skip_counter += 1
        if self.frame_skip_counter < SKIP_FRAMES:
            return None
        self.frame_skip_counter = 0
        
        # Detect faces with confidence scoring
        faces = self.detector(gray)
        if not faces:
            self._draw_no_face_message(frame)
            return frame, 0, 0, 0
        
        # Use the largest face (closest to camera)
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        self.face_detection_confidence = self._calculate_face_confidence(face, gray.shape)
        
        try:
            landmarks = self.predictor(gray, face)
            
            # Extract eye landmarks with error checking
            left_eye_points = self._get_eye_landmarks(landmarks, range(36, 42))
            right_eye_points = self._get_eye_landmarks(landmarks, range(42, 48))
            
            if not left_eye_points or not right_eye_points:
                return frame, 0, 0, 0
            
            # Calculate EAR with smoothing
            left_ear = get_smoothed_ear(left_eye_points)
            right_ear = get_smoothed_ear(right_eye_points)
            
            # Estimate gaze with quality assessment
            gaze_x, gaze_y, eye_quality = self._estimate_enhanced_gaze(
                gray, left_eye_points, right_eye_points
            )
            
            # Update tracking quality
            tracking_quality = self._calculate_tracking_quality(
                self.face_detection_confidence, eye_quality
            )
            
            # Move cursor only if quality is sufficient
            if tracking_quality > TRACKING_QUALITY_THRESHOLD:
                self._update_cursor_position(gaze_x, gaze_y)
            
            # Visualization
            if self.debug_mode:
                self._draw_debug_info(
                    frame, left_eye_points, right_eye_points, 
                    left_ear, right_ear, tracking_quality, gaze_x, gaze_y
                )
            
            return frame, left_ear, right_ear, tracking_quality
            
        except Exception as e:
            print(f"Error in frame processing: {e}")
            return frame, 0, 0, 0
    
    def _get_eye_landmarks(self, landmarks, point_range):
        """Extract eye landmarks with validation"""
        try:
            points = [(landmarks.part(i).x, landmarks.part(i).y) for i in point_range]
            # Validate points are reasonable
            if all(0 <= p[0] < 2000 and 0 <= p[1] < 2000 for p in points):
                return points
        except:
            pass
        return []
    
    def _estimate_enhanced_gaze(self, gray, left_eye_points, right_eye_points):
        """Enhanced gaze estimation with adaptive thresholding"""
        try:
            # Process left eye
            left_gaze, left_quality = self._process_single_eye(
                gray, left_eye_points, "left"
            )
            
            # Process right eye
            right_gaze, right_quality = self._process_single_eye(
                gray, right_eye_points, "right"
            )
            
            # Average the results with quality weighting
            total_quality = left_quality + right_quality
            if total_quality > 0:
                weight_left = left_quality / total_quality
                weight_right = right_quality / total_quality
                
                avg_x = left_gaze[0] * weight_left + right_gaze[0] * weight_right
                avg_y = left_gaze[1] * weight_left + right_gaze[1] * weight_right
            else:
                avg_x, avg_y = 0.5, 0.5
            
            return avg_x, avg_y, min(left_quality, right_quality)
            
        except Exception as e:
            print(f"Error in gaze estimation: {e}")
            return 0.5, 0.5, 0
    
    def _process_single_eye(self, gray, eye_points, side):
        """Process individual eye with enhanced pupil detection"""
        try:
            # Create eye region
            eye_np = np.array(eye_points)
            x, y, w, h = cv2.boundingRect(eye_np)
            
            # Add padding and ensure bounds
            padding = EYE_REGION_PADDING
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2 * padding)
            h = min(gray.shape[0] - y, h + 2 * padding)
            
            eye_region = gray[y:y+h, x:x+w]
            
            if eye_region.size == 0:
                return (0.5, 0.5), 0
            
            # Enhanced pupil detection
            pupil_pos, quality = get_enhanced_pupil_position(eye_region)
            
            # Normalize coordinates
            gaze_x = pupil_pos[0] / w if w > 0 else 0.5
            gaze_y = pupil_pos[1] / h if h > 0 else 0.5
            
            return (gaze_x, gaze_y), quality
            
        except Exception as e:
            return (0.5, 0.5), 0
    
    def _update_cursor_position(self, gaze_x, gaze_y):
        """Update cursor position with calibration and smoothing"""
        try:
            # Apply calibration if available
            if self.is_calibrated and self.calibration_matrix is not None:
                screen_x, screen_y = self._apply_calibration(gaze_x, gaze_y)
            else:
                # Fallback linear mapping
                screen_x = int((1 - gaze_x) * self.screen_width)
                screen_y = int(gaze_y * self.screen_height)
            
            # Apply screen boundaries
            screen_x = max(SCREEN_BORDER_MARGIN, 
                          min(self.screen_width - SCREEN_BORDER_MARGIN, screen_x))
            screen_y = max(SCREEN_BORDER_MARGIN, 
                          min(self.screen_height - SCREEN_BORDER_MARGIN, screen_y))
            
            # Smooth the movement
            smooth_x = smooth_position(self.prev_cursor_x, screen_x, SMOOTHING_ALPHA)
            smooth_y = smooth_position(self.prev_cursor_y, screen_y, SMOOTHING_ALPHA)
            
            # Apply dead zone
            if self._in_dead_zone(smooth_x, smooth_y):
                return
            
            # Move cursor
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
            self.prev_cursor_x, self.prev_cursor_y = smooth_x, smooth_y
            
        except Exception as e:
            print(f"Error updating cursor: {e}")
    
    def _in_dead_zone(self, x, y):
        """Check if position is in dead zone to reduce jitter"""
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        dead_zone_x = self.screen_width * DEAD_ZONE_SIZE
        dead_zone_y = self.screen_height * DEAD_ZONE_SIZE
        
        return (abs(x - center_x) < dead_zone_x and 
                abs(y - center_y) < dead_zone_y and
                abs(x - self.prev_cursor_x) < 10 and
                abs(y - self.prev_cursor_y) < 10)
    
    def _calculate_face_confidence(self, face, img_shape):
        """Calculate face detection confidence"""
        face_area = face.width() * face.height()
        img_area = img_shape[0] * img_shape[1]
        area_ratio = face_area / img_area
        
        # Ideal face should occupy 5-25% of image
        if 0.05 <= area_ratio <= 0.25:
            return min(1.0, area_ratio / 0.15)
        else:
            return max(0.1, 1.0 - abs(area_ratio - 0.15) / 0.15)
    
    def _calculate_tracking_quality(self, face_confidence, eye_quality):
        """Calculate overall tracking quality"""
        return (face_confidence * 0.3 + eye_quality * 0.7)
    
    def _draw_debug_info(self, frame, left_eye_points, right_eye_points, 
                        left_ear, right_ear, quality, gaze_x, gaze_y):
        """Enhanced debug visualization"""
        # Draw eye landmarks
        for point in left_eye_points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        for point in right_eye_points:
            cv2.circle(frame, point, 2, (0, 0, 255), -1)
        
        # Draw eye regions
        if left_eye_points:
            left_rect = cv2.boundingRect(np.array(left_eye_points))
            cv2.rectangle(frame, left_rect, (0, 255, 0), 1)
        
        if right_eye_points:
            right_rect = cv2.boundingRect(np.array(right_eye_points))
            cv2.rectangle(frame, right_rect, (0, 0, 255), 1)
        
        # Text info
        y_offset = 30
        cv2.putText(frame, f"L-EAR: {left_ear:.3f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (0, 255, 0), DEBUG_THICKNESS)
        y_offset += 25
        
        cv2.putText(frame, f"R-EAR: {right_ear:.3f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (0, 0, 255), DEBUG_THICKNESS)
        y_offset += 25
        
        cv2.putText(frame, f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (255, 255, 0), DEBUG_THICKNESS)
        y_offset += 25
        
        # Draw quality indicator
        color = (0, 255, 0) if quality > 0.7 else (0, 255, 255) if quality > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Quality: {quality:.2f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, color, DEBUG_THICKNESS)
        y_offset += 25
        
        # Draw calibration status
        status = "Calibrated" if self.is_calibrated else "Press 'C' to calibrate"
        if self.calibration_active:
            status = "Calibrating... Look at red dots"
        cv2.putText(frame, status, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (255, 255, 0), DEBUG_THICKNESS)
    
    def _draw_no_face_message(self, frame):
        """Draw message when no face is detected"""
        cv2.putText(frame, "No face detected - Position yourself in frame", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def start_calibration(self):
        """Start calibration process"""
        print("🎯 Starting calibration process...")
        self.calibration_active = True
        self.calibration_data = []
        # In a full implementation, this would show calibration points on screen
        
    def stop_calibration(self):
        """Stop calibration process"""
        self.calibration_active = False
        if len(self.calibration_data) > 5:
            self.is_calibrated = True
            print("✅ Calibration completed!")
        else:
            print("⚠️ Insufficient calibration data")
            
    def toggle_debug(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        print(f"🔍 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
    
    def calibrate(self):
        """Legacy calibration method"""
        self.start_calibration()
        
    def reset_calibration(self):
        """Reset calibration data"""
        self.calibration_data = []
        self.is_calibrated = False
        self.calibration_matrix = None
        self.calibration_active = False
        print("🔄 Calibration reset!")
    
    def _apply_calibration(self, gaze_x, gaze_y):
        """Apply calibration mapping (placeholder for polynomial mapping)"""
        # This would contain the actual calibration transformation
        # For now, return simple linear mapping
        return int((1 - gaze_x) * self.screen_width), int(gaze_y * self.screen_height)
# enhanced_blink_detector.py
# Enhanced blink detection with filtering and debouncing - FIXED VERSION

import time
from collections import deque
import pyautogui
from enhanced_config import *

class EnhancedBlinkDetector:
    def __init__(self):
        # Timing for click detection
        self.last_left_blink_time = 0
        self.last_right_blink_time = 0
        self.last_click_time = 0
        
        # Blink state tracking
        self.left_blink_state = False
        self.right_blink_state = False
        self.left_blink_start_time = 0
        self.right_blink_start_time = 0
        
        # FIXED: Track if eyes were open before blink
        self.left_was_open = False
        self.right_was_open = False
        
        # EAR filtering buffers
        self.left_ear_buffer = deque(maxlen=EAR_BUFFER_SIZE)
        self.right_ear_buffer = deque(maxlen=EAR_BUFFER_SIZE)
        
        # Consecutive frame counters for debouncing
        self.left_closed_frames = 0
        self.right_closed_frames = 0
        self.left_open_frames = 0
        self.right_open_frames = 0
        
        # Adaptive threshold tracking
        self.baseline_ear_left = EAR_THRESHOLD + 0.1
        self.baseline_ear_right = EAR_THRESHOLD + 0.1
        self.ear_samples_count = 0
        
        # FIXED: Track recent EAR history to validate blinks
        self.left_ear_history = deque(maxlen=20)
        self.right_ear_history = deque(maxlen=20)
        
    def detect_blink(self, ear, side="left"):
        """
        Enhanced blink detection with filtering and adaptive thresholding
        
        Args:
            ear: Eye Aspect Ratio value
            side: "left" or "right" eye
            
        Returns:
            bool: True if blink was detected and click was triggered
        """
        if ear <= 0:  # Invalid EAR value
            return False
        
        current_time = time.time()
        
        # Update adaptive baseline
        self._update_baseline_ear(ear, side)
        
        # Store EAR in history
        if side == "left":
            self.left_ear_history.append(ear)
        else:
            self.right_ear_history.append(ear)
        
        # Get adaptive threshold
        adaptive_threshold = self._get_adaptive_threshold(side)
        
        # Filter EAR value
        filtered_ear = self._filter_ear(ear, side)
        
        # Detect blink based on filtered EAR
        blink_detected = self._process_blink_detection(
            filtered_ear, side, adaptive_threshold, current_time
        )
        
        if blink_detected:
            # FIXED: Validate it's a real blink by checking EAR variation
            if self._validate_blink(side):
                self._trigger_click(side, current_time)
                return True
            
        return False
    
    def _validate_blink(self, side):
        """
        FIXED: Validate that the blink shows proper EAR variation
        A real blink has a clear drop and rise in EAR
        """
        history = list(self.left_ear_history) if side == "left" else list(self.right_ear_history)
        
        if len(history) < 10:
            return True  # Not enough data yet, allow it
        
        # Check if we have good variation (not stuck at one value)
        recent_values = history[-10:]
        max_ear = max(recent_values)
        min_ear = min(recent_values)
        variation = max_ear - min_ear
        
        # Real blinks show at least 0.10 variation in EAR
        return variation >= 0.10
    
    def _update_baseline_ear(self, ear, side):
        """Update baseline EAR for adaptive thresholding"""
        self.ear_samples_count += 1
        
        # Only update baseline when eyes are likely open
        if ear > EAR_THRESHOLD + 0.1:
            alpha = 0.05  # Learning rate
            if side == "left":
                self.baseline_ear_left = (1 - alpha) * self.baseline_ear_left + alpha * ear
            else:
                self.baseline_ear_right = (1 - alpha) * self.baseline_ear_right + alpha * ear
    
    def _get_adaptive_threshold(self, side):
        """Get adaptive threshold based on baseline EAR"""
        baseline = self.baseline_ear_left if side == "left" else self.baseline_ear_right
        # Threshold is typically 75% of baseline EAR (CHANGED from 80% for less sensitivity)
        return max(EAR_THRESHOLD, baseline * 0.75)
    
    def _filter_ear(self, ear, side):
        """Apply filtering to EAR value"""
        if side == "left":
            self.left_ear_buffer.append(ear)
            buffer = list(self.left_ear_buffer)
        else:
            self.right_ear_buffer.append(ear)
            buffer = list(self.right_ear_buffer)
        
        if len(buffer) < 3:
            return ear
        
        # Use median filter to reduce noise
        buffer_sorted = sorted(buffer)
        return buffer_sorted[len(buffer) // 2]
    
    def _process_blink_detection(self, ear, side, threshold, current_time):
        """Process blink detection with state machine"""
        is_closed = ear < threshold
        
        if side == "left":
            return self._update_blink_state(
                is_closed, current_time, "left"
            )
        else:
            return self._update_blink_state(
                is_closed, current_time, "right"
            )
    
    def _update_blink_state(self, is_closed, current_time, side):
        """
        FIXED: Update blink state machine with proper open-eye validation
        """
        if side == "left":
            if is_closed:
                self.left_closed_frames += 1
                self.left_open_frames = 0
                
                # Only start blink if we've confirmed open eyes first
                if (not self.left_blink_state and 
                    self.left_closed_frames >= EAR_CONSECUTIVE_FRAMES and
                    self.left_was_open):  # FIXED: Check eyes were open before
                    # Start of blink
                    self.left_blink_state = True
                    self.left_blink_start_time = current_time
                    self.left_was_open = False  # Reset flag
                    
            else:  # Eyes are open
                self.left_open_frames += 1
                self.left_closed_frames = 0
                
                # Mark that eyes are open (need sustained open frames)
                if self.left_open_frames >= EAR_CONSECUTIVE_FRAMES:
                    self.left_was_open = True
                
                if self.left_blink_state and self.left_open_frames >= EAR_CONSECUTIVE_FRAMES:
                    # End of blink
                    blink_duration = current_time - self.left_blink_start_time
                    self.left_blink_state = False
                    
                    # Validate blink duration
                    if MIN_BLINK_DURATION <= blink_duration <= MAX_BLINK_DURATION:
                        return True
        else:  # Right eye
            if is_closed:
                self.right_closed_frames += 1
                self.right_open_frames = 0
                
                # Only start blink if we've confirmed open eyes first
                if (not self.right_blink_state and 
                    self.right_closed_frames >= EAR_CONSECUTIVE_FRAMES and
                    self.right_was_open):  # FIXED: Check eyes were open before
                    self.right_blink_state = True
                    self.right_blink_start_time = current_time
                    self.right_was_open = False  # Reset flag
                    
            else:  # Eyes are open
                self.right_open_frames += 1
                self.right_closed_frames = 0
                
                # Mark that eyes are open (need sustained open frames)
                if self.right_open_frames >= EAR_CONSECUTIVE_FRAMES:
                    self.right_was_open = True
                
                if self.right_blink_state and self.right_open_frames >= EAR_CONSECUTIVE_FRAMES:
                    blink_duration = current_time - self.right_blink_start_time
                    self.right_blink_state = False
                    
                    if MIN_BLINK_DURATION <= blink_duration <= MAX_BLINK_DURATION:
                        return True
        
        return False
    
    def _trigger_click(self, side, current_time):
        """Trigger mouse click with cooldown and double-click detection"""
        # Check cooldown
        if current_time - self.last_click_time < CLICK_COOLDOWN:
            return
        
        try:
            if side == "left":
                time_since_last = current_time - self.last_left_blink_time
                self.last_left_blink_time = current_time
                
                if time_since_last < BLINK_TIME_THRESHOLD:
                    # Double-click
                    pyautogui.doubleClick(button='left')
                    print("Left double-click detected")
                else:
                    # Single click
                    pyautogui.click(button='left')
                    print("Left click detected")
                    
            else:
                time_since_last = current_time - self.last_right_blink_time
                self.last_right_blink_time = current_time
                
                if time_since_last < BLINK_TIME_THRESHOLD:
                    # Right double-click
                    pyautogui.doubleClick(button='right')
                    print("Right double-click detected")
                else:
                    # Right single click
                    pyautogui.click(button='right')
                    print("Right click detected")
            
            self.last_click_time = current_time
            
        except Exception as e:
            print(f"Error triggering click: {e}")
    
    def get_blink_statistics(self):
        """Return blink detection statistics"""
        return {
            'baseline_ear_left': self.baseline_ear_left,
            'baseline_ear_right': self.baseline_ear_right,
            'left_blink_active': self.left_blink_state,
            'right_blink_active': self.right_blink_state,
            'samples_processed': self.ear_samples_count,
            'left_ready': self.left_was_open,
            'right_ready': self.right_was_open
        }
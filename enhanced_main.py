import cv2
import dlib
import pyautogui
import numpy as np
import os
import sys
import time
from enhanced_eye_tracker import EnhancedEyeTracker
from true_gaze_tracker import TrueGazeTracker  # Import true gaze tracker
from enhanced_blink_detector import EnhancedBlinkDetector
from enhanced_config import *# enhanced_main.py
# Complete enhanced main script with cursor movement AND blink detection

import cv2
import dlib
import pyautogui
import numpy as np
import os
import sys
import time
from enhanced_eye_tracker import EnhancedEyeTracker
from true_gaze_tracker import TrueGazeTracker  # Import the true gaze tracker
from enhanced_blink_detector import EnhancedBlinkDetector
from enhanced_config import *

# Configuration: Choose which tracker to use
USE_TRUE_GAZE_TRACKING = True  # Set to True for pupil-based, False for eye position-based

class EnhancedEyeCursorController:
    def __init__(self):
        self.cap = None
        self.detector = None
        self.predictor = None
        self.eye_tracker = None
        self.blink_detector = None
        self.screen_width, self.screen_height = pyautogui.size()
        self.calibration_mode = False
        self.performance_stats = {
            'frames_processed': 0,
            'cursor_moves': 0,
            'blinks_detected': 0,
            'start_time': time.time()
        }
        
    def initialize_pyautogui(self):
        """Initialize PyAutoGUI with optimal settings"""
        print("🔧 Configuring PyAutoGUI...")
        
        # Configure PyAutoGUI for smooth operation
        pyautogui.FAILSAFE = ENABLE_FAILSAFE
        pyautogui.PAUSE = 0.001  # Minimal pause for maximum responsiveness
        
        # Test PyAutoGUI functionality
        try:
            current_pos = pyautogui.position()
            test_x, test_y = current_pos[0] + 1, current_pos[1] + 1
            pyautogui.moveTo(test_x, test_y, duration=0)
            time.sleep(0.01)
            pyautogui.moveTo(current_pos[0], current_pos[1], duration=0)
            print("✅ PyAutoGUI working correctly")
            return True
        except Exception as e:
            print(f"❌ PyAutoGUI error: {e}")
            print("💡 Try running as administrator or check permissions")
            return False
        
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        print("📹 Initializing camera...")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # Try other camera indices
            for i in range(1, 4):
                print(f"Trying camera index {i}...")
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    break
            else:
                print("❌ No camera found!")
                return False
        
        # Set optimal camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        # Test frame capture
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            print("❌ Cannot capture frames from camera")
            return False
            
        print(f"✅ Camera initialized: {test_frame.shape}")
        return True
    
    def load_models(self):
        """Load dlib models with comprehensive error handling"""
        print("🤖 Loading AI models...")
        
        try:
            # Initialize face detector
            self.detector = dlib.get_frontal_face_detector()
            print("✅ Face detector loaded")
            
            # Check for landmark predictor file
            if not os.path.exists(LANDMARK_MODEL_PATH):
                print(f"❌ Model file not found: {LANDMARK_MODEL_PATH}")
                print("📥 Please download from:")
                print("   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                print("   Extract and place in project folder")
                return False
                
            # Load landmark predictor
            self.predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)
            print("✅ Facial landmark predictor loaded")
            
            # Test the models with a dummy detection
            test_img = np.zeros((100, 100), dtype=np.uint8)
            faces = self.detector(test_img)
            print(f"✅ Models tested successfully ({len(faces)} faces in test)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("💡 Make sure dlib is properly installed: pip install dlib")
            return False
    
    def initialize_trackers(self):
        """Initialize eye tracker and blink detector with enhanced features"""
        print("👁️ Initializing tracking systems...")
        
        try:
            # Choose tracker based on configuration
            if USE_TRUE_GAZE_TRACKING:
                print("📍 Using TRUE GAZE tracking (pupil-based)")
                print("   Keep your head relatively still")
                print("   Move your EYES to control cursor")
                self.eye_tracker = TrueGazeTracker(
                    self.predictor, self.detector,
                    self.screen_width, self.screen_height
                )
            else:
                print("📍 Using enhanced eye position tracking")
                self.eye_tracker = EnhancedEyeTracker(
                    self.predictor, self.detector, 
                    self.screen_width, self.screen_height
                )
            
            # Initialize enhanced blink detector
            self.blink_detector = EnhancedBlinkDetector()
            
            print("✅ Tracking systems initialized")
            print(f"   - Screen resolution: {self.screen_width}x{self.screen_height}")
            print(f"   - EAR threshold: {EAR_THRESHOLD}")
            print(f"   - Smoothing: {SMOOTHING_ALPHA}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing trackers: {e}")
            return False
    
    def print_instructions(self):
        """Print detailed usage instructions"""
        print("\n" + "="*60)
        print("🎯 ENHANCED EYE CURSOR CONTROL")
        print("="*60)
        
        # Show tracking mode
        mode_name = get_tracking_mode_name()
        print(f"📍 Mode: {mode_name}")
        
        if USE_TRUE_GAZE_TRACKING:
            print("\n⚠️  TRUE GAZE MODE INSTRUCTIONS:")
            print("   1. Keep your HEAD STILL (very important!)")
            print("   2. Move your EYES to control cursor")
            print("   3. Look left/right/up/down, not move your head")
            print("   4. Sit in a comfortable, stable position")
        else:
            print("\n📍 EYE POSITION MODE:")
            print("   - Cursor follows eye position in frame")
            print("   - Head movement will move cursor")
        
        print("\n📋 CONTROLS:")
        print("   👀 Move eyes → Control cursor")
        print("   👁️  Left eye blink → Left-click")
        print("   👁️  Right eye blink → Right-click")
        print("   👁️👁️ Double-blink → Double-click")
        print("\n🎮 KEYBOARD:")
        print("   ESC     - Exit program")
        print("   C       - Start calibration")
        print("   R       - Reset calibration")
        print("   D       - Toggle debug info")
        print("   SPACE   - Test cursor movement")
        print("   S       - Show statistics")
        print("\n💡 TIPS:")
        print("   - Ensure GOOD LIGHTING on your face")
        print("   - Keep face CENTERED in camera view")
        print("   - Camera should be at EYE LEVEL")
        print("   - Blink DELIBERATELY for clicks")
        print("   - Use calibration (C) for better accuracy")
        print("="*60)
    
    def display_stats(self):
        """Display performance statistics"""
        elapsed = time.time() - self.performance_stats['start_time']
        fps = self.performance_stats['frames_processed'] / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 PERFORMANCE STATS:")
        print(f"   Runtime: {elapsed:.1f}s")
        print(f"   FPS: {fps:.1f}")
        print(f"   Frames: {self.performance_stats['frames_processed']}")
        print(f"   Cursor moves: {self.performance_stats['cursor_moves']}")
        print(f"   Blinks detected: {self.performance_stats['blinks_detected']}")
    
    def handle_calibration(self):
        """Handle calibration process"""
        if not self.calibration_mode:
            print("🎯 Starting calibration...")
            print("Look at the corners of your screen when prompted")
            self.calibration_mode = True
            if self.eye_tracker:
                self.eye_tracker.start_calibration()
        else:
            print("⏹️ Stopping calibration")
            self.calibration_mode = False
            if self.eye_tracker:
                self.eye_tracker.stop_calibration()
    
    def test_cursor_movement(self):
        """Test manual cursor movement"""
        print("🧪 Testing cursor movement...")
        try:
            original_pos = pyautogui.position()
            
            # Move in a small square pattern
            moves = [(50, 0), (0, 50), (-50, 0), (0, -50)]
            for dx, dy in moves:
                new_x = original_pos[0] + dx
                new_y = original_pos[1] + dy
                pyautogui.moveTo(new_x, new_y, duration=0.2)
                time.sleep(0.1)
            
            # Return to original position
            pyautogui.moveTo(original_pos[0], original_pos[1], duration=0.2)
            print("✅ Cursor movement test completed")
            
        except Exception as e:
            print(f"❌ Cursor movement test failed: {e}")
    
    def process_keyboard_input(self, key):
        """Process keyboard input with all enhanced controls"""
        if key == 27:  # ESC
            return False
        elif key == ord('c') or key == ord('C'):
            self.handle_calibration()
        elif key == ord('r') or key == ord('R'):
            if self.eye_tracker:
                self.eye_tracker.reset_calibration()
                print("🔄 Calibration reset")
        elif key == ord('d') or key == ord('D'):
            if self.eye_tracker:
                self.eye_tracker.toggle_debug()
                print("🔍 Debug mode toggled")
        elif key == ord(' '):  # Space
            self.test_cursor_movement()
        elif key == ord('s') or key == ord('S'):
            self.display_stats()
        
        return True
    
    def run(self):
        """Enhanced main execution loop"""
        print("🚀 Starting Enhanced Eye Cursor Control...")
        
        # Initialize all systems
        if not self.initialize_pyautogui():
            return False
            
        if not self.initialize_camera():
            return False
            
        if not self.load_models():
            self.cleanup()
            return False
            
        if not self.initialize_trackers():
            self.cleanup()
            return False
        
        self.print_instructions()
        
        # Main processing loop
        try:
            frame_count = 0
            last_fps_time = time.time()
            fps_counter = 0
            
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Failed to capture frame")
                    continue
                
                frame_count += 1
                self.performance_stats['frames_processed'] = frame_count
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Process frame with eye tracker
                try:
                    result = self.eye_tracker.process_frame(gray, frame)
                    
                    if result and len(result) >= 4:
                        processed_frame, left_ear, right_ear, tracking_quality = result
                        
                        # Update cursor position (handled internally by eye_tracker)
                        if tracking_quality > TRACKING_QUALITY_THRESHOLD:
                            self.performance_stats['cursor_moves'] += 1
                        
                        # Process blinks for clicking
                        if tracking_quality > TRACKING_QUALITY_THRESHOLD * 0.8:
                            # Detect blinks and handle clicking
                            left_blink = self.blink_detector.detect_blink(left_ear, "left")
                            right_blink = self.blink_detector.detect_blink(right_ear, "right")
                            
                            if left_blink or right_blink:
                                self.performance_stats['blinks_detected'] += 1
                        
                        # Use processed frame
                        frame = processed_frame
                        
                    else:
                        # No face detected or processing failed
                        cv2.putText(frame, "No face detected - Position yourself clearly", 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"⚠️ Frame processing error: {e}")
                    cv2.putText(frame, f"Processing Error: {str(e)[:50]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Calculate and display FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = fps_counter / (current_time - last_fps_time)
                    fps_counter = 0
                    last_fps_time = current_time
                    
                    # Display FPS on frame
                    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Enhanced Eye Cursor Control", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.process_keyboard_input(key):
                        break
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Unexpected error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Enhanced cleanup with statistics"""
        print("\n🧹 Cleaning up...")
        
        # Display final stats
        self.display_stats()
        
        # Release resources
        if self.cap:
            self.cap.release()
            print("✅ Camera released")
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Ensure windows are closed
        
        print("✅ Cleanup completed")
        print("👋 Thank you for using Enhanced Eye Cursor Control!")

def main():
    """Main entry point with system checks"""
    print("🔍 System Check...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher required")
        return
    
    # Check imports
    try:
        import cv2
        import dlib
        import pyautogui
        import numpy
        print("✅ All required packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("💡 Install with: pip install opencv-python dlib numpy pyautogui")
        return
    
    # Create and run controller
    controller = EnhancedEyeCursorController()
    success = controller.run()
    
    return success

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
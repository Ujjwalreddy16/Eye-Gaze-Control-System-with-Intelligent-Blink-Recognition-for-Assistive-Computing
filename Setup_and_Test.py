# setup_and_test.py
# Comprehensive setup and testing script for the enhanced eye cursor control

import sys
import os
import subprocess
import urllib.request
import bz2
import time

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 7):
        print(f"❌ Python 3.7+ required, found {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "opencv-python",
        "dlib", 
        "numpy",
        "pyautogui"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    packages = {
        'cv2': 'OpenCV',
        'dlib': 'dlib',
        'numpy': 'NumPy', 
        'pyautogui': 'PyAutoGUI'
    }
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name} import successful")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            return False
    
    return True

def download_shape_predictor():
    """Download the required shape predictor model"""
    print("📥 Checking for shape predictor model...")
    
    model_file = "shape_predictor_68_face_landmarks.dat"
    
    if os.path.exists(model_file):
        print(f"✅ {model_file} already exists")
        return True
    
    print("Downloading shape predictor model (99.7 MB)...")
    print("This may take a few minutes...")
    
    try:
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
        
        # Download
        urllib.request.urlretrieve(url, compressed_file)
        print("✅ Download completed")
        
        # Extract
        print("📂 Extracting...")
        with bz2.BZ2File(compressed_file, 'rb') as f_in:
            with open(model_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Cleanup
        os.remove(compressed_file)
        print(f"✅ {model_file} ready")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("💡 Please download manually from:")
        print("   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return False

def test_camera():
    """Test camera functionality"""
    print("📹 Testing camera...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot capture frames")
            cap.release()
            return False
        
        print(f"✅ Camera working: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_pyautogui():
    """Test PyAutoGUI functionality"""
    print("🖱️ Testing PyAutoGUI...")
    
    try:
        import pyautogui
        
        # Get screen size
        screen_size = pyautogui.size()
        print(f"Screen size: {screen_size}")
        
        # Get current position
        pos = pyautogui.position()
        print(f"Current cursor: {pos}")
        
        # Test small movement
        original_pos = pyautogui.position()
        test_pos = (original_pos[0] + 5, original_pos[1] + 5)
        
        pyautogui.moveTo(test_pos[0], test_pos[1], duration=0.1)
        time.sleep(0.1)
        
        new_pos = pyautogui.position()
        pyautogui.moveTo(original_pos[0], original_pos[1], duration=0.1)
        
        if abs(new_pos[0] - test_pos[0]) < 3 and abs(new_pos[1] - test_pos[1]) < 3:
            print("✅ PyAutoGUI cursor control working")
            return True
        else:
            print("❌ PyAutoGUI cursor movement failed")
            print("💡 Check permissions (run as admin on Windows, accessibility on macOS)")
            return False
            
    except Exception as e:
        print(f"❌ PyAutoGUI test failed: {e}")
        return False

def test_face_detection():
    """Test face detection with the model"""
    print("👤 Testing face detection...")
    
    try:
        import cv2
        import dlib
        import numpy as np
        
        # Load models
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Test with camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot test face detection - no camera")
            return False
        
        print("📸 Testing face detection for 5 seconds...")
        print("   Position your face in front of the camera")
        
        start_time = time.time()
        faces_detected = 0
        frames_processed = 0
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frames_processed += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            if faces:
                faces_detected += 1
                face = faces[0]
                landmarks = predictor(gray, face)
                
                # Test eye landmark extraction
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
                
                if len(left_eye) == 6 and len(right_eye) == 6:
                    print("✅ Eye landmarks detected successfully")
        
        cap.release()
        
        detection_rate = faces_detected / frames_processed if frames_processed > 0 else 0
        print(f"Face detection rate: {detection_rate:.1%}")
        
        if detection_rate > 0.5:
            print("✅ Face detection working well")
            return True
        else:
            print("⚠️ Face detection rate low - check lighting and positioning")
            return detection_rate > 0.1
            
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def run_comprehensive_test():
    """Run a comprehensive system test"""
    print("🧪 Running comprehensive system test...")
    
    try:
        from true_gaze_tracker import TrueGazeTracker
        from enhanced_blink_detector import EnhancedBlinkDetector
        from enhanced_config import USE_TRUE_GAZE_TRACKING, get_tracking_mode_name
        import cv2
        import dlib
        import pyautogui
        
        # Initialize components
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        screen_w, screen_h = pyautogui.size()
        
        # Choose tracker based on config
        if USE_TRUE_GAZE_TRACKING:
            print(f"Testing: {get_tracking_mode_name()}")
            eye_tracker = TrueGazeTracker(predictor, detector, screen_w, screen_h)
        else:
            from enhanced_eye_tracker import EnhancedEyeTracker
            eye_tracker = EnhancedEyeTracker(predictor, detector, screen_w, screen_h)
            
        blink_detector = EnhancedBlinkDetector()
        
        # Test with camera
        cap = cv2.VideoCapture(0)
        
        print("📸 Testing full system for 10 seconds...")
        if USE_TRUE_GAZE_TRACKING:
            print("   Keep your HEAD STILL and move your EYES")
        else:
            print("   Move your eyes and blink to test functionality")
        
        start_time = time.time()
        test_duration = 10
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process frame
            result = eye_tracker.process_frame(gray, frame)
            
            if result and len(result) >= 4:
                processed_frame, left_ear, right_ear, quality = result
                
                # Test blink detection
                blink_detector.detect_blink(left_ear, "left")
                blink_detector.detect_blink(right_ear, "right")
                
                # Show progress
                remaining = test_duration - (time.time() - start_time)
                cv2.putText(processed_frame, f"Test: {remaining:.1f}s remaining", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("System Test", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit early
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("✅ Comprehensive test completed")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup and test routine"""
    print("🚀 Enhanced Eye Cursor Control - Setup & Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("Package Imports", test_imports),
        ("Shape Predictor Model", download_shape_predictor),
        ("Camera", test_camera),
        ("PyAutoGUI", test_pyautogui),
        ("Face Detection", test_face_detection),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if not test_func():
                failed_tests.append(test_name)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed_tests.append(test_name)
    
    print(f"\n{'='*50}")
    print("📋 SETUP SUMMARY")
    print("="*50)
    
    if not failed_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ System is ready for enhanced eye cursor control")
        
        # Ask if user wants to run comprehensive test
        try:
            response = input("\nRun comprehensive system test? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                run_comprehensive_test()
        except KeyboardInterrupt:
            print("\n⏹️ Test cancelled by user")
        
        print("\n🚀 To start the eye cursor control:")
        print("   python enhanced_main.py")
        
    else:
        print(f"❌ {len(failed_tests)} test(s) failed:")
        for test in failed_tests:
            print(f"   - {test}")
        print("\n💡 Please fix the issues above before proceeding")
    
    print("\n📚 Need help? Check the troubleshooting guide or run:")
    print("   python debug_main.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
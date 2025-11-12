# enhanced_config.py
# Enhanced configuration with TRUE GAZE TRACKING support - FIXED VERSION

import os

# ============================================================================
# TRACKING MODE SELECTION
# ============================================================================
USE_TRUE_GAZE_TRACKING = True  # True = pupil-based, False = eye position-based

# ============================================================================
# CAMERA SETTINGS
# ============================================================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ============================================================================
# MODEL PATHS
# ============================================================================
LANDMARK_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

# ============================================================================
# EYE ASPECT RATIO (EAR) SETTINGS - For Blink Detection - FIXED
# ============================================================================
EAR_THRESHOLD = 0.21  # CHANGED: Lowered from 0.25 - less sensitive to false positives
EAR_CONSECUTIVE_FRAMES = 3  # CHANGED: Increased from 2 - need more confirmation frames
EAR_BUFFER_SIZE = 7  # CHANGED: Increased from 5 - better smoothing to reduce noise

# ============================================================================
# BLINK TIMING SETTINGS - FIXED
# ============================================================================
BLINK_TIME_THRESHOLD = 0.4  # Time window for double-click (seconds)
MIN_BLINK_DURATION = 0.08   # CHANGED: Increased from 0.05 - filters out noise/artifacts
MAX_BLINK_DURATION = 0.4    # CHANGED: Decreased from 0.5 - real blinks are quick
CLICK_COOLDOWN = 0.15       # CHANGED: Increased from 0.1 - prevent accidental rapid clicks

# ============================================================================
# PUPIL DETECTION SETTINGS - For True Gaze Tracking
# ============================================================================
PUPIL_THRESHOLD_MIN = 15
PUPIL_THRESHOLD_MAX = 35
ADAPTIVE_THRESHOLD = True  # Use adaptive thresholding
USE_HOUGH_CIRCLES = True   # Use Hough circle detection for pupil
USE_CONTOUR_DETECTION = True  # Use contour-based pupil detection

# Pupil detection weights (when combining methods)
PUPIL_WEIGHT_HOUGH = 0.5
PUPIL_WEIGHT_DARKEST = 0.3
PUPIL_WEIGHT_CONTOUR = 0.2

# ============================================================================
# GAZE TRACKING SETTINGS - FIXED
# ============================================================================
# Smoothing (lower = smoother but slower response)
SMOOTHING_ALPHA = 0.3  # CHANGED: Increased from 0.25 for slightly more responsiveness

# Cursor sensitivity multiplier
CURSOR_SENSITIVITY = 1.2  # CHANGED: Increased from 1.0 for better range

# Dead zone to reduce jitter (percentage of screen)
DEAD_ZONE_SIZE = 0.03  # CHANGED: Reduced from 0.05 for better responsiveness

# Minimum cursor movement threshold (pixels)
MIN_CURSOR_MOVEMENT = 2  # CHANGED: Reduced from 3 for smoother tracking

# Gaze range calibration (normalized 0-1 range) - FIXED
# These define the "usable" range of pupil movement
GAZE_RANGE_X_MIN = 0.15  # CHANGED: Expanded from 0.2
GAZE_RANGE_X_MAX = 0.85  # CHANGED: Expanded from 0.8
GAZE_RANGE_Y_MIN = 0.20  # CHANGED: Expanded from 0.3
GAZE_RANGE_Y_MAX = 0.80  # CHANGED: Expanded from 0.7

# Pupil position smoothing
PUPIL_HISTORY_SIZE = 5  # Number of frames to smooth
USE_MEDIAN_FILTER = True  # Use median filter for pupil position

# ============================================================================
# CALIBRATION SETTINGS
# ============================================================================
CALIBRATION_POINTS = 9  # 3x3 grid for calibration
CALIBRATION_DURATION = 2.0  # Seconds to hold gaze at each point
AUTO_CALIBRATION = False  # Enable automatic calibration improvement
ENABLE_DRIFT_CORRECTION = True  # Correct for gradual calibration drift
LEARNING_RATE = 0.01  # Learning rate for adaptive calibration

# ============================================================================
# SCREEN MAPPING SETTINGS
# ============================================================================
SCREEN_BORDER_MARGIN = 50  # Pixels from screen edge
USE_POLYNOMIAL_MAPPING = False  # Use polynomial mapping (advanced)
GAZE_SENSITIVITY_CURVE = 0.9  # Exponential curve (1.0 = linear, <1.0 = easier center control)

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
SKIP_FRAMES = 0  # Process every nth frame (0 = process all)
TRACKING_QUALITY_THRESHOLD = 0.5  # Minimum quality for cursor movement (0-1)

# Eye region padding
EYE_REGION_PADDING = 5  # Pixels to pad around detected eye region

# ============================================================================
# DEBUG AND VISUALIZATION
# ============================================================================
SHOW_DEBUG = True
SHOW_EYE_REGIONS = True
SHOW_PUPIL_DETECTION = True
SHOW_GAZE_VECTOR = True  # Show gaze direction
SHOW_CALIBRATION_POINTS = False

DEBUG_FONT_SCALE = 0.5
DEBUG_THICKNESS = 1
DEBUG_COLOR_LEFT_EYE = (0, 255, 0)  # Green
DEBUG_COLOR_RIGHT_EYE = (0, 255, 0)  # Green
DEBUG_COLOR_PUPIL = (0, 0, 255)  # Red
DEBUG_COLOR_GAZE = (255, 0, 255)  # Magenta

# ============================================================================
# FACE DETECTION SETTINGS
# ============================================================================
FACE_DETECTION_SCALE_FACTOR = 1.1
MIN_FACE_SIZE = (100, 100)
MAX_FACES = 1  # Only track one face

# Head movement detection
HEAD_MOVEMENT_THRESHOLD = 10  # Pixels - movement above this is head movement
COMPENSATE_HEAD_MOVEMENT = True  # Try to compensate for head movement

# ============================================================================
# SAFETY SETTINGS
# ============================================================================
ENABLE_FAILSAFE = True  # PyAutoGUI failsafe (move to corner to stop)
MAX_CURSOR_SPEED = 2000  # Maximum pixels per second

# ============================================================================
# ADVANCED SETTINGS - True Gaze Tracking
# ============================================================================
# Pupil detection algorithm parameters
GAUSSIAN_BLUR_SIZE = 7  # Must be odd number
MIN_PUPIL_RADIUS_PERCENT = 0.1  # Minimum pupil size (% of eye region)
MAX_PUPIL_RADIUS_PERCENT = 0.4  # Maximum pupil size (% of eye region)

# Contour detection
MIN_CONTOUR_AREA = 10  # Minimum area for pupil contour
CIRCULARITY_THRESHOLD = 0.6  # Minimum circularity score

# Hough circles parameters
HOUGH_DP = 1
HOUGH_MIN_DIST = 20
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30

# Baseline tracking (for adaptive thresholds)
BASELINE_SAMPLE_SIZE = 100  # Number of samples for baseline
BASELINE_UPDATE_RATE = 0.05  # How fast baseline adapts

# ============================================================================
# EXPERIMENTAL FEATURES
# ============================================================================
USE_HEAD_POSE_ESTIMATION = False  # Experimental: compensate using head pose
USE_DEEP_LEARNING_GAZE = False  # Experimental: use DL-based gaze estimation
ENABLE_SACCADE_DETECTION = False  # Experimental: detect rapid eye movements

# ============================================================================
# SYSTEM MESSAGES
# ============================================================================
SHOW_STARTUP_INFO = True
SHOW_PERFORMANCE_STATS = True
VERBOSE_LOGGING = False  # Detailed console output

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_tracking_mode_name():
    """Return human-readable tracking mode name"""
    return "True Gaze Tracking (Pupil-based)" if USE_TRUE_GAZE_TRACKING else "Eye Position Tracking"

def print_config_summary():
    """Print configuration summary"""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Tracking Mode: {get_tracking_mode_name()}")
    print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS")
    print(f"EAR Threshold: {EAR_THRESHOLD}")
    print(f"Smoothing: {SMOOTHING_ALPHA}")
    print(f"Sensitivity: {CURSOR_SENSITIVITY}")
    print(f"Debug Mode: {'ON' if SHOW_DEBUG else 'OFF'}")
    print(f"Pupil Detection: ", end="")
    methods = []
    if USE_HOUGH_CIRCLES:
        methods.append("Hough")
    if USE_CONTOUR_DETECTION:
        methods.append("Contour")
    methods.append("Darkest")
    print(", ".join(methods))
    print("="*60 + "\n")
# config.py

import os

# PATHS 
KNOWN_FACES_DIR = "faces"
UNKNOWN_SAVE_DIR = "logs/unknown_faces"
os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)

#  VIDEO PROCESSING 
PROCESS_EVERY_N_FRAMES = 6   # Process every 2nd frame (better accuracy, not too slow)
FRAME_HISTORY = 5            # Keep 5-frame history for smoothing
DECISION_THRESHOLD = 0.7     # Require 60% majority for stable decision
DISAPPEAR_FRAMES = 20        # How long before a face is considered gone

#  FACE RECOGNITION 
TOLERANCE = 0.40             # Stricter tolerance (was 0.45, improves accuracy)

#  ARDUINO 
ARDUINO_PORT = "COM10"
ARDUINO_BAUD = 9600

# ALERT 
ALERT_BEEP_FREQ = 1200       # Slightly higher pitch
ALERT_BEEP_DUR = 200         # Shorter beep, less annoying




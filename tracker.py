# tracker.py

from collections import deque
import threading

# You can import constants from config.py
from config import FRAME_HISTORY, DECISION_THRESHOLD, DISAPPEAR_FRAMES

tracks = {}
next_track_id = 0
tracks_lock = threading.Lock()

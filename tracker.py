from collections import deque
import threading
from config import FRAME_HISTORY, DECISION_THRESHOLD, DISAPPEAR_FRAMES

tracks = {}
next_track_id = 0
tracks_lock = threading.Lock()

def cleanup_tracks(frame_idx):
    """
    Remove or reset stale tracks based on disappearance or final decision.
    Call this from main loop after processing each frame.
    """
    global tracks
    with tracks_lock:
        for tid in list(tracks.keys()):
            t = tracks[tid]

            # If track has a final decision → reset history
            if t.get("final"):
                t["history"].clear()

            # If last update too old → drop track
            if frame_idx - t.get("last_seen", frame_idx) > DISAPPEAR_FRAMES:
                del tracks[tid]

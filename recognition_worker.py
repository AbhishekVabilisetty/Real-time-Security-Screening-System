# recognition_worker.py

import time
import numpy as np
import cv2
import face_recognition
import threading
from collections import deque
from tracker import tracks, next_track_id, tracks_lock
from config import FRAME_HISTORY, DECISION_THRESHOLD, DISAPPEAR_FRAMES, TOLERANCE
from face_utils import iou

# Shared input dict to be updated from main thread
recognition_input = {
    'small_frame': None,
    'frame_idx': 0,
    'should_run': True
}

recognition_lock = threading.Lock()

def recognition_worker(known_encodings, known_names):
    global next_track_id, tracks
    while True:
        with recognition_lock:
            if not recognition_input['should_run']:
                break
            small_frame = recognition_input['small_frame']
            frame_idx = recognition_input['frame_idx']
        if small_frame is None:
            time.sleep(0.01)
            continue

        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encs = face_recognition.face_encodings(rgb, face_locations, num_jitters=0)

        with tracks_lock:
            used_track_ids = set()
            for loc, enc in zip(face_locations, face_encs):
                matched_id = None
                best_iou = 0.0
                for tid, t in tracks.items():
                    i = iou(loc, t['bbox'])
                    if i > best_iou:
                        best_iou = i
                        matched_id = tid
                if best_iou < 0.2:
                    best_tid = None
                    best_dist = 1.0
                    for tid, t in tracks.items():
                        try:
                            d = np.linalg.norm(t['encoding'] - enc)
                            if d < best_dist:
                                best_dist = d
                                best_tid = tid
                        except Exception:
                            continue
                    if best_dist < 0.6:
                        matched_id = best_tid
                    else:
                        matched_id = None

                if matched_id is None:
                    tid = next_track_id
                    next_track_id += 1
                    tracks[tid] = {
                        'last_seen_frame': frame_idx,
                        'frames_missing': 0,
                        'history': deque(maxlen=FRAME_HISTORY),
                        'final': None,
                        'bbox': loc,
                        'encoding': enc
                    }
                    used_track_ids.add(tid)
                else:
                    t = tracks[matched_id]
                    t['last_seen_frame'] = frame_idx
                    t['frames_missing'] = 0
                    t['bbox'] = loc
                    t['encoding'] = enc
                    if known_encodings:
                        dists = face_recognition.face_distance(known_encodings, enc)
                        best_idx = np.argmin(dists)
                        if dists[best_idx] < TOLERANCE:
                            candidate = known_names[best_idx]
                        else:
                            candidate = "UNKNOWN"
                    else:
                        candidate = "UNKNOWN"
                    t['history'].append(candidate)
                    used_track_ids.add(matched_id)

            for tid, t in list(tracks.items()):
                if tid not in used_track_ids:
                    t['frames_missing'] += 1
                    if t['frames_missing'] > DISAPPEAR_FRAMES:
                        del tracks[tid]
                else:
                    if t['final'] is None and len(t['history']) == FRAME_HISTORY:
                        most_common = max(set(t['history']), key=t['history'].count)
                        conf = t['history'].count(most_common) / FRAME_HISTORY
                        if conf >= DECISION_THRESHOLD:
                            t['final'] = most_common
                            if most_common == "UNKNOWN":
                                t['to_save'] = True
                                t['alerted'] = False

        time.sleep(0.001)

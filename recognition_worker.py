# recognition_worker.py

import cv2
import time
import numpy as np
import face_recognition
from collections import deque
from tracker import tracks, tracks_lock
from config import FRAME_HISTORY, DECISION_THRESHOLD, DISAPPEAR_FRAMES, TOLERANCE
from face_utils import iou

def recognition_worker(frame_queue, stop_event, known_encodings, known_names, scale=0.5):
    """Consumes frames from queue and updates tracks with recognition results."""
    next_track_id = 0

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)  # get latest frame
        except:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encs = face_recognition.face_encodings(rgb, face_locations, num_jitters=0)

        with tracks_lock:
            used_track_ids = set()

            for loc, enc in zip(face_locations, face_encs):
                matched_id = None
                best_iou = 0.0

                # Try matching with existing tracks
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
                        'last_seen_frame': time.time(),
                        'frames_missing': 0,
                        'history': deque(maxlen=FRAME_HISTORY),
                        'final': None,
                        'bbox': loc,
                        'encoding': enc
                    }
                    used_track_ids.add(tid)
                else:
                    t = tracks[matched_id]
                    t['last_seen_frame'] = time.time()
                    t['frames_missing'] = 0
                    t['bbox'] = loc
                    t['encoding'] = enc

                    # recognition
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

                    # decision
                    if t['final'] is None and len(t['history']) == FRAME_HISTORY:
                        most_common = max(set(t['history']), key=t['history'].count)
                        conf = t['history'].count(most_common) / FRAME_HISTORY
                        if conf >= DECISION_THRESHOLD:
                            t['final'] = most_common
                            t['confidence'] = conf
                            if most_common == "UNKNOWN":
                                t['to_save'] = True
                                t['alerted'] = False

                    used_track_ids.add(matched_id)

            # Mark tracks not updated
            for tid, t in list(tracks.items()):
                if tid not in used_track_ids:
                    t['frames_missing'] += 1
                    if t['frames_missing'] > DISAPPEAR_FRAMES:
                        del tracks[tid]

        time.sleep(0.001)

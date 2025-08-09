1)Prepare Folder Structure
project_root/
├── faces/               # Known faces: subfolders by person name, with images inside
│   └── abhi/
│       ├── abhi_20250809_115936.jpg
│       └── ...
├── logs/
│   └── unknown_faces/   # Automatically created to save unknown face crops
├── optimized_face_rec.py
└── ...

BASH COMMANDS:
    

2)Suggested module structure:
face_recognition_project/
│
├── main.py                  # Main app script to run the video loop
├── face_utils.py            # Utility functions: load_known_faces, iou, save_unknown_crop
├── tracker.py               # Track management: tracks dict, update logic
├── recognition_worker.py    # Recognition thread logic
├── alerts.py                # Beep and Arduino alert related code
├── config.py                # All constants and config variables

1. config.py — All constants, paths, and settings
   # config.py

    import os
    
    KNOWN_FACES_DIR = "faces"
    UNKNOWN_SAVE_DIR = "logs/unknown_faces"
    os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)
    
    PROCESS_EVERY_N_FRAMES = 7
    FRAME_HISTORY = 5
    DECISION_THRESHOLD = 0.7
    DISAPPEAR_FRAMES = 30
    TOLERANCE = 0.45
    
    ARDUINO_PORT = "COM3"
    ARDUINO_BAUD = 9600
    
    ALERT_BEEP_FREQ = 1000
    ALERT_BEEP_DUR = 300
2. face_utils.py — utility functions

   # face_utils.py

    import os
    from datetime import datetime
    import cv2
    import face_recognition
    import numpy as np
    
    def load_known_faces(folder):
        encs = []
        names = []
        print("[INFO] Loading known faces from", folder)
    
        for root, dirs, files in os.walk(folder):
            person_name = os.path.basename(root)
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(root, f)
                    img = face_recognition.load_image_file(path)
                    e = face_recognition.face_encodings(img)
                    if e:
                        encs.append(e[0])
                        names.append(person_name)
                    else:
                        print("[WARN] no face found in", path)
    
        print(f"[INFO] Loaded {len(names)} known faces for {len(set(names))} people.")
        return encs, names
    
    def iou(boxA, boxB):
        tA, rA, bA, lA = boxA
        tB, rB, bB, lB = boxB
        xA = max(lA, lB)
        xB = min(rA, rB)
        yA = max(tA, tB)
        yB = min(bA, bB)
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        areaA = (rA - lA) * (bA - tA)
        areaB = (rB - lB) * (bB - tB)
        union = areaA + areaB - interArea
        if union == 0:
            return 0.0
        return interArea / union
    
    def save_unknown_crop(original_frame, small_loc, scale, prefix="unknown", save_dir=None):
        top, right, bottom, left = small_loc
        top_o = int(top / scale)
        right_o = int(right / scale)
        bottom_o = int(bottom / scale)
        left_o = int(left / scale)
        h, w = original_frame.shape[:2]
        top_o = max(0, min(h-1, top_o))
        bottom_o = max(0, min(h-1, bottom_o))
        left_o = max(0, min(w-1, left_o))
        right_o = max(0, min(w-1, right_o))
        if bottom_o <= top_o or right_o <= left_o:
            return None
        crop = original_frame[top_o:bottom_o, left_o:right_o]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_dir is None:
            save_dir = "logs/unknown_faces"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{prefix}_{timestamp}.jpg")
        cv2.imwrite(filename, crop)
        print("[LOG] saved unknown:", filename)
        return filename


3. alerts.py — Beep and Arduino alert


   # alerts.py

    import time
    
    try:
        import serial
    except Exception:
        serial = None
    
    try:
        import winsound
        def beep(freq, dur): winsound.Beep(freq, dur)
    except Exception:
        def beep(freq, dur): pass  # no-op on non-Windows
    
    def init_arduino(port, baud):
        if serial is None:
            print("[INFO] pyserial not installed; Arduino disabled.")
            return None
        try:
            arduino = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            print("[INFO] Arduino connected on", port)
            return arduino
        except Exception as e:
            print("[WARN] Arduino connect failed:", e)
            return None
    
    def alert(arduino, beep_freq, beep_dur):
        try:
            beep(beep_freq, beep_dur)
        except Exception:
            pass
        if arduino:
            try:
                arduino.write(b"ALERT\n")
                print("[INFO] ALERT sent to Arduino")
            except Exception as e:
                print("[WARN] Arduino send failed:", e)


4. tracker.py — Track management and thread-safe data

   # tracker.py

    from collections import deque
    import threading
    
    # You can import constants from config.py
    from config import FRAME_HISTORY, DECISION_THRESHOLD, DISAPPEAR_FRAMES
    
    tracks = {}
    next_track_id = 0
    tracks_lock = threading.Lock()
5). recognition_worker.py — The recognition thread function

   # recognition_worker.py

    import time
    import numpy as np
    import cv2
    import face_recognition
    
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

   
6). main.py — Main script to run camera loop and show UI

    # main.py
    
    import cv2
    import threading
    import time
    
    from face_utils import load_known_faces, save_unknown_crop
    from recognition_worker import recognition_worker, recognition_input, recognition_lock
    from tracker import tracks, tracks_lock
    from alerts import init_arduino, alert
    from config import *
    
    def main():
        known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
        arduino = init_arduino(ARDUINO_PORT, ARDUINO_BAUD)
    
        worker = threading.Thread(target=recognition_worker, args=(known_encodings, known_names), daemon=True)
        worker.start()
    
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            recognition_input['should_run'] = False
            worker.join()
            return
    
        frame_idx = 0
        scale = 0.5
        print("[INFO] Starting video. Press 'q' to quit.")
    
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
                    with recognition_lock:
                        recognition_input['small_frame'] = small.copy()
                        recognition_input['frame_idx'] = frame_idx
    
                with tracks_lock:
                    for tid, t in list(tracks.items()):
                        bbox = t['bbox']
                        top, right, bottom, left = bbox
                        top_d = int(top / scale)
                        right_d = int(right / scale)
                        bottom_d = int(bottom / scale)
                        left_d = int(left / scale)
                        label = t['final'] if t.get('final') else (t['history'][-1] if len(t['history']) > 0 else "...")
                        color = (0,255,0) if label != "UNKNOWN" else (0,0,255)
    
                        cv2.rectangle(frame, (left_d, top_d), (right_d, bottom_d), color, 2)
                        cv2.rectangle(frame, (left_d, bottom_d-25), (right_d, bottom_d), color, cv2.FILLED)
                        cv2.putText(frame, label, (left_d+6, bottom_d-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
                        if t.get('to_save') and not t.get('alerted'):
                            saved = save_unknown_crop(frame, bbox, scale, prefix=f"unknown_tid{tid}", save_dir=UNKNOWN_SAVE_DIR)
                            alert(arduino, ALERT_BEEP_FREQ, ALERT_BEEP_DUR)
                            t['alerted'] = True
    
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
        finally:
            recognition_input['should_run'] = False
            worker.join(timeout=1.0)
            cap.release()
            if arduino:
                try:
                    arduino.close()
                except:
                    pass
            cv2.destroyAllWindows()
            print("[INFO] Exiting.")
    
    if __name__ == "__main__":
        main()

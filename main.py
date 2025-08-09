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

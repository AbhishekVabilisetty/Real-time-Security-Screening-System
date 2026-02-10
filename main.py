import cv2
import threading
import queue

from face_utils import load_known_faces, load_encodings, save_encodings, save_unknown_crop
from recognition_worker import recognition_worker
from tracker import tracks, tracks_lock, cleanup_tracks
from alerts import init_arduino, alert
from config import *

frame_queue = queue.Queue(maxsize=5)

def frame_producer(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

def main():
    # Load pre-saved encodings if available
    known_encodings, known_names = load_encodings()

    # If no pre-saved encodings exist, generate and save them
    if not known_encodings:
        print("[WARN] No pre-saved encodings found — generating new ones...")
        known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
        save_encodings(known_encodings, known_names)
    else:
        # Check if new faces were added after encodings.pkl was last saved
        import os
        enc_time = os.path.getmtime("encodings.pkl")
        faces_time = max(
            os.path.getmtime(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(KNOWN_FACES_DIR)
            for f in filenames
        )
        if faces_time > enc_time:
            print("[INFO] New faces detected in dataset — regenerating encodings...")
            known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
            save_encodings(known_encodings, known_names)

    # Initialize Arduino connection
    arduino = init_arduino(ARDUINO_PORT, ARDUINO_BAUD)

    # Initialize camera
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    stop_event = threading.Event()

    producer = threading.Thread(target=frame_producer, args=(cap, frame_queue, stop_event), daemon=True)
    consumer = threading.Thread(target=recognition_worker, args=(frame_queue, stop_event, known_encodings, known_names), daemon=True)

    producer.start()
    consumer.start()

    scale = 0.5
    print("[INFO] Starting video. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            with tracks_lock:
                for tid, t in list(tracks.items()):
                    bbox = t['bbox']
                    top, right, bottom, left = bbox
                    top_d = int(top / scale)
                    right_d = int(right / scale)
                    bottom_d = int(bottom / scale)
                    left_d = int(left / scale)

                    if t.get('final'):
                        conf_pct = int(t.get('confidence', 1.0) * 100)
                        label = f"{t['final']} ({conf_pct}%)"
                    else:
                        label = t['history'][-1] if len(t['history']) > 0 else "..."

                    color = (0, 255, 0) if "UNKNOWN" not in label else (0, 0, 255)

                    cv2.rectangle(frame, (left_d, top_d), (right_d, bottom_d), color, 2)
                    cv2.rectangle(frame, (left_d, bottom_d - 25), (right_d, bottom_d), color, cv2.FILLED)
                    cv2.putText(frame, label, (left_d + 6, bottom_d - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if t.get('to_save') and not t.get('alerted'):
                        save_unknown_crop(frame, bbox, scale, prefix=f"unknown_tid{tid}", save_dir=UNKNOWN_SAVE_DIR)
                        alert(arduino)
                        t['alerted'] = True

            cleanup_tracks(0)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        producer.join(timeout=1.0)
        consumer.join(timeout=1.0)
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

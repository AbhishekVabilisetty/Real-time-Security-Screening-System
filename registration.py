import os
import cv2
import face_recognition
from datetime import datetime

OUTPUT_FOLDER = "faces"
NUM_IMAGES = 20
FACE_SIZE = (160, 160)
PADDING = 30

PERSON_NAME = input("Enter the name of the person: ").strip()
if not PERSON_NAME:
    print("[ERROR] Name cannot be empty.")
    exit()
    
person_path = os.path.join(OUTPUT_FOLDER, PERSON_NAME)
os.makedirs(person_path, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Cannot access camera")
    exit()

captured = 0
print("[INFO] Look at the camera. Press 'q' to quit anytime.")

while captured < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.putText(frame, f"Captured {captured}/{NUM_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Registration", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if len(face_locations) == 1:
        top, right, bottom, left = face_locations[0]
        top = max(0, top - PADDING)
        right = min(frame.shape[1], right + PADDING)
        bottom = min(frame.shape[0], bottom + PADDING)
        left = max(0, left - PADDING)

        face_img = frame[top:bottom, left:right]
        face_img = cv2.resize(face_img, FACE_SIZE)
        
        # --- VALIDATE BEFORE SAVING ---
        encodings = face_recognition.face_encodings(face_img)
        if len(encodings) == 0:
            print("[SKIP] Face not clear, not saving.")
            continue  # skip this frame

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(person_path, f"{PERSON_NAME}_{timestamp}.jpg")
        cv2.imwrite(save_path, face_img)
        captured += 1
        print(f"[SAVED] {save_path}")

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Captured {captured} high-quality images.")

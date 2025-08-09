import os
import cv2
import face_recognition
from datetime import datetime

# === CONFIG ===
OUTPUT_FOLDER = "faces"        # Folder where known faces are stored
PERSON_NAME = "abhi"           # Change to the person's name

# Create folder for the person if not exists
person_path = os.path.join(OUTPUT_FOLDER, PERSON_NAME)
os.makedirs(person_path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Press 's' to save face, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # frame = cv2.flip(frame, 1)  # Flip horizontally (mirror)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    # Draw box around detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Face Registration", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if len(face_locations) == 1:
            # Add padding around the face crop
            pad = 30
            top, right, bottom, left = face_locations[0]
            top = max(0, top - pad)
            right = min(frame.shape[1], right + pad)
            bottom = min(frame.shape[0], bottom + pad)
            left = max(0, left - pad)

            face_img = frame[top:bottom, left:right]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(person_path, f"{PERSON_NAME}_{timestamp}.jpg")
            cv2.imwrite(save_path, face_img)
            print(f"[SAVED] {save_path}")
        else:
            print("[WARNING] Make sure only ONE face is visible before saving.")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[DONE] Face registration completed.")

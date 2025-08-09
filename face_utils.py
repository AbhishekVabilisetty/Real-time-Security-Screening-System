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

def save_unknown_crop(original_frame, small_loc, scale, prefix="unknown", save_dir=None, padding=30):
    top, right, bottom, left = small_loc
    top_o = int(top / scale) - padding
    right_o = int(right / scale) + padding
    bottom_o = int(bottom / scale) + padding
    left_o = int(left / scale) - padding

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


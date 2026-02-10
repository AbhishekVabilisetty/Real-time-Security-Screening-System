# Real-time Security Screening System

> **An AI-powered real-time system for detecting and classifying prohibited objects from live video streams, designed for security and surveillance applications.**

---

## 📌 Overview

The **Real-time Security Screening System** is an advanced computer vision application built using **Python**, **OpenCV**, and **deep learning** techniques.
It captures live video feed from a camera, processes each frame, and detects potentially dangerous items — helping improve safety in public and private spaces.

The system is designed for:

* Airports and transportation hubs
* Public events and stadiums
* Corporate security checkpoints
* Educational institutions

---

## ✨ Features

* **🎥 Real-time object detection** — Instant processing from live camera feeds.
* **🧠 AI-powered classification** — Uses a trained deep learning model for accuracy.
* **⚡ High performance** — Optimized for low-latency inference.
* **📊 Visual feedback** — Bounding boxes and labels displayed on-screen.
* **🔒 Configurable object list** — Customize which items to detect.
* **📁 Image/Video input** — Works with both webcam streams and pre-recorded media.

---

## 🛠️ Tech Stack

* **Programming Language:** Python 3.x
* **Libraries & Frameworks:**

  * [OpenCV](https://opencv.org/) — For image and video processing
  * [face\_recognition](https://github.com/ageitgey/face_recognition) — For facial recognition
  * [NumPy](https://numpy.org/) — For numerical operations
* **Hardware Support:** CPU & GPU (CUDA acceleration optional)

---

## 📂 Project Structure

```
Real-time-Security-Screening-System/
│
├── models/              # Pre-trained or custom-trained detection models
├── data/                # Sample images/videos for testing
├── utils/               # Helper functions and scripts
├── main.py               # Main entry point for running the system
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AbhishekVabilisetty/Real-time-Security-Screening-System.git
cd Real-time-Security-Screening-System
```

### 2️⃣ Install Required Python Modules

```bash
pip install opencv-python numpy face_recognition
```

### 3️⃣ Install Standard Library Modules (No pip needed)

The following modules are part of Python’s standard library and do **not** require installation:

```text
os
time
threading
collections
datetime
```

These are automatically included with Python.

### 4️⃣ Download / Prepare Model

* Place the trained detection model in the `models/` directory.
* Update `main.py` with the correct model path.

---

## 🚀 Usage

### Run with Webcam

```bash
python main.py --source 0
```

### Run with Video File

```bash
python main.py --source path/to/video.mp4
```

**Options:**

* `--confidence` → Minimum detection confidence (default: 0.5)
* `--classes` → Path to class names file

---



## 📊 Model Training (Optional)

If you want to train your own model:

1. Prepare a dataset with labeled prohibited items.
2. Train using YOLOv5, TensorFlow Object Detection API, or similar.
3. Export weights and place them in `models/`.

---


## 👤 Author

**Abhishek Vabilisetty**
📧 Email: *(your email here)*
🔗 GitHub: [@AbhishekVabilisetty](https://github.com/AbhishekVabilisetty)

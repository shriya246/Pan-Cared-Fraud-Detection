"""# 🧠 Real-Time Object Detection Using OpenCV & TensorFlow

This project demonstrates a real-time object detection system using pre-trained deep learning models like SSD MobileNet v3 and YOLO. It supports image, video, and live webcam input sources.

---
### 1. Clone the repository
```bash
git clone https://github.com/your-username/object-detection-project.git
cd object-detection-project

2. Install dependencies
```bash
pip install -r requirements.txt

3. Run Detection
```bash
python scripts/e_run\ image.py

For video detection:
```bash
python scripts/f_run\ video.py

For webcam/live detection:
Run the notebook:
```bash
notebooks/Live\ Object\ Detection.ipynb

🧰 Tech Stack
Python
OpenCV
TensorFlow
NumPy

📌 Model Info
Model: SSD MobileNet v3 COCO
Config File: ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
Frozen Graph: frozen_inference_graph.pb
Labels: coco.names.txt


# Head Count and Image Recognition using YOLOv8 Model

## Project Overview

This project leverages the YOLOv8 model to detect and count heads in a video stream and identify individuals based on facial recognition. By utilizing advanced computer vision techniques, the system tracks individuals across frames and matches their faces with pre-stored facial encodings to provide identification. The model can be applied in various scenarios such as attendance systems, monitoring crowd density, and more.

### Business Objective
The main objective of this project is to **maximize the accuracy of head detection** and **minimize false positive identifications** in face recognition, ensuring precise headcount and individual identification.

### Maximize:
- The accuracy of headcount and identification in dynamic environments.
- The processing efficiency for real-time video analytics.

### Minimize:
- False detections and misidentifications.
- Computational load without compromising performance.

### Data Preprocessing
- **Image Augmentation**: The dataset includes augmented versions of facial images to improve model generalization.
- **Face Cropping**: Detected face regions are cropped and resized before feeding them to the face recognition module.
- **Encoding**: Face encodings are stored in a database for classification using a pre-trained SVM model.

### Technology Used:
- **YOLOv8** for head detection.
- **OpenCV** for video processing.
- **face_recognition** for face detection and encoding.
- **SVM Classifier** for face identification.
- **Numpy** for numerical operations.

### Impact of the Project:
- Enhanced surveillance systems capable of providing detailed insights into crowd monitoring.
- Improved real-time analytics in educational and corporate environments for headcount and attendance tracking.

---

## Steps in the Project:
1. **Video Capture**: Using OpenCV, the system captures frames from the video source.
2. **Head Detection**: YOLOv8 is applied to detect heads in each frame, with bounding boxes drawn around each detected head.
3. **Tracking**: The system tracks individuals across frames, ensuring that no person is counted twice.
4. **Face Identification**: For each detected head, the corresponding face is extracted, and facial recognition is performed using an SVM classifier.
5. **Head Counting**: Based on the bounding box location, individuals are counted as they cross predefined lines.
6. **Display Results**: The video output is annotated with the total headcount and identified names, saved as `output_final2.mp4`.

---

## Main File to Run:
- **`in_out.py`**: This is the main script for executing the headcount and face recognition process.

### Example Code Snippet:
```python
import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import face_recognition

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the trained SVM classifier
with open('svm_classifier_mega.pkl', 'rb') as f:
    svm_clf = pickle.load(f)

# Video Capture and Processing Logic
# ...
```

---

## Dataset Information:
The dataset contains facial images of students, augmented versions of these images, and updated facial encodings. Due to privacy concerns, the dataset is not provided publicly.

### Dataset Files:
- **Augmented_Clean_Files**: Contains the cleaned and augmented versions of the original facial images.
- **Augmented_Faces**: Processed facial images used for training the face recognition model.
- **Updated_Students**: Facial encodings of students stored for identification.

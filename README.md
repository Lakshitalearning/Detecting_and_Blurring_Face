# Live Face Blur Application

![Demo Screenshot](screenshots/demo.gif) <!-- Add actual screenshot later -->

A real-time face blurring application that detects faces using OpenCV's Haar Cascades and applies Gaussian blur to protect privacy.

## Features
- Real-time face detection from webcam feed
- Dual Haar Cascade classifiers (frontal + profile faces)
- Motion-stabilized face detection using previous frame data
- Adaptive Gaussian blurring with adjustable intensity
- Smooth face tracking between frames

## Tools & Technologies
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)

- **Python 3.8+**
- **OpenCV** (Computer Vision Library)
- **Haar Cascade Classifiers** (Pre-trained models)
  - haarcascade_frontalface_default.xml
  - haarcascade_profileface.xml
  
## Project Structure

- live-face-blur/
- ├── face_blur.py
- ├── haarcascades/           
- │   ├── frontalface.xml
- │   └── profileface.xml      
- └── README.md

## Installation

1. Clone repository:
```bash
git clone https://github.com/Lakshitalearning/Detecting_and_Blurring_Face.git
cd Detecting_and_Blurring_Face

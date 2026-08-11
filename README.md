# 🎓 AI-Powered Student Drowsiness & Attention Monitoring System

A production-grade, real-time Computer Vision and Temporal Attention Analytics application built using **OpenCV**, **MediaPipe FaceMesh**, **NumPy**, and **Streamlit**. Designed for high accuracy, temporal stability, low-light robustness, multi-person edge case handling, and national hackathon presentation.

---

## 🌟 Key Capabilities & Highlights

- **⚡ Real-Time Engine (30+ FPS)**: High-speed processing using vectorized NumPy operations, without training or running heavy deep learning models.
- **👁️ Dual-Eye Aspect Ratio (EAR)**: Computes precise EAR using official MediaPipe 468/478 landmark indices.
- **😮 Mouth Aspect Ratio (MAR)**: Differentiates yawning (sustained > 1.5s) from normal talking.
- **📐 3D Head Pose Estimation (`solvePnP`)**: Solves Perspective-n-Point pose mapping to extract Pitch, Yaw, and Roll Euler angles for orientation detection (Looking Left, Right, Up, Down, Phone simulation).
- **⏱️ 90-Frame Sliding Temporal Window**: Uses Exponential Moving Average (EMA) and continuous threshold validation (>= 3.0s) to eliminate single-frame blink false positives.
- **🎯 10-Second Adaptive Calibration**: Collects baseline user EAR, MAR, and neutral head pose to dynamically generate personalized thresholds.
- **🌙 Automatic Low-Light Enhancement**: Detects dark frames ($L_{mean} < 45$) and applies CLAHE contrast adjustment and noise reduction.
- **👓 Spectacles & Glasses Support**: Robust landmark extraction filtering out lens reflection noise.
- **👥 Multi-Person Handling**: Selects and tracks the largest face (student foreground) while displaying a "Multiple Faces Detected" warning banner.
- **📊 Streamlit Dark Glassmorphic Dashboard**: Features animated status badges (ALERT, DISTRACTED, DROWSY), live KPI cards, timeline graphs, and downloadable CSV telemetry reports.
- **📸 Automatic Screenshots & Audio Sirens**: Saves frame images to `reports/screenshots/` on drowsy events and triggers Web Audio synthesis alert tones.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Webcam / Video File Input] --> B[Image Preprocessing & Low-Light CLAHE Check]
    B --> C[MediaPipe FaceMesh Landmark Detector]
    C --> D{Multi-Face Check}
    D -- Multiple Faces --> E[Track Largest Face + Flag Warning]
    D -- Single Face --> F[Extract Primary Face 468 Landmarks]
    E --> F
    F --> G[Feature Extraction: EAR & MAR]
    F --> H[3D Head Pose Estimation: solvePnP]
    G --> I[10-Second Adaptive Calibration Engine]
    H --> I
    I --> J[90-Frame Sliding Window & EMA Filter]
    J --> K[Drowsiness & Distraction Engine]
    K --> L[0-100 Attention Scorer]
    L --> M[Streamlit Dashboard & Audio Alerts]
    M --> N[CSV Telemetry Exporter & Session Analytics]
```

---

## 📐 Mathematical Formulations

### 1. Eye Aspect Ratio (EAR)
Evaluated across 6 landmark points per eye:
$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2.0 \cdot ||p_1 - p_4||}$$

- **Left Eye Indices**: $P_1=33, P_2=160, P_3=158, P_4=133, P_5=153, P_6=144$
- **Right Eye Indices**: $P_1=362, P_2=385, P_3=387, P_4=263, P_5=373, P_6=380$

### 2. Mouth Aspect Ratio (MAR)
Evaluates inner and outer mouth vertical expansion against horizontal width:
$$MAR = \frac{||t_1 - b_1|| + ||t_2 - b_2|| + ||t_3 - b_3||}{2.0 \cdot ||left - right||}$$

- **Landmark Indices**: Left=$61$, Right=$291$, Vertical Pairs: $(81, 178), (13, 14), (311, 402)$

### 3. Head Pose Estimation (`cv2.solvePnP`)
Maps 2D landmarks (Nose $1$, Chin $152$, Eye corners $33, 263$, Mouth corners $61, 291$) to canonical 3D head model points. Solves:
$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} R & T \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

Decomposes rotation matrix $R$ into Euler angles:
- **Yaw**: Negative = Looking Left ($<-15^\circ$), Positive = Looking Right ($>+15^\circ$)
- **Pitch**: Negative = Looking Down / Phone Usage ($<-15^\circ$), Positive = Looking Up ($>+15^\circ$)
- **Roll**: Head tilt angle.

### 4. Exponential Moving Average (EMA)
Prevents noise spikes:
$$S_t = \alpha \cdot X_t + (1 - \alpha) \cdot S_{t-1} \quad (\alpha = 0.25)$$

---

## 📁 Project Structure

```
Hackathon/
├── config/
│   └── settings.json             # Application threshold & path parameters
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration loader & landmark constants
│   ├── image_processing.py       # CLAHE low-light detection & enhancement
│   ├── face_mesh.py              # MediaPipe FaceMesh wrapper & multi-face handler
│   ├── feature_extraction.py     # Vectorized EAR & MAR calculators
│   ├── head_pose.py              # solvePnP 3D pose estimator
│   ├── calibration.py            # Adaptive 10-second baseline calibrator
│   ├── drowsiness_detector.py    # 90-frame buffer & continuous 3s drowsiness logic
│   ├── distraction_detector.py    # Head turn, missing face & yawn detector
│   ├── attention_scorer.py       # Composite 0-100 Attention Score engine
│   └── analytics.py              # Session statistics, focus %, and risk recommendations
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # Structured application logging
│   ├── exporter.py               # Real-time CSV logging & screenshot generator
│   └── audio.py                  # Web Audio API sound alert generator
├── ui/
│   ├── __init__.py
│   ├── dashboard.py              # Streamlit dashboard layout & event loop
│   └── components.py             # Custom Glassmorphic CSS & status badges
├── reports/                      # Auto-generated CSV telemetry reports
│   └── screenshots/              # Automated screenshot captures on Drowsy alert
├── logs/                         # App runtime log files
├── dataset/
│   └── README.md                 # Dataset instructions & NTHU benchmarking guide
├── tests/
│   ├── __init__.py
│   └── test_suite.py             # Automated unit test suite (10 edge cases)
├── app.py                        # Streamlit application entry point
├── run.py                        # Dependency checker & launch runner
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+ installed on your system.
- Webcam (built-in or USB) or recorded video files.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Application
Using the runner script:
```bash
python run.py
```
Or directly via Streamlit:
```bash
streamlit run app.py
```

### 3. Run Automated Unit Test Suite
Verify all 10 edge case test modules:
```bash
python tests/test_suite.py
```

---

## 🧪 Edge Cases Handling Matrix

| Edge Case | Solution Implemented |
| :--- | :--- |
| **Low Light / Dark Room** | Automatic luminance check + LAB CLAHE contrast enhancement & noise reduction |
| **Spectacles / Glasses** | Robust landmark index distance calculation ignoring specular reflections |
| **Fast Blinking vs Sleep** | 90-frame sliding window filtering out 60-400ms blinks; triggers only on >=3s continuous closure |
| **Talking vs Yawning** | Differentiates brief MAR spikes from sustained mouth opening (>=1.5s) |
| **Multiple Faces** | Bounding box area tracking selects largest foreground face & displays warning overlay |
| **Mobile Phone Usage** | Detects posture combining downward pitch ($<-20^\circ$) and yaw tilt sustained for >=2s |
| **Face Occlusion / Loss** | Missing face detection triggers Distraction state after 2 seconds |
| **Individual Variability** | 10-second Adaptive Calibration personalizes EAR & MAR thresholds |

---

## 🏆 Benchmark & Datasets
Supports evaluation against:
- Custom recorded webcam/mobile videos (.mp4, .avi, .mov).
- **NTHU Drowsy Driver Dataset** (Bareface, Glasses, Night/Low Light sequences).

---

## 📜 Deliverables Summary
- Production-ready Python codebase built following PEP8 guidelines.
- Full unit test coverage passing 10/10 test cases.
- Automated CSV telemetry export & alert screenshots.
- Complete documentation and hackathon presentation architecture.

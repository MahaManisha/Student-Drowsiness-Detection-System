# Dataset & Benchmarking Instructions

## Overview
This system processes real-time facial streams without relying on pre-trained deep learning models. It extracts MediaPipe 468/478 3D landmark metrics directly from webcam frames or pre-recorded video files.

---

## 1. Custom Recorded Videos (Webcam / Mobile Camera)
You can record videos using your webcam or phone camera for testing and hackathon demonstrations:
- **Normal**: Face centered, eyes open, occasional blinks, neutral head pose.
- **Yawning**: Deep mouth opening sustained for >= 1.5 seconds.
- **Sleepy / Drowsy**: Prolonged eye closures sustained for >= 3.0 seconds.
- **Looking Away**: Head turned Left, Right, Up, or Down for >= 2.0 seconds.

### How to Run Pre-Recorded Videos:
1. Launch the app: `streamlit run app.py`
2. In the sidebar, select **Input Source**: `Upload Video File`.
3. Drag and drop your recorded `.mp4`, `.avi`, or `.mov` file.

---

## 2. NTHU Drowsy Driver Dataset Compatibility
The application is fully compatible with the **NTHU Drowsy Driver Dataset** (National Tsing Hua University Drowsy Driver Detection Dataset) for evaluation:

- **Supported Scenarios**:
  - Bareface
  - Glasses
  - Night / Low-Light condition
- **Supported State Labels**:
  - `Glasses_Normal` / `Bareface_Normal`
  - `Glasses_Yawning` / `Bareface_Yawning`
  - `Glasses_Sleepy` / `Bareface_Sleepy`

### Steps for NTHU Evaluation:
1. Download the NTHU evaluation video files.
2. Select **Upload Video File** in the sidebar.
3. Upload any NTHU video sequence (e.g. `night_bareface_sleepy.avi` or `glasses_yawning.avi`).
4. Watch the real-time telemetry metrics, automated low-light CLAHE enhancement, and CSV logging.

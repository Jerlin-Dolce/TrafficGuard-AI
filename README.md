# TrafficGuard AI — Full Starter Project

Windows-friendly traffic-video analysis using YOLO, ByteTrack, OpenCV and Streamlit.

## Features
- Upload a video in a browser dashboard (no manual renaming/copying needed).
- Detect cars, motorcycles, buses and trucks.
- Track vehicle IDs and count detections.
- Configure a virtual line and log crossings for human review.
- Save annotated video, CSV records and evidence snapshots.

**Important:** A virtual-line crossing is not proof of a legal traffic offence. Red-light detection requires reliable signal-state information and suitable camera calibration. Use footage you own or are allowed to analyze.

## Install on Windows
1. Extract this ZIP.
2. Open the extracted `TrafficGuardAI_Full` folder in File Explorer.
3. Click the address bar, type `cmd`, press Enter.
4. Run:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

A browser tab should open. If not, open the local URL printed in Command Prompt (usually http://localhost:8501). First run needs internet access to download YOLO model weights.

## Use
1. Click Browse files and choose a short traffic video (20–60 seconds is good for testing).
2. Adjust confidence and virtual-line position if needed.
3. Click Process video and wait.
4. Review the annotated video, tables and download buttons.

## Outputs
`outputs/tracked_traffic.mp4`, `outputs/vehicle_detections.csv`, `outputs/line_crossing_events.csv`, and `outputs/evidence/`.

If package imports fail, use the fresh `.venv` above instead of installing into a shared Anaconda environment.

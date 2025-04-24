# MediaPipe Object Detection Metrics

This project evaluates MediaPipe object detection model performance on videos and saves metrics to a results file.

## Setup Instructions

### 1. Clone the repository (if not already done):
```bash
git clone <repo-url>
cd mediapipe_object_detection
```

### 2. Create a virtual environment
#### On **macOS/Linux**:
```bash
python3 -m venv virtualvenv
source virtualvenv/bin/activate
```

#### On **Windows** (Command Prompt):
```bat
python -m venv virtualvenv
virtualvenv\Scripts\activate
```

#### On **Windows** (PowerShell):
```powershell
python -m venv virtualvenv
.\virtualvenv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

To generate a requirements.txt file from your current environment:
```bash
pip freeze > requirements.txt
```

### 4. Project Structure
Make sure your project has the following structure:
```
.
├── Data/                  # Place your videos here
├── Models/                # Contains the TensorFlow Lite model
│   └── efficientdet_lite0_pf16.tflite
├── resultados/            # Results will be saved here
└── metricas_mediapipe.py    # Main script
```

### 5. Place your video files in the `Data/` directory
Supported formats: .mp4, .avi, .mov, .mkv

### 6. Run the metrics script
```bash
python metricas_mediapipe.py
```

### 7. Results
Results will be saved in the `resultados/` directory with a timestamped filename in the format:
```
metrics_results_mediapipe_YYYYMMDD_HHMMSS.txt
```

The results file contains the following metrics for each video:
- CPU usage percentage
- CPU time used
- Wall-clock time elapsed
- Average confidence score for person detections
- Average inference time per frame
- Average processing time per frame

---

**Note:**
- Make sure you have Python 3.8+ installed.
- The MediaPipe model (`efficientdet_lite0_pf16.tflite`) should be present in the Models/ directory.
- If you encounter "Input timestamp must be monotonically increasing" errors, the script includes fixes to handle timestamps correctly.

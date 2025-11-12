# 🧠 Real-Time Webcam-Based Eye Gaze Control System  
**With Intelligent Blink Recognition for Assistive Computing**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.3-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Code of Conduct](https://img.shields.io/badge/Code%20of%20Conduct-Active-blue.svg)](CODE_OF_CONDUCT.md)
[![Security Policy](https://img.shields.io/badge/Security-Policy-orange.svg)](SECURITY.md)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

>
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
 ⚙️ An accessible, low-cost **eye-gaze control system** using standard webcams — featuring **real-time tracking** and **intelligent blink-based control** for users with motor impairments.

---

## 📄 Research Paper

**Real-Time Webcam-Based Eye Gaze Control System with Intelligent Blink Recognition for Assistive Computing**

**Authors:**  
K. Ujjwal Reddy, Karthik M, P. V. Koushik Reddy  
**Affiliation:** Sir M Visvesvaraya Institute of Technology, Bengaluru, India  

📘 *Published in:* [Conference Name], [Year]  
🔗 *Link:* _Coming soon_

---

## 🎯 Overview

This project enables **hands-free computer control** through real-time **eye and blink tracking**, requiring only a standard webcam — no infrared cameras, depth sensors, or GPUs needed.

### ✨ Key Highlights
- 🧩 **Multi-Algorithm Pupil Fusion** (Intensity + Hough + Contour)
- 👁️ **Adaptive Blink Recognition FSM** with 94.2% accuracy  
- ⚡ **Real-Time Operation:** 26.7 FPS on Intel i5 hardware  
- 🔧 **Configurable:** 50+ parameters for calibration and control  
- 💻 **Cross-Platform:** Works on Windows, Linux, and macOS  

---

## 📊 Performance Summary

| Metric | Value |
|--------|-------|
| Mean Positioning Error | 62 px |
| Click Detection Accuracy | 94.2% |
| False Positives | 0.3 / min |
| Frame Rate | 26.7 FPS |
| CPU Usage | < 60% |

---

## 🚀 Quick Start

### ✅ Requirements
- Python 3.9+
- Webcam (720p minimum, 1080p recommended)
- 4 GB+ RAM  
- Windows/macOS/Linux

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/eye-gaze-assistive-control.git
cd eye-gaze-assistive-control

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python download_models.py
```

---

## ▶️ Usage

### 1️⃣ Calibration
```bash
python main.py --calibrate
```
Follow the on-screen 9-point calibration and blink tests.  

### 2️⃣ Normal Operation
```bash
python main.py
```
**Controls:**
- 👁️ Look → Move cursor  
- 👁 Left blink → Left click  
- 👁 Right blink → Right click  
- 👁👁 Double blink → Double-click  
- 👁 Dwell gaze (2s) → Auto click  

---

## ⚙️ Configuration

Edit `config/config.yaml` for fine-tuning:

```yaml
# Gaze tracking
sensitivity: 1.5
smoothing_alpha: 0.3
nonlinearity_gamma: 1.15

# Blink detection
ear_threshold: 0.25
blink_duration_min: 80
blink_duration_max: 400
cooldown_period: 150

# Camera
resolution: [640, 480]
fps: 30
```

Full details: [`docs/configuration.md`](docs/configuration.md)

---

## 🧩 System Architecture

```
┌─────────────┐
│   Webcam    │ 640×480 @ 30 FPS
└──────┬──────┘
       ▼
┌─────────────────┐
│ Face Detection  │ HOG + dlib (12 ms)
└──────┬──────────┘
       ▼
┌─────────────────┐
│Facial Landmarks │ 68-point predictor
└──────┬──────────┘
   ┌───┴───┐
   ▼       ▼
┌──────┐ ┌───────┐
│ Gaze │ │ Blink │
│Track │ │ Detect│
└──┬───┘ └───┬───┘
    ▼         ▼
  ┌──────────┐
  │ Cursor + │
  │  Clicks  │
  └──────────┘
```

More in: [`docs/architecture.md`](docs/architecture.md)

---

## 🔬 Technical Insights

### 🧠 Multi-Algorithm Fusion
Combines:
1. **Intensity-Extrema** → Fast under contrast  
2. **Circular Hough Transform** → Accurate under strong edges  
3. **Contour Analysis** → Robust to occlusion  

Weighted fusion formula:
\[
(x_f, y_f) = \frac{\sum Q_i x_i}{\sum Q_i}, \quad \frac{\sum Q_i y_i}{\sum Q_i}
\]

### 👁️ Blink FSM
- Duration: 80–400 ms  
- EAR Drop ≥ 25%  
- Cooldown: 150 ms  
→ Reduces false positives by **67%**

---

## 🧪 Evaluation Results

| Method | Mean Error | Variance | Failure Rate |
|--------|------------|-----------|---------------|
| Intensity-only | 89 ± 34 px | 1156 | 12.0% |
| Hough-only | 76 ± 28 px | 784 | 8.0% |
| Contour-only | 82 ± 31 px | 961 | 10.0% |
| Fixed-weight | 71 ± 26 px | 676 | 5.0% |
| **Ours (Quality-weighted)** | **62 ± 18 px** | **324** | **2.0%** |

---

## 🏗️ Project Structure

```
Eye-Gaze-Control-System-with-Intelligent-Blink-Recognition-for-Assistive-Computing/
├── enhanced_main.py                  # Main entry point
├── enhanced_eye_tracker.py           # Eye and gaze tracking module
├── enhanced_blink_detector.py        # Blink recognition logic
├── enhanced_config.py                # Configuration parameters
├── enhanced_utils.py                 # Utility functions
├── true_gaze_tracker.py              # Advanced gaze estimation logic
├── shape_predictor_68_face_landmarks.dat  # Dlib facial landmark model
├── Setup_and_Test.py                 # Environment setup and test script
├── README.md                         # Documentation
├── LICENSE                           # MIT License
├── CONTRIBUTING.md                   # Contribution guide
├── CODE_OF_CONDUCT.md                # Community guidelines
├── SECURITY.md                       # Security reporting policy
├── .gitignore                        # Ignored files for Git
└── __pycache__/                      # Python bytecode cache
```

eye-gaze-assistive-control/
├── main.py
├── src/
│   ├── gaze_tracker.py
│   ├── blink_detector.py
│   ├── pupil_fusion.py
│   ├── calibration.py
│   └── utils.py
├── config/
│   ├── default.yaml
│   └── profiles/
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── docs/
├── tests/
└── requirements.txt
```

---

## 🐳 Docker Deployment

```bash
docker build -t eye-gaze-control .
docker run --device=/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix eye-gaze-control
```

---

## 🤝 Contributing

We welcome contributions in:
- New pupil detection algorithms  
- Calibration-free gaze estimation  
- 3D head pose correction  
- Performance benchmarking  

**Steps:**
1. Fork this repo  
2. Create branch → `feature/my-feature`  
3. Commit → `git commit -m "Add my feature"`  
4. Push → `git push origin feature/my-feature`  
5. Open a Pull Request  

See [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 📜 License

Licensed under **MIT License** — see [`LICENSE`](LICENSE).  
> 🪶 Open, accessible, and free for both academic and commercial use.

---

## 📚 Citation

```bibtex
@inproceedings{reddy2024eyegaze,
  title={Real-Time Webcam-Based Eye Gaze Control System with Intelligent Blink Recognition for Assistive Computing},
  author={Reddy, K. Ujjwal and M, Karthik and Reddy, P. V. Koushik},
  booktitle={Proceedings of [Conference Name]},
  year={2024},
  organization={IEEE}
}
```

---

## 🙏 Acknowledgments
- Sir M Visvesvaraya Institute of Technology – AI/ML Department  
- OpenCV & dlib open-source communities  
- All study participants for their valuable feedback  

---

## ⚠️ Disclaimer
This system is a **research prototype** for assistive technology.  
Not certified for clinical or medical use. Please test carefully before deploying in accessibility-critical contexts.

---

## 🗺️ Roadmap
- [x] v1.0 – Initial release  
- [ ] v1.1 – Add MobileNet-based CNN detector  
- [ ] v1.2 – 3D head-pose correction  
- [ ] v2.0 – Calibration-free gaze mapping  
- [ ] v2.1 – Multi-modal fusion (voice + gaze + gesture)  

---

## 📈 GitHub Stats

[![GitHub Stars](https://img.shields.io/github/stars/Ujjwalreddy16/Eye-Gaze-Control-System-with-Intelligent-Blink-Recognition-for-Assistive-Computing?style=social)](https://github.com/Ujjwalreddy16/Eye-Gaze-Control-System-with-Intelligent-Blink-Recognition-for-Assistive-Computing/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Ujjwalreddy16/Eye-Gaze-Control-System-with-Intelligent-Blink-Recognition-for-Assistive-Computing?style=social)](https://github.com/Ujjwalreddy16/Eye-Gaze-Control-System-with-Intelligent-Blink-Recognition-for-Assistive-Computing/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/Ujjwalreddy16/Eye-Gaze-Control-System-with-Intelligent-Blink-Recognition-for-Assistive-Computing)](https://github.com/Ujjwalreddy16/Eye-Gaze-Control-System-with-Intelligent-Blink-Recognition-for-Assistive-Computing/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/Ujjwalreddy16/Eye-Gaze-Control-System-with-Intelligent-Blink-Recognition-for-Assistive-Computing)](https://github.com/Ujjwalreddy16/Eye-Gaze-Control-System-with-Intelligent-Blink-Recognition-for-Assistive-Computing/pulls)

---

**💡 Made with passion for accessibility — empowering users through vision-based computing.**



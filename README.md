# 🚗 Vehicle Detection & Counting System

A real-time vehicle detection and counting system powered by **YOLOv8** and **OpenCV**. It identifies vehicles (cars, motorcycles, buses, and trucks) from a video feed or file, tracks them across frames, and counts how many cross a configurable line — all while showing live stats on screen.

---

## What Does It Do?

- Detects vehicles in real time using a YOLOv8 model
- Tracks each vehicle across frames using a lightweight centroid tracker
- Counts vehicles the moment they cross a virtual line on screen
- Displays live per-class counts (car / motorcycle / bus / truck) and FPS
- Saves the annotated output to a video file (optional)

---

## Prerequisites

Before you get started, make sure you have the following installed on your system:

- **Python 3.8 or higher** — [Download here](https://www.python.org/downloads/)
- **pip** — comes bundled with Python
- A working **webcam** (if you want to use live input), or a video file to test with

---

## Setup & Installation

### 1. Clone or Download the Project

```bash
git clone https://github.com/your-username/vehicle-detection.git
cd vehicle-detection
```

Or simply download the ZIP and extract it somewhere on your computer.

---

### 2. (Recommended) Create a Virtual Environment

This keeps dependencies isolated and avoids conflicts with other Python projects:

```bash
python -m venv venv
```

Then activate it:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

---

### 3. Install Dependencies

```bash
pip install ultralytics opencv-python numpy
```

That's it — `ultralytics` will automatically pull in PyTorch and everything else YOLOv8 needs.

> **Note:** The first time you run the script, it will automatically download the YOLOv8 model weights (e.g., `yolov8n.pt`, ~6 MB). You'll need an internet connection for that initial download.

---

## Running the Project

### Basic Usage (Webcam)

To start detecting vehicles from your default webcam:

```bash
python vehicle_detector.py
```

Press **`Q`** to quit at any time.

---

### Use a Video File Instead

```bash
python vehicle_detector.py --source path/to/your/video.mp4
```

---

### Full List of Options

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Video source: `0` for webcam, or a path/URL to a video file |
| `--model` | `yolov8n.pt` | YOLOv8 model to use (see Model Options below) |
| `--output` | `output.mp4` | Path to save the annotated output video. Leave empty (`""`) to disable saving |
| `--imgsz` | `640` | Image size used for inference (larger = more accurate but slower) |
| `--confidence` | `0.3` | Minimum detection confidence (0.0–1.0) |
| `--iou` | `0.45` | IoU threshold for Non-Maximum Suppression |
| `--line_position` | `0.5` | Vertical position of the counting line as a fraction of frame height (0 = top, 1 = bottom) |

**Example — custom settings:**

```bash
python vehicle_detector.py \
  --source traffic.mp4 \
  --model yolov8s.pt \
  --confidence 0.4 \
  --line_position 0.6 \
  --output counted_traffic.mp4
```

---

## Choosing a Model

YOLOv8 comes in several sizes. Smaller models are faster; larger ones are more accurate:

| Model | Speed | Accuracy | Best For |
|---|---|---|---|
| `yolov8n.pt` | Fastest | Lower | Webcam / low-end hardware |
| `yolov8s.pt` | Fast | Good | General use |
| `yolov8m.pt` | Medium | Better | Higher-quality footage |
| `yolov8l.pt` | Slow | High | Accuracy-critical tasks |
| `yolov8x.pt` | Slowest | Highest | Maximum accuracy |

The model file will be downloaded automatically the first time you use it.

---

## Understanding the Output

When the program runs, you'll see a window with:

- **Bounding boxes** (green) drawn around detected vehicles, labeled with their type
- **Red dots** at the centroid (center) of each tracked vehicle, with an ID number
- **A yellow horizontal line** — vehicles are counted when they cross this line
- **Live count display** at the top: `Counts - Car:3  Motorcycle:1  Bus:0  Truck:2`
- **FPS counter** in the top-right corner

At the end of the session (or when the video finishes), the final counts are printed to the terminal:

```
Final counts: {'car': 12, 'motorcycle': 4, 'bus': 1, 'truck': 3}
```

---

## Project Structure

```
vehicle-detection/
├── vehicle_detector.py     # Main script — run this
├── output.mp4              # Output video (generated after running)
└── README.md               # This file
```

---

## Troubleshooting

**"Cannot open source 0" error**
Your webcam isn't being detected. Try specifying the source explicitly (`--source 1` or `--source 2`) or use a video file instead.

**Very low FPS or laggy performance**
Switch to a smaller model (`--model yolov8n.pt`) or reduce the image size (`--imgsz 320`).

**Vehicles not being detected**
Lower the confidence threshold (`--confidence 0.2`) or try a larger model for better accuracy.

**ModuleNotFoundError**
Make sure your virtual environment is activated and you've run `pip install ultralytics opencv-python numpy`.

---

## License

This project is provided for educational and research purposes. The YOLOv8 model is subject to [Ultralytics' license terms](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

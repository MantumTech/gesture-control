# Gesture-Controlled Mouse

Control your computer's mouse cursor using hand gestures and your webcam. No special hardware needed — just a laptop with a camera.

## Features

- **Move cursor** — point with your index finger
- **Left click** — pinch thumb and index finger
- **Right click** — pinch thumb, index, and middle finger
- **Scroll** — open hand gesture
- **Zoom** — two-hand pinch
- **Horizontal scroll** — hand roll gesture
- Works on **macOS, Windows, and Linux**

## Quick Start

```bash
# Clone the repo
git clone https://github.com/MantumTech/gesture-control.git
cd gesture-control

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run it
python gesture_mouse.py
```

Press `q` in the webcam window to quit. Move your mouse to any screen corner as an emergency stop.

## Requirements

- Python 3.8+
- Webcam (built-in or USB)
- macOS, Windows, or Linux

## Learn to Build Your Own

Want to understand how it works and build your own version? Check out the **[Build Guide](https://mantumtech.github.io/gesture-control/guide.html)** — a step-by-step project walkthrough covering computer vision, hand tracking, and mouse automation.

## Safety

- **Always keep the fail-safe enabled** — moving your mouse to any screen corner will stop the program
- Your webcam feed is processed locally and never transmitted anywhere
- Close the app when not in use

## License

MIT License — free to use, modify, and share.

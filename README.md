# Gesture-Controlled Mouse

Control your computer's mouse cursor using hand gestures and your webcam. No special hardware needed — just a laptop with a camera.

## Features

- Multiple gesture modes
- Left click, right click, double click, drag
- Scroll, horizontal scroll, and zoom
- Configurable settings via JSON
- Built-in calibration wizard
- Debug overlay for tuning
- Works on **macOS, Windows, and Linux**

## How to Use

The app has two gesture modes. You can switch between them in `settings.json` by changing `"interface_mode"` to `"finger"` or `"palm"`.

### Palm Mode (default)

| Gesture | Action |
|---------|--------|
| Open hand | Move cursor (follows palm center) |
| Fist (quick) | Left click |
| Fist (hold) | Drag |
| Peace sign (index + middle up) | Double click |
| Pinch (index + thumb) | Right click |
| Open hand (5 fingers) | Reset anchor |

### Finger Mode

| Gesture | Action |
|---------|--------|
| Point with index finger | Move cursor (follows fingertip) |
| Pinch index + thumb (quick) | Left click |
| Pinch index + thumb (hold) | Drag |
| Two quick pinches | Double click |
| Pinch index + middle + thumb | Right click |
| Both hands index tips together | Zoom in/out |
| Roll hand sideways | Horizontal scroll |
| Open hand (5 fingers) | Reset anchor |

### Controls

- Press `q` in the webcam window to quit
- Move your mouse to any screen corner as an **emergency stop**
- The app includes a calibration wizard on first run to tune sensitivity to your setup

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

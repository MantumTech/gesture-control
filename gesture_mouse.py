#!/usr/bin/env python3
"""
Gesture-Based Mouse Control Application

Uses webcam hand tracking (MediaPipe) to control the mouse cursor via gestures.
Desktop prototype using PyAutoGUI — designed for easy conversion to Raspberry Pi
Zero 2 W USB HID device by flipping RASPBERRY_PI_MODE and swapping MouseOutput internals.

Gestures:
  - Index fingertip position → move cursor
  - Index + thumb pinch → left click (short) / drag (hold)
  - Index + middle + thumb pinch → right click
  - Both hands index tips touching → zoom mode
  - Open hand (all 5 fingers) → reset dynamic anchor
  - Hand roll (rotated like holding apple) → horizontal scroll

Usage:
  python gesture_mouse.py

Requirements:
  pip install opencv-python mediapipe pyautogui
  (tkinter is included with Python from python.org)
"""

import dataclasses
import json
import math
import os
import platform
import queue
import struct
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Flag — flip to True for Raspberry Pi HID mode
# ═══════════════════════════════════════════════════════════════════════════════

RASPBERRY_PI_MODE = False

# ═══════════════════════════════════════════════════════════════════════════════
# Platform Detection
# ═══════════════════════════════════════════════════════════════════════════════

PLATFORM = platform.system()  # "Darwin", "Windows", "Linux"
IS_MACOS = PLATFORM == "Darwin"
IS_WINDOWS = PLATFORM == "Windows"

# ═══════════════════════════════════════════════════════════════════════════════
# Startup Checks
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("ERROR: tkinter is not available.")
    if IS_MACOS:
        print("If you installed Python via Homebrew, tkinter is not included.")
        print("Either install Python from https://www.python.org/downloads/")
        print("or run: brew install python-tk@3.x (match your Python version).")
    else:
        print("Install tkinter for your system:")
        print("  Debian/Ubuntu: sudo apt install python3-tk")
        print("  Fedora: sudo dnf install python3-tkinter")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: MediaPipe not installed. Run: pip install mediapipe")
    sys.exit(1)

import numpy as np

if not RASPBERRY_PI_MODE:
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
    except ImportError:
        print("ERROR: PyAutoGUI not installed. Run: pip install pyautogui")
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Gesture states
STATE_IDLE = "IDLE"
STATE_MOVING = "MOVING"
STATE_LEFT_CLICK = "LEFT_CLICK"
STATE_DRAGGING = "DRAGGING"
STATE_RIGHT_CLICK = "RIGHT_CLICK"
STATE_ANCHOR_RESET = "ANCHOR_RESET"
STATE_ZOOMING = "ZOOMING"
STATE_HSCROLL = "H_SCROLL"

# Debug overlay colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_ORANGE = (0, 165, 255)

# GUI colors
BG_COLOR = "#1e1e1e"
FG_COLOR = "#d4d4d4"
ACCENT_COLOR = "#007acc"
ENTRY_BG = "#2d2d2d"
BUTTON_BG = "#333333"
BUTTON_ACTIVE_BG = "#444444"
SLIDER_TROUGH = "#404040"
SECTION_BG = "#252525"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class GestureResult:
    """Output from the gesture state machine for a single frame."""
    state: str = STATE_IDLE
    cursor_dx: float = 0.0
    cursor_dy: float = 0.0
    cursor_abs_x: Optional[int] = None
    cursor_abs_y: Optional[int] = None
    left_click: bool = False
    right_click: bool = False
    scroll_v: int = 0
    scroll_h: int = 0
    zoom_amount: int = 0
    debug_info: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class FrameResult:
    """Data passed from camera thread to GUI thread each frame."""
    fps: float = 0.0
    gesture_name: str = STATE_IDLE
    frame: Optional[np.ndarray] = None
    debug_info: Dict[str, Any] = dataclasses.field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# SettingsManager
# ═══════════════════════════════════════════════════════════════════════════════

class SettingsManager:
    """Manages persistent user settings with JSON serialization."""

    SETTINGS_FILE = "settings.json"

    DEFAULTS = {
        "tracking_mode": "static",  # "static" (trackpad), "dynamic" (joystick), "framed" (4-corner calibration)
        "interface_mode": "finger",  # "finger" (original) or "palm" (fist/peace)
        "smoothing_factor": 0.3,
        "pinch_threshold": 0.05,
        "scroll_sensitivity": 5.0,
        "hscroll_sensitivity": 3.0,
        "click_cooldown_ms": 400,
        "gesture_toggles": {
            "left_click": True,
            "right_click": True,
            "zoom": True,
            "horizontal_scroll": True,
        },
        "show_debug_overlay": True,
    }

    # Validation ranges
    RANGES = {
        "smoothing_factor": (0.05, 0.95),
        "pinch_threshold": (0.01, 0.15),
        "scroll_sensitivity": (1.0, 20.0),
        "hscroll_sensitivity": (1.0, 20.0),
        "click_cooldown_ms": (100, 1500),
    }

    def __init__(self, settings_dir: str):
        self._file_path = os.path.join(settings_dir, self.SETTINGS_FILE)
        self._settings: Dict[str, Any] = {}
        self._deep_copy_defaults()
        self.load()

    def _deep_copy_defaults(self):
        """Deep copy defaults to avoid shared mutable state."""
        self._settings = json.loads(json.dumps(self.DEFAULTS))

    def load(self) -> None:
        """Load settings from JSON file, falling back to defaults for invalid values."""
        if not os.path.exists(self._file_path):
            return
        try:
            with open(self._file_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            self._apply_validated(data)
        except (json.JSONDecodeError, OSError):
            pass  # Keep defaults

    def _apply_validated(self, data: dict) -> None:
        """Apply loaded data with validation, keeping defaults for invalid values."""
        if "tracking_mode" in data and data["tracking_mode"] in ("static", "dynamic", "framed"):
            self._settings["tracking_mode"] = data["tracking_mode"]

        if "interface_mode" in data and data["interface_mode"] in ("finger", "palm"):
            self._settings["interface_mode"] = data["interface_mode"]

        for key in ("smoothing_factor", "pinch_threshold", "scroll_sensitivity",
                     "hscroll_sensitivity", "click_cooldown_ms"):
            if key in data:
                try:
                    val = float(data[key])
                    lo, hi = self.RANGES[key]
                    self._settings[key] = max(lo, min(hi, val))
                except (TypeError, ValueError):
                    pass

        if "gesture_toggles" in data and isinstance(data["gesture_toggles"], dict):
            for gkey in self.DEFAULTS["gesture_toggles"]:
                if gkey in data["gesture_toggles"]:
                    val = data["gesture_toggles"][gkey]
                    if isinstance(val, bool):
                        self._settings["gesture_toggles"][gkey] = val

        if "show_debug_overlay" in data and isinstance(data["show_debug_overlay"], bool):
            self._settings["show_debug_overlay"] = data["show_debug_overlay"]

    def save(self) -> None:
        """Atomically save settings to JSON file."""
        dir_path = os.path.dirname(self._file_path)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(self._settings, f, indent=2)
            os.replace(tmp_path, self._file_path)
        except OSError as e:
            print(f"Warning: Could not save settings: {e}")

    def get(self, key: str) -> Any:
        """Get a setting value."""
        return self._settings.get(key)

    def get_toggle(self, gesture: str) -> bool:
        """Get whether a specific gesture is enabled."""
        return self._settings.get("gesture_toggles", {}).get(gesture, True)

    def set(self, key: str, value: Any) -> None:
        """Set a setting value with validation and auto-save."""
        if key in self.RANGES:
            try:
                value = float(value)
                lo, hi = self.RANGES[key]
                value = max(lo, min(hi, value))
            except (TypeError, ValueError):
                return
        self._settings[key] = value
        self.save()

    def set_toggle(self, gesture: str, enabled: bool) -> None:
        """Set whether a specific gesture is enabled."""
        if gesture in self._settings.get("gesture_toggles", {}):
            self._settings["gesture_toggles"][gesture] = bool(enabled)
            self.save()


# ═══════════════════════════════════════════════════════════════════════════════
# MouseOutput
# ═══════════════════════════════════════════════════════════════════════════════

class MouseOutput:
    """Abstraction layer for all mouse output.

    Desktop mode uses PyAutoGUI. Raspberry Pi mode writes HID reports
    to /dev/hidg0. Swap internals here for Pi conversion — no other
    code needs to change.
    """

    def __init__(self, pi_mode: bool):
        self._pi_mode = pi_mode
        self._hid_fd = None

        if not pi_mode:
            self._pyautogui = pyautogui

    def move_relative(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        if self._pi_mode:
            dx = max(-127, min(127, int(dx)))
            dy = max(-127, min(127, int(dy)))
            self._hid_write(struct.pack("bbbb", 0, dx, dy, 0))
        else:
            try:
                self._pyautogui.moveRel(dx, dy, _pause=False)
            except Exception:
                pass

    def move_absolute(self, x: int, y: int) -> None:
        if self._pi_mode:
            # Basic HID mouse uses relative — absolute requires
            # a digitizer descriptor. Stub for future implementation.
            pass
        else:
            try:
                self._pyautogui.moveTo(x, y, _pause=False)
            except Exception:
                pass

    def left_click(self) -> None:
        if self._pi_mode:
            self._hid_write(struct.pack("bbbb", 0x01, 0, 0, 0))
            time.sleep(0.05)
            self._hid_write(struct.pack("bbbb", 0x00, 0, 0, 0))
        else:
            try:
                self._pyautogui.click(_pause=False)
            except Exception:
                pass

    def double_click(self) -> None:
        if self._pi_mode:
            for _ in range(2):
                self._hid_write(struct.pack("bbbb", 0x01, 0, 0, 0))
                time.sleep(0.05)
                self._hid_write(struct.pack("bbbb", 0x00, 0, 0, 0))
                time.sleep(0.05)
        elif IS_MACOS:
            try:
                # Use native Quartz CGEvents for proper double-click on macOS
                from Quartz.CoreGraphics import (
                    CGEventCreateMouseEvent, CGEventPost, CGEventSetIntegerValueField,
                    kCGEventLeftMouseDown, kCGEventLeftMouseUp,
                    kCGMouseButtonLeft, kCGHIDEventTap,
                    kCGMouseEventClickState,
                )
                from Quartz.CoreGraphics import CGPointMake
                x, y = self._pyautogui.position()
                point = CGPointMake(float(x), float(y))

                # First click (clickState=1)
                down1 = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, point, kCGMouseButtonLeft)
                CGEventSetIntegerValueField(down1, kCGMouseEventClickState, 1)
                CGEventPost(kCGHIDEventTap, down1)
                up1 = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, point, kCGMouseButtonLeft)
                CGEventSetIntegerValueField(up1, kCGMouseEventClickState, 1)
                CGEventPost(kCGHIDEventTap, up1)

                time.sleep(0.05)

                # Second click (clickState=2 tells macOS this is a double-click)
                down2 = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, point, kCGMouseButtonLeft)
                CGEventSetIntegerValueField(down2, kCGMouseEventClickState, 2)
                CGEventPost(kCGHIDEventTap, down2)
                up2 = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, point, kCGMouseButtonLeft)
                CGEventSetIntegerValueField(up2, kCGMouseEventClickState, 2)
                CGEventPost(kCGHIDEventTap, up2)
            except Exception:
                # Fallback to pyautogui
                try:
                    x, y = self._pyautogui.position()
                    self._pyautogui.doubleClick(x=x, y=y, interval=0.1, _pause=False)
                except Exception:
                    pass
        else:
            try:
                x, y = self._pyautogui.position()
                self._pyautogui.doubleClick(x=x, y=y, interval=0.1, _pause=False)
            except Exception:
                pass

    def mouse_down(self) -> None:
        if self._pi_mode:
            self._hid_write(struct.pack("bbbb", 0x01, 0, 0, 0))
        else:
            try:
                self._pyautogui.mouseDown(_pause=False)
            except Exception:
                pass

    def mouse_up(self) -> None:
        if self._pi_mode:
            self._hid_write(struct.pack("bbbb", 0x00, 0, 0, 0))
        else:
            try:
                self._pyautogui.mouseUp(_pause=False)
            except Exception:
                pass

    def right_click(self) -> None:
        if self._pi_mode:
            self._hid_write(struct.pack("bbbb", 0x02, 0, 0, 0))
            time.sleep(0.05)
            self._hid_write(struct.pack("bbbb", 0x00, 0, 0, 0))
        else:
            try:
                self._pyautogui.rightClick(_pause=False)
            except Exception:
                pass

    def scroll_vertical(self, amount: int) -> None:
        if amount == 0:
            return
        if self._pi_mode:
            amount = max(-127, min(127, int(amount)))
            self._hid_write(struct.pack("bbbb", 0, 0, 0, amount))
        else:
            scroll_amount = int(amount)
            if IS_MACOS:
                scroll_amount = -scroll_amount
            try:
                self._pyautogui.scroll(scroll_amount, _pause=False)
            except Exception:
                pass

    def scroll_horizontal(self, amount: int) -> None:
        if amount == 0:
            return
        if self._pi_mode:
            # Extended HID report needed for horizontal scroll — stub
            pass
        else:
            try:
                self._pyautogui.hscroll(int(amount), _pause=False)
            except Exception:
                pass

    def zoom(self, amount: int) -> None:
        """Zoom via modifier+scroll. CMD on macOS, CTRL on others."""
        if amount == 0:
            return
        if self._pi_mode:
            # Requires keyboard HID gadget alongside mouse — stub
            pass
        else:
            modifier = "command" if IS_MACOS else "ctrl"
            scroll_amount = int(amount)
            if IS_MACOS:
                scroll_amount = -scroll_amount
            try:
                with self._pyautogui.hold(modifier):
                    self._pyautogui.scroll(scroll_amount, _pause=False)
            except Exception:
                pass

    def get_screen_size(self) -> Tuple[int, int]:
        if self._pi_mode:
            return (1920, 1080)
        try:
            return self._pyautogui.size()
        except Exception:
            return (1920, 1080)

    def _hid_write(self, report: bytes) -> None:
        """Write a raw HID report to /dev/hidg0."""
        try:
            if self._hid_fd is None:
                self._hid_fd = open("/dev/hidg0", "wb", buffering=0)
            self._hid_fd.write(report)
        except PermissionError:
            print("ERROR: Cannot write to /dev/hidg0. Run with sudo or check permissions.")
        except FileNotFoundError:
            print("ERROR: /dev/hidg0 not found. USB gadget mode not configured.")
        except OSError as e:
            print(f"ERROR: HID write failed: {e}")

    def cleanup(self) -> None:
        # Release mouse button in case drag was active
        try:
            self.mouse_up()
        except Exception:
            pass
        if self._hid_fd is not None:
            try:
                self._hid_fd.close()
            except OSError:
                pass
            self._hid_fd = None


# ═══════════════════════════════════════════════════════════════════════════════
# ExponentialSmoother
# ═══════════════════════════════════════════════════════════════════════════════

class ExponentialSmoother:
    """Exponential moving average filter for reducing cursor jitter."""

    def __init__(self):
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None

    def smooth(self, raw_x: float, raw_y: float, alpha: float) -> Tuple[float, float]:
        """Apply exponential smoothing.

        Args:
            raw_x, raw_y: Raw input values.
            alpha: Smoothing factor in (0, 1]. Lower = smoother, higher = more responsive.

        Returns:
            Smoothed (x, y) values.
        """
        if self._prev_x is None:
            self._prev_x = raw_x
            self._prev_y = raw_y
            return (raw_x, raw_y)

        sx = alpha * raw_x + (1.0 - alpha) * self._prev_x
        sy = alpha * raw_y + (1.0 - alpha) * self._prev_y
        self._prev_x = sx
        self._prev_y = sy
        return (sx, sy)

    def reset(self):
        """Reset the filter (call when hand disappears)."""
        self._prev_x = None
        self._prev_y = None


# ═══════════════════════════════════════════════════════════════════════════════
# HandAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class HandAnalyzer:
    """Pure computation on MediaPipe hand landmarks. All static methods."""

    @staticmethod
    def pinch_distance(landmarks, tip_a: int, tip_b: int) -> float:
        """Euclidean distance between two landmarks (including Z depth)."""
        a = landmarks.landmark[tip_a]
        b = landmarks.landmark[tip_b]
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def fingertip_position(landmarks, tip_idx: int) -> Tuple[float, float]:
        """Get normalized (x, y) position of a fingertip."""
        lm = landmarks.landmark[tip_idx]
        return (lm.x, lm.y)

    @staticmethod
    def hand_roll_angle(landmarks) -> float:
        """Calculate hand roll angle using the knuckle vector (INDEX_MCP to PINKY_MCP).

        Returns angle in degrees. ~0 or ~180 for neutral upright hand,
        deviates when hand rotates around the wrist axis.
        """
        idx_mcp = landmarks.landmark[INDEX_MCP]
        pinky_mcp = landmarks.landmark[PINKY_MCP]
        dx = pinky_mcp.x - idx_mcp.x
        dy = pinky_mcp.y - idx_mcp.y
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def is_open_hand(landmarks) -> bool:
        """Check if all fingers are extended (open palm)."""
        lm = landmarks.landmark
        fingers_extended = (
            lm[INDEX_TIP].y < lm[INDEX_MCP].y
            and lm[MIDDLE_TIP].y < lm[MIDDLE_MCP].y
            and lm[RING_TIP].y < lm[RING_MCP].y
            and lm[PINKY_TIP].y < lm[PINKY_MCP].y
        )
        thumb_extended = abs(lm[THUMB_TIP].x - lm[WRIST].x) > abs(
            lm[THUMB_MCP].x - lm[WRIST].x
        )
        return fingers_extended and thumb_extended

    @staticmethod
    def is_four_fingers(landmarks) -> bool:
        """Check if 4 fingers are extended but thumb is tucked in."""
        lm = landmarks.landmark
        fingers_extended = (
            lm[INDEX_TIP].y < lm[INDEX_MCP].y
            and lm[MIDDLE_TIP].y < lm[MIDDLE_MCP].y
            and lm[RING_TIP].y < lm[RING_MCP].y
            and lm[PINKY_TIP].y < lm[PINKY_MCP].y
        )
        thumb_tucked = abs(lm[THUMB_TIP].x - lm[WRIST].x) <= abs(
            lm[THUMB_MCP].x - lm[WRIST].x
        )
        return fingers_extended and thumb_tucked

    @staticmethod
    def is_index_pointing(landmarks) -> bool:
        """Check if index finger is extended while others are curled."""
        lm = landmarks.landmark
        index_up = lm[INDEX_TIP].y < lm[INDEX_PIP].y
        middle_down = lm[MIDDLE_TIP].y > lm[MIDDLE_PIP].y
        ring_down = lm[RING_TIP].y > lm[RING_PIP].y
        pinky_down = lm[PINKY_TIP].y > lm[PINKY_PIP].y
        return index_up and middle_down and ring_down and pinky_down

    @staticmethod
    def palm_center(landmarks) -> Tuple[float, float]:
        """Get the center of the palm as average of wrist and MCP joints."""
        lm = landmarks.landmark
        points = [lm[WRIST], lm[INDEX_MCP], lm[MIDDLE_MCP], lm[RING_MCP], lm[PINKY_MCP]]
        cx = sum(p.x for p in points) / len(points)
        cy = sum(p.y for p in points) / len(points)
        return (cx, cy)

    @staticmethod
    def is_fist(landmarks) -> bool:
        """Check if hand is closed into a fist (all fingertips curled below PIP joints)."""
        lm = landmarks.landmark
        index_curled = lm[INDEX_TIP].y > lm[INDEX_PIP].y
        middle_curled = lm[MIDDLE_TIP].y > lm[MIDDLE_PIP].y
        ring_curled = lm[RING_TIP].y > lm[RING_PIP].y
        pinky_curled = lm[PINKY_TIP].y > lm[PINKY_PIP].y
        thumb_curled = (
            abs(lm[THUMB_TIP].x - lm[WRIST].x)
            < abs(lm[THUMB_MCP].x - lm[WRIST].x) * 1.2
        )
        return index_curled and middle_curled and ring_curled and pinky_curled and thumb_curled

    @staticmethod
    def is_peace_sign(landmarks) -> bool:
        """Check if two fingers are up (index + middle extended) with others not fully extended.

        Uses a relaxed check: index and middle must be clearly above MCP,
        ring and pinky just need to NOT be as extended (tip above PIP is ok, just not above MCP).
        """
        lm = landmarks.landmark
        index_up = lm[INDEX_TIP].y < lm[INDEX_MCP].y
        middle_up = lm[MIDDLE_TIP].y < lm[MIDDLE_MCP].y
        # Ring and pinky: just need to not be fully extended (allow partial curl)
        ring_not_up = lm[RING_TIP].y > lm[RING_PIP].y or lm[RING_TIP].y > lm[RING_MCP].y
        pinky_not_up = lm[PINKY_TIP].y > lm[PINKY_PIP].y or lm[PINKY_TIP].y > lm[PINKY_MCP].y
        return index_up and middle_up and ring_not_up and pinky_not_up

    @staticmethod
    def wrist_y(landmarks) -> float:
        """Get the Y coordinate of the wrist landmark."""
        return landmarks.landmark[WRIST].y

    @staticmethod
    def two_hand_index_distance(hand1_landmarks, hand2_landmarks) -> float:
        """Distance between index fingertips of two hands."""
        tip1 = hand1_landmarks.landmark[INDEX_TIP]
        tip2 = hand2_landmarks.landmark[INDEX_TIP]
        dx = tip1.x - tip2.x
        dy = tip1.y - tip2.y
        return math.sqrt(dx * dx + dy * dy)


# ═══════════════════════════════════════════════════════════════════════════════
# DynamicAnchorTracker
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicAnchorTracker:
    """Relative cursor movement — behaves like a trackpad.

    When the index finger first appears, that position becomes the anchor.
    Moving from the anchor moves the cursor. Lifting and re-pointing resets.
    """

    def __init__(self):
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None
        self._smoother = ExponentialSmoother()
        self._sensitivity = 2.5

    def update(
        self,
        index_tip_x: float,
        index_tip_y: float,
        hand_visible: bool,
        screen_w: int,
        screen_h: int,
        smoothing_alpha: float,
    ) -> Tuple[float, float]:
        """Compute relative cursor movement (dx, dy) in screen pixels."""
        if not hand_visible:
            self._prev_x = None
            self._prev_y = None
            self._smoother.reset()
            return (0.0, 0.0)

        if self._prev_x is None:
            self._prev_x = index_tip_x
            self._prev_y = index_tip_y
            return (0.0, 0.0)

        raw_dx = (index_tip_x - self._prev_x) * screen_w * self._sensitivity
        raw_dy = (index_tip_y - self._prev_y) * screen_h * self._sensitivity
        self._prev_x = index_tip_x
        self._prev_y = index_tip_y

        sx, sy = self._smoother.smooth(raw_dx, raw_dy, smoothing_alpha)
        return (sx, sy)

    def reset(self):
        """Force an anchor reset."""
        self._prev_x = None
        self._prev_y = None
        self._smoother.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# JoystickAnchorTracker (Dynamic Anchor)
# ═══════════════════════════════════════════════════════════════════════════════

class JoystickAnchorTracker:
    """Joystick-style cursor movement — distance from anchor controls speed.

    The anchor is set when the hand first appears (or manually reset).
    Moving the hand away from the anchor moves the cursor in that direction.
    The farther from the anchor, the faster the cursor moves.
    Returning to within the dead zone stops the cursor.
    """

    def __init__(self):
        self._anchor_x: Optional[float] = None
        self._anchor_y: Optional[float] = None
        self._smoother = ExponentialSmoother()
        self._speed_scale = 30.0    # Max cursor speed multiplier
        self._dead_zone = 0.02      # Normalized distance within which cursor doesn't move
        self._max_radius = 0.15     # Distance at which speed is maxed out
        self._exponent = 2.5        # Exponential curve: >1 = slow near center, fast at edges

    def update(
        self,
        hand_x: float,
        hand_y: float,
        hand_visible: bool,
        screen_w: int,
        screen_h: int,
        smoothing_alpha: float,
    ) -> Tuple[float, float]:
        """Compute cursor movement based on distance and direction from anchor."""
        if not hand_visible:
            self._smoother.reset()
            return (0.0, 0.0)

        if self._anchor_x is None:
            self._anchor_x = hand_x
            self._anchor_y = hand_y
            return (0.0, 0.0)

        dx = hand_x - self._anchor_x
        dy = hand_y - self._anchor_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self._dead_zone:
            # Inside dead zone — no movement, let smoother decay
            sx, sy = self._smoother.smooth(0.0, 0.0, smoothing_alpha)
            return (sx, sy)

        # Normalize direction
        dir_x = dx / dist
        dir_y = dy / dist

        # Exponential speed curve: slow near center, accelerates toward edges
        effective_dist = min(dist - self._dead_zone, self._max_radius - self._dead_zone)
        linear_factor = effective_dist / (self._max_radius - self._dead_zone)  # 0.0 to 1.0
        speed_factor = linear_factor ** self._exponent  # Exponential: e.g. 0.5 -> 0.18 at exp=2.5

        raw_dx = dir_x * speed_factor * self._speed_scale
        raw_dy = dir_y * speed_factor * self._speed_scale

        sx, sy = self._smoother.smooth(raw_dx, raw_dy, smoothing_alpha)
        return (sx, sy)

    def reset(self):
        """Reset the anchor to be set on next update."""
        self._anchor_x = None
        self._anchor_y = None
        self._smoother.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# CalibrationWizard
# ═══════════════════════════════════════════════════════════════════════════════

class CalibrationWizard:
    """4-corner calibration for static anchor mode with perspective transform."""

    CALIBRATION_FILE = "calibration.json"
    CORNERS = ["TOP LEFT", "TOP RIGHT", "BOTTOM RIGHT", "BOTTOM LEFT"]
    COUNTDOWN_SECONDS = 3.0

    def __init__(self, settings_dir: str):
        self._file_path = os.path.join(settings_dir, self.CALIBRATION_FILE)
        self.active = False
        self.step = 0
        self.points: List[Tuple[float, float]] = []
        self.countdown: float = 0.0
        self.countdown_active = False
        self.transform_matrix: Optional[np.ndarray] = None
        self._screen_size: Tuple[int, int] = (1920, 1080)
        self._smoother = ExponentialSmoother()

    def start(self, screen_size: Tuple[int, int]) -> None:
        """Begin the calibration wizard."""
        self.active = True
        self.step = 0
        self.points = []
        self.countdown = 0.0
        self.countdown_active = False
        self._screen_size = screen_size
        self._smoother.reset()

    def update(
        self, index_tip_pos: Optional[Tuple[float, float]], pinch_detected: bool, dt: float
    ) -> str:
        """Update calibration state. Returns status message for overlay."""
        if not self.active:
            return ""

        corner_name = self.CORNERS[self.step]

        if index_tip_pos is None:
            self.countdown_active = False
            self.countdown = 0.0
            return f"Point at {corner_name} corner"

        if not self.countdown_active:
            if pinch_detected:
                self.countdown_active = True
                self.countdown = self.COUNTDOWN_SECONDS
            return f"Point at {corner_name} — pinch to start countdown"

        self.countdown -= dt
        if self.countdown > 0:
            return f"Capturing {corner_name} in {self.countdown:.1f}s — hold steady"

        # Countdown finished — capture this point
        self.points.append(index_tip_pos)
        self.countdown_active = False
        self.countdown = 0.0
        self.step += 1

        if self.step >= 4:
            self._compute_transform()
            self.save()
            self.active = False
            return "Calibration complete!"

        return f"Captured! Next: {self.CORNERS[self.step]}"

    def _compute_transform(self) -> None:
        """Compute perspective transform from 4 calibration points."""
        src = np.array(self.points, dtype=np.float32)
        sw, sh = self._screen_size
        dst = np.array(
            [[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32
        )
        self.transform_matrix = cv2.getPerspectiveTransform(src, dst)

    def transform_point(self, camera_x: float, camera_y: float) -> Tuple[int, int]:
        """Map a camera-normalized point to screen coordinates."""
        if self.transform_matrix is None:
            return (0, 0)
        point = np.array([[[camera_x, camera_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.transform_matrix)
        sw, sh = self._screen_size
        sx = int(np.clip(transformed[0][0][0], 0, sw - 1))
        sy = int(np.clip(transformed[0][0][1], 0, sh - 1))
        return (sx, sy)

    def save(self) -> None:
        """Save calibration data to JSON."""
        if self.transform_matrix is None:
            return
        data = {
            "camera_points": [list(p) for p in self.points],
            "screen_size": list(self._screen_size),
            "transform_matrix": self.transform_matrix.tolist(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        dir_path = os.path.dirname(self._file_path)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self._file_path)
        except OSError as e:
            print(f"Warning: Could not save calibration: {e}")

    def load(self) -> bool:
        """Load calibration data from JSON. Returns True if successful."""
        if not os.path.exists(self._file_path):
            return False
        try:
            with open(self._file_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return False

            points = data.get("camera_points")
            screen = data.get("screen_size")
            matrix = data.get("transform_matrix")

            if not (isinstance(points, list) and len(points) == 4):
                return False
            if not (isinstance(screen, list) and len(screen) == 2):
                return False
            if not (isinstance(matrix, list) and len(matrix) == 3):
                return False

            self.points = [tuple(p) for p in points]
            self._screen_size = tuple(screen)
            self.transform_matrix = np.array(matrix, dtype=np.float64)
            return True
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# GestureCalibrationWizard
# ═══════════════════════════════════════════════════════════════════════════════

class GestureCalibrationWizard:
    """Interactive calibration exercises to tune gesture thresholds to the user's hand.

    Steps:
      1. REST BASELINE — hold hand open and still for 3 seconds
      2. PINCH CALIBRATION — pinch index+thumb together 5 times
      3. DOUBLE-CLICK — do two quick pinches 5 times
      4. DRAG — pinch and hold for 2 seconds, 3 times
      5. MOVEMENT RANGE — move hand to edges of comfortable range
    """

    CALIB_FILE = "gesture_calibration.json"

    # Exercise definitions
    STEP_REST = 0
    STEP_PINCH = 1
    STEP_DOUBLE_CLICK = 2
    STEP_DRAG = 3
    STEP_MOVEMENT = 4
    STEP_DONE = 5

    STEP_NAMES = [
        "Rest Baseline",
        "Pinch Calibration",
        "Double-Click Timing",
        "Drag Calibration",
        "Movement Range",
        "Complete",
    ]

    def __init__(self, settings_dir: str):
        self._file_path = os.path.join(settings_dir, self.CALIB_FILE)
        self.active = False
        self.step = 0
        self._reset_exercise_state()

    def _reset_exercise_state(self):
        """Reset all per-exercise tracking."""
        # Rest baseline
        self._rest_samples: List[float] = []  # pinch distances while hand is open
        self._rest_timer = 0.0
        self._rest_duration = 3.0

        # Pinch calibration
        self._pinch_distances: List[float] = []  # pinch distances during pinch
        self._pinch_count = 0
        self._pinch_target = 5
        self._was_pinched = False
        self._open_distances: List[float] = []  # distances when hand open (between pinches)

        # Double-click timing
        self._dc_count = 0
        self._dc_target = 5
        self._dc_was_pinched = False
        self._dc_first_pinch_time = 0.0
        self._dc_has_first = False
        self._dc_intervals: List[float] = []

        # Drag calibration
        self._drag_count = 0
        self._drag_target = 3
        self._drag_was_pinched = False
        self._drag_hold_start = 0.0
        self._drag_hold_times: List[float] = []

        # Movement range
        self._move_timer = 0.0
        self._move_duration = 5.0
        self._move_positions: List[Tuple[float, float]] = []

        # Results
        self._results: Dict[str, Any] = {}

    def start(self):
        """Begin the gesture calibration wizard."""
        self.active = True
        self.step = self.STEP_REST
        self._reset_exercise_state()

    def update(self, hand_landmarks, dt: float) -> str:
        """Process one frame during calibration. Returns status message for overlay."""
        if not self.active:
            return ""

        step_num = self.step + 1
        total = self.STEP_DONE
        prefix = f"[{step_num}/{total}] "

        if self.step == self.STEP_REST:
            return prefix + self._update_rest(hand_landmarks, dt)
        elif self.step == self.STEP_PINCH:
            return prefix + self._update_pinch(hand_landmarks, dt)
        elif self.step == self.STEP_DOUBLE_CLICK:
            return prefix + self._update_double_click(hand_landmarks, dt)
        elif self.step == self.STEP_DRAG:
            return prefix + self._update_drag(hand_landmarks, dt)
        elif self.step == self.STEP_MOVEMENT:
            return prefix + self._update_movement(hand_landmarks, dt)
        else:
            self.active = False
            return "Calibration complete! Settings applied."

    def _update_rest(self, hand_landmarks, dt: float) -> str:
        """Step 1: Hold hand open and still to capture baseline distances."""
        if hand_landmarks is None:
            self._rest_timer = 0.0
            self._rest_samples.clear()
            return "REST BASELINE: Show your open hand to the camera"

        if not HandAnalyzer.is_open_hand(hand_landmarks):
            self._rest_timer = 0.0
            self._rest_samples.clear()
            return "REST BASELINE: Open all fingers wide"

        self._rest_timer += dt
        dist = HandAnalyzer.pinch_distance(hand_landmarks, THUMB_TIP, INDEX_TIP)
        self._rest_samples.append(dist)

        remaining = max(0, self._rest_duration - self._rest_timer)
        if remaining > 0:
            return f"REST BASELINE: Hold still... {remaining:.1f}s"

        # Done — compute baseline
        if self._rest_samples:
            self._results["rest_avg_dist"] = sum(self._rest_samples) / len(self._rest_samples)
            self._results["rest_min_dist"] = min(self._rest_samples)
        self.step = self.STEP_PINCH
        return "Baseline captured! Next: Pinch Exercise"

    def _update_pinch(self, hand_landmarks, dt: float) -> str:
        """Step 2: Pinch index+thumb together 5 times to measure pinch distance."""
        if hand_landmarks is None:
            return f"PINCH ({self._pinch_count}/{self._pinch_target}): Show your hand"

        dist = HandAnalyzer.pinch_distance(hand_landmarks, THUMB_TIP, INDEX_TIP)
        rest_avg = self._results.get("rest_avg_dist", 0.15)

        # Use a generous threshold for detecting pinch during calibration
        is_pinched = dist < (rest_avg * 0.5)

        if is_pinched and not self._was_pinched:
            # Pinch just started — record the distance
            self._pinch_distances.append(dist)
            self._pinch_count += 1
            self._was_pinched = True
        elif not is_pinched and self._was_pinched:
            # Released — record open distance
            self._open_distances.append(dist)
            self._was_pinched = False

        if self._pinch_count >= self._pinch_target:
            if self._pinch_distances:
                avg_pinch = sum(self._pinch_distances) / len(self._pinch_distances)
                max_pinch = max(self._pinch_distances)
                self._results["avg_pinch_dist"] = avg_pinch
                self._results["max_pinch_dist"] = max_pinch
                if self._open_distances:
                    min_open = min(self._open_distances)
                    # Threshold = midpoint between max pinch distance and min open distance
                    self._results["pinch_threshold"] = (max_pinch + min_open) / 2.0
                else:
                    self._results["pinch_threshold"] = max_pinch * 1.3
            self.step = self.STEP_DOUBLE_CLICK
            return "Pinch calibrated! Next: Double-Click Exercise"

        status = "PINCH" if is_pinched else "OPEN"
        return f"PINCH ({self._pinch_count}/{self._pinch_target}): Pinch index+thumb together then release [{status}]"

    def _update_double_click(self, hand_landmarks, dt: float) -> str:
        """Step 3: Double-pinch 5 times to calibrate double-click timing window."""
        if hand_landmarks is None:
            return f"DOUBLE-CLICK ({self._dc_count}/{self._dc_target}): Show your hand"

        rest_avg = self._results.get("rest_avg_dist", 0.15)
        threshold = self._results.get("pinch_threshold", rest_avg * 0.5)
        dist = HandAnalyzer.pinch_distance(hand_landmarks, THUMB_TIP, INDEX_TIP)
        is_pinched = dist < threshold

        now = time.time()

        if is_pinched and not self._dc_was_pinched:
            # Pinch just started
            if not self._dc_has_first:
                # First pinch of the pair
                self._dc_first_pinch_time = now
                self._dc_has_first = True
            else:
                # Second pinch — record interval
                interval = now - self._dc_first_pinch_time
                if interval < 2.0:  # Sanity check: must be under 2 seconds
                    self._dc_intervals.append(interval)
                    self._dc_count += 1
                self._dc_has_first = False
            self._dc_was_pinched = True
        elif not is_pinched and self._dc_was_pinched:
            self._dc_was_pinched = False

        # Reset if too long since first pinch
        if self._dc_has_first and (now - self._dc_first_pinch_time) > 2.0:
            self._dc_has_first = False

        if self._dc_count >= self._dc_target:
            if self._dc_intervals:
                avg_interval = sum(self._dc_intervals) / len(self._dc_intervals)
                max_interval = max(self._dc_intervals)
                # Set window to max interval + 30% margin
                self._results["double_click_window"] = min(max_interval * 1.3, 1.0)
                self._results["avg_dc_interval"] = avg_interval
            self.step = self.STEP_DRAG
            return "Double-click calibrated! Next: Drag Exercise"

        waiting = " (now pinch AGAIN quickly!)" if self._dc_has_first else ""
        return f"DOUBLE-CLICK ({self._dc_count}/{self._dc_target}): Pinch twice quickly{waiting}"

    def _update_drag(self, hand_landmarks, dt: float) -> str:
        """Step 4: Pinch and hold for ~2 seconds, 3 times, to calibrate drag threshold."""
        if hand_landmarks is None:
            return f"DRAG ({self._drag_count}/{self._drag_target}): Show your hand"

        rest_avg = self._results.get("rest_avg_dist", 0.15)
        threshold = self._results.get("pinch_threshold", rest_avg * 0.5)
        dist = HandAnalyzer.pinch_distance(hand_landmarks, THUMB_TIP, INDEX_TIP)
        is_pinched = dist < threshold

        if is_pinched:
            if not self._drag_was_pinched:
                # Pinch just started
                self._drag_hold_start = time.time()
                self._drag_was_pinched = True
            hold_time = time.time() - self._drag_hold_start
            return f"DRAG ({self._drag_count}/{self._drag_target}): Holding... {hold_time:.1f}s (hold for ~2s)"
        else:
            if self._drag_was_pinched:
                # Just released — record hold time
                hold_time = time.time() - self._drag_hold_start
                if hold_time >= 0.3:  # Must be at least 300ms to count
                    self._drag_hold_times.append(hold_time)
                    self._drag_count += 1
                self._drag_was_pinched = False

        if self._drag_count >= self._drag_target:
            if self._drag_hold_times:
                # Drag threshold = quickest intentional hold * 0.5 (so it triggers before they expect)
                min_hold = min(self._drag_hold_times)
                self._results["drag_threshold"] = max(0.1, min(min_hold * 0.5, 0.3))
            self.step = self.STEP_MOVEMENT
            return "Drag calibrated! Next: Movement Range"

        return f"DRAG ({self._drag_count}/{self._drag_target}): Pinch and HOLD for ~2 seconds, then release"

    def _update_movement(self, hand_landmarks, dt: float) -> str:
        """Step 5: Move hand around comfortable range to calibrate sensitivity."""
        if hand_landmarks is None:
            self._move_timer = 0.0
            self._move_positions.clear()
            return "MOVEMENT: Show your hand and move it around"

        self._move_timer += dt
        thumb_pos = HandAnalyzer.fingertip_position(hand_landmarks, THUMB_TIP)
        self._move_positions.append(thumb_pos)

        remaining = max(0, self._move_duration - self._move_timer)
        if remaining > 0:
            return f"MOVEMENT: Move your hand to all edges of your comfort zone... {remaining:.1f}s"

        # Compute range
        if len(self._move_positions) > 10:
            xs = [p[0] for p in self._move_positions]
            ys = [p[1] for p in self._move_positions]
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            self._results["movement_x_range"] = x_range
            self._results["movement_y_range"] = y_range
            # Sensitivity: smaller range = need more sensitivity
            # Target: a full sweep maps to ~80% of screen
            avg_range = (x_range + y_range) / 2.0
            if avg_range > 0.01:
                self._results["sensitivity"] = min(max(0.8 / avg_range, 1.5), 5.0)

        self.step = self.STEP_DONE
        return "Movement calibrated!"

    def apply_to_settings(self, settings: SettingsManager) -> Dict[str, Any]:
        """Apply calibrated values to settings. Returns dict of what was changed."""
        applied = {}

        if "pinch_threshold" in self._results:
            val = self._results["pinch_threshold"]
            lo, hi = SettingsManager.RANGES["pinch_threshold"]
            val = max(lo, min(hi, val))
            settings.set("pinch_threshold", val)
            applied["pinch_threshold"] = val

        if "double_click_window" in self._results:
            applied["double_click_window"] = self._results["double_click_window"]

        if "drag_threshold" in self._results:
            applied["drag_threshold"] = self._results["drag_threshold"]

        if "sensitivity" in self._results:
            applied["sensitivity"] = self._results["sensitivity"]

        # Save calibration data to file
        self._save()
        return applied

    def get_results(self) -> Dict[str, Any]:
        """Get raw calibration results."""
        return dict(self._results)

    def _save(self) -> None:
        """Save calibration results to JSON."""
        data = {
            "results": self._results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        dir_path = os.path.dirname(self._file_path)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self._file_path)
        except OSError as e:
            print(f"Warning: Could not save gesture calibration: {e}")

    def load(self) -> bool:
        """Load previous calibration results. Returns True if successful."""
        if not os.path.exists(self._file_path):
            return False
        try:
            with open(self._file_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                self._results = data["results"]
                return True
        except (json.JSONDecodeError, OSError):
            pass
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# GestureStateMachine
# ═══════════════════════════════════════════════════════════════════════════════

class GestureStateMachine:
    """Finite state machine for gesture recognition. One gesture at a time."""

    # Hysteresis multiplier for exiting a pinch state
    PINCH_EXIT_MULT = 1.3
    # Thresholds for scroll detection
    # Thresholds for horizontal scroll (hand roll)
    ROLL_ENTER_DEG = 35.0
    ROLL_EXIT_DEG = 20.0
    ROLL_NEUTRAL_RIGHT = 0.0    # Neutral angle for right hand
    ROLL_NEUTRAL_LEFT = 180.0   # Neutral angle for left hand
    # Zoom thresholds
    ZOOM_ENTER_DIST = 0.35       # Hands must be clearly separated to enter zoom
    ZOOM_EXIT_DIST = 0.8
    ZOOM_CONFIRM_FRAMES = 5      # Require N consecutive frames of 2 hands before zoom

    def __init__(self, settings: SettingsManager, mouse_output: MouseOutput,
                 calibration: CalibrationWizard):
        self._settings = settings
        self._mouse = mouse_output
        self._calibration = calibration
        self.state = STATE_IDLE
        self._click_cooldown = 0.0
        self._zoom_prev_dist: Optional[float] = None
        self._two_hand_frames = 0
        self._static_tracker = DynamicAnchorTracker()   # "static" = trackpad-style
        self._dynamic_tracker = JoystickAnchorTracker()  # "dynamic" = joystick-style
        self._framed_smoother = ExponentialSmoother()    # "framed" = 4-corner calibration
        self._pinch_was_active = False
        self._right_pinch_was_active = False
        self._drag_active = False
        self._pinch_hold_time = 0.0
        self._DRAG_THRESHOLD_SEC = 0.15  # Hold pinch this long to start drag
        self._last_click_time = 0.0
        self._DOUBLE_CLICK_WINDOW = 0.4  # Second pinch within this window = double click
        # Release confirmation: require consecutive exit frames before releasing
        self._release_confirm_frames = 0
        self._RELEASE_CONFIRM_REQUIRED = 4  # Must see 4 consecutive "released" frames
        self._hand_lost_frames = 0
        self._HAND_LOST_GRACE = 8  # Allow hand to disappear for up to 8 frames during click/drag

    def update(self, hands_data: list, handedness_data: list, dt: float) -> GestureResult:
        """Process one frame of hand data and return the gesture result."""
        result = GestureResult()
        result.debug_info = {}

        # Decrease click cooldown
        if self._click_cooldown > 0:
            self._click_cooldown -= dt

        num_hands = len(hands_data)
        screen_w, screen_h = self._mouse.get_screen_size()
        pinch_threshold = self._settings.get("pinch_threshold")
        smoothing = self._settings.get("smoothing_factor")
        scroll_sens = self._settings.get("scroll_sensitivity")
        hscroll_sens = self._settings.get("hscroll_sensitivity")

        # No hands visible
        if num_hands == 0:
            if self.state in (STATE_LEFT_CLICK, STATE_DRAGGING):
                # Grace period: don't release immediately if hand tracking drops
                self._hand_lost_frames += 1
                if self._hand_lost_frames < self._HAND_LOST_GRACE:
                    result.state = self.state
                    return result
            self._hand_lost_frames = 0
            self._release_confirm_frames = 0
            self._transition_to_idle(result)
            self._dynamic_tracker.reset()
            return result

        # Hand is visible — reset lost counter
        self._hand_lost_frames = 0

        # Get primary hand (first detected)
        primary = hands_data[0]
        primary_handedness = "Right"
        if handedness_data and len(handedness_data) > 0:
            primary_handedness = handedness_data[0].classification[0].label

        # Compute metrics for primary hand
        index_thumb_dist = HandAnalyzer.pinch_distance(primary, THUMB_TIP, INDEX_TIP)
        middle_thumb_dist = HandAnalyzer.pinch_distance(primary, THUMB_TIP, MIDDLE_TIP)
        thumb_pos = HandAnalyzer.fingertip_position(primary, THUMB_TIP)
        roll_angle = HandAnalyzer.hand_roll_angle(primary)
        open_hand = HandAnalyzer.is_open_hand(primary)
        # Roll deviation from neutral
        neutral = self.ROLL_NEUTRAL_RIGHT if primary_handedness == "Right" else self.ROLL_NEUTRAL_LEFT
        roll_deviation = self._angle_diff(roll_angle, neutral)

        # Pack debug info
        result.debug_info = {
            "index_thumb_dist": index_thumb_dist,
            "middle_thumb_dist": middle_thumb_dist,
            "roll_angle": roll_angle,
            "roll_deviation": roll_deviation,
            "open_hand": open_hand,
            "num_hands": num_hands,
            "thumb_pos": thumb_pos,
        }

        # ── Priority 1: Zoom (two hands) ─────────────────────────────
        # Require several consecutive frames of 2 hands to avoid ghost detections
        if num_hands >= 2:
            self._two_hand_frames += 1
        else:
            self._two_hand_frames = 0

        if (num_hands >= 2
                and self._two_hand_frames >= self.ZOOM_CONFIRM_FRAMES
                and self._settings.get_toggle("zoom")):
            two_hand_dist = HandAnalyzer.two_hand_index_distance(
                hands_data[0], hands_data[1]
            )
            result.debug_info["two_hand_dist"] = two_hand_dist

            if self.state != STATE_ZOOMING and two_hand_dist < self.ZOOM_ENTER_DIST:
                self.state = STATE_ZOOMING
                self._zoom_prev_dist = two_hand_dist
                self._dynamic_tracker.reset()

            if self.state == STATE_ZOOMING:
                if two_hand_dist > self.ZOOM_EXIT_DIST:
                    self.state = STATE_IDLE
                    self._zoom_prev_dist = None
                else:
                    delta = two_hand_dist - (self._zoom_prev_dist or two_hand_dist)
                    self._zoom_prev_dist = two_hand_dist
                    zoom_amount = int(delta * 80 * scroll_sens)
                    result.zoom_amount = zoom_amount
                    result.state = STATE_ZOOMING
                    self._mouse.zoom(zoom_amount)
                    return result

        if self.state == STATE_ZOOMING and num_hands < 2:
            # Dropped to one hand — exit zoom cleanly
            self.state = STATE_IDLE
            self._zoom_prev_dist = None

        # ── Priority 2: Right Click (index+middle both touching thumb) ──
        # Check right click FIRST since both-pinch also triggers index pinch
        both_pinch = (index_thumb_dist < pinch_threshold
                      and middle_thumb_dist < pinch_threshold)
        both_pinch_exit = (index_thumb_dist > pinch_threshold * self.PINCH_EXIT_MULT
                           or middle_thumb_dist > pinch_threshold * self.PINCH_EXIT_MULT)

        if self._settings.get_toggle("right_click"):
            if both_pinch and self.state not in (STATE_RIGHT_CLICK, STATE_LEFT_CLICK, STATE_DRAGGING, STATE_ZOOMING):
                if not self._right_pinch_was_active and self._click_cooldown <= 0:
                    result.right_click = True
                    self._click_cooldown = self._settings.get("click_cooldown_ms") / 1000.0
                    self._mouse.right_click()
                self._right_pinch_was_active = True
                self.state = STATE_RIGHT_CLICK
                result.state = STATE_RIGHT_CLICK
                return result

        if self.state == STATE_RIGHT_CLICK and both_pinch_exit:
            self._right_pinch_was_active = False
            self.state = STATE_IDLE

        # ── Priority 3: Left Click / Drag (index+thumb only, not middle) ──
        left_pinch = (index_thumb_dist < pinch_threshold
                      and middle_thumb_dist >= pinch_threshold)
        left_pinch_exit = index_thumb_dist > pinch_threshold * self.PINCH_EXIT_MULT

        if self._settings.get_toggle("left_click"):
            if left_pinch and self.state not in (STATE_LEFT_CLICK, STATE_DRAGGING, STATE_RIGHT_CLICK, STATE_ZOOMING):
                # Pinch just started — click immediately
                self._pinch_was_active = True
                self._pinch_hold_time = 0.0
                now = time.time()
                if self._click_cooldown <= 0:
                    if (now - self._last_click_time) < self._DOUBLE_CLICK_WINDOW:
                        # Second pinch within window — double click
                        result.left_click = True
                        self._click_cooldown = self._settings.get("click_cooldown_ms") / 1000.0
                        self._mouse.double_click()
                        self._last_click_time = 0.0  # Reset so third pinch isn't another double
                    else:
                        # First pinch — single click
                        result.left_click = True
                        self._click_cooldown = self._settings.get("click_cooldown_ms") / 1000.0
                        self._mouse.left_click()
                        self._last_click_time = now
                self.state = STATE_LEFT_CLICK
                result.state = STATE_LEFT_CLICK
                return result

            if left_pinch and self.state == STATE_LEFT_CLICK:
                # Still holding pinch — accumulate hold time
                self._pinch_hold_time += dt
                if self._pinch_hold_time >= self._DRAG_THRESHOLD_SEC:
                    # Transition to drag: press mouse button down
                    self._mouse.mouse_down()
                    self._drag_active = True
                    self.state = STATE_DRAGGING
                    result.state = STATE_DRAGGING
                    # Fall through to cursor movement below
                else:
                    result.state = STATE_LEFT_CLICK
                    return result

            if left_pinch and self.state == STATE_DRAGGING:
                # Dragging — state stays, fall through to cursor movement
                result.state = STATE_DRAGGING

        if self.state in (STATE_LEFT_CLICK, STATE_DRAGGING) and left_pinch_exit:
            # Finger appears released — require consecutive frames to confirm
            self._release_confirm_frames += 1
            if self._release_confirm_frames >= self._RELEASE_CONFIRM_REQUIRED:
                if self.state == STATE_DRAGGING:
                    self._mouse.mouse_up()
                    self._drag_active = False
                self._pinch_was_active = False
                self._pinch_hold_time = 0.0
                self._release_confirm_frames = 0
                self.state = STATE_IDLE
            else:
                # Not yet confirmed — stay in current state
                result.state = self.state
        elif self.state in (STATE_LEFT_CLICK, STATE_DRAGGING) and not left_pinch_exit:
            # Still pinching or in hysteresis zone — reset confirmation counter
            self._release_confirm_frames = 0

        # ── Priority 4: Horizontal Scroll (hand roll) ────────────────
        if self._settings.get_toggle("horizontal_scroll"):
            if abs(roll_deviation) > self.ROLL_ENTER_DEG and self.state not in (
                STATE_LEFT_CLICK, STATE_DRAGGING, STATE_RIGHT_CLICK, STATE_ZOOMING
            ):
                self.state = STATE_HSCROLL
                scroll_amount = int(
                    (roll_deviation - self.ROLL_ENTER_DEG * (1 if roll_deviation > 0 else -1))
                    * hscroll_sens * 0.05
                )
                result.scroll_h = scroll_amount
                result.state = STATE_HSCROLL
                self._mouse.scroll_horizontal(scroll_amount)
                self._dynamic_tracker.reset()
                return result

        if self.state == STATE_HSCROLL and abs(roll_deviation) < self.ROLL_EXIT_DEG:
            self.state = STATE_IDLE

        # ── Priority 5: Anchor Reset (open hand — all 5 fingers) ─────
        if open_hand and self.state not in (
            STATE_LEFT_CLICK, STATE_DRAGGING, STATE_RIGHT_CLICK,
            STATE_ZOOMING, STATE_HSCROLL, STATE_ANCHOR_RESET
        ):
            self._static_tracker.reset()
            self._dynamic_tracker.reset()
            self.state = STATE_ANCHOR_RESET
            result.state = STATE_ANCHOR_RESET
            return result

        if self.state == STATE_ANCHOR_RESET and not open_hand:
            self.state = STATE_IDLE

        # ── Priority 6: Cursor Movement (also during drag) ─────────────
        if self.state in (STATE_IDLE, STATE_MOVING, STATE_DRAGGING):
            if self.state != STATE_DRAGGING:
                self.state = STATE_MOVING
                result.state = STATE_MOVING

            tracking_mode = self._settings.get("tracking_mode")
            ix, iy = thumb_pos

            if tracking_mode == "framed" and self._calibration.transform_matrix is not None:
                # Framed anchor — absolute position via 4-corner calibration
                abs_x, abs_y = self._calibration.transform_point(ix, iy)
                sx, sy = self._framed_smoother.smooth(
                    float(abs_x), float(abs_y), smoothing
                )
                result.cursor_abs_x = int(sx)
                result.cursor_abs_y = int(sy)
                self._mouse.move_absolute(int(sx), int(sy))
            elif tracking_mode == "dynamic":
                # Dynamic anchor — joystick-style, distance from anchor = speed
                dx, dy = self._dynamic_tracker.update(
                    ix, iy, True, screen_w, screen_h, smoothing
                )
                result.cursor_dx = dx
                result.cursor_dy = dy
                self._mouse.move_relative(int(dx), int(dy))
            else:
                # Static anchor — trackpad-style relative movement
                dx, dy = self._static_tracker.update(
                    ix, iy, True, screen_w, screen_h, smoothing
                )
                result.cursor_dx = dx
                result.cursor_dy = dy
                self._mouse.move_relative(int(dx), int(dy))

            return result

        result.state = self.state
        return result

    def _transition_to_idle(self, result: GestureResult) -> None:
        """Reset to idle state."""
        if self._drag_active:
            self._mouse.mouse_up()
            self._drag_active = False
        self.state = STATE_IDLE
        result.state = STATE_IDLE
        self._pinch_was_active = False
        self._right_pinch_was_active = False
        self._pinch_hold_time = 0.0
        self._zoom_prev_dist = None
        self._static_tracker.reset()
        self._dynamic_tracker.reset()
        self._framed_smoother.reset()

    @staticmethod
    def _angle_diff(angle: float, neutral: float) -> float:
        """Compute the signed angular difference, wrapped to [-180, 180]."""
        diff = angle - neutral
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def reset(self):
        if self._drag_active:
            self._mouse.mouse_up()
            self._drag_active = False
        self.state = STATE_IDLE
        self._click_cooldown = 0.0
        self._zoom_prev_dist = None
        self._static_tracker.reset()
        self._dynamic_tracker.reset()
        self._framed_smoother.reset()
        self._pinch_was_active = False
        self._right_pinch_was_active = False
        self._pinch_hold_time = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PalmModeStateMachine
# ═══════════════════════════════════════════════════════════════════════════════

# Palm mode gesture states
STATE_PALM_FIST = "FIST_CLICK"
STATE_PALM_DRAG = "FIST_DRAG"
STATE_PALM_PEACE = "PEACE_DBLCLICK"
STATE_PALM_RIGHT = "PINCH_RCLICK"


class PalmModeStateMachine:
    """Alternative gesture mode using palm center for movement.

    Gestures:
      - Palm center → cursor movement
      - Fist (short) → left click
      - Fist (hold) → drag
      - Peace sign → double left click
      - Index+thumb pinch → right click
    """

    def __init__(self, settings: SettingsManager, mouse_output: MouseOutput):
        self._settings = settings
        self._mouse = mouse_output
        self.state = STATE_IDLE
        self._click_cooldown = 0.0
        self._static_tracker = DynamicAnchorTracker()    # static = trackpad-style
        self._dynamic_tracker = JoystickAnchorTracker()  # dynamic = joystick-style

        # Fist click/drag
        self._fist_was_active = False
        self._fist_hold_time = 0.0
        self._DRAG_THRESHOLD_SEC = 0.2
        self._drag_active = False

        # Release confirmation (same approach as main SM)
        self._release_confirm_frames = 0
        self._RELEASE_CONFIRM_REQUIRED = 4
        self._hand_lost_frames = 0
        self._HAND_LOST_GRACE = 8

        # Peace sign debounce
        self._peace_cooldown = 0.0

        # Right click (pinch) debounce
        self._pinch_was_active = False

    def update(self, hands_data: list, handedness_data: list, dt: float) -> GestureResult:
        """Process one frame and return gesture result."""
        result = GestureResult()
        result.debug_info = {}

        if self._click_cooldown > 0:
            self._click_cooldown -= dt
        if self._peace_cooldown > 0:
            self._peace_cooldown -= dt

        screen_w, screen_h = self._mouse.get_screen_size()
        smoothing = self._settings.get("smoothing_factor")
        pinch_threshold = self._settings.get("pinch_threshold")

        num_hands = len(hands_data)

        # No hands
        if num_hands == 0:
            if self.state in (STATE_PALM_FIST, STATE_PALM_DRAG):
                self._hand_lost_frames += 1
                if self._hand_lost_frames < self._HAND_LOST_GRACE:
                    result.state = self.state
                    return result
            self._hand_lost_frames = 0
            self._release_confirm_frames = 0
            self._transition_to_idle(result)
            self._dynamic_tracker.reset()
            return result

        self._hand_lost_frames = 0
        primary = hands_data[0]

        # Compute gesture detections
        fist = HandAnalyzer.is_fist(primary)
        peace = HandAnalyzer.is_peace_sign(primary)
        open_hand = HandAnalyzer.is_open_hand(primary)
        index_thumb_dist = HandAnalyzer.pinch_distance(primary, THUMB_TIP, INDEX_TIP)
        middle_thumb_dist = HandAnalyzer.pinch_distance(primary, THUMB_TIP, MIDDLE_TIP)
        palm_x, palm_y = HandAnalyzer.palm_center(primary)

        # Pinch = index touches thumb but NOT fist (middle finger away from thumb)
        left_pinch = (index_thumb_dist < pinch_threshold
                      and middle_thumb_dist >= pinch_threshold
                      and not fist)

        result.debug_info = {
            "index_thumb_dist": index_thumb_dist,
            "middle_thumb_dist": middle_thumb_dist,
            "fist": fist,
            "peace": peace,
            "open_hand": open_hand,
            "palm_x": palm_x,
            "palm_y": palm_y,
        }

        # ── Priority 1: Peace sign → double click ──
        if peace and self.state not in (STATE_PALM_FIST, STATE_PALM_DRAG, STATE_PALM_RIGHT):
            if self._peace_cooldown <= 0:
                print("[PALM] Peace sign detected — firing double click")
                self._mouse.double_click()
                self._peace_cooldown = 0.8  # Prevent repeat
                result.left_click = True
                result.state = STATE_PALM_PEACE
                self.state = STATE_PALM_PEACE
                return result

        if self.state == STATE_PALM_PEACE and not peace:
            self.state = STATE_IDLE

        # ── Priority 2: Right click (pinch, but not during peace) ──
        if left_pinch and not peace and self.state not in (STATE_PALM_FIST, STATE_PALM_DRAG, STATE_PALM_PEACE):
            if not self._pinch_was_active and self._click_cooldown <= 0:
                self._mouse.right_click()
                result.right_click = True
                self._click_cooldown = self._settings.get("click_cooldown_ms") / 1000.0
            self._pinch_was_active = True
            result.state = STATE_PALM_RIGHT
            self.state = STATE_PALM_RIGHT
            return result

        if self.state == STATE_PALM_RIGHT and not left_pinch:
            self._pinch_was_active = False
            self.state = STATE_IDLE

        # ── Priority 3: Fist → left click / drag ──
        if fist and self.state not in (STATE_PALM_FIST, STATE_PALM_DRAG, STATE_PALM_RIGHT, STATE_PALM_PEACE):
            # Fist just started
            self._fist_was_active = True
            self._fist_hold_time = 0.0
            if self._click_cooldown <= 0:
                self._mouse.left_click()
                result.left_click = True
                self._click_cooldown = self._settings.get("click_cooldown_ms") / 1000.0
            self.state = STATE_PALM_FIST
            result.state = STATE_PALM_FIST
            return result

        if fist and self.state == STATE_PALM_FIST:
            # Still holding fist — accumulate for drag
            self._fist_hold_time += dt
            if self._fist_hold_time >= self._DRAG_THRESHOLD_SEC:
                self._mouse.mouse_down()
                self._drag_active = True
                self.state = STATE_PALM_DRAG
                result.state = STATE_PALM_DRAG
                # Fall through to cursor movement
            else:
                result.state = STATE_PALM_FIST
                return result

        if fist and self.state == STATE_PALM_DRAG:
            result.state = STATE_PALM_DRAG
            # Fall through to cursor movement while dragging

        # Release fist (with confirmation)
        if self.state in (STATE_PALM_FIST, STATE_PALM_DRAG) and not fist:
            self._release_confirm_frames += 1
            if self._release_confirm_frames >= self._RELEASE_CONFIRM_REQUIRED:
                if self._drag_active:
                    self._mouse.mouse_up()
                    self._drag_active = False
                self._fist_was_active = False
                self._fist_hold_time = 0.0
                self._release_confirm_frames = 0
                self.state = STATE_IDLE
                self._static_tracker.reset()
                self._dynamic_tracker.reset()
            else:
                result.state = self.state
        elif self.state in (STATE_PALM_FIST, STATE_PALM_DRAG) and fist:
            self._release_confirm_frames = 0

        # ── Cursor movement (palm center) ──
        if self.state not in (STATE_PALM_FIST, STATE_PALM_RIGHT, STATE_PALM_PEACE):
            tracking_mode = self._settings.get("tracking_mode")
            if tracking_mode == "dynamic":
                tracker = self._dynamic_tracker
            else:
                tracker = self._static_tracker
            dx, dy = tracker.update(
                palm_x, palm_y, True, screen_w, screen_h, smoothing
            )
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                self._mouse.move_relative(dx, dy)
                result.cursor_dx = dx
                result.cursor_dy = dy
                if self.state == STATE_IDLE:
                    result.state = STATE_MOVING

        # ── Four fingers (thumb tucked) → reset anchor (but not during peace) ──
        four_fingers = HandAnalyzer.is_four_fingers(primary)
        if four_fingers and not peace and self.state == STATE_IDLE:
            self._static_tracker.reset()
            self._dynamic_tracker.reset()
            result.state = STATE_ANCHOR_RESET

        return result

    def _transition_to_idle(self, result: GestureResult) -> None:
        if self._drag_active:
            self._mouse.mouse_up()
            self._drag_active = False
        self.state = STATE_IDLE
        result.state = STATE_IDLE
        self._fist_was_active = False
        self._pinch_was_active = False
        self._fist_hold_time = 0.0
        self._static_tracker.reset()
        self._dynamic_tracker.reset()

    def reset(self):
        if self._drag_active:
            self._mouse.mouse_up()
            self._drag_active = False
        self.state = STATE_IDLE
        self._click_cooldown = 0.0
        self._static_tracker.reset()
        self._dynamic_tracker.reset()
        self._fist_was_active = False
        self._pinch_was_active = False
        self._fist_hold_time = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DebugOverlay
# ═══════════════════════════════════════════════════════════════════════════════

class DebugOverlay:
    """Renders debug information onto the OpenCV camera frame."""

    @staticmethod
    def draw(
        frame: np.ndarray,
        gesture_result: GestureResult,
        fps: float,
        tracking_mode: str,
        hands_data: list,
        calibration_active: bool,
        calibration_msg: str,
        show: bool,
    ) -> np.ndarray:
        if not show and not calibration_active:
            return frame

        h, w = frame.shape[:2]

        if show:
            # FPS
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2,
            )
            # Active gesture
            state_color = COLOR_YELLOW if gesture_result.state != STATE_IDLE else COLOR_WHITE
            cv2.putText(
                frame, f"Gesture: {gesture_result.state}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2,
            )
            # Tracking mode
            cv2.putText(
                frame, f"Mode: {tracking_mode}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1,
            )

            # Pinch distances
            debug = gesture_result.debug_info
            if "index_thumb_dist" in debug:
                dist = debug["index_thumb_dist"]
                cv2.putText(
                    frame, f"I+T: {dist:.3f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1,
                )
            if "middle_thumb_dist" in debug:
                dist = debug["middle_thumb_dist"]
                cv2.putText(
                    frame, f"M+T: {dist:.3f}", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1,
                )
            if "roll_deviation" in debug:
                rd = debug["roll_deviation"]
                cv2.putText(
                    frame, f"Roll: {rd:.1f} deg", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1,
                )
            if "two_hand_dist" in debug:
                td = debug["two_hand_dist"]
                cv2.putText(
                    frame, f"Zoom dist: {td:.3f}", (10, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAGENTA, 1,
                )

            # Draw hand landmarks
            for hand_landmarks in hands_data:
                for idx, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if idx in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP):
                        cv2.circle(frame, (cx, cy), 8, COLOR_RED, -1)
                    elif idx in (WRIST, INDEX_MCP, PINKY_MCP):
                        cv2.circle(frame, (cx, cy), 6, COLOR_BLUE, -1)
                    else:
                        cv2.circle(frame, (cx, cy), 3, COLOR_GREEN, -1)

                # Draw pinch lines
                thumb = hand_landmarks.landmark[THUMB_TIP]
                index = hand_landmarks.landmark[INDEX_TIP]
                middle = hand_landmarks.landmark[MIDDLE_TIP]

                tx, ty = int(thumb.x * w), int(thumb.y * h)
                ix, iy = int(index.x * w), int(index.y * h)
                mx, my = int(middle.x * w), int(middle.y * h)

                # Color pinch lines based on distance
                it_dist = debug.get("index_thumb_dist", 1.0)
                mt_dist = debug.get("middle_thumb_dist", 1.0)
                it_color = COLOR_RED if it_dist < 0.05 else COLOR_GREEN
                mt_color = COLOR_RED if mt_dist < 0.05 else COLOR_GREEN

                cv2.line(frame, (tx, ty), (ix, iy), it_color, 2)
                cv2.line(frame, (tx, ty), (mx, my), mt_color, 2)

                # Draw roll indicator arc
                idx_mcp = hand_landmarks.landmark[INDEX_MCP]
                pinky_mcp = hand_landmarks.landmark[PINKY_MCP]
                imx, imy = int(idx_mcp.x * w), int(idx_mcp.y * h)
                pmx, pmy = int(pinky_mcp.x * w), int(pinky_mcp.y * h)
                cv2.line(frame, (imx, imy), (pmx, pmy), COLOR_ORANGE, 2)

            # Zoom indicator between hands
            if len(hands_data) >= 2:
                t1 = hands_data[0].landmark[INDEX_TIP]
                t2 = hands_data[1].landmark[INDEX_TIP]
                p1 = (int(t1.x * w), int(t1.y * h))
                p2 = (int(t2.x * w), int(t2.y * h))
                zoom_color = COLOR_MAGENTA if gesture_result.state == STATE_ZOOMING else COLOR_WHITE
                cv2.line(frame, p1, p2, zoom_color, 2)

        # Calibration overlay
        if calibration_active and calibration_msg:
            # Dark banner at top
            cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
            cv2.putText(
                frame, calibration_msg, (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2,
            )

        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# CameraThread
# ═══════════════════════════════════════════════════════════════════════════════

class CameraThread(threading.Thread):
    """Background thread for camera capture, hand tracking, and gesture processing."""

    def __init__(
        self,
        result_queue: queue.Queue,
        settings: SettingsManager,
        mouse_output: MouseOutput,
        calibration: CalibrationWizard,
        gesture_calibration: GestureCalibrationWizard,
    ):
        super().__init__(daemon=True)
        self._queue = result_queue
        self._settings = settings
        self._mouse = mouse_output
        self._calibration = calibration
        self._gesture_calibration = gesture_calibration
        self._running = True
        self._gesture_sm = GestureStateMachine(settings, mouse_output, calibration)
        self._palm_sm = PalmModeStateMachine(settings, mouse_output)

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:
            print(f"ERROR in camera thread: {e}")
            import traceback
            traceback.print_exc()

    def _run_inner(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open camera.")
            if IS_MACOS:
                print("Grant Camera permissions in:")
                print("  System Settings > Privacy & Security > Camera")
            return

        # Give the camera time to warm up
        time.sleep(0.5)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        overlay = DebugOverlay()
        prev_time = time.time()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Mirror the frame for natural movement
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Timing
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            fps = 1.0 / dt if dt > 0 else 0.0

            # Extract hand data
            hands_list = results.multi_hand_landmarks or []
            handedness_list = results.multi_handedness or []

            # Handle calibration wizards if active
            calibration_msg = ""
            if self._gesture_calibration.active:
                hand_lm = hands_list[0] if hands_list else None
                calibration_msg = self._gesture_calibration.update(hand_lm, dt)
                gesture_result = GestureResult()
                # Check if just finished
                if not self._gesture_calibration.active:
                    applied = self._gesture_calibration.apply_to_settings(self._settings)
                    # Apply drag/double-click thresholds to gesture state machine
                    if "drag_threshold" in applied:
                        self._gesture_sm._DRAG_THRESHOLD_SEC = applied["drag_threshold"]
                    if "double_click_window" in applied:
                        self._gesture_sm._DOUBLE_CLICK_WINDOW = applied["double_click_window"]
                    if "sensitivity" in applied:
                        self._gesture_sm._dynamic_tracker._sensitivity = applied["sensitivity"]
            elif self._calibration.active:
                index_pos = None
                pinch_detected = False
                if hands_list:
                    index_pos = HandAnalyzer.fingertip_position(hands_list[0], INDEX_TIP)
                    pinch_detected = (
                        HandAnalyzer.pinch_distance(hands_list[0], THUMB_TIP, INDEX_TIP)
                        < self._settings.get("pinch_threshold")
                    )
                calibration_msg = self._calibration.update(index_pos, pinch_detected, dt)
                gesture_result = GestureResult()
            else:
                # Normal gesture processing — route to correct state machine
                if self._settings.get("interface_mode") == "palm":
                    gesture_result = self._palm_sm.update(hands_list, handedness_list, dt)
                else:
                    gesture_result = self._gesture_sm.update(hands_list, handedness_list, dt)

            # Draw debug overlay
            show_debug = self._settings.get("show_debug_overlay")
            any_calibration_active = self._calibration.active or self._gesture_calibration.active
            annotated = overlay.draw(
                frame,
                gesture_result,
                fps,
                self._settings.get("tracking_mode"),
                hands_list,
                any_calibration_active,
                calibration_msg,
                show_debug,
            )

            # Send frame result to GUI
            frame_result = FrameResult(
                fps=fps,
                gesture_name=gesture_result.state,
                frame=annotated,
                debug_info=gesture_result.debug_info,
            )
            try:
                self._queue.put_nowait(frame_result)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(frame_result)
                except queue.Empty:
                    pass

        cap.release()
        hands.close()

    def stop(self) -> None:
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
# SettingsGUI
# ═══════════════════════════════════════════════════════════════════════════════

class SettingsGUI:
    """Dark-themed tkinter settings panel running on the main thread."""

    def __init__(
        self,
        settings: SettingsManager,
        camera_thread: CameraThread,
        calibration: CalibrationWizard,
        gesture_calibration: GestureCalibrationWizard,
        mouse_output: MouseOutput,
        result_queue: queue.Queue,
    ):
        self._settings = settings
        self._camera_thread = camera_thread
        self._calibration = calibration
        self._gesture_calibration = gesture_calibration
        self._mouse = mouse_output
        self._queue = result_queue

        self.root = tk.Tk()
        self.root.title("Gesture Mouse Control")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Variables for live readouts
        self._gesture_var = tk.StringVar(value="IDLE")
        self._fps_var = tk.StringVar(value="0.0")

        self._build_ui()
        self._poll_id = self.root.after(16, self._poll_camera_results)

    def _build_ui(self) -> None:
        main_frame = tk.Frame(self.root, bg=BG_COLOR, padx=15, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ── Tracking Mode Section ────────────────────────────────────
        self._section_label(main_frame, "Tracking Mode")
        mode_frame = tk.Frame(main_frame, bg=SECTION_BG, padx=10, pady=8)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        self._tracking_var = tk.StringVar(value=self._settings.get("tracking_mode"))
        for mode, label in [
            ("static", "Static Anchor  —  trackpad-style relative movement"),
            ("dynamic", "Dynamic Anchor  —  joystick-style, distance from anchor = speed"),
            ("framed", "Framed Anchor  —  4-corner calibrated absolute position"),
        ]:
            rb = tk.Radiobutton(
                mode_frame, text=label, variable=self._tracking_var, value=mode,
                bg=SECTION_BG, fg=FG_COLOR, selectcolor=ENTRY_BG,
                activebackground=SECTION_BG, activeforeground=FG_COLOR,
                command=lambda: self._on_mode_change(),
            )
            rb.pack(anchor=tk.W)

        btn = tk.Button(
            mode_frame, text="Run Framed Calibration", bg=BUTTON_BG, fg=FG_COLOR,
            activebackground=BUTTON_ACTIVE_BG, activeforeground=FG_COLOR,
            relief=tk.FLAT, padx=10, pady=4,
            command=self._on_calibrate,
        )
        btn.pack(anchor=tk.W, pady=(6, 0))

        gesture_btn = tk.Button(
            mode_frame, text="Calibrate Gestures", bg=BUTTON_BG, fg=FG_COLOR,
            activebackground=BUTTON_ACTIVE_BG, activeforeground=FG_COLOR,
            relief=tk.FLAT, padx=10, pady=4,
            command=self._on_gesture_calibrate,
        )
        gesture_btn.pack(anchor=tk.W, pady=(4, 0))

        # ── Interface Mode Section ───────────────────────────────────
        self._section_label(main_frame, "Interface Mode")
        iface_frame = tk.Frame(main_frame, bg=SECTION_BG, padx=10, pady=8)
        iface_frame.pack(fill=tk.X, pady=(0, 10))

        self._iface_var = tk.StringVar(value=self._settings.get("interface_mode"))
        for mode, label, desc in [
            ("finger", "Finger Mode", "Thumb tracks cursor, pinch=click"),
            ("palm", "Palm Mode", "Palm tracks cursor, fist=click, peace=dblclick"),
        ]:
            rb = tk.Radiobutton(
                iface_frame, text=f"{label}  —  {desc}",
                variable=self._iface_var, value=mode,
                bg=SECTION_BG, fg=FG_COLOR, selectcolor=ENTRY_BG,
                activebackground=SECTION_BG, activeforeground=FG_COLOR,
                command=self._on_iface_change,
            )
            rb.pack(anchor=tk.W)

        # ── Settings Sliders Section ─────────────────────────────────
        self._section_label(main_frame, "Settings")
        sliders_frame = tk.Frame(main_frame, bg=SECTION_BG, padx=10, pady=8)
        sliders_frame.pack(fill=tk.X, pady=(0, 10))

        slider_configs = [
            ("smoothing_factor", "Smoothing", 0.05, 0.95, 0.01),
            ("pinch_threshold", "Pinch Threshold", 0.01, 0.15, 0.005),
            ("scroll_sensitivity", "Scroll Sensitivity", 1.0, 20.0, 0.5),
            ("hscroll_sensitivity", "H-Scroll Sensitivity", 1.0, 20.0, 0.5),
            ("click_cooldown_ms", "Click Cooldown (ms)", 100, 1500, 50),
        ]
        self._slider_vars: Dict[str, tk.DoubleVar] = {}

        for key, label, lo, hi, resolution in slider_configs:
            row = tk.Frame(sliders_frame, bg=SECTION_BG)
            row.pack(fill=tk.X, pady=2)

            lbl = tk.Label(row, text=label, bg=SECTION_BG, fg=FG_COLOR, width=18, anchor=tk.W)
            lbl.pack(side=tk.LEFT)

            var = tk.DoubleVar(value=self._settings.get(key))
            self._slider_vars[key] = var

            val_label = tk.Label(row, text=self._format_val(key, var.get()),
                                bg=SECTION_BG, fg=ACCENT_COLOR, width=8, anchor=tk.E)
            val_label.pack(side=tk.RIGHT)

            slider = tk.Scale(
                row, from_=lo, to=hi, resolution=resolution, orient=tk.HORIZONTAL,
                variable=var, showvalue=False, length=180,
                bg=SECTION_BG, fg=FG_COLOR, troughcolor=SLIDER_TROUGH,
                highlightthickness=0, borderwidth=0,
                command=lambda v, k=key, vl=val_label: self._on_slider_change(k, v, vl),
            )
            slider.pack(side=tk.RIGHT, padx=(5, 5))

        # ── Gesture Toggles Section ──────────────────────────────────
        self._section_label(main_frame, "Gesture Toggles")
        toggles_frame = tk.Frame(main_frame, bg=SECTION_BG, padx=10, pady=8)
        toggles_frame.pack(fill=tk.X, pady=(0, 10))

        gesture_labels = {
            "left_click": "Left Click",
            "right_click": "Right Click",
            "zoom": "Zoom",
            "horizontal_scroll": "Horizontal Scroll",
        }
        self._toggle_vars: Dict[str, tk.BooleanVar] = {}

        row_frame = None
        for i, (key, label) in enumerate(gesture_labels.items()):
            if i % 2 == 0:
                row_frame = tk.Frame(toggles_frame, bg=SECTION_BG)
                row_frame.pack(fill=tk.X, pady=1)

            var = tk.BooleanVar(value=self._settings.get_toggle(key))
            self._toggle_vars[key] = var

            cb = tk.Checkbutton(
                row_frame, text=label, variable=var,
                bg=SECTION_BG, fg=FG_COLOR, selectcolor=ENTRY_BG,
                activebackground=SECTION_BG, activeforeground=FG_COLOR,
                command=lambda k=key: self._on_toggle_change(k),
            )
            cb.pack(side=tk.LEFT, padx=(0, 20))

        # ── Debug Section ────────────────────────────────────────────
        self._section_label(main_frame, "Debug")
        debug_frame = tk.Frame(main_frame, bg=SECTION_BG, padx=10, pady=8)
        debug_frame.pack(fill=tk.X, pady=(0, 10))

        self._debug_var = tk.BooleanVar(value=self._settings.get("show_debug_overlay"))
        cb = tk.Checkbutton(
            debug_frame, text="Show Debug Overlay", variable=self._debug_var,
            bg=SECTION_BG, fg=FG_COLOR, selectcolor=ENTRY_BG,
            activebackground=SECTION_BG, activeforeground=FG_COLOR,
            command=self._on_debug_toggle,
        )
        cb.pack(anchor=tk.W)

        # Live readouts
        readout_frame = tk.Frame(debug_frame, bg=SECTION_BG)
        readout_frame.pack(fill=tk.X, pady=(8, 0))

        tk.Label(readout_frame, text="Gesture:", bg=SECTION_BG, fg=FG_COLOR).pack(
            side=tk.LEFT
        )
        tk.Label(
            readout_frame, textvariable=self._gesture_var, bg=SECTION_BG, fg=ACCENT_COLOR,
            width=14, anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(5, 15))

        tk.Label(readout_frame, text="FPS:", bg=SECTION_BG, fg=FG_COLOR).pack(side=tk.LEFT)
        tk.Label(
            readout_frame, textvariable=self._fps_var, bg=SECTION_BG, fg=ACCENT_COLOR,
            width=6, anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(5, 0))

    def _section_label(self, parent: tk.Frame, text: str) -> None:
        lbl = tk.Label(parent, text=text, bg=BG_COLOR, fg=ACCENT_COLOR,
                       font=("TkDefaultFont", 11, "bold"))
        lbl.pack(anchor=tk.W, pady=(8, 2))

    @staticmethod
    def _format_val(key: str, val: float) -> str:
        if key == "click_cooldown_ms":
            return f"{int(val)} ms"
        if key in ("scroll_sensitivity", "hscroll_sensitivity"):
            return f"{val:.1f}"
        return f"{val:.3f}"

    def _on_slider_change(self, key: str, value: str, val_label: tk.Label) -> None:
        fval = float(value)
        self._settings.set(key, fval)
        val_label.config(text=self._format_val(key, fval))

    def _on_mode_change(self) -> None:
        self._settings.set("tracking_mode", self._tracking_var.get())

    def _on_toggle_change(self, gesture: str) -> None:
        self._settings.set_toggle(gesture, self._toggle_vars[gesture].get())

    def _on_debug_toggle(self) -> None:
        self._settings.set("show_debug_overlay", self._debug_var.get())

    def _on_calibrate(self) -> None:
        screen_size = self._mouse.get_screen_size()
        self._calibration.start(screen_size)
        self._tracking_var.set("framed")
        self._settings.set("tracking_mode", "framed")

    def _on_iface_change(self) -> None:
        self._settings.set("interface_mode", self._iface_var.get())

    def _on_gesture_calibrate(self) -> None:
        self._gesture_calibration.start()

    def _poll_camera_results(self) -> None:
        """Drain the frame queue and update GUI + OpenCV window."""
        latest: Optional[FrameResult] = None
        try:
            while True:
                latest = self._queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            self._gesture_var.set(latest.gesture_name)
            self._fps_var.set(f"{latest.fps:.1f}")

            if latest.frame is not None and self._settings.get("show_debug_overlay"):
                cv2.imshow("Gesture Mouse - Camera", latest.frame)
                cv2.waitKey(1)
            elif latest.frame is not None and not self._settings.get("show_debug_overlay"):
                # Close the window if overlay is disabled
                cv2.destroyWindow("Gesture Mouse - Camera")

        self._poll_id = self.root.after(16, self._poll_camera_results)

    def _on_close(self) -> None:
        """Handle window close — clean shutdown."""
        self._camera_thread.stop()
        self._camera_thread.join(timeout=3.0)
        self._mouse.cleanup()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self) -> None:
        """Start the tkinter main loop (blocks)."""
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def check_accessibility() -> None:
    """Check if PyAutoGUI can control the mouse on macOS."""
    if IS_MACOS and not RASPBERRY_PI_MODE:
        try:
            pyautogui.position()
        except Exception:
            print("=" * 60)
            print("WARNING: PyAutoGUI cannot control the mouse.")
            print("On macOS, grant Accessibility permissions:")
            print("  System Settings > Privacy & Security > Accessibility")
            print("  Add your terminal app (Terminal, iTerm2, VS Code, etc.)")
            print("=" * 60)


def check_camera() -> bool:
    """Quick check that the camera is accessible.

    Note: On macOS, opening and immediately closing the camera can cause
    the subsequent real open to fail, so we just check availability
    without fully opening the device.
    """
    import subprocess
    if IS_MACOS:
        # On macOS, check that a camera device exists via system_profiler
        try:
            result = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True, text=True, timeout=5,
            )
            if "Camera" not in result.stdout and "FaceTime" not in result.stdout:
                print("=" * 60)
                print("ERROR: No camera detected.")
                print("=" * 60)
                return False
        except Exception:
            pass  # Fall through — let the camera thread handle errors
        return True
    else:
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        if opened:
            cap.release()
        else:
            print("=" * 60)
            print("ERROR: Cannot access camera.")
            print("=" * 60)
        return opened


def main():
    """Application entry point."""
    print("Gesture Mouse Control — Starting up...")

    # Check permissions
    check_accessibility()
    if not check_camera():
        sys.exit(1)

    # Initialize components
    script_dir = os.path.dirname(os.path.abspath(__file__))

    settings = SettingsManager(script_dir)
    mouse_out = MouseOutput(RASPBERRY_PI_MODE)
    calibration = CalibrationWizard(script_dir)
    calibration.load()
    gesture_calibration = GestureCalibrationWizard(script_dir)
    gesture_calibration.load()

    screen_size = mouse_out.get_screen_size()
    calibration._screen_size = screen_size

    result_queue = queue.Queue(maxsize=2)

    # Start camera thread
    camera_thread = CameraThread(result_queue, settings, mouse_out, calibration, gesture_calibration)
    camera_thread.start()

    # Run GUI on main thread (blocks until window is closed)
    gui = SettingsGUI(settings, camera_thread, calibration, gesture_calibration, mouse_out, result_queue)
    gui.run()

    print("Gesture Mouse Control — Shut down.")


if __name__ == "__main__":
    main()

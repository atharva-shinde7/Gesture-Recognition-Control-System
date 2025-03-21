# Gesture Recognition Control System

A computer vision system that enables controlling your computer through hand gestures. The application uses your webcam to track hand movements and translate them into mouse actions.

## Features

- **Cursor Movement**: Point with your index finger to move the mouse cursor
- **Left Click**: Extend index finger while folding middle finger
- **Right Click**: Extend middle finger while folding index finger
- **Double Click**: Extend both index and middle fingers
- **Screenshot**: Form a pinch gesture with thumb and index finger

## Requirements

- Python 3.8+
- Webcam

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/atharva-shinde7/Gesture-Recognition-Control-System.git
   cd GestureRecognition
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```
python main.py
```

- Position your hand in front of the webcam
- Use the gestures described in the Features section
- Press 'q' to quit the application

## Project Structure

- `main.py`: Main application with gesture detection and mouse control logic
- `util.py`: Utility functions for calculating angles and distances
- `requirements.txt`: Project dependencies

## License

MIT

## Acknowledgments

- MediaPipe for hand tracking technology
- OpenCV for computer vision capabilities 
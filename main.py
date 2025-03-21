"""
Hand Gesture Recognition and Control System
This module implements a computer vision system to control mouse actions through hand gestures.
It uses a webcam to detect hand movements and translates them into various computer controls.
"""

# Standard library imports
import random

# Third-party imports
import cv2
import mediapipe as mp
import pyautogui
from pynput.mouse import Button, Controller

# Local imports
import util

# Initialize mouse controller
mouse = Controller()

# Get screen dimensions for coordinate mapping
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,      # For video processing (not static images)
    model_complexity=1,           # Balanced between speed and accuracy
    min_detection_confidence=0.7, # Threshold for hand detection
    min_tracking_confidence=0.7,  # Threshold for hand tracking between frames
    max_num_hands=1               # Only track one hand for simplicity
)


def find_finger_tip(processed):
    """
    Locate the index finger tip from processed hand landmarks.
    
    Args:
        processed: MediaPipe hand processing results
        
    Returns:
        tuple: Normalized (x, y) coordinates of the index finger tip, or (None, None) if not detected
    """
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip.x, index_finger_tip.y  # Return (x, y) instead of the object
    return None, None


# Buffer to store recent positions for smooth movement
recent_positions = []

def move_mouse(x, y, smoothing=5):
    """
    Move the mouse cursor with smoothing effect.
    
    Args:
        x: Normalized x-coordinate (0-1) to map to screen width
        y: Normalized y-coordinate (0-1) to map to screen height
        smoothing: Number of recent positions to average for smooth movement
    """
    global recent_positions
    
    if x is not None and y is not None:
        # Convert normalized coordinates to screen pixels
        x = int(x * screen_width)
        y = int(y * screen_height)
        
        # Add new position to buffer
        recent_positions.append((x, y))
        
        # Keep only the last `smoothing` positions
        if len(recent_positions) > smoothing:
            recent_positions.pop(0)
        
        # Compute smoothed position by averaging recent positions
        avg_x = int(sum(pos[0] for pos in recent_positions) / len(recent_positions))
        avg_y = int(sum(pos[1] for pos in recent_positions) / len(recent_positions))

        pyautogui.moveTo(avg_x, avg_y)


def is_left_click(landmark_list, thumb_index_dist):
    """
    Detect left click gesture: index finger extended, middle finger folded.
    
    Args:
        landmark_list: List of hand landmarks coordinates
        thumb_index_dist: Distance between thumb and index finger
        
    Returns:
        bool: True if left click gesture is detected, False otherwise
    """
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and  # Index finger extended
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and # Middle finger folded
            thumb_index_dist > 50  # Thumb not touching index finger
    )


def is_right_click(landmark_list, thumb_index_dist):
    """
    Detect right click gesture: middle finger extended, index finger folded.
    
    Args:
        landmark_list: List of hand landmarks coordinates
        thumb_index_dist: Distance between thumb and index finger
        
    Returns:
        bool: True if right click gesture is detected, False otherwise
    """
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and  # Middle finger extended
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and    # Index finger folded
            thumb_index_dist > 50  # Thumb not touching index finger
    )


def is_double_click(landmark_list, thumb_index_dist):
    """
    Detect double click gesture: both index and middle fingers extended.
    
    Args:
        landmark_list: List of hand landmarks coordinates
        thumb_index_dist: Distance between thumb and index finger
        
    Returns:
        bool: True if double click gesture is detected, False otherwise
    """
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and    # Index finger extended
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and  # Middle finger extended
            thumb_index_dist > 50  # Thumb not touching index finger
    )


def is_screenshot(landmark_list, thumb_index_dist):
    """
    Detect screenshot gesture: index and middle fingers extended with thumb-index pinch.
    
    Args:
        landmark_list: List of hand landmarks coordinates
        thumb_index_dist: Distance between thumb and index finger
        
    Returns:
        bool: True if screenshot gesture is detected, False otherwise
    """
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and    # Index finger extended
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and  # Middle finger extended
            thumb_index_dist < 50  # Thumb touching or close to index finger (pinch)
    )


# Buffer for smoothing mouse movement
recent_positions = []  
# Smoothing factor for responsiveness (lower = more responsive but less smooth)
SMOOTHING_FACTOR = 3  

def detect_gesture(frame, landmark_list, processed):
    """
    Detect and perform actions based on recognized hand gestures.
    
    Args:
        frame: Current video frame for visualization
        landmark_list: List of hand landmarks coordinates
        processed: MediaPipe hand processing results
    """
    if len(landmark_list) >= 21:  # Full hand detected with all 21 landmarks
        x, y = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        # Check for cursor movement gesture (index finger extended, others folded)
        if x is not None and y is not None and thumb_index_dist < 50:
            # Make sure only the index finger is pointing and others are folded
            if util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:  # Index finger fully extended
                if util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90:  # Middle finger folded
                    if util.get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) > 90:  # Ring finger folded
                        if util.get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) > 90:  # Pinky folded
                            # Now, move the cursor smoothly
                            global recent_positions
                            # Convert normalized coordinates to screen pixels
                            x = int(x * screen_width)
                            y = int(y * screen_height)

                            # Add new position to buffer
                            recent_positions.append((x, y))
                            if len(recent_positions) > SMOOTHING_FACTOR:
                                recent_positions.pop(0)

                            # Calculate average position for smoothing
                            avg_x = sum(pos[0] for pos in recent_positions) // len(recent_positions)
                            avg_y = sum(pos[1] for pos in recent_positions) // len(recent_positions)

                            # Reduce latency by limiting redundant movements (only move if significant change)
                            if len(recent_positions) >= 2 and abs(avg_x - x) > 5 and abs(avg_y - y) > 5:
                                pyautogui.moveTo(avg_x, avg_y, duration=0.01)

        # Detect click gestures only when index finger isn't used for cursor movement
        elif is_left_click(landmark_list, thumb_index_dist):
            # Perform left click and show label on screen
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif is_right_click(landmark_list, thumb_index_dist):
            # Perform right click and show label on screen
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif is_double_click(landmark_list, thumb_index_dist):
            # Perform double click and show label on screen
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        elif is_screenshot(landmark_list, thumb_index_dist):
            # Take a screenshot with random filename and show label on screen
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def main():
    """
    Main function to run the gesture recognition system.
    Initializes the webcam, processes frames, and detects gestures.
    """
    # Initialize drawing utilities for hand landmarks visualization
    draw = mp.solutions.drawing_utils
    # Initialize webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip the frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            # Convert to RGB for MediaPipe processing
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame to detect hands
            processed = hands.process(frameRGB)

            # Extract landmarks from detected hand
            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                # Draw the hand landmarks and connections on the frame
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                # Store normalized coordinates of landmarks
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            # Detect and perform actions based on hand gestures
            detect_gesture(frame, landmark_list, processed)

            # Display the processed frame
            cv2.imshow('Frame', frame)
            # Exit when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release resources when done
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
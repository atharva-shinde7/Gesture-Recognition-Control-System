"""
Utility functions for hand gesture recognition.
This module provides helper functions for calculating angles and distances between hand landmarks.
"""

import numpy as np

def get_angle(a, b, c):
    """
    Calculate the angle between three points (landmarks).
    Used to determine finger bending angles.
    
    Args:
        a: First point coordinates (x, y)
        b: Middle point coordinates (x, y) - this is the vertex of the angle
        c: Last point coordinates (x, y)
        
    Returns:
        float: Angle in degrees between the three points
    """
    # Calculate angle using arctan2 for accurate angle measurement
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    # Convert to absolute degrees (0-180)
    angle = np.abs(np.degrees(radians))
    return angle


def get_distance(landmark_ist):
    """
    Calculate the Euclidean distance between two landmarks.
    Used to determine how far apart two fingers are.
    
    Args:
        landmark_ist: List containing two landmarks, each with (x, y) coordinates
        
    Returns:
        float: Normalized distance value (0-1000)
        None: If input list has less than 2 landmarks
    """
    if len(landmark_ist) < 2:
        return
    # Extract coordinates
    (x1, y1), (x2, y2) = landmark_ist[0], landmark_ist[1]
    # Calculate Euclidean distance
    L = np.hypot(x2 - x1, y2 - y1)
    # Normalize distance to range 0-1000 for easier thresholding
    return np.interp(L, [0, 1], [0, 1000])
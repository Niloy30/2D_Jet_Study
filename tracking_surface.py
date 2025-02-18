# DEPRECATED. DON'T USE. Will delete after I'm done refering to it.

import os

import numpy as np
import pandas as pd

from edge_detection import detect_edges

frame_folder = r"C:\Users\niloy\Desktop\Shots 02112025\20250211_083548"

GAUSSIAN_BLUR_KERNEL = (5, 5)  # Kernel size for Gaussian blur
CANNY_THRESHOLD1 = 10  # Lower threshold for Canny edge detection
CANNY_THRESHOLD2 = 150  # Upper threshold for Canny edge detection
GAMMA = 0.1  # Gamma correction factor (1.0 = no change)
SPLINE_SMOOTHING = 1500  # Spline smoothing factor (higher = smoother, fewer details)
INTERP_RESOLUTION = 200  # Number of interpolation points for spline
LEVELS_ALPHA = 10  # Contrast scaling factor
LEVELS_BETA = 2  # Brightness adjustment


# Get list of all .bmp files in the folder
frames = [f for f in os.listdir(frame_folder) if f.lower().endswith(".bmp")]


# Pre-allocate array for edge vs frames for speed
n = len(frames)
edges = np.empty((n, 2, 200))

# Print the list of .bmp files
for i in range(n - 1):
    FRAME1_PATH = os.path.join(frame_folder, frames[i])
    FRAME2_PATH = os.path.join(frame_folder, frames[i + 1])
    try:
        edge = detect_edges(
            FRAME1_PATH,
            FRAME2_PATH,
            GAMMA,
            GAUSSIAN_BLUR_KERNEL,
            LEVELS_ALPHA,
            LEVELS_BETA,
            CANNY_THRESHOLD1,
            CANNY_THRESHOLD2,
            SPLINE_SMOOTHING,
            INTERP_RESOLUTION,
        )
        edges[i] = edge
    # plt.plot(edge[0], edge[1], "b-", linewidth=2)  # Plot spline
    # image = cv2.imread(FRAME1_PATH)
    # plt.imshow(image)
    # plt.title(f"Frame {i}")
    # plt.pause(0.005)
    # plt.clf()
    except:
        print(f"Frame {i} Didn't work")

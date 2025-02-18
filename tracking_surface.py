import os

import matplotlib.pyplot as plt
import numpy as np

from edge_detection import detect_edges

frame_folder = r"C:\Users\niloy\Desktop\Shots 02112025\20250211_082312"

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

edges = np.empty(len(frames))


# Print the list of .bmp files
for i in range(len(frames) - 1):
    FRAME1_PATH = os.path.join(frame_folder, frames[i])
    FRAME2_PATH = os.path.join(frame_folder, frames[i + 1])

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

    # plt.plot(edge[0], edge[1], "r-", linewidth=2)  # Plot spline
    # plt.title(f"Frame {i}")
    # plt.xlim(0, 500)
    # plt.ylim(0, 500)
    # plt.gca().invert_yaxis()  # This explicitly flips the y-axis
    # plt.pause(0.005)
    # plt.clf()


plt.show()

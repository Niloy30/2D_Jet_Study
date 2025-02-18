"""
Filename: main.py
Author: Niloy Barua
Date: 2025-02-17
Version: 1.2
Description: Surface tracking using frame differencing.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep

# ================= Configuration Parameters =================
FRAME1_PATH = r"E:\FDL\2D Jet Study Experiments\2025-02-11\20250211_082312\20250211_082312000100.bmp"
FRAME2_PATH = r"E:\FDL\2D Jet Study Experiments\2025-02-11\20250211_082312\20250211_082312000101.bmp"

GAUSSIAN_BLUR_KERNEL = (5, 5)  # Kernel size for Gaussian blur
CANNY_THRESHOLD1 = 10  # Lower threshold for Canny edge detection
CANNY_THRESHOLD2 = 150  # Upper threshold for Canny edge detection
GAMMA = 1.25  # Gamma correction factor (1.0 = no change)
SPLINE_SMOOTHING = 1000  # Spline smoothing factor (higher = smoother, fewer details)
INTERP_RESOLUTION = 200  # Number of interpolation points for spline

# ================== Load and Preprocess Frames ==================
frame1 = cv2.imread(FRAME1_PATH, cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread(FRAME2_PATH, cv2.IMREAD_GRAYSCALE)

# Compute frame difference
frame_diff = cv2.absdiff(frame1, frame2)


# Apply Gamma Correction (Optional for contrast adjustment)
def apply_gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


frame_diff = apply_gamma_correction(frame_diff, GAMMA)

cv2.imshow("Frame Difference", frame_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Gaussian blur
blurred = cv2.GaussianBlur(frame_diff, GAUSSIAN_BLUR_KERNEL, 0)

# Edge detection (Canny)
edges = cv2.Canny(blurred, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

# Get nonzero edge points
y, x = np.nonzero(edges)  # Get pixel indices where edges are detected

# Sort x values for spline fitting
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]

# Fit spline
tck, u = splprep([x_sorted, y_sorted], s=SPLINE_SMOOTHING)

# Generate biased u_fine with more points near the center

u_linear = np.linspace(-1, 1, INTERP_RESOLUTION)  # Linear space from -1 to 1
u_fine = (u_linear + 1) / 2  # Normalize to range [0,1]

# Interpolate spline
x_smooth, y_smooth = splev(u_fine, tck)

# Plot results
plt.imshow(frame_diff, cmap="gray")  # Show frame difference as background
plt.plot(x_smooth, y_smooth, "r-", linewidth=2)  # Plot spline
plt.scatter(x_sorted, y_sorted, s=2, c="blue")  # Scatter plot of detected edge points
plt.show()

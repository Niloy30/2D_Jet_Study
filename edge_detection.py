import cv2
import numpy as np
from scipy.interpolate import splev, splprep


def preprocess_frame(image, gamma=1, blur_kernel=(1, 1), alpha=1, beta=1):
    """
    Apply gamma correction, Gaussian blur, and levels adjustment.
    """
    # Gamma Correction
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    gamma_corrected = cv2.LUT(image, table)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gamma_corrected, blur_kernel, 0)

    # Levels Adjustment (Contrast and Brightness)
    adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)

    return adjusted


def detect_edges(
    frame1_path,
    frame2_path,
    gamma,
    blur_kernel,
    alpha,
    beta,
    canny_threshold1,
    canny_threshold2,
    spline_smoothing,
    interp_resolution,
):
    """
    Perform edge detection and return the spline interpolated edge points.
    """
    # Load and preprocess frames
    frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

    f1_processed = preprocess_frame(frame1, gamma, blur_kernel, alpha, beta)
    f2_processed = preprocess_frame(frame2, gamma, blur_kernel, alpha, beta)

    # Compute frame difference
    frame_diff = cv2.absdiff(f1_processed, f2_processed)

    # Edge detection (Canny)
    edges = cv2.Canny(frame_diff, canny_threshold1, canny_threshold2)

    # Get nonzero edge points
    y, x = np.nonzero(edges)

    # Sort x values for spline fitting
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Fit spline
    tck, u = splprep([x_sorted, y_sorted], s=spline_smoothing)

    u_linear = np.linspace(-1, 1, interp_resolution)  # Linear space from -1 to 1
    u_fine = (u_linear + 1) / 2  # Normalize to range [0,1]

    # Interpolate spline
    x_smooth, y_smooth = splev(u_fine, tck)
    # Flip the y-coordinates for Matplotlib
    y_smooth = frame1.shape[0] - y_smooth

    edge = np.array([x_smooth, y_smooth])
    return edge

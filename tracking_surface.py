import os

import numpy as np

from edge_detection import detect_edges


def process_frames(
    frame_folder,
    M_rot,
    output_size,
    rotation_angle,
    gaussian_blur_kernel=(5, 5),
    canny_threshold1=10,
    canny_threshold2=150,
    gamma=0.1,
    spline_smoothing=1500,
    interp_resolution=200,
    levels_alpha=10,
    levels_beta=2,
    save_edges=False,
    save_path="edges.npy",
):
    """
    Processes a sequence of frames for edge detection.

    Parameters:
        frame_folder (str): Path to the folder containing BMP frames.
        gaussian_blur_kernel (tuple): Kernel size for Gaussian blur.
        canny_threshold1 (int): Lower threshold for Canny edge detection.
        canny_threshold2 (int): Upper threshold for Canny edge detection.
        gamma (float): Gamma correction factor.
        spline_smoothing (int): Spline smoothing factor.
        interp_resolution (int): Number of interpolation points for spline.
        levels_alpha (int): Contrast scaling factor.
        levels_beta (int): Brightness adjustment.
        save_edges (bool): Whether to save the edges array as a .npy file.
        save_path (str): Path to save the .npy file if save_edges is True.

    Returns:
        np.ndarray: The computed edges array.
    """
    frames = [f for f in os.listdir(frame_folder) if f.lower().endswith(".bmp")]
    frames.sort()  # Ensure frames are processed in order

    number_frames = len(frames)
    if number_frames < 2:
        raise ValueError("At least two frames are required for edge detection.")

    edges = np.full((2, interp_resolution, number_frames - 1), np.nan)

    counter = 0

    for i in range(number_frames - 1):
        frame1_path = os.path.join(frame_folder, frames[i])
        frame2_path = os.path.join(frame_folder, frames[i + 1])
        try:
            edge = detect_edges(
                frame1_path,
                frame2_path,
                gamma,
                gaussian_blur_kernel,
                levels_alpha,
                levels_beta,
                canny_threshold1,
                canny_threshold2,
                spline_smoothing,
                interp_resolution,
                M_rot,
                output_size,
                rotation_angle,
            )
            edges[:, :, counter] = edge
            counter += 1
        except Exception as e:
            pass
            # print(f"Frame {i} processing failed: {e}")

    edges = edges[:, :, 0:counter]

    if save_edges:
        np.save(rf"{save_path}\free_surface_data.npy", edges)

    return edges

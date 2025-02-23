import os

import cv2
import numpy as np


def detect_circles(
    experiment_path, save_obstacle_data=True, save_path=None, show=False
):
    """
    Detects circles in the first .bmp file found in a folder using the Hough Circle Transform.

    Converts OpenCV's coordinate system (origin at top-left) to Matplotlib's (origin at bottom-left).

    Parameters:
        experiment_path (str): Path to the folder containing frames.
        save_obstacle_data (bool): Whether to save detected circles as a .npy file.
        save_path (str): Folder to save the circles.npy file and overlayed image.

    Returns:
        np.ndarray: Array of detected circles with (x, y, radius) values in Matplotlib coordinates.
    """
    # Find the first .bmp file in the directory
    files = [f for f in os.listdir(experiment_path) if f.endswith(".bmp")]
    if not files:
        raise FileNotFoundError("No .bmp files found in the specified directory.")

    image_path = os.path.join(experiment_path, files[0])

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100,
    )

    # Convert OpenCV coordinates to Matplotlib's coordinate system
    if circles is not None:
        circles = np.uint16(
            np.around(circles)
        )  # Round the detected circle values and convert them to integers
        img_height = image.shape[
            0
        ]  # Get the height of the image (number of rows in pixels)

        for circle in circles[0, :]:  # Loop through each detected circle
            circle[1] = (
                img_height - circle[1]
            )  # Convert y-coordinate: OpenCV (top-left origin) → Matplotlib (bottom-left origin)

            # Draw the outer circle in blue
            cv2.circle(
                image, (circle[0], img_height - circle[1]), circle[2], (255, 0, 0), 2
            )

            # Draw the center of the circle in red
            cv2.circle(image, (circle[0], img_height - circle[1]), 2, (0, 0, 255), 3)

    # Save detected circles and overlayed image if requested
    if save_obstacle_data and save_path:
        np.save(os.path.join(save_path, "obstacle_data.npy"), circles)
        cv2.imwrite(os.path.join(save_path, "obstacle.png"), image)

    # Display the result
    if show:
        cv2.imshow("Detected Circles", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return circles[0] if circles is not None else None


# Example usage
# circles = detect_circles("./Sample Frames", save_obstacle_data=True, save_path="./Results")
# print(circles)

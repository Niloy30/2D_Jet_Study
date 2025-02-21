import cv2
import numpy as np

# Load the image
# path = r"./Sample Frames/test_multiple_obstacles.png"
path = r"./Sample Frames/20250211_082312000001.bmp"
image = cv2.imread(path)

# Check if the image is loaded properly
if image is None:
    raise FileNotFoundError(f"Could not load image at path: {path}")

# Convert the image to grayscale (HoughCircles requires a single-channel image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and smooth the image
blurred = cv2.GaussianBlur(gray, (9, 9), 2)  # Kernel size (9,9), sigma = 2

# Apply Canny edge detection to highlight edges in the image
edges = cv2.Canny(blurred, 50, 150)  # Thresholds set to detect strong and weak edges

# Use Hough Circle Transform to detect circles in the blurred image
# cv2.HoughCircles(...) returns circles defined by (x,y,r)
circles = cv2.HoughCircles(
    blurred,  # Input image (must be grayscale)
    cv2.HOUGH_GRADIENT,  # Detection method (Hough Gradient)
    dp=1.2,  # Inverse ratio of resolution (higher means less accuracy)
    minDist=50,  # Minimum distance between detected circles
    param1=50,  # Upper threshold for internal Canny edge detection
    param2=30,  # Threshold for center detection (lower means more circles)
    minRadius=20,  # Minimum circle radius
    maxRadius=100,  # Maximum circle radius
)


# If circles are detected, process and draw them
if circles is not None:
    circles = np.uint16(
        np.around(circles)
    )  # Convert detected circles to integer values
    for circle in circles[0, :]:  # Loop through all detected circles
        # Draw the outer circle (green)
        cv2.circle(image, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
        # Draw the center of the circle (red)
        cv2.circle(image, (circle[0], circle[1]), 2, (0, 0, 255), 3)

# # Display the image with detected circles
# cv2.imshow("Detected Circles", image)

# # Wait for a key press before closing the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(circles[0])

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = r"E:\FDL\2D Jet Study Experiments\05162025\192.168.0.10_C001H001S0003.bmp"


image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the chessboard pattern size
pattern_size = (12, 19)


# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

# Draw and display the corners if found
if ret:
    cv2.drawChessboardCorners(image, pattern_size, corners, ret)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Chessboard Corners")
    plt.axis("off")
    plt.show()
else:
    print("Chessboard pattern not found.")

dist = (max(corners[:, :, 1]).item() - min(corners[:, :, 1]).item()) / pattern_size[1]
# print("Distance is", dist)
scaling = 5 / dist
print("Scaling is ", scaling)


def get_transformation_matrix(image_path, pattern_size):
    """
    Detects a checkerboard grid and returns the rotation matrix that straightens the grid.

    Returns:
        M_rot: Affine rotation matrix to straighten the image.
        (w, h): Original image dimensions.
        angle_deg: Rotation angle (in degrees).
        corners: Detected corner positions.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    ret, corners = cv2.findChessboardCorners(img, pattern_size, None)
    if not ret:
        raise ValueError("Checkerboard not detected.")

    # Refine corner detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

    # Get the vector between the first and last corner in the first row
    corners = corners.reshape(-1, 2)
    num_cols = pattern_size[0]
    pt1 = corners[0]
    pt2 = corners[num_cols - 1]
    dx, dy = pt2 - pt1

    # Calculate angle in degrees
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    print(angle_deg)
    # Image center for rotation
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    print(min(-angle_deg, 180 - angle_deg, type=abs))
    # Calculate the matrix to rotate image to upright (angle -> 0)
    M_rot = cv2.getRotationMatrix2D(center, min(-angle_deg, 180 - angle_deg), 1.0)

    return M_rot, (w, h), angle_deg, corners


pattern_size = (12, 21)
M_rot, (w, h), angle, corners = get_transformation_matrix(image_path, pattern_size)
rotated_img = cv2.warpAffine(cv2.imread(image_path), M_rot, (w, h))
cv2.imshow("Straightened", rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

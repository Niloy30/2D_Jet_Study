# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_transformation_matrix(image_path, pattern_size):
    """Detects a checkerboard grid and returns the rotation transformation matrix."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        raise ValueError("Checkerboard not detected.")

    corners = corners.reshape(-1, 2)
    pt1, pt2 = corners[0], corners[1]
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    rotation_angle = np.degrees(np.arctan2(dy, dx))  # Negate to correct orientation

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    return M_rot, (w, h), rotation_angle


def apply_transformation(image_path, M_rot, output_size):
    """Applies the given transformation matrix to an image and returns the corrected image."""
    img = cv2.imread(image_path)
    corrected_img = cv2.warpAffine(img, M_rot, output_size)
    return corrected_img


def crop_image(img, crop_ratio=0.85):
    """Crops the inner portion of the image based on the given crop ratio."""
    h, w = img.shape[:2]
    crop_margin_w = int(w * (1 - crop_ratio) / 2)
    crop_margin_h = int(h * (1 - crop_ratio) / 2)
    cropped_img = img[
        crop_margin_h : h - crop_margin_h, crop_margin_w : w - crop_margin_w
    ]
    return cropped_img


# Example usage
image_with_grid = "Sample Frames/calibration_grid.bmp"
new_image = "Sample Frames/shot_20250317_123839000229.bmp"
pattern_size = (13, 18)

M_rot, output_size, rotation_angle = get_transformation_matrix(
    image_with_grid, pattern_size
)
rotated_img = apply_transformation(new_image, M_rot, output_size)
cropped_img = crop_image(rotated_img)

# Save and display result
output_path = "corrected_image.bmp"
cv2.imwrite(output_path, cropped_img)

plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"Corrected & Cropped Image ({rotation_angle:.2f}°)")
plt.show()

# %%

# %%
import cv2
import numpy as np


def get_transformation_matrix(image_path, pattern_size):
    """Detects a checkerboard grid and returns the rotation transformation matrix."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    ret, corners = cv2.findChessboardCorners(img, pattern_size, None)
    if not ret:
        raise ValueError("Checkerboard not detected.")

    corners = corners.reshape(-1, 2)
    pt1, pt2 = corners[0], corners[1]
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    rotation_angle = min(
        np.degrees(np.arctan2(dy, dx)), 180 - np.degrees(np.arctan2(dy, dx))
    )

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    return M_rot, (w, h), rotation_angle, corners


def apply_transformation(image_path, M_rot, output_size):
    """Applies the given transformation matrix to an image and returns the corrected image."""
    img = cv2.imread(image_path)
    corrected_img = cv2.warpAffine(img, M_rot, output_size)
    return corrected_img


def crop_image(img, crop_ratio=0.95):
    """Crops the inner portion of the image based on the given crop ratio."""
    h, w = img.shape[:2]
    crop_margin_w = int(w * (1 - crop_ratio) / 2)
    crop_margin_h = int(h * (1 - crop_ratio) / 2)
    cropped_img = img[
        crop_margin_h : h - crop_margin_h, crop_margin_w : w - crop_margin_w
    ]
    return cropped_img


def get_scaling(image_with_grid, pattern_size):
    M_rot, output_size, rotation_angle, corners = get_transformation_matrix(
        image_with_grid, pattern_size
    )
    rotated_img = apply_transformation(image_with_grid, M_rot, output_size)
    #cropped_img = crop_image(rotated_img)
    ret, corners = cv2.findChessboardCorners(rotated_img, pattern_size, None)
    dist = (max(corners[:, :, 1]).item() - min(corners[:, :, 1]).item()) / pattern_size[
        1
    ]
    # print("Distance is", dist)
    scaling = 5 / dist
    return scaling


# # Example usage
# image_with_grid = "Sample Frames/calibration_grid.bmp"
# # new_image = "Sample Frames/shot_20250317_123839000229.bmp"
# new_image = "Sample Frames/calibration_grid.bmp"
# pattern_size = (13, 18)

# M_rot, output_size, rotation_angle, corners = get_transformation_matrix(
#     image_with_grid, pattern_size
# )
# rotated_img = apply_transformation(new_image, M_rot, output_size)
# cropped_img = crop_image(rotated_img)

# # Save and display result
# output_path = "corrected_image.bmp"
# cv2.imwrite(output_path, cropped_img)

# plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.title(f"Corrected & Cropped Image ({rotation_angle:.2f}°)")
# plt.show()
# # %%
# ret, corners = cv2.findChessboardCorners(cropped_img, (13, 15), None)
# plt.scatter(corners[:, :, 0], corners[:, :, 1], marker=".")
# plt.show()

# # %%

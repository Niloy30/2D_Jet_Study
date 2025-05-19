import cv2
import matplotlib.pyplot as plt
import numpy as np


def mask_image_borders(img, crop_ratio=1):
    """Masks the outer portion of the image with black pixels based on the given crop ratio."""
    h, w = img.shape[:2]
    crop_margin_w = int(w * (1 - crop_ratio) / 2)
    crop_margin_h = int(h * (1 - crop_ratio) / 2)

    masked_img = np.zeros_like(img)
    masked_img[crop_margin_h:h - crop_margin_h, crop_margin_w:w - crop_margin_w] = \
        img[crop_margin_h:h - crop_margin_h, crop_margin_w:w - crop_margin_w]

    return masked_img

# ====== CONFIGURATION ======
file_path = r'C:\Users\niloy\Desktop\Experiments\05022025\20250502_142543\20250502_142543000001.bmp'  # <- Replace this with your image path
crop_ratio = 0.95  # <- Adjust the crop ratio (0 to 1)

# ====== LOAD AND PROCESS ======
img = cv2.imread(file_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {file_path}")

# Convert BGR (OpenCV default) to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
masked_img = mask_image_borders(img_rgb, crop_ratio=crop_ratio)

# ====== DISPLAY ======
plt.imshow(masked_img)
plt.title(f"Masked Image (crop_ratio={crop_ratio})")
plt.axis('off')
plt.show()

import os
from multiprocessing import Pool, cpu_count

import cv2
from PIL import Image
from tqdm import tqdm

# Enable OpenCL (GPU Acceleration if supported by OpenCV)
cv2.ocl.setUseOpenCL(True)


def load_image(image_path):
    """Load and process an image"""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert OpenCV format to Pillow format
    return Image.fromarray(img)


def convert_to_gif(
    input_folder, output_folder, output_filename="output.gif", fps=10, loop=0
):
    """Convert BMP images in a folder to a GIF"""
    images = sorted(
        [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.endswith(".bmp")
        ]
    )

    if not images:
        print("No BMP images found in the specified folder.")
        return

    print(f"Processing {len(images)} images with {cpu_count()} cores...")

    # Use multiprocessing to load images in parallel with a progress bar
    with Pool(cpu_count()) as pool:
        frames = list(
            tqdm(
                pool.imap(load_image, images), total=len(images), desc="Loading Images"
            )
        )

    # Remove None values in case of failed loads
    frames = [f for f in frames if f is not None]

    if not frames:
        print("Failed to load images.")
        return

    # Calculate duration per frame (ms) based on FPS
    duration = int(1000 / fps)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    # Save the frames as a GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )
    print(f"GIF saved as {output_path}")


if __name__ == "__main__":
    input_folder = input(
        "Enter input folder path (where BMP images are located): "
    ).strip()
    output_folder = input(
        "Enter output folder path (where GIF will be saved): "
    ).strip()
    fps = int(input("Enter FPS (frames per second): "))

    convert_to_gif(input_folder, output_folder, fps=fps)

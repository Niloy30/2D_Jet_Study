import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\Users\niloy\Desktop\Experiments\192.168.0.10_C001H001S0004.bmp"


image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the chessboard pattern size
pattern_size = (9, 10)


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
print(scaling)
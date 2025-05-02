# %%
import cv2
import matplotlib.pyplot as plt

# %%
img = cv2.imread(
    r"C:\Users\niloy\Desktop\Experiments\04162025\192.168.0.10_C001H001S0003.bmp"
)

plt.hlines(30, 0, 1024)
plt.hlines(985, 0, 1024)
plt.imshow(img)

plt.show()


scaling = (225 - 110) / (985 - 30)
print(scaling)

# %%

frame_1 = cv2.imread(r"C:\Users\niloy\Desktop\Experiments\04162025\20250416_092059\20250416_092059000001.bmp")

frame_2 = cv2.imread(r"C:\Users\niloy\Desktop\Experiments\04162025\20250416_092059\20250416_092059000580.bmp")

frame_3 = cv2.imread(r"C:\Users\niloy\Desktop\Experiments\04162025\192.168.0.10_C001H001S0004.bmp")

# Create figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# First subplot
axes[0].imshow(frame_1)
axes[0].hlines(1010,0,1024)
axes[0].set_title("Initial Frame. At Rest")
# Second subplot
axes[1].imshow(frame_2)
axes[1].hlines(345,0,1024)
axes[1].set_title("Final Frame. During Experiment")
# Third subplot
axes[2].imshow(frame_3)
axes[2].hlines(475,0,1024)
axes[2].set_title("Post Experiment. At rest")
# Overall layout adjustments
plt.tight_layout()
plt.show()

scaling = 0.12041884816753927
print((1010-345) * scaling)
print((1010-475) * scaling)

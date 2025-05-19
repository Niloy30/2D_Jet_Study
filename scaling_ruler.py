import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk


class LineAdjusterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Line Adjuster with Scaling")

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        # Load image button
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=5)

        # Scaling label
        self.scaling_label = tk.Label(root, text="Scaling: N/A", font=("Arial", 14))
        self.scaling_label.pack(pady=5)

        # Frame for value entries
        self.entry_frame = tk.Frame(root)
        self.entry_frame.pack(pady=5)

        tk.Label(self.entry_frame, text="Top Value:").pack(side=tk.LEFT, padx=5)
        self.top_value_entry = tk.Entry(self.entry_frame, width=5)
        self.top_value_entry.pack(side=tk.LEFT, padx=5)
        self.top_value_entry.insert(0, "225")

        tk.Label(self.entry_frame, text="Bottom Value:").pack(side=tk.LEFT, padx=5)
        self.bottom_value_entry = tk.Entry(self.entry_frame, width=5)
        self.bottom_value_entry.pack(side=tk.LEFT, padx=5)
        self.bottom_value_entry.insert(0, "110")

        # Sliders for line positions
        self.slider_frame = tk.Frame(root)
        self.slider_frame.pack(pady=10)

        self.y1_slider = tk.Scale(self.slider_frame, from_=0, to=1000, label="Top Line Y", orient=tk.HORIZONTAL, command=self.update_lines)
        self.y1_slider.pack(side=tk.LEFT, padx=10)

        self.y2_slider = tk.Scale(self.slider_frame, from_=0, to=1000, label="Bottom Line Y", orient=tk.HORIZONTAL, command=self.update_lines)
        self.y2_slider.pack(side=tk.LEFT, padx=10)

        self.image = None
        self.tk_img = None
        self.y1 = 30
        self.y2 = 985

        # Trigger updates when entry content changes
        self.top_value_entry.bind("<KeyRelease>", self.update_lines)
        self.bottom_value_entry.bind("<KeyRelease>", self.update_lines)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        # Load and convert the image
        bgr_image = cv2.imread(file_path)
        self.image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Resize image for display
        self.display_image = self.image.copy()
        self.show_image()

        # Update slider bounds
        height = self.image.shape[0]
        self.y1_slider.config(to=height-1)
        self.y2_slider.config(to=height-1)
        self.y1_slider.set(30)
        self.y2_slider.set(height - 30)

    def show_image(self):
        if self.image is None:
            return

        img_with_lines = self.display_image.copy()

        # Draw horizontal lines
        cv2.line(img_with_lines, (0, self.y1), (img_with_lines.shape[1], self.y1), (255, 0, 0), 2)
        cv2.line(img_with_lines, (0, self.y2), (img_with_lines.shape[1], self.y2), (0, 255, 0), 2)

        # Convert to PIL image and display
        img_pil = Image.fromarray(img_with_lines)
        img_pil = img_pil.resize((800, 600), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def update_lines(self, _=None):
        if self.image is None:
            return

        self.y1 = self.y1_slider.get()
        self.y2 = self.y2_slider.get()

        try:
            top_val = float(self.top_value_entry.get())
            bot_val = float(self.bottom_value_entry.get())
        except ValueError:
            self.scaling_label.config(text="Scaling: Invalid input")
            return

        if self.y1 == self.y2:
            self.scaling_label.config(text="Scaling: Undefined (div by zero)")
        else:
            scaling = (top_val - bot_val) / (self.y2 - self.y1)
            self.scaling_label.config(text=f"Scaling: {scaling:.5f}")

        self.show_image()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = LineAdjusterApp(root)
    root.mainloop()

# # %%

# frame_1 = cv2.imread(r"C:\Users\niloy\Desktop\Experiments\04162025\20250416_092059\20250416_092059000001.bmp")

# frame_2 = cv2.imread(r"C:\Users\niloy\Desktop\Experiments\04162025\20250416_092059\20250416_092059000580.bmp")

# frame_3 = cv2.imread(r"C:\Users\niloy\Desktop\Experiments\04162025\192.168.0.10_C001H001S0004.bmp")

# # Create figure and subplots
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# # First subplot
# axes[0].imshow(frame_1)
# axes[0].hlines(1010,0,1024)
# axes[0].set_title("Initial Frame. At Rest")
# # Second subplot
# axes[1].imshow(frame_2)
# axes[1].hlines(345,0,1024)
# axes[1].set_title("Final Frame. During Experiment")
# # Third subplot
# axes[2].imshow(frame_3)
# axes[2].hlines(475,0,1024)
# axes[2].set_title("Post Experiment. At rest")
# # Overall layout adjustments
# plt.tight_layout()
# plt.show()

# scaling = 0.12041884816753927
# print((1010-345) * scaling)
# print((1010-475) * scaling)

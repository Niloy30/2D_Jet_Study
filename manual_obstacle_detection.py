import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk


class CircleDrawerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Circle Drawer")
        self.root.geometry("900x650")
        self.root.minsize(400, 400)
        self.crop_percent = 0

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Upload + Save + Crop
        top_frame = tk.Frame(root)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure((0, 1, 2), weight=1)

        self.upload_btn = tk.Button(top_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.save_btn = tk.Button(top_frame, text="Save", command=self.save_image, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.crop_entry = tk.Entry(top_frame)
        self.crop_entry.insert(0, "0")  # default 0%
        self.crop_entry.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        self.crop_entry.bind("<Return>", self.on_crop_change)

        # Main layout
        main_frame = tk.Frame(root)
        main_frame.grid(row=1, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=4)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Label(main_frame, bg="gray")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        slider_frame = tk.Frame(main_frame)
        slider_frame.grid(row=0, column=1, sticky="nsew")

        self.setup_slider_with_arrows(slider_frame, "X")
        self.setup_slider_with_arrows(slider_frame, "Y")
        self.setup_slider_with_arrows(slider_frame, "Radius")

        self.root.bind("<Configure>", self.on_resize)

    def setup_slider_with_arrows(self, parent, label):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)

        slider = tk.Scale(frame, from_=0, to=1000, label=label, orient="horizontal", command=self.update_image)
        slider.pack(fill=tk.X)

        entry = tk.Entry(frame)
        entry.pack(fill=tk.X)
        entry.insert(0, "0")
        entry.bind("<Return>", self.update_slider_from_entry)

        arrow_frame = tk.Frame(frame)
        arrow_frame.pack()

        btn_down = tk.Button(arrow_frame, text="–", width=2, command=lambda l=label: self.adjust_slider(l, -1))
        btn_up = tk.Button(arrow_frame, text="+", width=2, command=lambda l=label: self.adjust_slider(l, +1))
        btn_down.pack(side=tk.LEFT, padx=2)
        btn_up.pack(side=tk.LEFT, padx=2)

        if label == "X":
            self.x_slider = slider
            self.x_entry = entry
        elif label == "Y":
            self.y_slider = slider
            self.y_entry = entry
        elif label == "Radius":
            self.r_slider = slider
            self.r_entry = entry

    def adjust_slider(self, label, delta):
        if label == "X":
            self.x_slider.set(self.x_slider.get() + delta)
        elif label == "Y":
            self.y_slider.set(self.y_slider.get() + delta)
        elif label == "Radius":
            self.r_slider.set(max(0, self.r_slider.get() + delta))
        self.update_image()

    def upload_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        self.original = cv2.imread(path)
        self.process_cropped_image()
        self.save_btn.config(state=tk.NORMAL)

    def process_cropped_image(self):
        if not hasattr(self, 'original'):
            return

        crop = max(0, min(100, self.crop_percent)) / 100.0
        h, w = self.original.shape[:2]
        crop_w = int(w * (1 - crop))
        crop_h = int(h * (1 - crop))

        center = (w // 2, h // 2)
        self.image = cv2.getRectSubPix(self.original, (crop_w, crop_h), center)
        self.height, self.width = self.image.shape[:2]

        self.x_slider.config(to=self.width)
        self.y_slider.config(to=self.height)
        self.r_slider.config(to=min(self.width, self.height) // 2)

        # Estimate initial circle using HoughCircles (Script 1)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=30,
            minRadius=10, maxRadius=min(self.width, self.height)//2
        )

        if circles is not None:
            circle = np.round(circles[0, 0]).astype("int")
            x, y, r = circle
        else:
            # Fallback: use center with 1/4 radius
            x, y, r = self.width // 2, self.height // 2, min(self.width, self.height) // 4

        self.x_slider.set(x)
        self.y_slider.set(y)
        self.r_slider.set(r)

        self.update_image()

    def on_crop_change(self, event=None):
        try:
            value = float(self.crop_entry.get())
            self.crop_percent = max(0, min(100, value))
            self.process_cropped_image()
        except ValueError:
            pass

    def update_image(self, event=None):
        if not hasattr(self, 'image'):
            return
        self.drawn_image = self.image.copy()
        x = self.x_slider.get()
        y = self.y_slider.get()
        r = self.r_slider.get()

        self.x_entry.delete(0, tk.END)
        self.x_entry.insert(0, str(x))
        self.y_entry.delete(0, tk.END)
        self.y_entry.insert(0, str(y))
        self.r_entry.delete(0, tk.END)
        self.r_entry.insert(0, str(r))

        cv2.circle(self.drawn_image, (x, y), r, (0, 0, 255), 2)
        self.show_image()

    def show_image(self):
        if not hasattr(self, 'drawn_image'):
            return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            return
        resized = self.resize_to_fit(self.drawn_image, canvas_width, canvas_height)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.config(image=self.tk_img)

    def resize_to_fit(self, image, max_width, max_height):
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        return cv2.resize(image, (int(w * scale), int(h * scale)))

    def on_resize(self, event):
        self.show_image()

    def update_slider_from_entry(self, event):
        try:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            r = int(self.r_entry.get())

            self.x_slider.set(x)
            self.y_slider.set(y)
            self.r_slider.set(r)

            self.update_image()
        except ValueError:
            pass

    def save_image(self):
        if not hasattr(self, 'image'):
            return

        x = self.x_slider.get()
        y = self.y_slider.get()
        r = self.r_slider.get()

        circle_coords = np.array([[[x, y, r]]], dtype=np.float32)

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )

        if path:
            np.savetxt(path, circle_coords.reshape(-1, 3), fmt="%f", header="x, y, radius")


def run_manual_circle_detection():
    root = tk.Tk()
    app = CircleDrawerApp(root)
    root.mainloop()


# Uncomment to run directly
run_manual_circle_detection()

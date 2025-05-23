import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


class SVMPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SVM Surface Feature Classifier with Controls")

        # Initialize parameters
        self.C = 100.0
        self.gamma = 0.0004
        self.m_threshold = 21.0

        # Load Button
        self.load_button = tk.Button(root, text="Load Excel File", command=self.load_data)
        self.load_button.pack(pady=10)

        # Sliders Frame
        self.slider_frame = tk.Frame(root)
        self.slider_frame.pack(pady=5)

        # C Slider
        self.c_slider = self.create_slider("C", 1, 500, self.C, self.update_c)
        # Gamma Slider (log scale)
        self.gamma_slider = self.create_slider("Gamma (x10⁻⁴)", 1, 100, int(self.gamma * 10000), self.update_gamma)
        # Threshold Slider
        self.thresh_slider = self.create_slider("m Threshold", 0, 50, self.m_threshold, self.update_threshold)

        # Plot Button
        self.plot_button = tk.Button(root, text="Plot SVM Decision Boundary", command=self.plot_data, state=tk.DISABLED)
        self.plot_button.pack(pady=10)

        # Placeholder for figure
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = None

    def create_slider(self, label, frm, to, val, command):
        frame = tk.Frame(self.slider_frame)
        frame.pack()
        tk.Label(frame, text=label).pack(side=tk.LEFT, padx=5)
        slider = tk.Scale(frame, from_=frm, to=to, orient=tk.HORIZONTAL, resolution=0.1, length=300, command=command)
        slider.set(val)
        slider.pack(side=tk.LEFT)
        return slider

    def update_c(self, val):
        self.C = float(val)
        self.plot_data()

    def update_gamma(self, val):
        self.gamma = float(val) / 10000  # convert back from x10⁻⁴
        self.plot_data()

    def update_threshold(self, val):
        self.m_threshold = float(val)
        self.plot_data()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return

        try:
            self.df = pd.read_excel(file_path)
            self.plot_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "File loaded successfully!")
            self.plot_data()  # Auto-plot after loading
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

    def plot_data(self):
        if not hasattr(self, 'df'):
            return

        h = self.df["h"]
        m = self.df["m"]
        phase = self.df["Phase"]
        who = self.df["Who"]

        X = np.column_stack((h, m))
        y = LabelEncoder().fit_transform(phase)
        weights = np.where(m > self.m_threshold, 0.025, 1.0)

        clf = SVC(kernel="rbf", C=self.C, gamma=self.gamma)
        clf.fit(X, y, sample_weight=weights)

        colors = ["blue" if t == "Gravity Wave" else "red" for t in phase]
        alphas = [0.7 if person == "Rubert" else 0.7 for person in who]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title("Surface Features by Height, m, Diameter, and Type")
        ax.set_xlabel("Height (h)")
        ax.set_ylabel("m")
        ax.grid(True, linestyle="--", alpha=0.5)

        for xi, yi, ci, ai in zip(h, m, colors, alphas):
            ax.scatter(xi, yi, s=50, c=ci, alpha=ai, edgecolors="k")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                             np.linspace(*ylim, num=200))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contour(xx, yy, Z, colors="k", levels=[0], alpha=0.8, linestyles=["-"])

        red_patch = mpatches.Patch(color="red", label="Jet")
        blue_patch = mpatches.Patch(color="blue", label="Gravity Wave")
        ax.legend(handles=[blue_patch, red_patch])

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(padx=10, pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = SVMPlotApp(root)
    root.mainloop()

"""AXON interactive native digit laboratory. Does not use third-party ML models."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .preprocess import image_to_features
from .corrections import save_correction

BG = "#10151c"
PANEL = "#1b222c"
WHITE = "#e9f1f8"
MUTED = "#98a7b8"
BLUE = "#7bb8ff"


class DigitDrawer:
    def __init__(self, model):
        self.model = model
        self.size = 28 if model.sizes[0] == 784 else 8
        self.root = tk.Tk()
        self.root.title("AXON — Neural Network Laboratory")
        self.root.geometry("1090x730")
        self.root.minsize(970, 680)
        self.root.configure(bg=BG)
        self.canvas_size = 320
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.sample_features = None
        self.last = None
        self.preview = None
        self.sample_pointer = 0
        self.rows = []
        self.dataset = None
        self.explorers = []
        self.actual_label = tk.StringVar(value="6")
        self.build()
        self.next_sample()

    def build(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=30, pady=(20, 6))
        tk.Label(top, text="AXON", font=("Helvetica", 25, "bold"), bg=BG, fg=WHITE).pack(side="left")
        tk.Label(top, text="  /  NEURAL NETWORK LAB", font=("Helvetica", 11), bg=BG, fg=MUTED).pack(side="left", pady=(12,0))
        tk.Label(self.root, text=f"NumPy-only backpropagation  /  {self.size} × {self.size} pixels  /  10 digit classes",
                 font=("Helvetica", 11), bg=BG, fg=MUTED, anchor="w").pack(fill="x", padx=30, pady=(0,17))

        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=30)
        left = tk.Frame(content, bg=BG)
        left.pack(side="left", anchor="n")
        tk.Label(left, text="01  /  DRAW A DIGIT", font=("Helvetica", 10, "bold"), bg=BG, fg=BLUE).pack(anchor="w", pady=(0,10))
        self.canvas = tk.Canvas(left, width=self.canvas_size, height=self.canvas_size, bg="#03070b",
                                highlightthickness=1, highlightbackground="#354050", cursor="pencil")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        buttons = tk.Frame(left, bg=BG)
        buttons.pack(fill="x", pady=(11,0))
        self.button(buttons, "PREDICT", self.predict, primary=True).pack(side="left", padx=(0,7))
        self.button(buttons, "CLEAR", self.clear).pack(side="left", padx=(0,7))
        self.button(buttons, "NEXT SAMPLE", self.next_sample).pack(side="left")
        explore = tk.Frame(left, bg=BG)
        explore.pack(fill="x", pady=(10, 0))
        self.button(explore, "VIEW LIVE NEURAL NETWORK  ↗", self.open_explorer, primary=True).pack(side="left")
        tk.Label(left, text="If AXON guesses wrong, save the correct answer:",
                 font=("Helvetica", 10), bg=BG, fg=MUTED).pack(anchor="w", pady=(19,8))
        corrections = tk.Frame(left, bg=BG)
        corrections.pack(fill="x")
        tk.Label(corrections, text="ACTUAL DIGIT", font=("Helvetica", 9, "bold"),
                 bg=BG, fg=WHITE).pack(side="left", padx=(0,9))
        self.actual = tk.Spinbox(corrections, from_=0, to=9, wrap=True, textvariable=self.actual_label,
                                  width=3, font=("Helvetica", 13), justify="center")
        self.actual.pack(side="left", padx=(0,10))
        self.button(corrections, "SAVE CORRECTION", self.save_feedback).pack(side="left")
        tk.Label(left, text="Saved privately on this Mac. Retrain with: python -m axon personalize",
                 font=("Helvetica", 9), bg=BG, fg=MUTED, wraplength=320,
                 justify="left").pack(anchor="w", pady=(13,0))

        right = tk.Frame(content, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(35, 0), anchor="n")
        tk.Label(right, text="02  /  PREDICTION", font=("Helvetica", 10, "bold"),
                 bg=BG, fg=BLUE).pack(anchor="w", pady=(0,10))
        result = tk.Frame(right, bg=PANEL, padx=20, pady=18)
        result.pack(fill="x")
        self.result = tk.Label(result, text="—", font=("Helvetica", 54, "bold"), bg=PANEL, fg=WHITE)
        self.result.pack(side="left")
        details = tk.Frame(result, bg=PANEL)
        details.pack(side="left", padx=20)
        self.result_caption = tk.Label(details, text="Awaiting input", font=("Helvetica", 12, "bold"),
                                        bg=PANEL, fg=WHITE)
        self.result_caption.pack(anchor="w")
        self.result_score = tk.Label(details, text="Softmax score: —", font=("Helvetica", 10),
                                      bg=PANEL, fg=MUTED)
        self.result_score.pack(anchor="w", pady=(7,0))
        tk.Label(right, text="SOFTMAX OUTPUTS  /  0–9", font=("Helvetica", 10, "bold"),
                 bg=BG, fg=MUTED).pack(anchor="w", pady=(21,10))
        for digit in range(10):
            row = tk.Frame(right, bg=BG)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=str(digit), font=("Helvetica", 11), bg=BG, fg=WHITE, width=2).pack(side="left")
            bar = tk.Canvas(row, height=13, bg="#28313d", highlightthickness=0)
            bar.pack(side="left", fill="x", expand=True, padx=(10,11))
            val = tk.Label(row, text="0.0%", font=("Helvetica", 10), bg=BG, fg=MUTED, width=8, anchor="e")
            val.pack(side="right")
            self.rows.append((bar,val))
        self.status = tk.Label(self.root, text="Loading...", font=("Helvetica", 9),
                               bg=BG, fg=MUTED, anchor="w")
        self.status.pack(fill="x", padx=30, pady=(7,15))

    def button(self, parent, label, command, *, primary=False):
        # Labels are used for custom controls; native macOS Tk buttons can ignore
        # dark colors, resulting in unreadable white-on-white button text.
        button = tk.Label(parent, text=label, font=("Helvetica", 10, "bold"),
                          bg=BLUE if primary else PANEL, fg="#091520" if primary else WHITE,
                          padx=12, pady=10, cursor="hand2")
        button.bind("<Button-1>", lambda _event: command())
        return button

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.sample_features = None
        self.last = None
        self.preview = None
        self.result.configure(text="—")
        self.result_caption.configure(text="Awaiting input", fg=WHITE)
        self.result_score.configure(text="Softmax score: —")
        for bar, val in self.rows:
            bar.delete("all")
            val.configure(text="0.0%")
        self.status.configure(text="FREEHAND  /  Draw a digit, then choose PREDICT")

    def start_stroke(self, event):
        if self.sample_features is not None:
            self.clear()
        self.last = (event.x,event.y)
        self.stroke(event)

    def stroke(self, event):
        p = (max(0, min(self.canvas_size-1,event.x)), max(0,min(self.canvas_size-1,event.y)))
        r=10
        self.canvas.create_oval(p[0]-r,p[1]-r,p[0]+r,p[1]+r,fill="white",outline="white")
        self.pen.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=255)
        if self.last is not None:
            self.canvas.create_line(*self.last,*p,fill="white",width=20,capstyle=tk.ROUND,smooth=True)
            self.pen.line((*self.last,*p),fill=255,width=20)
        self.last=p

    def end_stroke(self, _event):
        self.last=None

    def next_sample(self):
        self.clear()
        if self.dataset is None:
            try:
                if self.size == 28:
                    from .mnist import load_mnist
                    self.dataset = load_mnist(seed=int(self.model.metadata.get("split_seed",42)))
                else:
                    from .data import load_digits_data
                    self.dataset = load_digits_data(seed=int(self.model.metadata.get("split_seed",42)))
            except (RuntimeError, ValueError) as exc:
                self.status.configure(text=f"Reference samples unavailable: {exc}")
                return
        index = self.sample_pointer % len(self.dataset.y_test)
        self.sample_pointer += 1
        self.sample_features = self.dataset.x_test[index:index+1]
        a = np.uint8(np.clip(self.sample_features.reshape(self.size,self.size)*255,0,255))
        image = Image.fromarray(a,mode="L").resize((self.canvas_size,self.canvas_size), Image.Resampling.NEAREST)
        self.preview = ImageTk.PhotoImage(image)
        self.canvas.create_image(0,0,image=self.preview,anchor="nw")
        self.predict()
        self.status.configure(text=f"REFERENCE DATA / ACTUAL DIGIT {self.dataset.y_test[index]} / TEST IMAGE #{index+1}")

    def features(self):
        return self.sample_features if self.sample_features is not None else image_to_features(self.image,size=self.size)

    def predict(self):
        x=self.features()
        if x.max() < .02:
            self.result_caption.configure(text="Draw a digit first")
            return
        probs = self.model.predict_proba(x)[0]
        self._sync_explorers(x)
        best = int(probs.argmax())
        ordered = np.sort(probs)
        uncertain = probs[best] < 0.60 or ordered[-1]-ordered[-2] < 0.2
        self.result.configure(text=str(best))
        self.result_caption.configure(text="Uncertain · possible digit" if uncertain else "Predicted digit",
                                      fg="#f3c67a" if uncertain else WHITE)
        self.result_score.configure(text=f"Softmax score: {probs[best]:.1%} (not calibrated)")
        for i,(bar,val) in enumerate(self.rows):
            bar.update_idletasks()
            bar.delete("all")
            bar.create_rectangle(0,0,max(bar.winfo_width(),100)*float(probs[i]),13,
                                 fill=BLUE if i==best else "#526e90", outline="")
            val.configure(text=f"{probs[i]*100:.1f}%")
        if self.sample_features is None:
            self.status.configure(text="FREEHAND / If incorrect, specify actual digit and save the correction")

    def open_explorer(self):
        from .neuroviz import open_network_explorer
        try:
            explorer = open_network_explorer(self.root, self.model, self.features())
            self.explorers.append(explorer)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Network explorer", str(exc))

    def _sync_explorers(self, x):
        alive = []
        for explorer in self.explorers:
            if explorer.window.winfo_exists():
                explorer.update_input(x)
                alive.append(explorer)
        self.explorers = alive

    def save_feedback(self):
        if self.sample_features is not None:
            messagebox.showinfo("AXON", "Draw your own digit before saving a correction.")
            return
        try:
            label=int(self.actual_label.get())
            path = save_correction(self.image,label)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Cannot save",str(exc))
            return
        self.status.configure(text=f"SAVED CORRECTION: {path}  /  Run python -m axon personalize to learn it")

    def run(self):
        self.root.mainloop()


def launch(model):
    if model.sizes[0] not in (64,784) or model.sizes[-1] != 10:
        raise ValueError("AXON desktop demo requires an 8x8 or 28x28 digit model")
    DigitDrawer(model).run()

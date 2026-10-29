"""Inspect real activations and weighted connections from a NumPy AXON network.

This is an *inference* trace: weights do not change while it is animated.
Only a bounded number of neurons and edges are rendered, for clarity and speed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NetworkTrace:
    sizes: tuple[int, ...]
    activations: tuple[np.ndarray, ...]
    preactivations: tuple[np.ndarray, ...]
    probabilities: np.ndarray
    predicted: int


def trace_forward(model, features: np.ndarray) -> NetworkTrace:
    """Use the *actual trained model* and return one input's full forward pass."""
    x = np.asarray(features, dtype=model.dtype)
    if x.shape != (1, model.sizes[0]):
        raise ValueError(f"expected a single sample (1, {model.sizes[0]}), got {x.shape}")
    logits, (acts, zs) = model.forward(x, cache=True)
    probs = model.softmax(logits)[0].copy()
    outputs = [np.asarray(a[0], dtype=np.float64).copy() for a in acts]
    outputs[-1] = probs.copy()  # Render output *probabilities*, not raw logits.
    return NetworkTrace(tuple(model.sizes), tuple(outputs),
                        tuple(np.asarray(z[0], dtype=np.float64).copy() for z in zs),
                        probs, int(probs.argmax()))


def displayed_neurons(values: np.ndarray, limit: int = 26) -> np.ndarray:
    """Keep high-signal neurons plus evenly spaced context when a layer is huge."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or limit < 2:
        raise ValueError("expected 1D activations and limit >= 2")
    if len(a) <= limit:
        return np.arange(len(a), dtype=np.int64)
    prominent = min(limit * 3 // 4, limit)
    # Stable tie breaking, so empty inputs don't appear to flicker.
    ranking = np.lexsort((np.arange(len(a)), -np.abs(a)))
    idx = set(map(int, ranking[:prominent]))
    for j in np.linspace(0, len(a) - 1, limit * 3, dtype=int):
        if len(idx) == limit:
            break
        idx.add(int(j))
    return np.array(sorted(idx), dtype=np.int64)


def strongest_edges(model, trace: NetworkTrace, layer: int,
                    source_ids: np.ndarray, target_ids: np.ndarray,
                    per_target: int = 4) -> list[tuple[int, int, float, float]]:
    """Return (source, target, trained_weight, actual signal=a*weight)."""
    if not 0 <= layer < len(model.weights) or per_target < 1:
        raise ValueError("invalid layer or per_target")
    w = model.weights[layer]
    ids = np.asarray(source_ids, dtype=int)
    targets = np.asarray(target_ids, dtype=int)
    if np.any(ids < 0) or np.any(ids >= w.shape[0]) or np.any(targets < 0) or np.any(targets >= w.shape[1]):
        raise ValueError("neuron index outside layer")
    if not len(ids) or not len(targets):
        return []
    out = []
    active = trace.activations[layer]
    for target in targets:
        weights = w[ids, target]
        contributions = active[ids] * weights
        # If zero signal, still show the strongest dormant synapses in muted tones.
        strength = np.abs(contributions) if np.any(contributions) else np.abs(weights)
        order = np.argsort(-strength, kind="stable")[:per_target]
        for k in order:
            out.append((int(ids[k]), int(target), float(weights[k]), float(contributions[k])))
    return out


def neuron_details(model, trace: NetworkTrace, layer: int, neuron: int) -> dict:
    if not 0 <= layer < len(trace.sizes) or not 0 <= neuron < trace.sizes[layer]:
        raise ValueError("neuron is outside the network")
    detail = {"layer": layer, "index": neuron, "activation": float(trace.activations[layer][neuron])}
    if layer > 0:
        weights = model.weights[layer - 1][:, neuron]
        signals = trace.activations[layer - 1] * weights
        detail.update(bias=float(model.biases[layer - 1][0, neuron]),
                      preactivation=float(trace.preactivations[layer - 1][neuron]),
                      inputs=len(weights),
                      strongest_input=int(np.argmax(np.abs(signals))),
                      strongest_contribution=float(signals[np.argmax(np.abs(signals))]))
    if layer < len(trace.sizes) - 1:
        detail["outputs"] = len(model.weights[layer][neuron])
    return detail


# Tk is imported only when opening the desktop explorer; headless tests use the functions above.
def open_network_explorer(parent, model, features):
    import tkinter as tk

    BG = "#0b1017"
    PANEL = "#141e2a"
    TEXT = "#e5edf6"
    MUTED = "#91a4b9"
    BLUE = "#75b6ff"
    POSITIVE = "#62cbb4"
    NEGATIVE = "#ff927d"

    class NetworkExplorer:
        def __init__(self):
            self.model = model
            self.trace = trace_forward(model, features)
            self.window = tk.Toplevel(parent)
            self.window.title("AXON — Neural Network Explorer")
            self.window.configure(bg=BG)
            self.window.geometry("1260x800")
            self.window.minsize(930, 620)
            self.layer_visible = 0
            self.selected = None
            self.nodes = {}
            self.after_id = None
            self.resize_id = None
            self.show_n = 26
            self.fan_in = 4
            self.playing = False
            self._build()
            self.window.protocol("WM_DELETE_WINDOW", self.close)
            self.draw()
            self.play()

        def _label(self, parent, message, size=10, color=TEXT, bold=False):
            return tk.Label(parent, text=message, bg=parent.cget("bg"), fg=color,
                            font=("Helvetica", size, "bold" if bold else "normal"))

        def _button(self, parent, name, callback, primary=False):
            b = tk.Label(parent, text=name, bg=BLUE if primary else PANEL,
                         fg="#06101b" if primary else TEXT, font=("Helvetica", 10, "bold"),
                         padx=13, pady=9, cursor="hand2")
            b.bind("<Button-1>", lambda e: callback())
            b.pack(side="left", padx=(0, 9))
            return b

        def _build(self):
            header = tk.Frame(self.window, bg=BG)
            header.pack(fill="x", padx=24, pady=(20, 4))
            self._label(header, "AXON", 23, TEXT, True).pack(side="left")
            self._label(header, "   /   NETWORK EXPLORER", 11, BLUE, True).pack(side="left", pady=(9, 0))
            self.header_stats = self._label(header, f"{sum(w.size+b.size for w,b in zip(self.model.weights, self.model.biases)):,} PARAMETERS   •   PREDICTION {self.trace.predicted}",
                                            10, MUTED)
            self.header_stats.pack(side="right", pady=(9, 0))
            self._label(self.window, "REAL WEIGHTS  /  REAL ACTIVATIONS  /  INFERENCE FLOW — NOT TRAINING",
                        10, MUTED).pack(anchor="w", padx=24, pady=(0, 13))
            controls = tk.Frame(self.window, bg=BG)
            controls.pack(fill="x", padx=24, pady=(0, 12))
            self.play_button = self._button(controls, "REPLAY FLOW", self.play, primary=True)
            self._button(controls, "NEXT LAYER", self.step)
            self._button(controls, "SHOW ALL", self.show_all)
            self.density_button = self._button(controls, "26 NODES / LAYER", self.toggle_density)
            self.edge_button = self._button(controls, "4 LINKS / NEURON", self.toggle_edges)
            self.canvas = tk.Canvas(self.window, bg="#0e1620", highlightthickness=1,
                                    highlightbackground="#263548", cursor="crosshair")
            self.canvas.pack(fill="both", expand=True, padx=24)
            self.canvas.bind("<Button-1>", self.select)
            self.canvas.bind("<Configure>", self.schedule_redraw)
            footer = tk.Frame(self.window, bg=BG)
            footer.pack(fill="x", padx=24, pady=(12, 18))
            self.info = self._label(footer, "Click a neuron to inspect its activation and strongest input.", 10, TEXT)
            self.info.pack(anchor="w")
            self._label(footer, "GREEN: positive weight   •   CORAL: negative weight   •   BRIGHTER: stronger signal   •   Only sampled neurons/links are shown",
                        9, MUTED).pack(anchor="w", pady=(6, 0))

        def stop(self):
            self.playing = False
            if self.after_id is not None:
                self.window.after_cancel(self.after_id)
                self.after_id = None

        def close(self):
            self.stop()
            if self.resize_id is not None:
                self.window.after_cancel(self.resize_id)
                self.resize_id = None
            self.window.destroy()

        def update_input(self, x):
            self.trace = trace_forward(self.model, x)
            self.header_stats.config(text=f"{sum(w.size+b.size for w,b in zip(self.model.weights, self.model.biases)):,} PARAMETERS   •   PREDICTION {self.trace.predicted}")
            self.info.config(text="New input loaded. Click a neuron to inspect its actual activation.")
            self.selected = None
            self.layer_visible = len(self.model.weights)
            self.draw()

        def play(self):
            self.stop()
            self.layer_visible = 0
            self.selected = None
            self.playing = True
            self.draw()
            self.after_id = self.window.after(510, self._animate)

        def _animate(self):
            self.after_id = None
            if not self.playing or not self.window.winfo_exists():
                return
            if self.layer_visible < len(self.model.weights):
                self.layer_visible += 1
                self.draw()
                self.after_id = self.window.after(540, self._animate)
            else:
                self.playing = False

        def step(self):
            self.stop()
            self.layer_visible = min(len(self.model.weights), self.layer_visible + 1)
            self.draw()

        def show_all(self):
            self.stop()
            self.layer_visible = len(self.model.weights)
            self.draw()

        def toggle_density(self):
            self.show_n = 36 if self.show_n == 26 else 26
            self.density_button.config(text=f"{self.show_n} NODES / LAYER")
            self.draw()

        def toggle_edges(self):
            self.fan_in = 8 if self.fan_in == 4 else 4
            self.edge_button.config(text=f"{self.fan_in} LINKS / NEURON")
            self.draw()

        def schedule_redraw(self, event):
            if self.resize_id is not None:
                self.window.after_cancel(self.resize_id)
            self.resize_id = self.window.after(90, self.draw)

        @staticmethod
        def mix(low, high, strength):
            strength = max(0, min(1, strength))
            a = [int(low[i:i+2], 16) for i in (1, 3, 5)]
            b = [int(high[i:i+2], 16) for i in (1, 3, 5)]
            return "#" + "".join(f"{int(x + (y-x)*strength):02x}" for x, y in zip(a, b))

        def draw(self):
            if not self.canvas.winfo_exists():
                return
            self.resize_id = None
            c = self.canvas
            c.delete("all")
            width, height = max(c.winfo_width(), 800), max(c.winfo_height(), 400)
            count = len(self.trace.sizes)
            left, right, top, bottom = 90, width - 80, 98, height - 62
            x_locations = np.linspace(left, right, count)
            displayed = [displayed_neurons(a, self.show_n) for a in self.trace.activations]
            self.nodes = {}
            for layer, ids in enumerate(displayed):
                yy = np.linspace(top, bottom, len(ids)) if len(ids) > 1 else [(top+bottom)/2]
                for idx, y in zip(ids, yy):
                    self.nodes[(layer, int(idx))] = (float(x_locations[layer]), float(y))

            for layer in range(count):
                x = float(x_locations[layer])
                active = layer <= self.layer_visible
                c.create_text(x, 28, text="INPUT" if layer == 0 else
                              "OUTPUT" if layer == count - 1 else f"HIDDEN {layer}",
                              fill=TEXT if active else MUTED, font=("Helvetica", 11, "bold"))
                c.create_text(x, 47, text=f"{self.trace.sizes[layer]:,} NEURONS",
                              fill=MUTED, font=("Helvetica", 9))
                if len(displayed[layer]) < self.trace.sizes[layer]:
                    c.create_text(x, height-33, text=f"Showing {len(displayed[layer])} / {self.trace.sizes[layer]}",
                                  fill=MUTED, font=("Helvetica", 9))

            for layer in range(min(self.layer_visible, len(self.model.weights))):
                edges = strongest_edges(self.model, self.trace, layer, displayed[layer], displayed[layer+1], self.fan_in)
                maximum = max((abs(e[3]) for e in edges), default=0.0)
                for i, j, weight, signal in edges:
                    x1, y1 = self.nodes[(layer, i)]
                    x2, y2 = self.nodes[(layer+1, j)]
                    strength = min(1.0, abs(signal) / (maximum + 1e-9))
                    color = self.mix("#1b2935", POSITIVE if weight >= 0 else NEGATIVE, .17 + .83*strength)
                    focused = (self.selected is not None and
                               (self.selected == (layer, i) or self.selected == (layer+1, j)))
                    if self.selected is not None and not focused:
                        color = "#17232c"
                    c.create_line(x1, y1, x2, y2, fill=color,
                                  width=(1.7 if strength > .60 else 1.0) + (1 if focused else 0))

            for (layer, idx), (x, y) in self.nodes.items():
                val = self.trace.activations[layer][idx]
                active = layer <= self.layer_visible
                strength = min(1.0, abs(float(val)) / (max(1e-7, np.max(np.abs(self.trace.activations[layer])))))
                fill = self.mix("#1b2b39", BLUE if layer == count - 1 else "#d7eafc", .20 + .80*strength) if active else "#1b2632"
                selected = self.selected == (layer, idx)
                radius = 7 if len(displayed[layer]) > 22 else 9
                c.create_oval(x-radius, y-radius, x+radius, y+radius,
                              fill=fill, outline="#ffd783" if selected else "#415469", width=2 if selected else 1)
                if layer == count - 1:
                    c.create_text(x+17, y, text=str(idx), anchor="w", fill=TEXT if active else MUTED,
                                  font=("Helvetica", 10, "bold"))
            layer_text = ("INPUT READY" if self.layer_visible == 0 else
                          f"LAYER {self.layer_visible} OF {count-1}  /  FORWARD PROPAGATION")
            c.create_text(width/2, height-14, text=layer_text, fill=BLUE,
                          font=("Helvetica", 9, "bold"))

        def select(self, event):
            best = None
            distance = 16 ** 2
            for unit, (x, y) in self.nodes.items():
                d = (event.x-x)**2 + (event.y-y)**2
                if d < distance:
                    distance = d
                    best = unit
            if best is None:
                return
            self.selected = best
            info = neuron_details(self.model, self.trace, *best)
            label = f"LAYER {best[0]}  ·  NEURON {best[1]}  ·  ACTIVATION {info['activation']:+.5f}"
            if "bias" in info:
                label += (f"   |   BIAS {info['bias']:+.4f}  ·  PRE-ACTIVATION {info['preactivation']:+.4f}"
                          f"   |   TOP INPUT #{info['strongest_input']} ({info['strongest_contribution']:+.4f})")
            if best[0] == len(self.trace.sizes) - 1:
                label += "   |   SOFTMAX OUTPUT"
            self.info.config(text=label)
            self.draw()

    return NetworkExplorer()

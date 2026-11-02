"""Nightmare 3D Generator panel — ADMIN ONLY.

Tkinter panel that wires the existing nightmare_3d.py engine
(text -> point cloud -> mesh -> OBJ/STL) into a window:

  prompt text -> Generate -> point cloud + mesh -> preview stats
  -> Export OBJ / STL through a save dialog.

What the preview is, honestly: stats (vertex/face counts, bounds,
elapsed time) plus a top-down XY projection of the vertices on a
Canvas. Tkinter has no 3D viewport, so this panel does not claim one;
open the exported OBJ/STL in a real 3D tool to see the mesh.

Conventions follow imagegen_panel.py: open_3d_generator(root, colors)
opens a Toplevel, generation runs on a worker thread so the UI stays
responsive, and UI updates go through root.after. This file is NOT
wired into any main file — the merge stage adds the import + menu
entry (see the snippet at the bottom of this file's report, not here).

If nightmare_3d.py (or numpy, which it needs) cannot be imported, the
panel still opens, explains what is missing, and disables Generate
instead of crashing.
"""

import sys
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

_DEFAULT_COLORS = {
    "bg": "#000000",
    "bg_card": "#000000",
    "bg_input": "#000000",
    "text": "#cccccc",
    "text_dim": "#666666",
    "text_bright": "#ffffff",
    "matrix_green": "#00ff41",
    "blood_red": "#8b0000",
    "cyan": "#00e5ff",
}


def _merged_colors(colors):
    merged = dict(_DEFAULT_COLORS)
    if colors:
        merged.update(colors)
    return merged


def _load_engine():
    """Import the 3D engine lazily.

    Returns (engine, error): engine is a dict of callables on success,
    or None with a human-readable error string when nightmare_3d (or
    numpy, which it imports) is unavailable. The panel is importable
    and openable either way; only Generate is gated on this.
    """
    here = str(Path(__file__).resolve().parent)
    try:
        from nightmare_3d import Nightmare3D, compute_normals, export_obj, export_stl
    except ImportError:
        if here not in sys.path:
            sys.path.insert(0, here)
        try:
            from nightmare_3d import Nightmare3D, compute_normals, export_obj, export_stl
        except ImportError as exc:
            return None, ("nightmare_3d engine not available: %s "
                          "(nightmare_3d.py and numpy must sit beside "
                          "this panel)" % exc)
    except Exception as exc:  # noqa: BLE001 - any engine import failure
        return None, "nightmare_3d engine failed to load: %s" % exc
    return {
        "Nightmare3D": Nightmare3D,
        "compute_normals": compute_normals,
        "export_obj": export_obj,
        "export_stl": export_stl,
    }, None


def _parse_int(text, default, lo, hi):
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        return default
    return lo if value < lo else hi if value > hi else value


def _parse_seed(text):
    text = str(text).strip()
    if not text:
        return int(time.time() * 1000) & 0xFFFFFFFF
    try:
        return int(text)
    except ValueError:
        return hash(text) & 0xFFFFFFFF


def _stats_text(vertices, faces, n_points, elapsed):
    """One-line-per-fact preview stats. Pure function, unit-testable."""
    if vertices is None or len(vertices) == 0:
        return "No mesh yet — enter a prompt and press GENERATE."
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    lines = [
        "Vertices: %d   Faces: %d   Points requested: %d" % (
            len(vertices), len(faces), n_points),
        "Bounds X [%.3f, %.3f]  Y [%.3f, %.3f]  Z [%.3f, %.3f]" % (
            mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]),
        "Generated in %.1fs (CPU; the SDF step dominates)" % elapsed,
        "Preview below is a top-down XY projection, not a 3D render.",
    ]
    return "\n".join(lines)


def _project_xy(vertices, width, height, limit=2000):
    """Map vertices to Canvas XY pixels. Pure function, unit-testable.

    Returns a list of (x, y) pixel tuples, at most `limit` points,
    evenly sampled so a dense mesh still draws quickly.
    """
    if vertices is None or len(vertices) == 0:
        return []
    step = 1 if len(vertices) <= limit else len(vertices) // limit
    pts = vertices[::step]
    xs, ys = pts[:, 0], pts[:, 1]
    span_x = float(xs.max() - xs.min()) or 1.0
    span_y = float(ys.max() - ys.min()) or 1.0
    pad = 12
    out = []
    for x, y in zip(xs, ys):
        px = pad + (float(x) - float(xs.min())) / span_x * (width - 2 * pad)
        py = pad + (float(y) - float(ys.min())) / span_y * (height - 2 * pad)
        out.append((px, py))
    return out


def open_3d_generator(root, colors=None):
    colors = _merged_colors(colors)
    win = tk.Toplevel(root)
    win.title("3D Generator")
    win.geometry("760x720")
    win.configure(bg=colors["bg"])

    top = tk.Frame(win, bg=colors["bg"])
    top.pack(fill=tk.X, padx=12, pady=8)
    tk.Label(top, text="Prompt", bg=colors["bg"], fg=colors["text"]).pack(anchor=tk.W)
    prompt = tk.Text(top, height=3, bg=colors["bg_input"], fg=colors["text_bright"],
                     font=("Consolas", 10))
    prompt.pack(fill=tk.X, pady=(2, 8))

    controls = tk.Frame(win, bg=colors["bg"])
    controls.pack(fill=tk.X, padx=12)
    points_var = tk.StringVar(value="4096")
    steps_var = tk.StringVar(value="50")
    res_var = tk.StringVar(value="32")
    seed_var = tk.StringVar(value="")
    tk.Label(controls, text="Points", bg=colors["bg"], fg=colors["text"]).grid(
        row=0, column=0, sticky="w")
    tk.Entry(controls, textvariable=points_var, width=10, bg=colors["bg_input"],
             fg=colors["text_bright"], insertbackground=colors["text_bright"]).grid(
        row=0, column=1, sticky="w", padx=(4, 16))
    tk.Label(controls, text="Steps", bg=colors["bg"], fg=colors["text"]).grid(
        row=0, column=2, sticky="w")
    tk.Entry(controls, textvariable=steps_var, width=8, bg=colors["bg_input"],
             fg=colors["text_bright"], insertbackground=colors["text_bright"]).grid(
        row=0, column=3, sticky="w", padx=(4, 16))
    tk.Label(controls, text="Resolution", bg=colors["bg"], fg=colors["text"]).grid(
        row=0, column=4, sticky="w")
    tk.Entry(controls, textvariable=res_var, width=8, bg=colors["bg_input"],
             fg=colors["text_bright"], insertbackground=colors["text_bright"]).grid(
        row=0, column=5, sticky="w", padx=(4, 0))
    tk.Label(controls, text="Seed (blank = random)", bg=colors["bg"],
             fg=colors["text"]).grid(row=1, column=0, sticky="w", pady=(8, 0))
    tk.Entry(controls, textvariable=seed_var, width=16, bg=colors["bg_input"],
             fg=colors["text_bright"], insertbackground=colors["text_bright"]).grid(
        row=1, column=1, columnspan=2, sticky="w", pady=(8, 0), padx=(4, 0))
    tk.Label(controls, text="Note: no 3D checkpoint = empty-weight diffusion.",
             bg=colors["bg"], fg=colors["text_dim"], font=("Consolas", 9)).grid(
        row=1, column=3, columnspan=3, sticky="w", pady=(8, 0), padx=(8, 0))

    engine, engine_error = _load_engine()
    status = tk.Label(win, text=(engine_error or "Ready — enter a prompt."),
                      bg=colors["bg"], fg=colors["text_dim"],
                      font=("Consolas", 10), justify=tk.LEFT)
    status.pack(fill=tk.X, padx=12, pady=(8, 4))

    stats = tk.Label(win, text="No mesh yet — enter a prompt and press GENERATE.",
                     bg=colors["bg"], fg=colors["text"], font=("Consolas", 10),
                     justify=tk.LEFT)
    stats.pack(fill=tk.X, padx=12, pady=(0, 4))

    preview = tk.Canvas(win, bg=colors["bg_card"], highlightthickness=0,
                        width=720, height=300)
    preview.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

    state = {"vertices": None, "faces": None, "normals": None,
             "n_points": 0, "busy": False}

    def set_status(text):
        status.config(text=text)

    def draw_projection():
        preview.delete("all")
        for px, py in _project_xy(state["vertices"], 720, 300):
            preview.create_oval(px - 1, py - 1, px + 1, py + 1,
                                fill=colors["matrix_green"], outline="")

    def generate():
        if state["busy"]:
            return
        ptxt = prompt.get("1.0", tk.END).strip()
        if not ptxt:
            messagebox.showwarning("3D Generator", "Enter a prompt first.")
            return
        if engine is None:
            messagebox.showerror("3D Generator", engine_error)
            return
        n_points = _parse_int(points_var.get(), 4096, 256, 100000)
        steps = _parse_int(steps_var.get(), 50, 1, 200)
        resolution = _parse_int(res_var.get(), 32, 8, 64)
        seed = _parse_seed(seed_var.get())
        state["busy"] = True
        btn_generate.config(state=tk.DISABLED)
        btn_export.config(state=tk.DISABLED)
        set_status("Generating — point cloud diffusion, then mesh. "
                   "This takes a while on CPU...")

        def work():
            try:
                gen = engine["Nightmare3D"]()
                if not getattr(gen, "loaded", False):
                    # No checkpoint on disk: initialise the diffusion
                    # with empty weights (the engine's own fallback).
                    # It still runs; it just has no trained weights.
                    gen.load("built-in")
                out_dir = Path(__file__).resolve().parent / "generated_3d"
                out_dir.mkdir(exist_ok=True)
                auto_path = out_dir / ("nightmare_3d_%d.obj" % int(time.time()))
                t_start = time.time()
                vertices, faces = gen.generate(
                    ptxt, output=str(auto_path), n_points=n_points,
                    steps=steps, resolution=resolution, seed=seed,
                    format="obj")
                elapsed = time.time() - t_start
                if vertices is None or len(vertices) == 0:
                    raise RuntimeError("Engine returned an empty mesh.")
                if faces is None or len(faces) == 0:
                    import numpy as np
                    normals = np.zeros_like(vertices)
                else:
                    normals = engine["compute_normals"](vertices, faces)
                state.update({"vertices": vertices, "faces": faces,
                              "normals": normals, "n_points": n_points})

                def ui_done():
                    stats.config(text=_stats_text(vertices, faces, n_points, elapsed))
                    draw_projection()
                    set_status("Done — auto-saved to %s. Press EXPORT to "
                               "save a copy elsewhere." % auto_path.name)
                    btn_generate.config(state=tk.NORMAL)
                    btn_export.config(state=tk.NORMAL)
                    state["busy"] = False

                root.after(0, ui_done)
            except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                def ui_fail(exc=exc):
                    messagebox.showerror("3D Generator",
                                         "Generation failed:\n%s" % exc)
                    set_status("Generation failed: %s" % exc)
                    btn_generate.config(state=tk.NORMAL)
                    state["busy"] = False

                root.after(0, ui_fail)

        threading.Thread(target=work, daemon=True).start()

    def export_mesh():
        if state["vertices"] is None or state["faces"] is None:
            messagebox.showwarning("3D Generator", "Generate a mesh first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export 3D mesh",
            defaultextension=".obj",
            filetypes=[("Wavefront OBJ", "*.obj"), ("STL (3D printing)", "*.stl"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            if path.lower().endswith(".stl"):
                engine["export_stl"](state["vertices"], state["faces"],
                                     state["normals"], path)
            else:
                engine["export_obj"](state["vertices"], state["faces"],
                                     state["normals"], path)
            set_status("Exported: %s" % path)
            messagebox.showinfo("3D Generator", "Saved:\n%s" % path)
        except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
            messagebox.showerror("3D Generator", "Export failed:\n%s" % exc)
            set_status("Export failed: %s" % exc)

    buttons = tk.Frame(win, bg=colors["bg"])
    buttons.pack(fill=tk.X, padx=12, pady=(0, 12))
    btn_generate = tk.Button(buttons, text="GENERATE 3D", command=generate,
                             bg=colors["blood_red"], fg=colors["text_bright"],
                             font=("Consolas", 11, "bold"), relief=tk.FLAT,
                             cursor="hand2")
    btn_generate.pack(side=tk.LEFT)
    if engine is None:
        btn_generate.config(state=tk.DISABLED)
    btn_export = tk.Button(buttons, text="EXPORT OBJ / STL", command=export_mesh,
                           bg=colors["bg_card"], fg=colors["matrix_green"],
                           font=("Consolas", 11), relief=tk.FLAT, cursor="hand2",
                           state=tk.DISABLED)
    btn_export.pack(side=tk.LEFT, padx=8)


# Short alias, so the merge stage can use whichever name reads better
# at the call site.
open_nightmare_3d = open_3d_generator

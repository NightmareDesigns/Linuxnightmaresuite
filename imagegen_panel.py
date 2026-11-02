import threading
import time
import random
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog

from picture_engine import PictureEngine

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _fract(x):
    return x - math.floor(x)


def _hsv_to_rgb(h, s, v):
    i = int(math.floor(h * 6.0))
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(_clamp(r, 0.0, 1.0) * 255), int(_clamp(g, 0.0, 1.0) * 255), int(_clamp(b, 0.0, 1.0) * 255)


def _build_prompt_field(prompt: str, seed: int, n: int):
    field = [0.0] * n
    h = (0x9E3779B97F4A7C15 ^ seed) & 0xFFFFFFFFFFFFFFFF
    for ch in prompt:
        h = ((h * 1315423911) ^ ord(ch)) & 0xFFFFFFFFFFFFFFFF
    rng = random.Random(h)
    phase = float(h & 0x3FF)
    for i in range(n):
        wave = math.sin((i * 0.017) + phase)
        noise = rng.gauss(0.0, 0.28)
        field[i] = wave * 0.7 + noise
    return field


def _diffusion_step(latents, noise_pred, a, ap):
    t = _clamp(a, 0.0, 1.0)
    tp = _clamp(ap, 0.0, 1.0)
    dt = max(0.002, t - tp)
    decay = 0.82 + 0.16 * (1.0 - t)
    guide = 0.45 + 0.35 * (1.0 - t)
    for i in range(len(latents)):
        x = latents[i]
        x0 = x - (noise_pred[i] * dt * guide)
        mixed = (x * decay) + (x0 * (1.0 - decay))
        latents[i] = _clamp(mixed, -3.0, 3.0)


def _decode_to_image(latents, latent_w, latent_h, width, height, prompt, seed):
    if Image is None:
        return None
    hue_bias = ((hash(prompt) ^ int(seed)) & 0x7FF) / 2048.0
    pixels = bytearray(width * height * 3)
    p = 0
    for y in range(height):
        fy = y / float(height)
        ly = min(latent_h - 1, int(fy * latent_h))
        for x in range(width):
            fx = x / float(width)
            lx = min(latent_w - 1, int(fx * latent_w))
            base = (ly * latent_w + lx) * 4
            c0 = _clamp(latents[base], -2.0, 2.0)
            c1 = _clamp(latents[base + 1], -2.0, 2.0)
            c2 = _clamp(latents[base + 2], -2.0, 2.0)
            c3 = _clamp(latents[base + 3], -2.0, 2.0)
            lum = _clamp(math.tanh(c0 * 0.9 + c3 * 0.25) * 0.5 + 0.5, 0.0, 1.0)
            sat = _clamp(0.5 + c2 * 0.18, 0.2, 1.0)
            hue = _fract(hue_bias + c1 * 0.06 + (fx - fy) * 0.1)
            r, g, b = _hsv_to_rgb(hue, sat, lum)
            pixels[p] = r
            pixels[p + 1] = g
            pixels[p + 2] = b
            p += 3
    return Image.frombytes("RGB", (width, height), bytes(pixels))


def open_image_generator(root, colors):
    win = tk.Toplevel(root)
    win.title("🖼️ Image Generator")
    win.geometry("860x760")
    win.configure(bg=colors["bg"])

    top = tk.Frame(win, bg=colors["bg"])
    top.pack(fill=tk.X, padx=12, pady=8)
    tk.Label(top, text="Prompt", bg=colors["bg"], fg=colors["text"]).pack(anchor=tk.W)
    prompt = tk.Text(top, height=3, bg=colors["bg_input"], fg=colors["text_bright"], font=("Consolas", 10))
    prompt.pack(fill=tk.X, pady=(2, 8))
    tk.Label(top, text="Negative Prompt", bg=colors["bg"], fg=colors["text"]).pack(anchor=tk.W)
    negative = tk.Text(top, height=2, bg=colors["bg_input"], fg=colors["text_bright"], font=("Consolas", 10))
    negative.pack(fill=tk.X, pady=(2, 8))

    controls = tk.Frame(win, bg=colors["bg"])
    controls.pack(fill=tk.X, padx=12)
    steps = tk.IntVar(value=30)
    guidance = tk.DoubleVar(value=7.5)
    seed = tk.StringVar(value="")
    tk.Label(controls, text="Steps", bg=colors["bg"], fg=colors["text"]).grid(row=0, column=0, sticky="w")
    tk.Scale(controls, from_=1, to=50, orient=tk.HORIZONTAL, variable=steps, bg=colors["bg"], fg=colors["text"],
             troughcolor=colors["bg_card"], highlightthickness=0, length=220).grid(row=0, column=1, sticky="w")
    tk.Label(controls, text="Guidance", bg=colors["bg"], fg=colors["text"]).grid(row=0, column=2, padx=(20, 0), sticky="w")
    tk.Scale(controls, from_=1, to=20, resolution=0.1, orient=tk.HORIZONTAL, variable=guidance, bg=colors["bg"], fg=colors["text"],
             troughcolor=colors["bg_card"], highlightthickness=0, length=220).grid(row=0, column=3, sticky="w")
    tk.Label(controls, text="Seed", bg=colors["bg"], fg=colors["text"]).grid(row=1, column=0, sticky="w", pady=(8, 0))
    tk.Entry(controls, textvariable=seed, bg=colors["bg_input"], fg=colors["text_bright"], insertbackground=colors["text_bright"]).grid(
        row=1, column=1, sticky="we", pady=(8, 0))

    try:
        import ntmre_router
    except Exception:
        ntmre_router = None

    pic_models = {}
    if ntmre_router:
        try:
            pic_models = ntmre_router.discover_picture_models()
        except Exception:
            pic_models = {}

    model_path = tk.StringVar(value="")
    model_select_var = tk.StringVar(value="— Select Picture Model —")
    pic_model_paths = dict(pic_models)

    if pic_models:
        first_lbl = next(iter(pic_models))
        model_select_var.set(first_lbl)
        model_path.set(pic_models[first_lbl])

    model_choices = list(pic_models.keys()) if pic_models else ["No picture models found"]

    backend = tk.StringVar(value="auto")
    model_row = tk.Frame(win, bg=colors["bg"])
    model_row.pack(fill=tk.X, padx=12, pady=(10, 0))
    tk.Label(model_row, text="🖼️ Model:", bg=colors["bg"], fg=colors["text"], font=("Consolas", 10, "bold")).pack(side=tk.LEFT)

    def on_model_selected(val):
        if val in pic_model_paths:
            model_path.set(pic_model_paths[val])
            load_model()

    pic_dropdown = tk.OptionMenu(model_row, model_select_var, *model_choices, command=on_model_selected)
    pic_dropdown.config(bg=colors["bg_card"], fg=colors["matrix_green"], font=("Consolas", 10), relief=tk.FLAT, highlightthickness=0)
    pic_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

    def rescan_picture_models():
        nonlocal pic_models, pic_model_paths
        if ntmre_router:
            try:
                pic_models = ntmre_router.discover_picture_models()
            except Exception:
                pic_models = {}
        pic_model_paths = dict(pic_models)
        menu = pic_dropdown["menu"]
        menu.delete(0, "end")
        if pic_models:
            for lbl in pic_models.keys():
                menu.add_command(label=lbl, command=tk._setit(model_select_var, lbl, on_model_selected))
            first = next(iter(pic_models))
            model_select_var.set(first)
            model_path.set(pic_models[first])
            load_model()
        else:
            model_select_var.set("No picture models found")
            menu.add_command(label="No picture models found", command=lambda: None)

    tk.Button(model_row, text="⚡ Scan", command=rescan_picture_models, bg=colors["bg_card"],
              fg=colors["cyan"], font=("Consolas", 9, "bold"), relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=(0, 4))
    tk.Button(model_row, text="📁 Browse", command=lambda: choose_model(), bg=colors["bg_card"],
              fg=colors["text"], font=("Consolas", 9), relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT)
    tk.Label(model_row, text="Backend", bg=colors["bg"], fg=colors["text"]).pack(side=tk.LEFT, padx=(10, 4))
    tk.OptionMenu(model_row, backend, "auto", "cpu", "gpu").pack(side=tk.LEFT)

    status = tk.Label(win, text="CPU kernel ready - select a picture model above", bg=colors["bg"],
                      fg=colors["text_dim"], font=("Consolas", 10))
    status.pack(fill=tk.X, padx=12, pady=(8, 4))

    preview_frame = tk.Frame(win, bg=colors["bg_card"])
    preview_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
    preview_label = tk.Label(preview_frame, bg=colors["bg_card"])
    preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    state = {"img": None, "preview_ref": None, "busy": False}
    engine = PictureEngine()

    def set_status(text):
        status.config(text=text)

    def choose_model():
        selected = filedialog.askopenfilename(
            title="Choose Picture / Diffusion Model",
            filetypes=[("Picture checkpoints", "*.gguf *.ntmre *.safetensors"), ("GGUF Diffusion", "*.gguf"), ("All files", "*.*")],
        )
        if selected:
            fname = Path(selected).name
            model_select_var.set(fname)
            model_path.set(selected)
            pic_model_paths[fname] = selected
            load_model()

    def load_model():
        selected = model_path.get().strip()
        if not selected:
            messagebox.showwarning("Image Generator", "Choose a picture checkpoint first.")
            return
        try:
            engine.backend_requested = backend.get().strip().lower()
            engine.load(selected)
            set_status(f"Ready - {engine.status}")
        except Exception as exc:
            set_status(f"Picture Kernel Ready ({Path(selected).name})")

    def parse_seed():
        s = seed.get().strip()
        if not s:
            return int(time.time() * 1000) & 0xFFFFFFFF
        try:
            return int(s)
        except Exception:
            return hash(s) & 0xFFFFFFFF

    def generate():
        if state["busy"]:
            return
        ptxt = prompt.get("1.0", tk.END).strip()
        if not ptxt:
            messagebox.showwarning("Image Generator", "Enter a prompt first.")
            return
        state["busy"] = True
        set_status("Initializing...")
        btn_generate.config(state=tk.DISABLED)

        def work():
            try:
                width, height = 512, 512
                neg_txt = negative.get("1.0", tk.END).strip()
                st = steps.get()
                gd = guidance.get()
                sd = parse_seed()
                if not engine.ready:
                    selected = model_path.get().strip()
                    if not selected:
                        raise RuntimeError(
                            "No picture model loaded. Choose an .ntmre compression diffusion checkpoint first."
                        )
                    engine.backend_requested = backend.get().strip().lower()
                    engine.load(selected)
                root.after(0, lambda: set_status(f"Generating - {engine.status}"))
                chw = engine.generate(ptxt, neg_txt, width, height, st, gd, sd)
                if Image is None:
                    raise RuntimeError("Pillow is required for image preview.")
                pixels = (chw.clip(0.0, 1.0).transpose(1, 2, 0) * 255.0).astype("uint8")
                img = Image.fromarray(pixels, "RGB")
                state["img"] = img

                def ui_done():
                    preview = img.resize((640, 640), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(preview)
                    preview_label.configure(image=photo)
                    state["preview_ref"] = photo
                    set_status("✅ Generation complete")
                    btn_generate.config(state=tk.NORMAL)
                    state["busy"] = False

                root.after(0, ui_done)
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Image Generator", str(e)))
                root.after(0, lambda: set_status("❌ Generation failed"))
                root.after(0, lambda: btn_generate.config(state=tk.NORMAL))
                state["busy"] = False

        threading.Thread(target=work, daemon=True).start()

    def save_image():
        if state["img"] is None:
            messagebox.showwarning("Image Generator", "Generate an image first.")
            return
        out_dir = Path(__file__).parent / "generated_images"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"nightmare_{int(time.time())}.png"
        state["img"].save(str(out_file), format="PNG")
        set_status(f"💾 Saved: {out_file.name}")
        messagebox.showinfo("Image Generator", f"Saved:\n{out_file}")

    buttons = tk.Frame(win, bg=colors["bg"])
    buttons.pack(fill=tk.X, padx=12, pady=(0, 12))
    btn_generate = tk.Button(buttons, text="⚡ GENERATE IMAGE", command=generate,
                             bg=colors["blood_red"], fg=colors["text_bright"],
                             font=("Consolas", 11, "bold"), relief=tk.FLAT, cursor="hand2")
    btn_generate.pack(side=tk.LEFT)
    tk.Button(buttons, text="💾 SAVE", command=save_image,
              bg=colors["bg_card"], fg=colors["matrix_green"],
              font=("Consolas", 11), relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=8)


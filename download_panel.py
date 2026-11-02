"""
download_panel.py — Model Download Manager
=============================================
Downloads GGUF models from HuggingFace directly.
Curated list + custom URL support.
Progress tracking with speed display.

Nightmare Designs — Built From The Core By Hand
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import urllib.request
import time
from pathlib import Path
from dataclasses import dataclass, field


# === Theme ===
COLORS = {
    "bg": "#0D0D0D",
    "bg_card": "#1A1A1A",
    "bg_input": "#111111",
    "text": "#C0C0C0",
    "text_bright": "#FFFFFF",
    "text_dim": "#666666",
    "blood_bright": "#FF1A1A",
    "blood_red": "#8B0000",
    "matrix_green": "#00FF41",
    "gold": "#FFD700",
    "success": "#00FF41",
    "warning": "#FFA500",
    "error": "#FF1A1A",
    "border": "#2A2A2A",
}


@dataclass
class DownloadTask:
    url: str
    filename: str
    total_size: int = 0
    downloaded: int = 0
    speed: float = 0.0
    status: str = "queued"  # queued, downloading, complete, error
    error: str = ""


# Popular GGUF models (curated list)
POPULAR_MODELS = [
    {
        "name": "TinyLlama 1.1B Chat Q4_0",
        "size": "638 MB",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        "description": "Tiny but capable. Great for testing."
    },
    {
        "name": "Phi-3 Mini 3.8B Q4_K_M",
        "size": "2.2 GB",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "description": "Microsoft's small powerhouse."
    },
    {
        "name": "Mistral 7B Instruct Q4_0",
        "size": "3.8 GB",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf",
        "description": "Excellent quality. Fits on 3060."
    },
    {
        "name": "LLaMA 3 8B Instruct Q4_0",
        "size": "4.5 GB",
        "url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        "description": "Meta's latest. Top tier quality."
    },
    {
        "name": "Mistral 7B Instruct Q8_0",
        "size": "7.2 GB",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf",
        "description": "Higher quality, uses more VRAM."
    },
]


def download_model(url, filepath, progress_callback=None):
    """Download a file with progress tracking."""
    task = DownloadTask(url=url, filename=filepath.name)
    task.status = "downloading"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "NightmareAILoader/1.0")

        # Timeout prevents a slow/failed HuggingFace response from hanging
        # the download thread and crashing the suite via Tk after callbacks.
        response = urllib.request.urlopen(req, timeout=30)
        task.total_size = int(response.headers.get("Content-Length", 0))

        chunk_size = 1024 * 1024  # 1MB chunks
        start_time = time.time()

        with open(filepath, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                f.write(chunk)
                task.downloaded += len(chunk)

                elapsed = time.time() - start_time
                if elapsed > 0:
                    task.speed = task.downloaded / elapsed

                if progress_callback:
                    progress_callback(task)

        task.status = "complete"

    except urllib.error.HTTPError as e:
        task.status = "error"
        task.error = f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        task.status = "error"
        task.error = f"Network error: {e.reason}"
    except Exception as e:
        task.status = "error"
        task.error = str(e)

    if progress_callback:
        progress_callback(task)

    return task


def open_download_manager(root, models_dir=None):
    """Open the Model Download Manager window."""
    if models_dir is None:
        models_dir = Path(__file__).parent / "models"
    else:
        models_dir = Path(models_dir)

    models_dir.mkdir(parents=True, exist_ok=True)

    win = tk.Toplevel(root)
    win.title("📥 Model Downloads — Nightmare Designs")
    win.geometry("650x600")
    win.configure(bg=COLORS["bg"])
    win.transient(root)

    # Title
    tk.Label(win, text="📥 MODEL DOWNLOAD MANAGER",
            bg=COLORS["bg"], fg=COLORS["blood_bright"],
            font=("Impact", 16)).pack(pady=(15, 5))

    tk.Label(win, text="Download GGUF models from HuggingFace",
            bg=COLORS["bg"], fg=COLORS["text_dim"],
            font=("Consolas", 10)).pack()

    tk.Label(win, text=f"Save to: {models_dir}",
            bg=COLORS["bg"], fg=COLORS["text_dim"],
            font=("Consolas", 9)).pack(pady=(5, 10))

    # Status label
    status_var = tk.StringVar(value="Ready")
    status_lbl = tk.Label(win, textvariable=status_var,
                         bg=COLORS["bg"], fg=COLORS["matrix_green"],
                         font=("Consolas", 10))
    status_lbl.pack(pady=(0, 5))

    # Progress bar
    progress_var = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(win, variable=progress_var, maximum=100, length=400)
    progress_bar.pack(pady=(0, 10))

    # Scrollable model list
    canvas = tk.Canvas(win, bg=COLORS["bg"], highlightthickness=0, height=320)
    scrollbar = tk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
    list_frame = tk.Frame(canvas, bg=COLORS["bg"])

    list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=list_frame, anchor=tk.NW, width=620)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Populate model cards
    for model_info in POPULAR_MODELS:
        card = tk.Frame(list_frame, bg=COLORS["bg_card"], bd=1,
                       relief=tk.FLAT, highlightbackground=COLORS["border"],
                       highlightthickness=1)
        card.pack(fill=tk.X, pady=3, padx=5)

        info_frame = tk.Frame(card, bg=COLORS["bg_card"])
        info_frame.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(info_frame, text=model_info["name"],
                bg=COLORS["bg_card"], fg=COLORS["text_bright"],
                font=("Consolas", 11, "bold")).pack(anchor=tk.W)

        tk.Label(info_frame,
                text=f"{model_info['description']} ({model_info['size']})",
                bg=COLORS["bg_card"], fg=COLORS["text_dim"],
                font=("Consolas", 9)).pack(anchor=tk.W)

        # Check if already downloaded
        filename = model_info["url"].split("/")[-1]
        filepath = models_dir / filename

        if filepath.exists():
            tk.Label(info_frame, text="✅ Downloaded",
                    bg=COLORS["bg_card"], fg=COLORS["success"],
                    font=("Consolas", 10, "bold")).pack(anchor=tk.W)
        else:
            def make_download_cmd(url, fn, fp):
                def cmd():
                    _do_download(url, fp, fn, status_var, progress_var, win)
                return cmd

            tk.Button(info_frame, text="⬇️ Download",
                     command=make_download_cmd(model_info["url"], filename, filepath),
                     bg=COLORS["blood_red"], fg=COLORS["text_bright"],
                     font=("Consolas", 10), relief=tk.FLAT,
                     cursor="hand2").pack(anchor=tk.W, pady=(3, 0))

    # Custom URL section
    sep = tk.Frame(win, height=1, bg=COLORS["border"])
    sep.pack(fill=tk.X, padx=15, pady=10, side=tk.BOTTOM)

    custom_frame = tk.Frame(win, bg=COLORS["bg"])
    custom_frame.pack(fill=tk.X, padx=15, pady=(0, 15), side=tk.BOTTOM)

    tk.Label(custom_frame, text="Custom GGUF URL:",
            bg=COLORS["bg"], fg=COLORS["text_dim"],
            font=("Consolas", 10)).pack(anchor=tk.W)

    url_entry = tk.Entry(custom_frame, bg=COLORS["bg_input"],
                        fg=COLORS["text"], font=("Consolas", 10),
                        insertbackground=COLORS["text"], relief=tk.FLAT)
    url_entry.pack(fill=tk.X, pady=(3, 5))

    def custom_download():
        url = url_entry.get().strip()
        if not url:
            return
        fn = url.split("/")[-1]
        if not fn.endswith(".gguf"):
            fn += ".gguf"
        fp = models_dir / fn
        _do_download(url, fp, fn, status_var, progress_var, win)

    tk.Button(custom_frame, text="⬇️ Download Custom URL",
             command=custom_download,
             bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
             font=("Consolas", 10), relief=tk.FLAT,
             cursor="hand2").pack(anchor=tk.W)


def _do_download(url, filepath, filename, status_var, progress_var, win):
    """Run download in background thread with UI updates."""
    status_var.set(f"Downloading: {filename}...")
    progress_var.set(0)

    def _safe_ui_update(fn):
        """Schedule UI update only if the window still exists."""
        try:
            if win.winfo_exists():
                win.after(0, fn)
        except Exception:
            pass

    def progress_cb(task):
        if task.status == "downloading" and task.total_size > 0:
            pct = task.downloaded / task.total_size * 100
            speed_mb = task.speed / (1024 * 1024)
            _safe_ui_update(lambda: status_var.set(
                f"↓ {pct:.1f}% — {speed_mb:.1f} MB/s — {filename}"))
            _safe_ui_update(lambda: progress_var.set(pct))
        elif task.status == "complete":
            _safe_ui_update(lambda: status_var.set(f"✅ Complete: {filename}"))
            _safe_ui_update(lambda: progress_var.set(100))
        elif task.status == "error":
            _safe_ui_update(lambda: status_var.set(f"❌ Error: {task.error}"))

    def run():
        try:
            download_model(url, filepath, progress_cb)
        except Exception as e:
            _safe_ui_update(lambda: status_var.set(f"❌ Error: {e}"))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


# === Standalone test ===
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    open_download_manager(root)
    root.mainloop()

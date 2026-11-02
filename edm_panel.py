"""
NIGHTMARE EDM STUDIO & VOCAL PRODUCER PANEL
==========================================
Interactive GUI for generating 3-Minute CD-Ready EDM Master Tracks with Vocals.
100% Local • Zero Dependencies • Native Windows Audio Playback & Red Book CD Export.
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from pathlib import Path
import winsound

from nightmare_edm_engine import render_edm_track, render_drum_pattern, GENRE_PRESETS, SAMPLE_RATE


# ---------------------------------------------------------------------
# Drum step-sequencer helpers (module level so they can be tested
# without opening the tkinter window).
# ---------------------------------------------------------------------
# Grid rows: (pattern track key, label). 16 steps = 1 bar of 16ths.
DRUM_GRID_TRACKS = (
    ("kick", "Kick"),
    ("snare", "Snare"),
    ("closed_hat", "Closed Hat"),
    ("open_hat", "Open Hat"),
)

# Four-on-the-floor default so the grid is never empty on first open.
DEFAULT_DRUM_GRID = {
    "kick":       [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "snare":      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "closed_hat": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "open_hat":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# Display name -> drum_machine.py DEATHSTEP_PATTERNS key.
DRUM_PRESET_NAMES = {
    "Deathstep Basic": "basic",
    "Deathstep Heavy": "heavy",
    "Buildup": "buildup",
    "Drop Hit": "drop",
}


def pattern_grid_to_dict(grid):
    """Convert a grid {track: 16 truthy/falsy values} to an engine pattern.

    Returns {track: [16 floats 0.0/1.0]}. Raises ValueError on an
    unknown track or a row that is not 16 steps, so a broken grid can
    never silently render the wrong beat.
    """
    pattern = {}
    for track, steps in grid.items():
        if track not in dict(DRUM_GRID_TRACKS):
            raise ValueError(f"unknown drum track: {track!r}")
        values = [1.0 if v else 0.0 for v in steps]
        if len(values) != 16:
            raise ValueError(f"drum track {track!r} must have 16 steps, got {len(values)}")
        pattern[track] = values
    for track, _label in DRUM_GRID_TRACKS:
        pattern.setdefault(track, [0.0] * 16)
    return pattern


def drum_machine_preset_to_grid(preset_name):
    """Load a drum_machine.py preset into grid form, or None.

    Only the four grid tracks are mapped (kick/snare/closed/open hat);
    the preset's clap/crash/tom tracks are not shown on this grid.
    Returns None when drum_machine.py is missing or the name is unknown,
    so the panel never breaks on a missing optional file.
    """
    try:
        from drum_machine import DEATHSTEP_PATTERNS
    except Exception:
        return None
    key = DRUM_PRESET_NAMES.get(preset_name)
    if key is None:
        return None
    preset = DEATHSTEP_PATTERNS.get(key)
    if preset is None:
        return None
    mapping = {"kick": "kick", "snare": "snare",
               "hihat_closed": "closed_hat", "hihat_open": "open_hat"}
    grid = {}
    for dm_track, grid_track in mapping.items():
        grid[grid_track] = [1 if v > 0 else 0 for v in preset.tracks.get(dm_track, [])]
    return grid


# ---------------------------------------------------------------------
# Drum pattern save/load (JSON, module level so it can be tested
# without opening the tkinter window). Small scope only: the 16-step
# grid + preset name. No stems, no MIDI, no velocity — those remain
# unfinished and are NOT claimed here.
# ---------------------------------------------------------------------
def save_pattern_json(grid, path, preset_name=""):
    """Write a validated 16-step grid to a JSON file.

    grid is {track: [16 truthy/falsy values]}. Validation is via
    pattern_grid_to_dict, so an unknown track or a non-16 row raises
    ValueError and nothing is written.
    """
    import json
    pattern = pattern_grid_to_dict(grid)
    data = {
        "format": "nightmare-drum-pattern",
        "version": 1,
        "steps": 16,
        "preset": str(preset_name or ""),
        "grid": {track: [int(v) for v in steps]
                 for track, steps in pattern.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return path


def load_pattern_json(path):
    """Read a pattern JSON file, with validation, or raise ValueError.

    Returns {track: [16 ints 0/1]}, with missing tracks filled with 0.
    Unknown tracks, non-16 rows, non-0/1 values, or malformed JSON
    all raise ValueError — a broken file can never silently load a
    wrong beat.
    """
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"cannot read pattern file: {e}")
    if not isinstance(data, dict):
        raise ValueError("pattern file must contain a JSON object")
    raw = data.get("grid")
    if not isinstance(raw, dict):
        raise ValueError("pattern file has no 'grid' object")
    grid = {}
    for track, steps in raw.items():
        if track not in dict(DRUM_GRID_TRACKS):
            raise ValueError(f"unknown drum track in file: {track!r}")
        if not isinstance(steps, list) or len(steps) != 16:
            raise ValueError(f"drum track {track!r} must have 16 steps")
        row = []
        for v in steps:
            if v is True:
                row.append(1)
            elif v is False:
                row.append(0)
            elif isinstance(v, int) and v in (0, 1):
                row.append(v)
            else:
                raise ValueError(f"drum track {track!r} has a non-0/1 step: {v!r}")
        grid[track] = row
    # pattern_grid_to_dict fills missing tracks with silence.
    pattern_grid_to_dict({t: [bool(v) for v in row] for t, row in grid.items()})
    for track, _label in DRUM_GRID_TRACKS:
        grid.setdefault(track, [0] * 16)
    return grid


def open_edm_studio(root, colors):
    win = tk.Toplevel(root)
    win.title("🎵 Nightmare EDM Music Studio — 3-Minute CD Ready Producer")
    win.geometry("920x1060")
    win.minsize(820, 920)
    win.configure(bg=colors.get("bg", "#0a0a0f"))

    # State variables
    current_wav_path = [None]
    is_rendering = [False]
    is_playing = [False]
    playback_thread = [None]
    playback_start_time = [0]

    # --- TOP HEADER ---
    top = tk.Frame(win, bg=colors["bg"])
    top.pack(fill=tk.X, padx=16, pady=(12, 6))

    head_row = tk.Frame(top, bg=colors["bg"])
    head_row.pack(fill=tk.X, pady=(0, 4))
    tk.Label(head_row, text="🎵 NIGHTMARE EDM STUDIO & VOCAL ENGINE", bg=colors["bg"],
             fg=colors.get("matrix_green", "#00ff66"), font=("Consolas", 14, "bold")).pack(side=tk.LEFT)
    tk.Label(head_row, text="[RED BOOK CD • 44.1kHz 16-BIT • 3:00 MIN • AIR-GAPPED]",
             bg=colors["bg"], fg=colors.get("text_dim", "#707080"), font=("Consolas", 9)).pack(side=tk.LEFT, padx=10)

    # --- MAIN CONFIGURATION CONTAINER ---
    main_frame = tk.Frame(win, bg=colors["bg"])
    main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

    # Track Title Row
    title_frame = tk.Frame(main_frame, bg=colors["bg"])
    title_frame.pack(fill=tk.X, pady=(0, 8))
    tk.Label(title_frame, text="Track Title:", bg=colors["bg"], fg=colors["text"], font=("Consolas", 10, "bold"), width=14, anchor=tk.W).pack(side=tk.LEFT)
    title_entry = tk.Entry(title_frame, bg=colors["bg_input"], fg=colors["text_bright"], font=("Consolas", 11), insertbackground=colors["text_bright"])
    title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    title_entry.insert(0, "Cyber_Odyssey_2026_Master")

    # Options Grid
    grid_frame = tk.LabelFrame(main_frame, text=" 🎛️ Production Parameters ", bg=colors["bg"], fg=colors.get("matrix_green", "#00ff66"), font=("Consolas", 10, "bold"), padx=10, pady=8)
    grid_frame.pack(fill=tk.X, pady=6)

    # Genre
    tk.Label(grid_frame, text="EDM Sub-Genre:", bg=colors["bg"], fg=colors["text"], font=("Consolas", 9, "bold")).grid(row=0, column=0, sticky=tk.W, pady=4)
    genre_var = tk.StringVar(value="Cyberpunk_Bass")
    genre_menu = ttk.Combobox(grid_frame, textvariable=genre_var, values=list(GENRE_PRESETS.keys()), state="readonly", width=22)
    genre_menu.grid(row=0, column=1, sticky=tk.W, padx=(4, 20), pady=4)

    # Tempo / BPM
    tk.Label(grid_frame, text="Tempo (BPM):", bg=colors["bg"], fg=colors["text"], font=("Consolas", 9, "bold")).grid(row=0, column=2, sticky=tk.W, pady=4)
    bpm_var = tk.StringVar(value="128")
    bpm_spin = tk.Spinbox(grid_frame, from_=120, to=150, textvariable=bpm_var, width=8, bg=colors["bg_input"], fg=colors["text_bright"], font=("Consolas", 10))
    bpm_spin.grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)

    # Vocal Mode
    tk.Label(grid_frame, text="Vocal Engine:", bg=colors["bg"], fg=colors["text"], font=("Consolas", 9, "bold")).grid(row=1, column=0, sticky=tk.W, pady=4)
    vocal_var = tk.StringVar(value="Vocoder_Cyber")
    vocal_modes = [
        "Vocoder_Cyber",
        "Formant_Singing_Voice",
        "Festival_Vocal_Chops",
        "Full_Verse_Chorus_Duet"
    ]
    vocal_menu = ttk.Combobox(grid_frame, textvariable=vocal_var, values=vocal_modes, state="readonly", width=22)
    vocal_menu.grid(row=1, column=1, sticky=tk.W, padx=(4, 20), pady=4)

    # Master Profile
    tk.Label(grid_frame, text="Mastering Target:", bg=colors["bg"], fg=colors["text"], font=("Consolas", 9, "bold")).grid(row=1, column=2, sticky=tk.W, pady=4)
    master_var = tk.StringVar(value="Club / Festival Loudness (-9 LUFS)")
    master_menu = ttk.Combobox(grid_frame, textvariable=master_var, values=[
        "Club / Festival Loudness (-9 LUFS)",
        "Streaming Radio Balanced (-14 LUFS)",
        "Audiophile Dynamic Master"
    ], state="readonly", width=28)
    master_menu.grid(row=1, column=3, sticky=tk.W, padx=4, pady=4)

    # Custom Lyrics / Prompt Box
    lyrics_frame = tk.LabelFrame(main_frame, text=" 🎤 Vocal Lyrics & Theme Prompts ", bg=colors["bg"], fg=colors.get("matrix_green", "#00ff66"), font=("Consolas", 10, "bold"), padx=10, pady=8)
    lyrics_frame.pack(fill=tk.X, pady=6)

    lyrics_text = tk.Text(lyrics_frame, height=4, bg=colors["bg_input"], fg=colors["text_bright"],
                          font=("Consolas", 10), insertbackground=colors["text_bright"])
    lyrics_text.pack(fill=tk.X, pady=(2, 6))
    lyrics_text.insert("1.0", "[Verse 1] Electric shadows in the neon rain, pulses running through my cyber vein\n[Build] Feel the bass line rise, open up your eyes\n[Chorus] Drop into the matrix, lost inside the beat, higher than the city street!")

    # Preset Lyrics Helpers
    btn_box = tk.Frame(lyrics_frame, bg=colors["bg"])
    btn_box.pack(fill=tk.X)

    def load_theme_lyrics(theme):
        themes = {
            "Cyberpunk": "[Verse] Neon towers in the midnight glow, binary currents start to flow\n[Build] Accelerate the clock, feel the grid unlock\n[Drop] Cybernetic energy, take control of reality!",
            "Festival Anthem": "[Verse] Hands up reaching for the laser light, we are immortal in the sound tonight\n[Build] 3, 2, 1 — Let the bass ignite!\n[Drop] We own the world, we feel alive!",
            "Trance Dream": "[Verse] Drifting across the starry sea, melody setting our spirits free\n[Build] Rising higher than the sky\n[Drop] Infinite horizon, infinite euphoria!"
        }
        lyrics_text.delete("1.0", tk.END)
        lyrics_text.insert("1.0", themes.get(theme, ""))

    tk.Button(btn_box, text="⚡ Cyberpunk Theme", bg=colors.get("bg_panel", "#14141e"), fg=colors["text"],
              font=("Consolas", 9), command=lambda: load_theme_lyrics("Cyberpunk")).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_box, text="🎪 Festival Anthem", bg=colors.get("bg_panel", "#14141e"), fg=colors["text"],
              font=("Consolas", 9), command=lambda: load_theme_lyrics("Festival Anthem")).pack(side=tk.LEFT, padx=6)
    tk.Button(btn_box, text="🌌 Trance Dream", bg=colors.get("bg_panel", "#14141e"), fg=colors["text"],
              font=("Consolas", 9), command=lambda: load_theme_lyrics("Trance Dream")).pack(side=tk.LEFT, padx=6)

    # --- PROGRESS & STATUS DISPLAY ---
    status_frame = tk.LabelFrame(main_frame, text=" 📟 Engine Telemetry & Arrangement Progress ", bg=colors["bg"], fg=colors.get("matrix_green", "#00ff66"), font=("Consolas", 10, "bold"), padx=10, pady=8)
    status_frame.pack(fill=tk.X, pady=6)

    progress_bar = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, mode="determinate")
    progress_bar.pack(fill=tk.X, pady=(2, 6))

    status_lbl = tk.Label(status_frame, text="Engine Idle. Ready to synthesize 3-Minute CD Master.", bg=colors["bg"], fg=colors["text_dim"], font=("Consolas", 9), anchor=tk.W)
    status_lbl.pack(fill=tk.X)

    # --- AUDIO PLAYER & MASTER CONTROLS ---
    player_frame = tk.LabelFrame(main_frame, text=" 🎚️ CD Master Playback & Export ", bg=colors["bg"], fg=colors.get("matrix_green", "#00ff66"), font=("Consolas", 10, "bold"), padx=10, pady=8)
    player_frame.pack(fill=tk.X, pady=6)

    play_row = tk.Frame(player_frame, bg=colors["bg"])
    play_row.pack(fill=tk.X, pady=4)

    time_lbl = tk.Label(play_row, text="00:00 / 03:00", bg=colors["bg"], fg=colors.get("matrix_green", "#00ff66"), font=("Consolas", 12, "bold"), width=16)
    time_lbl.pack(side=tk.LEFT, padx=(0, 10))

    def play_audio():
        if not current_wav_path[0] or not os.path.exists(current_wav_path[0]):
            messagebox.showwarning("No Audio", "Please render a track first before playback.")
            return
        try:
            winsound.PlaySound(current_wav_path[0], winsound.SND_ASYNC | winsound.SND_FILENAME)
            is_playing[0] = True
            playback_start_time[0] = time.time()
            status_lbl.config(text=f"▶️ Playing: {os.path.basename(current_wav_path[0])}", fg=colors.get("matrix_green", "#00ff66"))
            update_playback_timer()
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    def stop_audio():
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
            is_playing[0] = False
            time_lbl.config(text="00:00 / 03:00")
            status_lbl.config(text="⏹️ Playback stopped.", fg=colors["text_dim"])
        except Exception:
            pass

    def update_playback_timer():
        if is_playing[0]:
            elapsed = time.time() - playback_start_time[0]
            if elapsed > 180.0:
                stop_audio()
            else:
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                time_lbl.config(text=f"{mins:02d}:{secs:02d} / 03:00")
                win.after(500, update_playback_timer)

    def open_folder():
        if current_wav_path[0] and os.path.exists(current_wav_path[0]):
            os.startfile(os.path.dirname(current_wav_path[0]))
        else:
            desk = os.path.join(os.path.expanduser("~"), "Desktop")
            os.startfile(desk)

    play_btn = tk.Button(play_row, text=" ▶️ Play Master ", bg=colors.get("matrix_green", "#00ff66"), fg="#000000",
                         font=("Consolas", 10, "bold"), command=play_audio, padx=10)
    play_btn.pack(side=tk.LEFT, padx=4)

    stop_btn = tk.Button(play_row, text=" ⏹️ Stop ", bg=colors.get("bg_panel", "#14141e"), fg=colors["text_bright"],
                         font=("Consolas", 10), command=stop_audio, padx=8)
    stop_btn.pack(side=tk.LEFT, padx=4)

    folder_btn = tk.Button(play_row, text=" 📂 Open Folder ", bg=colors.get("bg_panel", "#14141e"), fg=colors["text_bright"],
                           font=("Consolas", 10), command=open_folder, padx=8)
    folder_btn.pack(side=tk.LEFT, padx=4)

    # --- DRUM STEP SEQUENCER: user beat input (was missing entirely) ---
    # Before this section the panel could only render a fixed 3-minute
    # track and play that WAV back. This grid lets the user tap in their
    # own 16-step beat, hear it (Preview renders it to WAV and plays it
    # with the same Play Master path), and optionally drop it into the
    # full-track render's Drop sections.
    seq_frame = tk.LabelFrame(main_frame, text=" 🥁 Drum Step Sequencer — tap your own beat (16 steps = 1 bar) ",
                              bg=colors["bg"], fg=colors.get("matrix_green", "#00ff66"),
                              font=("Consolas", 10, "bold"), padx=10, pady=8)
    seq_frame.pack(fill=tk.X, pady=6)

    hdr_row = tk.Frame(seq_frame, bg=colors["bg"])
    hdr_row.pack(fill=tk.X)
    tk.Label(hdr_row, text="Step:", bg=colors["bg"], fg=colors["text_dim"],
             font=("Consolas", 9, "bold"), width=12, anchor=tk.W).pack(side=tk.LEFT)
    for _step_no in range(16):
        tk.Label(hdr_row, text=str(_step_no + 1), bg=colors["bg"], fg=colors["text_dim"],
                 font=("Consolas", 8), width=2).pack(side=tk.LEFT, padx=1)

    grid_vars = {}
    for _track, _label in DRUM_GRID_TRACKS:
        grid_row = tk.Frame(seq_frame, bg=colors["bg"])
        grid_row.pack(fill=tk.X)
        tk.Label(grid_row, text=_label, bg=colors["bg"], fg=colors["text"],
                 font=("Consolas", 9, "bold"), width=12, anchor=tk.W).pack(side=tk.LEFT)
        _vars = []
        for _step in range(16):
            _var = tk.BooleanVar(value=bool(DEFAULT_DRUM_GRID[_track][_step]))
            _btn = tk.Checkbutton(grid_row, variable=_var, indicatoron=False, width=2,
                                  selectcolor=colors.get("matrix_green", "#00ff66"),
                                  bg=colors.get("bg_panel", "#14141e"),
                                  fg=colors["text_bright"],
                                  font=("Consolas", 9, "bold"))
            _btn.pack(side=tk.LEFT, padx=1)
            _vars.append(_var)
        grid_vars[_track] = _vars

    use_in_drops_var = tk.BooleanVar(value=False)
    tk.Checkbutton(seq_frame, text="Use this pattern for the Drops in the full-track render",
                   variable=use_in_drops_var, bg=colors["bg"], fg=colors["text"],
                   selectcolor=colors.get("bg_panel", "#14141e"),
                   font=("Consolas", 9)).pack(anchor=tk.W, pady=(6, 2))

    seq_btn_row = tk.Frame(seq_frame, bg=colors["bg"])
    seq_btn_row.pack(fill=tk.X, pady=2)

    preset_var = tk.StringVar(value="Deathstep Basic")
    preset_menu = ttk.Combobox(seq_btn_row, textvariable=preset_var,
                               values=list(DRUM_PRESET_NAMES.keys()),
                               state="readonly", width=18)
    preset_menu.pack(side=tk.LEFT, padx=(0, 4))

    def load_preset():
        grid = drum_machine_preset_to_grid(preset_var.get())
        if grid is None:
            messagebox.showwarning("Preset unavailable",
                                   "drum_machine.py presets could not be loaded.")
            return
        for _track, _vars in grid_vars.items():
            for _step, _var in enumerate(_vars):
                _var.set(bool(grid[_track][_step]))
        status_lbl.config(text=f"🥁 Loaded preset: {preset_var.get()}",
                          fg=colors.get("matrix_green", "#00ff66"))

    def clear_pattern():
        for _vars in grid_vars.values():
            for _var in _vars:
                _var.set(False)
        status_lbl.config(text="🥁 Pattern cleared.", fg=colors["text_dim"])

    pattern_busy = [False]

    def preview_pattern():
        if pattern_busy[0] or is_rendering[0]:
            return
        pattern_busy[0] = True
        stop_audio()
        try:
            bpm = float(bpm_var.get())
        except ValueError:
            bpm = 128.0
        title = title_entry.get().strip() or "EDM_Track_Master"
        pattern = pattern_grid_to_dict(
            {t: [v.get() for v in vars_] for t, vars_ in grid_vars.items()})
        output_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        out_wav = os.path.join(output_dir, f"{title}_pattern.wav")
        status_lbl.config(text="🥁 Rendering pattern preview (4 bars)...",
                          fg=colors.get("matrix_green", "#00ff66"))

        def worker():
            try:
                out = render_drum_pattern(out_wav, pattern, bpm=bpm, loops=4)
                current_wav_path[0] = out

                def on_done():
                    pattern_busy[0] = False
                    status_lbl.config(text=f"🥁 Pattern ready — playing: {os.path.basename(out)}",
                                      fg=colors.get("matrix_green", "#00ff66"))
                    play_audio()
                win.after(0, on_done)
            except Exception as e:
                def on_err(err=e):
                    pattern_busy[0] = False
                    status_lbl.config(text=f"❌ Pattern error: {err}", fg="#ff3333")
                    messagebox.showerror("Pattern Error", str(err))
                win.after(0, on_err)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    tk.Button(seq_btn_row, text=" ▶ Preview Pattern ", bg=colors.get("matrix_green", "#00ff66"),
              fg="#000000", font=("Consolas", 10, "bold"),
              command=preview_pattern, padx=8).pack(side=tk.LEFT, padx=4)
    tk.Button(seq_btn_row, text=" Load Preset ", bg=colors.get("bg_panel", "#14141e"),
              fg=colors["text_bright"], font=("Consolas", 10),
              command=load_preset, padx=8).pack(side=tk.LEFT, padx=4)
    tk.Button(seq_btn_row, text=" Clear ", bg=colors.get("bg_panel", "#14141e"),
              fg=colors["text_bright"], font=("Consolas", 10),
              command=clear_pattern, padx=8).pack(side=tk.LEFT, padx=4)

    def save_pattern():
        path = filedialog.asksaveasfilename(
            title="Save drum pattern",
            defaultextension=".json",
            filetypes=[("JSON pattern", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            grid = {t: [v.get() for v in vars_] for t, vars_ in grid_vars.items()}
            save_pattern_json(grid, path, preset_name=preset_var.get())
            status_lbl.config(text=f"🥁 Pattern saved: {os.path.basename(path)}",
                              fg=colors.get("matrix_green", "#00ff66"))
        except Exception as e:
            messagebox.showerror("Save Pattern Error", str(e))

    def load_pattern():
        path = filedialog.askopenfilename(
            title="Load drum pattern",
            filetypes=[("JSON pattern", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            grid = load_pattern_json(path)
        except Exception as e:
            messagebox.showerror("Load Pattern Error", str(e))
            return
        for _track, _vars in grid_vars.items():
            for _step, _var in enumerate(_vars):
                _var.set(bool(grid[_track][_step]))
        status_lbl.config(text=f"🥁 Pattern loaded: {os.path.basename(path)}",
                          fg=colors.get("matrix_green", "#00ff66"))

    tk.Button(seq_btn_row, text=" Save Pattern ", bg=colors.get("bg_panel", "#14141e"),
              fg=colors["text_bright"], font=("Consolas", 10),
              command=save_pattern, padx=8).pack(side=tk.LEFT, padx=4)
    tk.Button(seq_btn_row, text=" Load Pattern ", bg=colors.get("bg_panel", "#14141e"),
              fg=colors["text_bright"], font=("Consolas", 10),
              command=load_pattern, padx=8).pack(side=tk.LEFT, padx=4)

    # --- RENDER MASTER BUTTON ---
    # Mastering: panel labels keep the LUFS-style names the user knows,
    # but the engine wire below is drive/peak only (loud/balanced/
    # audiophile) — NOT a measured LUFS chain. Do not claim otherwise.
    _MASTER_LEVELS = {
        "Club / Festival Loudness (-9 LUFS)": "loud",
        "Streaming Radio Balanced (-14 LUFS)": "balanced",
        "Audiophile Dynamic Master": "audiophile",
    }
    def start_render():
        if is_rendering[0]:
            return
        is_rendering[0] = True
        stop_audio()
        
        genre = genre_var.get()
        try:
            bpm = float(bpm_var.get())
        except ValueError:
            bpm = 128.0
        vocal_mode = vocal_var.get()
        if use_in_drops_var.get():
            drum_pattern = pattern_grid_to_dict(
                {t: [v.get() for v in vars_] for t, vars_ in grid_vars.items()})
        else:
            drum_pattern = None
        title = title_entry.get().strip() or "EDM_Track_Master"
        lyrics = lyrics_text.get("1.0", tk.END).strip()
        
        output_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        out_wav = os.path.join(output_dir, f"{title}.wav")
        
        render_btn.config(state=tk.DISABLED, text="⏳ RENDERING 3-MINUTE MASTER...", bg="#333333")
        progress_bar["value"] = 0

        def worker():
            def cb(prog, msg):
                def update():
                    progress_bar["value"] = int(prog * 100)
                    status_lbl.config(text=f"[{int(prog*100)}%] {msg}", fg=colors.get("matrix_green", "#00ff66"))
                win.after(0, update)

            try:
                out = render_edm_track(
                    output_wav_path=out_wav,
                    genre=genre,
                    bpm=bpm,
                    vocal_mode=vocal_mode,
                    custom_lyrics=lyrics,
                    callback=cb,
                    drum_pattern=drum_pattern,
                    master_level=_MASTER_LEVELS.get(master_var.get(), "balanced")
                )
                current_wav_path[0] = out
                def on_done():
                    is_rendering[0] = False
                    render_btn.config(state=tk.NORMAL, text=" ⚡ RENDER 3-MINUTE CD MASTER (WAV) ⚡ ", bg=colors.get("matrix_green", "#00ff66"))
                    status_lbl.config(text=f"✅ Finished! Master Exported to: {out}", fg=colors.get("matrix_green", "#00ff66"))
                    messagebox.showinfo("CD Master Ready", f"Successfully rendered 3-Minute CD Ready EDM Master!\n\nPath:\n{out}\n\nSpecs: 44,100 Hz • 16-Bit PCM Stereo (Red Book Standard)")
                win.after(0, on_done)
            except Exception as e:
                def on_err(err=e):
                    is_rendering[0] = False
                    render_btn.config(state=tk.NORMAL, text=" ⚡ RENDER 3-MINUTE CD MASTER (WAV) ⚡ ", bg=colors.get("matrix_green", "#00ff66"))
                    status_lbl.config(text=f"❌ Error: {err}", fg="#ff3333")
                    messagebox.showerror("Render Error", str(err))
                win.after(0, on_err)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    render_btn = tk.Button(main_frame, text=" ⚡ RENDER 3-MINUTE CD MASTER (WAV) ⚡ ", bg=colors.get("matrix_green", "#00ff66"),
                           fg="#000000", font=("Consolas", 12, "bold"), pady=8, command=start_render)
    render_btn.pack(fill=tk.X, pady=(10, 4))

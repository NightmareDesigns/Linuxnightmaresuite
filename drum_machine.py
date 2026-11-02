"""
💀 NIGHTMARE STUDIO — Drum Machine
=====================================
Synthesized drums from scratch + step sequencer.
Every drum sound generated with NumPy — no samples needed.

Drum Sounds:
  - Kick (sub-heavy, punchy, 808-style)
  - Snare (layered noise + body)
  - Clap
  - Hi-hat (open & closed)
  - Crash cymbal
  - Ride cymbal
  - Tom (high, mid, low)
  - Percussion FX (metallic, glitch, impact)

Step Sequencer:
  - 16/32/64 steps per pattern
  - Per-step velocity
  - Pattern chaining for arrangement
  - Swing/shuffle

Author: Nightmare Designs 💀
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

SAMPLE_RATE = 44100
TWO_PI = 2 * np.pi


# ============================================================
# SYNTHESIZED DRUM SOUNDS — From Scratch
# ============================================================

def drum_kick(pitch: float = 55.0, punch: float = 0.8, decay: float = 0.4,
              click: float = 0.3, sub: float = 0.8, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthesized kick drum — pitch sweep + sub sine + transient click.
    
    Args:
        pitch: Base pitch in Hz (55 = deep, 80 = punchy)
        punch: Initial transient intensity
        decay: Overall decay time in seconds
        click: High-frequency click at attack
        sub: Sub-bass amount
    """
    n_samples = int(sr * decay)
    t = np.arange(n_samples) / sr
    
    # Pitch envelope — starts high, sweeps down fast
    pitch_env = pitch + 200 * punch * np.exp(-t * 40)
    
    # Body — sine with pitch sweep
    phase = np.cumsum(TWO_PI * pitch_env / sr)
    body = np.sin(phase) * np.exp(-t * (3.0 / decay))
    
    # Sub — pure low sine
    sub_osc = np.sin(TWO_PI * pitch * 0.5 * t) * np.exp(-t * (2.0 / decay)) * sub
    
    # Click — short burst of noise
    click_env = np.exp(-t * 200) * click
    click_noise = np.random.randn(n_samples) * click_env
    
    # Combine
    kick = body * 0.7 + sub_osc * 0.5 + click_noise * 0.3
    
    # Compress hard for punch
    kick = np.tanh(kick * 2.0) * 0.9
    
    return kick.astype(np.float32)


def drum_snare(tone: float = 200.0, noise_amount: float = 0.7, decay: float = 0.25,
               ring: float = 0.3, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthesized snare — tonal body + noise layer.
    """
    n_samples = int(sr * decay)
    t = np.arange(n_samples) / sr
    
    # Tonal body (two detuned oscillators)
    body_env = np.exp(-t * 20)
    body1 = np.sin(TWO_PI * tone * t) * body_env
    body2 = np.sin(TWO_PI * tone * 1.5 * t) * body_env * 0.5
    
    # Noise layer (filtered)
    noise_env = np.exp(-t * (8.0 / decay))
    noise = np.random.randn(n_samples) * noise_env * noise_amount
    
    # Ring (resonant overtone)
    ring_osc = np.sin(TWO_PI * tone * 3.46 * t) * np.exp(-t * 30) * ring
    
    snare = body1 + body2 + noise + ring_osc
    snare = np.tanh(snare * 1.5) * 0.85
    
    return snare.astype(np.float32)


def drum_clap(decay: float = 0.2, layers: int = 4, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesized hand clap — layered noise bursts."""
    n_samples = int(sr * decay)
    t = np.arange(n_samples) / sr
    
    clap = np.zeros(n_samples)
    
    # Multiple short noise bursts (simulates multiple hands)
    for i in range(layers):
        offset = int(sr * 0.005 * i)  # 5ms apart
        if offset < n_samples:
            burst_len = min(int(sr * 0.01), n_samples - offset)
            burst = np.random.randn(burst_len) * np.exp(-np.linspace(0, 5, burst_len))
            clap[offset:offset + burst_len] += burst
    
    # Overall envelope
    env = np.exp(-t * 15)
    clap *= env
    
    # Bandpass filter effect (claps are mid-heavy)
    # Simple: reduce lows and highs
    from synth_engine import filter_bandpass
    clap = filter_bandpass(clap, 1200, 0.5, sr)
    
    return (clap / (np.max(np.abs(clap)) + 1e-10) * 0.8).astype(np.float32)


def drum_hihat_closed(decay: float = 0.05, tone: float = 8000.0,
                      sr: int = SAMPLE_RATE) -> np.ndarray:
    """Closed hi-hat — short burst of filtered noise."""
    n_samples = int(sr * decay)
    t = np.arange(n_samples) / sr
    
    # Multiple square wave oscillators at inharmonic frequencies (metallic)
    sig = np.zeros(n_samples)
    freqs = [tone, tone * 1.34, tone * 1.68, tone * 1.91, tone * 2.13, tone * 2.52]
    for f in freqs:
        sig += np.sign(np.sin(TWO_PI * f * t))
    sig /= len(freqs)
    
    # Envelope
    env = np.exp(-t * (60.0 / decay))
    sig *= env
    
    # High-pass to remove any low content
    from synth_engine import filter_highpass
    sig = filter_highpass(sig, 5000, 0.3, sr)
    
    return (sig * 0.5).astype(np.float32)


def drum_hihat_open(decay: float = 0.3, tone: float = 8000.0,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """Open hi-hat — longer decay, same character."""
    return drum_hihat_closed(decay=decay, tone=tone, sr=sr)


def drum_crash(decay: float = 2.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Crash cymbal — noise + metallic inharmonics, long decay."""
    n_samples = int(sr * decay)
    t = np.arange(n_samples) / sr
    
    # Metallic oscillators
    sig = np.zeros(n_samples)
    freqs = [4200, 5163, 6375, 7834, 9100, 10300]
    for f in freqs:
        sig += np.sin(TWO_PI * f * t + np.random.rand() * TWO_PI) * np.exp(-t * (1.5 / decay))
    
    # Noise component
    noise = np.random.randn(n_samples) * np.exp(-t * (3.0 / decay)) * 0.3
    
    sig = (sig / len(freqs) + noise) * 0.5
    
    return sig.astype(np.float32)


def drum_tom(pitch: float = 100.0, decay: float = 0.3, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Tom drum — pitched body with fast pitch drop."""
    n_samples = int(sr * decay)
    t = np.arange(n_samples) / sr
    
    # Pitch sweep
    freq_env = pitch + 80 * np.exp(-t * 25)
    phase = np.cumsum(TWO_PI * freq_env / sr)
    body = np.sin(phase) * np.exp(-t * (6.0 / decay))
    
    # Add some attack noise
    noise = np.random.randn(n_samples) * np.exp(-t * 50) * 0.2
    
    return ((body + noise) * 0.7).astype(np.float32)


def drum_impact(decay: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Heavy impact — sub boom + noise explosion. For drops."""
    n_samples = int(sr * decay)
    t = np.arange(n_samples) / sr
    
    # Sub boom (40Hz sine with long decay)
    sub = np.sin(TWO_PI * 40 * t) * np.exp(-t * 2)
    
    # Explosion noise (fast attack, medium decay)
    noise = np.random.randn(n_samples) * np.exp(-t * 5) * 0.6
    
    # Combine
    impact = sub * 0.7 + noise * 0.5
    impact = np.tanh(impact * 2.0)
    
    return impact.astype(np.float32)


# ============================================================
# STEP SEQUENCER
# ============================================================

class DrumPattern:
    """A single drum pattern — 16/32/64 steps."""
    
    def __init__(self, steps: int = 16, name: str = "Pattern 1"):
        self.name = name
        self.steps = steps
        
        # Each track: list of velocities per step (0 = off, 0.1-1.0 = velocity)
        self.tracks: Dict[str, List[float]] = {
            "kick": [0.0] * steps,
            "snare": [0.0] * steps,
            "clap": [0.0] * steps,
            "hihat_closed": [0.0] * steps,
            "hihat_open": [0.0] * steps,
            "crash": [0.0] * steps,
            "tom_low": [0.0] * steps,
            "tom_mid": [0.0] * steps,
            "tom_high": [0.0] * steps,
            "impact": [0.0] * steps,
        }
    
    def set_step(self, track: str, step: int, velocity: float = 1.0):
        """Set a step on/off with velocity."""
        if track in self.tracks and 0 <= step < self.steps:
            self.tracks[track][step] = velocity
    
    def clear_track(self, track: str):
        """Clear all steps on a track."""
        if track in self.tracks:
            self.tracks[track] = [0.0] * self.steps
    
    def clear_all(self):
        """Clear the entire pattern."""
        for track in self.tracks:
            self.tracks[track] = [0.0] * self.steps


class DrumMachine:
    """
    Full drum machine — pattern sequencer + sound synthesis.
    Renders patterns to audio at a given BPM.
    """
    
    # Sound generators for each track
    SOUND_GENERATORS = {
        "kick": lambda vel: drum_kick(pitch=55, punch=vel, decay=0.4),
        "snare": lambda vel: drum_snare(noise_amount=0.6 + vel * 0.3, decay=0.2),
        "clap": lambda vel: drum_clap(decay=0.15) * vel,
        "hihat_closed": lambda vel: drum_hihat_closed(decay=0.04 + vel * 0.02) * vel,
        "hihat_open": lambda vel: drum_hihat_open(decay=0.2 + vel * 0.1) * vel,
        "crash": lambda vel: drum_crash(decay=1.5) * vel,
        "tom_low": lambda vel: drum_tom(pitch=80, decay=0.3) * vel,
        "tom_mid": lambda vel: drum_tom(pitch=120, decay=0.25) * vel,
        "tom_high": lambda vel: drum_tom(pitch=180, decay=0.2) * vel,
        "impact": lambda vel: drum_impact(decay=1.0) * vel,
    }
    
    def __init__(self, bpm: float = 145.0):
        self.bpm = bpm
        self.patterns: List[DrumPattern] = [DrumPattern()]
        self.active_pattern = 0
        self.swing = 0.0  # 0-1, amount of swing
    
    def get_step_duration(self) -> float:
        """Duration of one step in seconds (16th note at given BPM)."""
        return 60.0 / self.bpm / 4.0  # 16th note
    
    def render_pattern(self, pattern_idx: int = None, sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Render a complete drum pattern to audio.
        
        Returns: numpy array of mixed audio
        """
        if pattern_idx is None:
            pattern_idx = self.active_pattern
        
        pattern = self.patterns[pattern_idx]
        step_duration = self.get_step_duration()
        total_duration = step_duration * pattern.steps
        n_samples = int(sr * total_duration)
        
        output = np.zeros(n_samples, dtype=np.float32)
        
        for track_name, steps in pattern.tracks.items():
            if track_name not in self.SOUND_GENERATORS:
                continue
            
            for step_idx, velocity in enumerate(steps):
                if velocity <= 0:
                    continue
                
                # Calculate step position with swing
                step_time = step_idx * step_duration
                if self.swing > 0 and step_idx % 2 == 1:
                    # Swing: push every other 16th note later
                    step_time += step_duration * self.swing * 0.5
                
                start_sample = int(step_time * sr)
                
                # Generate the drum sound
                sound = self.SOUND_GENERATORS[track_name](velocity)
                
                # Place in output
                end_sample = min(start_sample + len(sound), n_samples)
                sound_len = end_sample - start_sample
                if sound_len > 0 and start_sample < n_samples:
                    output[start_sample:end_sample] += sound[:sound_len]
        
        # Soft clip to prevent clipping from layered hits
        output = np.tanh(output)
        
        return output
    
    def render_bars(self, n_bars: int = 4, pattern_sequence: List[int] = None,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Render multiple bars (patterns) in sequence.
        
        Args:
            n_bars: Number of bars to render
            pattern_sequence: List of pattern indices to play in order
        """
        if pattern_sequence is None:
            pattern_sequence = [self.active_pattern] * n_bars
        
        chunks = []
        for pat_idx in pattern_sequence:
            if pat_idx < len(self.patterns):
                chunks.append(self.render_pattern(pat_idx, sr))
        
        if not chunks:
            return np.zeros(int(sr * 4 * 60 / self.bpm), dtype=np.float32)
        
        return np.concatenate(chunks)


# ============================================================
# PRESET PATTERNS — Deathstep
# ============================================================

def pattern_deathstep_basic() -> DrumPattern:
    """Basic deathstep half-time pattern."""
    p = DrumPattern(16, "Deathstep Basic")
    
    # Kick on 1 and (sometimes) 9
    p.set_step("kick", 0, 1.0)
    p.set_step("kick", 8, 0.6)
    
    # Snare on 5 and 13 (half-time feel)
    p.set_step("snare", 4, 1.0)
    p.set_step("snare", 12, 0.9)
    
    # Hi-hats
    for i in range(0, 16, 2):
        p.set_step("hihat_closed", i, 0.5 + (i % 4) * 0.1)
    p.set_step("hihat_open", 6, 0.6)
    p.set_step("hihat_open", 14, 0.5)
    
    return p


def pattern_deathstep_heavy() -> DrumPattern:
    """Heavy pattern with double kicks and ghost snares."""
    p = DrumPattern(16, "Deathstep Heavy")
    
    # Double kick
    p.set_step("kick", 0, 1.0)
    p.set_step("kick", 1, 0.7)
    p.set_step("kick", 8, 0.9)
    p.set_step("kick", 10, 0.6)
    
    # Snare
    p.set_step("snare", 4, 1.0)
    p.set_step("snare", 12, 1.0)
    p.set_step("snare", 7, 0.3)  # Ghost note
    
    # Clap layered on snare
    p.set_step("clap", 4, 0.8)
    p.set_step("clap", 12, 0.7)
    
    # Fast hats
    for i in range(16):
        p.set_step("hihat_closed", i, 0.3 + 0.1 * (i % 2))
    p.set_step("hihat_open", 3, 0.5)
    p.set_step("hihat_open", 11, 0.5)
    
    return p


def pattern_buildup() -> DrumPattern:
    """Build-up pattern — accelerating hits."""
    p = DrumPattern(16, "Buildup")
    
    # Snare roll (accelerating)
    p.set_step("snare", 8, 0.4)
    p.set_step("snare", 10, 0.5)
    p.set_step("snare", 11, 0.6)
    p.set_step("snare", 12, 0.7)
    p.set_step("snare", 13, 0.8)
    p.set_step("snare", 14, 0.9)
    p.set_step("snare", 15, 1.0)
    
    # Kick sparse
    p.set_step("kick", 0, 0.8)
    
    # Crash at end
    p.set_step("crash", 15, 0.9)
    
    return p


def pattern_drop() -> DrumPattern:
    """Drop pattern — impact + heavy kick."""
    p = DrumPattern(16, "Drop Hit")
    
    # Impact on beat 1
    p.set_step("impact", 0, 1.0)
    p.set_step("kick", 0, 1.0)
    p.set_step("crash", 0, 0.9)
    
    # Rest sparse — let the bass breathe
    p.set_step("kick", 8, 0.8)
    p.set_step("snare", 12, 0.9)
    
    return p


DEATHSTEP_PATTERNS = {
    "basic": pattern_deathstep_basic(),
    "heavy": pattern_deathstep_heavy(),
    "buildup": pattern_buildup(),
    "drop": pattern_drop(),
}

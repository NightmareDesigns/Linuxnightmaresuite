"""
💀 NIGHTMARE STUDIO — Synthesizer Engine
==========================================
Pure NumPy audio synthesis from scratch.
Wavetable, FM, Subtractive, and Noise generators.
No external audio libraries. Just math.

Oscillators:
  - Sine, Saw, Square, Triangle, Pulse
  - Wavetable (morphable)
  - FM synthesis (operator stacks)
  - Noise (white, pink, brown)
  - Super saw (detuned unison)

Modulators:
  - LFO (all waveforms)
  - ADSR envelope
  - Pitch envelope
  - Filter envelope

Filters:
  - Low-pass (resonant, 2-pole and 4-pole)
  - High-pass
  - Band-pass
  - Notch
  - Moog ladder filter emulation

Author: Nightmare Designs 💀
"""

import numpy as np
from typing import Optional, Tuple, List

# Audio constants
SAMPLE_RATE = 44100
TWO_PI = 2.0 * np.pi


# ============================================================
# OSCILLATORS — Raw Waveform Generation
# ============================================================

def osc_sine(freq: float, duration: float, phase: float = 0.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Pure sine wave."""
    t = np.arange(int(sr * duration)) / sr
    return np.sin(TWO_PI * freq * t + phase)


def osc_saw(freq: float, duration: float, phase: float = 0.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Sawtooth wave (band-limited via additive synthesis for anti-aliasing)."""
    t = np.arange(int(sr * duration)) / sr
    n_samples = len(t)
    signal = np.zeros(n_samples)
    
    # Additive synthesis — sum harmonics (band-limited to Nyquist)
    max_harmonic = int(sr / (2 * freq))
    for k in range(1, min(max_harmonic + 1, 64)):  # Cap at 64 harmonics for speed
        signal += ((-1) ** (k + 1)) * np.sin(TWO_PI * k * freq * t + phase * k) / k
    
    return signal * (2.0 / np.pi)


def osc_square(freq: float, duration: float, duty: float = 0.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Square/pulse wave with variable duty cycle."""
    t = np.arange(int(sr * duration)) / sr
    phase = (freq * t) % 1.0
    return np.where(phase < duty, 1.0, -1.0)


def osc_triangle(freq: float, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Triangle wave."""
    t = np.arange(int(sr * duration)) / sr
    phase = (freq * t) % 1.0
    return 4.0 * np.abs(phase - 0.5) - 1.0


def osc_pulse(freq: float, duration: float, width: float = 0.1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Narrow pulse wave (good for plucks and digital sounds)."""
    return osc_square(freq, duration, duty=width, sr=sr)


def osc_super_saw(freq: float, duration: float, voices: int = 7, detune: float = 0.02, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Super saw — multiple detuned saw waves stacked.
    Classic trance/EDM lead sound. Also great for heavy walls of sound.
    
    Args:
        voices: Number of detuned oscillators (odd number centers one at exact freq)
        detune: Maximum detune amount (0.02 = 2% = ~34 cents)
    """
    t = np.arange(int(sr * duration)) / sr
    signal = np.zeros(len(t))
    
    for i in range(voices):
        # Spread detune evenly across voices
        detune_factor = 1.0 + detune * (2.0 * i / (voices - 1) - 1.0) if voices > 1 else 1.0
        voice_freq = freq * detune_factor
        # Simple saw (phase accumulation for speed)
        phase = (voice_freq * t) % 1.0
        signal += 2.0 * phase - 1.0
    
    return signal / voices


def osc_noise_white(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """White noise — equal energy at all frequencies."""
    return np.random.uniform(-1.0, 1.0, int(sr * duration)).astype(np.float32)


def osc_noise_pink(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Pink noise (1/f) — more low-frequency content. Good for risers/sweeps."""
    n_samples = int(sr * duration)
    white = np.random.randn(n_samples)
    
    # Voss-McCartney algorithm (approximate)
    pink = np.zeros(n_samples)
    b = np.zeros(7)
    for i in range(n_samples):
        white_val = white[i]
        # Update each octave at different rates
        for j in range(7):
            if i % (2 ** j) == 0:
                b[j] = white[i] * 0.5
        pink[i] = np.sum(b) + white_val * 0.5
    
    # Normalize
    pink /= np.max(np.abs(pink)) + 1e-10
    return pink.astype(np.float32)


def osc_noise_brown(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Brown/red noise — random walk. Very bass-heavy."""
    n_samples = int(sr * duration)
    white = np.random.randn(n_samples) * 0.02
    brown = np.cumsum(white)
    brown /= np.max(np.abs(brown)) + 1e-10
    return brown.astype(np.float32)


# ============================================================
# WAVETABLE SYNTHESIS
# ============================================================

class Wavetable:
    """
    Wavetable oscillator — morph between wave shapes.
    Perfect for evolving bass sounds (deathstep growls).
    """
    
    TABLE_SIZE = 2048
    
    def __init__(self):
        self.tables: List[np.ndarray] = []
        self._build_default_tables()
    
    def _build_default_tables(self):
        """Build a set of morphable wavetables."""
        t = np.linspace(0, 1, self.TABLE_SIZE, endpoint=False)
        
        # Table 0: Sine
        self.tables.append(np.sin(TWO_PI * t))
        
        # Table 1: Soft saw
        saw = np.zeros(self.TABLE_SIZE)
        for k in range(1, 16):
            saw += ((-1) ** (k + 1)) * np.sin(TWO_PI * k * t) / k
        self.tables.append(saw * (2.0 / np.pi))
        
        # Table 2: Hard saw (more harmonics)
        saw2 = np.zeros(self.TABLE_SIZE)
        for k in range(1, 64):
            saw2 += ((-1) ** (k + 1)) * np.sin(TWO_PI * k * t) / k
        self.tables.append(saw2 * (2.0 / np.pi))
        
        # Table 3: Square
        sq = np.zeros(self.TABLE_SIZE)
        for k in range(1, 32, 2):
            sq += np.sin(TWO_PI * k * t) / k
        self.tables.append(sq * (4.0 / np.pi))
        
        # Table 4: PWM 25%
        self.tables.append(np.where(t < 0.25, 1.0, -1.0))
        
        # Table 5: Triangle
        self.tables.append(4.0 * np.abs(t - 0.5) - 1.0)
        
        # Table 6: Formant vowel "A" (growl base)
        formant = np.sin(TWO_PI * t) + 0.5 * np.sin(TWO_PI * 3 * t) + 0.25 * np.sin(TWO_PI * 5 * t)
        self.tables.append(formant / np.max(np.abs(formant)))
        
        # Table 7: Digital harsh (bit-crushed sine)
        digital = np.round(np.sin(TWO_PI * t) * 4) / 4
        self.tables.append(digital)
        
        # Table 8: Metallic (inharmonic overtones)
        metal = np.sin(TWO_PI * t) + 0.7 * np.sin(TWO_PI * 2.76 * t) + 0.5 * np.sin(TWO_PI * 4.53 * t)
        self.tables.append(metal / np.max(np.abs(metal)))
    
    def generate(self, freq: float, duration: float, position: float = 0.0,
                 morph_start: float = 0.0, morph_end: float = 0.0,
                 sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Generate audio from wavetable with morphing.
        
        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            position: Static wavetable position (0.0 to 1.0)
            morph_start: Start morph position (overrides position if != morph_end)
            morph_end: End morph position
        """
        n_samples = int(sr * duration)
        n_tables = len(self.tables)
        
        # Phase accumulator
        phase_inc = freq * self.TABLE_SIZE / sr
        phases = np.cumsum(np.full(n_samples, phase_inc)) % self.TABLE_SIZE
        phase_indices = phases.astype(np.int32)
        
        # Morph position over time
        if morph_start != morph_end:
            morph = np.linspace(morph_start, morph_end, n_samples)
        else:
            morph = np.full(n_samples, position)
        
        # Scale morph to table indices
        table_pos = morph * (n_tables - 1)
        table_idx_low = np.clip(table_pos.astype(np.int32), 0, n_tables - 2)
        table_frac = table_pos - table_idx_low
        
        # Interpolate between adjacent tables
        output = np.zeros(n_samples)
        for i in range(n_samples):
            idx = phase_indices[i]
            t_low = table_idx_low[i]
            frac = table_frac[i]
            
            sample_low = self.tables[t_low][idx]
            sample_high = self.tables[t_low + 1][idx]
            output[i] = sample_low * (1.0 - frac) + sample_high * frac
        
        return output.astype(np.float32)


# ============================================================
# FM SYNTHESIS
# ============================================================

def fm_synthesis(carrier_freq: float, mod_freq: float, mod_depth: float,
                 duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    2-operator FM synthesis.
    
    Args:
        carrier_freq: Carrier frequency (heard pitch)
        mod_freq: Modulator frequency (creates harmonics)
        mod_depth: Modulation index (higher = more harmonics/metallic)
    """
    t = np.arange(int(sr * duration)) / sr
    modulator = mod_depth * np.sin(TWO_PI * mod_freq * t)
    carrier = np.sin(TWO_PI * carrier_freq * t + modulator)
    return carrier.astype(np.float32)


def fm_3op(freq: float, ratios: Tuple[float, float, float],
           depths: Tuple[float, float], duration: float,
           sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    3-operator FM: Op3 → Op2 → Op1 (carrier).
    Great for metallic/bell sounds and aggressive timbres.
    """
    t = np.arange(int(sr * duration)) / sr
    
    f1, f2, f3 = freq * ratios[0], freq * ratios[1], freq * ratios[2]
    
    # Op3 modulates Op2
    op3 = depths[1] * np.sin(TWO_PI * f3 * t)
    # Op2 modulates Op1
    op2 = depths[0] * np.sin(TWO_PI * f2 * t + op3)
    # Op1 is the carrier (output)
    op1 = np.sin(TWO_PI * f1 * t + op2)
    
    return op1.astype(np.float32)


# ============================================================
# ENVELOPES
# ============================================================

class ADSR:
    """
    Attack-Decay-Sustain-Release envelope.
    The backbone of every synth sound.
    """
    
    def __init__(self, attack: float = 0.01, decay: float = 0.1,
                 sustain: float = 0.7, release: float = 0.2):
        self.attack = attack    # seconds
        self.decay = decay      # seconds
        self.sustain = sustain  # level 0-1
        self.release = release  # seconds
    
    def generate(self, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Generate envelope shape over given duration."""
        n_samples = int(sr * duration)
        envelope = np.zeros(n_samples)
        
        attack_samples = int(self.attack * sr)
        decay_samples = int(self.decay * sr)
        release_samples = int(self.release * sr)
        sustain_samples = max(0, n_samples - attack_samples - decay_samples - release_samples)
        
        idx = 0
        
        # Attack: 0 → 1
        if attack_samples > 0:
            end = min(idx + attack_samples, n_samples)
            envelope[idx:end] = np.linspace(0, 1, end - idx)
            idx = end
        
        # Decay: 1 → sustain
        if idx < n_samples and decay_samples > 0:
            end = min(idx + decay_samples, n_samples)
            envelope[idx:end] = np.linspace(1, self.sustain, end - idx)
            idx = end
        
        # Sustain: hold at sustain level
        if idx < n_samples and sustain_samples > 0:
            end = min(idx + sustain_samples, n_samples)
            envelope[idx:end] = self.sustain
            idx = end
        
        # Release: sustain → 0
        if idx < n_samples:
            envelope[idx:] = np.linspace(self.sustain, 0, n_samples - idx)
        
        return envelope.astype(np.float32)


class LFO:
    """Low-Frequency Oscillator for modulation."""
    
    def __init__(self, rate: float = 4.0, shape: str = "sine", depth: float = 1.0):
        self.rate = rate      # Hz
        self.shape = shape    # "sine", "triangle", "square", "saw", "random"
        self.depth = depth    # 0-1
    
    def generate(self, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Generate LFO signal (bipolar: -depth to +depth)."""
        t = np.arange(int(sr * duration)) / sr
        
        if self.shape == "sine":
            signal = np.sin(TWO_PI * self.rate * t)
        elif self.shape == "triangle":
            phase = (self.rate * t) % 1.0
            signal = 4.0 * np.abs(phase - 0.5) - 1.0
        elif self.shape == "square":
            signal = np.sign(np.sin(TWO_PI * self.rate * t))
        elif self.shape == "saw":
            phase = (self.rate * t) % 1.0
            signal = 2.0 * phase - 1.0
        elif self.shape == "random" or self.shape == "s&h":
            # Sample & Hold
            samples_per_cycle = int(sr / self.rate)
            signal = np.zeros(len(t))
            for i in range(0, len(t), samples_per_cycle):
                val = np.random.uniform(-1, 1)
                signal[i:i + samples_per_cycle] = val
        else:
            signal = np.sin(TWO_PI * self.rate * t)
        
        return (signal * self.depth).astype(np.float32)


# ============================================================
# FILTERS — Digital IIR Implementations
# ============================================================

def filter_lowpass(signal: np.ndarray, cutoff: float, resonance: float = 0.5,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    2-pole resonant low-pass filter (State Variable Filter).
    
    Args:
        cutoff: Cutoff frequency in Hz
        resonance: Resonance 0-1 (0.95+ for self-oscillation)
    """
    output = np.zeros_like(signal)
    
    # SVF coefficients
    f = 2.0 * np.sin(np.pi * min(cutoff, sr * 0.49) / sr)
    q = 1.0 - resonance * 0.99  # Prevent division by zero
    
    low = 0.0
    band = 0.0
    
    for i in range(len(signal)):
        high = signal[i] - low - q * band
        band += f * high
        low += f * band
        output[i] = low
    
    return output.astype(np.float32)


def filter_highpass(signal: np.ndarray, cutoff: float, resonance: float = 0.5,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """2-pole resonant high-pass filter."""
    output = np.zeros_like(signal)
    
    f = 2.0 * np.sin(np.pi * min(cutoff, sr * 0.49) / sr)
    q = 1.0 - resonance * 0.99
    
    low = 0.0
    band = 0.0
    
    for i in range(len(signal)):
        high = signal[i] - low - q * band
        band += f * high
        low += f * band
        output[i] = high
    
    return output.astype(np.float32)


def filter_bandpass(signal: np.ndarray, cutoff: float, resonance: float = 0.5,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """2-pole band-pass filter."""
    output = np.zeros_like(signal)
    
    f = 2.0 * np.sin(np.pi * min(cutoff, sr * 0.49) / sr)
    q = 1.0 - resonance * 0.99
    
    low = 0.0
    band = 0.0
    
    for i in range(len(signal)):
        high = signal[i] - low - q * band
        band += f * high
        low += f * band
        output[i] = band
    
    return output.astype(np.float32)


def filter_moog(signal: np.ndarray, cutoff: float, resonance: float = 0.7,
                sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Moog ladder filter emulation (4-pole, 24dB/oct).
    The fat, warm filter sound. Perfect for bass.
    """
    output = np.zeros_like(signal)
    
    # Cutoff frequency to coefficient
    fc = 2.0 * cutoff / sr
    fc = min(fc, 0.99)
    
    # Feedback amount from resonance (0-4 for Moog, >4 = self-oscillation)
    fb = resonance * 4.0
    
    # 4 cascaded 1-pole filters
    y1 = y2 = y3 = y4 = 0.0
    old_y3 = 0.0
    
    for i in range(len(signal)):
        # Input with feedback
        x = signal[i] - fb * y4
        x = np.tanh(x)  # Soft clipping (nonlinear like real Moog)
        
        # 4 cascaded RC stages
        y1 += fc * (x - y1)
        y2 += fc * (y1 - y2)
        y3 += fc * (y2 - y3)
        y4 += fc * (y3 - y4)
        
        output[i] = y4
    
    return output.astype(np.float32)


def filter_notch(signal: np.ndarray, cutoff: float, width: float = 100.0,
                 sr: int = SAMPLE_RATE) -> np.ndarray:
    """Notch (band-reject) filter."""
    lp = filter_lowpass(signal, cutoff - width / 2, 0.3, sr)
    hp = filter_highpass(signal, cutoff + width / 2, 0.3, sr)
    return lp + hp


def filter_sweep(signal: np.ndarray, start_freq: float, end_freq: float,
                 resonance: float = 0.6, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Low-pass filter with sweeping cutoff (automation).
    Essential for dubstep wobbles and builds.
    """
    output = np.zeros_like(signal)
    n = len(signal)
    
    # Exponential sweep (sounds more natural than linear)
    freqs = np.exp(np.linspace(np.log(max(start_freq, 20)), np.log(min(end_freq, sr * 0.49)), n))
    
    low = 0.0
    band = 0.0
    q = 1.0 - resonance * 0.99
    
    for i in range(n):
        f = 2.0 * np.sin(np.pi * freqs[i] / sr)
        high = signal[i] - low - q * band
        band += f * high
        low += f * band
        output[i] = low
    
    return output.astype(np.float32)


# ============================================================
# SYNTH PATCHES — Pre-built sounds for Deathstep
# ============================================================

class SynthPatch:
    """A complete synth voice: oscillator + filter + envelope."""
    
    def __init__(self, name: str = "Init"):
        self.name = name
        self.osc_type = "saw"  # "sine", "saw", "square", "triangle", "supersow", "wavetable", "fm", "noise"
        self.osc_voices = 1
        self.osc_detune = 0.0
        
        # Filter
        self.filter_type = "lowpass"  # "lowpass", "highpass", "bandpass", "moog"
        self.filter_cutoff = 8000.0
        self.filter_resonance = 0.3
        self.filter_env_amount = 0.0  # How much envelope affects cutoff
        
        # Envelopes
        self.amp_env = ADSR(0.01, 0.1, 0.7, 0.2)
        self.filter_env = ADSR(0.01, 0.3, 0.3, 0.2)
        
        # LFO
        self.lfo_rate = 0.0
        self.lfo_depth = 0.0
        self.lfo_target = "cutoff"  # "cutoff", "pitch", "amp"
        
        # Wavetable specific
        self.wavetable_pos = 0.0
        self.wavetable_morph_start = 0.0
        self.wavetable_morph_end = 0.0
        
        # FM specific
        self.fm_ratio = 2.0
        self.fm_depth = 1.0
    
    def render(self, freq: float, duration: float, velocity: float = 1.0,
               sr: int = SAMPLE_RATE) -> np.ndarray:
        """Render a complete note with this patch."""
        n_samples = int(sr * duration)
        
        # Generate base oscillator
        if self.osc_type == "sine":
            signal = osc_sine(freq, duration, sr=sr)
        elif self.osc_type == "saw":
            if self.osc_voices > 1:
                signal = osc_super_saw(freq, duration, self.osc_voices, self.osc_detune, sr)
            else:
                signal = osc_saw(freq, duration, sr=sr)
        elif self.osc_type == "square":
            signal = osc_square(freq, duration, sr=sr)
        elif self.osc_type == "triangle":
            signal = osc_triangle(freq, duration, sr=sr)
        elif self.osc_type == "supersaw":
            signal = osc_super_saw(freq, duration, self.osc_voices or 7, self.osc_detune or 0.02, sr)
        elif self.osc_type == "wavetable":
            wt = Wavetable()
            signal = wt.generate(freq, duration, self.wavetable_pos,
                                self.wavetable_morph_start, self.wavetable_morph_end, sr)
        elif self.osc_type == "fm":
            signal = fm_synthesis(freq, freq * self.fm_ratio, self.fm_depth, duration, sr)
        elif self.osc_type == "noise":
            signal = osc_noise_white(duration, sr)
        else:
            signal = osc_saw(freq, duration, sr=sr)
        
        # Ensure correct length
        signal = signal[:n_samples]
        if len(signal) < n_samples:
            signal = np.pad(signal, (0, n_samples - len(signal)))
        
        # Apply LFO modulation
        if self.lfo_rate > 0 and self.lfo_depth > 0:
            lfo = LFO(self.lfo_rate, "sine", self.lfo_depth)
            lfo_signal = lfo.generate(duration, sr)
            
            if self.lfo_target == "pitch":
                # Pitch modulation (vibrato) — regenerate with modulated freq
                # Simplified: apply as phase modulation
                t = np.arange(n_samples) / sr
                mod = lfo_signal * freq * 0.1
                signal = signal * np.cos(TWO_PI * np.cumsum(mod) / sr)
            elif self.lfo_target == "amp":
                signal *= (1.0 + lfo_signal) * 0.5
            # cutoff handled in filter section
        
        # Apply filter
        if self.filter_cutoff < sr * 0.49:
            # Filter envelope modulation
            f_env = self.filter_env.generate(duration, sr)
            effective_cutoff = self.filter_cutoff + self.filter_env_amount * f_env.mean() * self.filter_cutoff
            
            # LFO on cutoff (wobble!)
            if self.lfo_target == "cutoff" and self.lfo_rate > 0:
                lfo = LFO(self.lfo_rate, "sine", self.lfo_depth)
                lfo_signal = lfo.generate(duration, sr)
                # Sweep the filter with LFO
                signal = filter_sweep(signal,
                                     effective_cutoff * 0.2,
                                     effective_cutoff,
                                     self.filter_resonance, sr)
            else:
                if self.filter_type == "lowpass":
                    signal = filter_lowpass(signal, effective_cutoff, self.filter_resonance, sr)
                elif self.filter_type == "highpass":
                    signal = filter_highpass(signal, effective_cutoff, self.filter_resonance, sr)
                elif self.filter_type == "bandpass":
                    signal = filter_bandpass(signal, effective_cutoff, self.filter_resonance, sr)
                elif self.filter_type == "moog":
                    signal = filter_moog(signal, effective_cutoff, self.filter_resonance, sr)
        
        # Apply amplitude envelope
        amp_env = self.amp_env.generate(duration, sr)
        signal *= amp_env
        
        # Apply velocity
        signal *= velocity
        
        return signal.astype(np.float32)


# ============================================================
# PRESET PATCHES — Deathstep Sounds
# ============================================================

def patch_sub_bass() -> SynthPatch:
    """Deep sub bass (30-80 Hz sine). Shakes the room."""
    p = SynthPatch("Sub Bass")
    p.osc_type = "sine"
    p.filter_cutoff = 200.0
    p.filter_type = "moog"
    p.filter_resonance = 0.2
    p.amp_env = ADSR(0.005, 0.05, 0.9, 0.3)
    return p


def patch_growl_bass() -> SynthPatch:
    """Deathstep growl bass — wavetable morph with filter wobble."""
    p = SynthPatch("Growl Bass")
    p.osc_type = "wavetable"
    p.wavetable_morph_start = 0.2
    p.wavetable_morph_end = 0.8
    p.filter_type = "moog"
    p.filter_cutoff = 2000.0
    p.filter_resonance = 0.7
    p.lfo_rate = 4.0
    p.lfo_depth = 0.8
    p.lfo_target = "cutoff"
    p.amp_env = ADSR(0.001, 0.05, 0.95, 0.1)
    return p


def patch_screech_bass() -> SynthPatch:
    """High-pitched screeching bass — FM + heavy resonance."""
    p = SynthPatch("Screech Bass")
    p.osc_type = "fm"
    p.fm_ratio = 3.0
    p.fm_depth = 5.0
    p.filter_type = "bandpass"
    p.filter_cutoff = 3000.0
    p.filter_resonance = 0.85
    p.filter_env_amount = 4000.0
    p.amp_env = ADSR(0.001, 0.02, 0.9, 0.05)
    p.filter_env = ADSR(0.001, 0.5, 0.2, 0.1)
    return p


def patch_dark_pad() -> SynthPatch:
    """Dark atmospheric pad — slow attack, detuned."""
    p = SynthPatch("Dark Pad")
    p.osc_type = "supersaw"
    p.osc_voices = 5
    p.osc_detune = 0.01
    p.filter_type = "lowpass"
    p.filter_cutoff = 2000.0
    p.filter_resonance = 0.1
    p.amp_env = ADSR(1.0, 0.5, 0.6, 2.0)
    return p


def patch_digital_lead() -> SynthPatch:
    """Harsh digital lead — square + high resonance."""
    p = SynthPatch("Digital Lead")
    p.osc_type = "square"
    p.filter_type = "lowpass"
    p.filter_cutoff = 5000.0
    p.filter_resonance = 0.6
    p.filter_env_amount = 3000.0
    p.amp_env = ADSR(0.002, 0.1, 0.7, 0.15)
    p.filter_env = ADSR(0.002, 0.2, 0.3, 0.1)
    return p


def patch_metallic_hit() -> SynthPatch:
    """Metallic percussive hit — FM inharmonic."""
    p = SynthPatch("Metallic Hit")
    p.osc_type = "fm"
    p.fm_ratio = 2.76
    p.fm_depth = 8.0
    p.filter_type = "highpass"
    p.filter_cutoff = 500.0
    p.filter_resonance = 0.3
    p.amp_env = ADSR(0.001, 0.3, 0.0, 0.1)
    return p


def patch_riser() -> SynthPatch:
    """Noise riser — filtered noise sweep for buildups."""
    p = SynthPatch("Riser")
    p.osc_type = "noise"
    p.filter_type = "bandpass"
    p.filter_cutoff = 1000.0
    p.filter_resonance = 0.8
    p.filter_env_amount = 8000.0
    p.amp_env = ADSR(2.0, 0.0, 1.0, 0.5)
    p.filter_env = ADSR(4.0, 0.0, 1.0, 0.1)
    return p


# All preset patches
DEATHSTEP_PATCHES = {
    "sub_bass": patch_sub_bass(),
    "growl_bass": patch_growl_bass(),
    "screech_bass": patch_screech_bass(),
    "dark_pad": patch_dark_pad(),
    "digital_lead": patch_digital_lead(),
    "metallic_hit": patch_metallic_hit(),
    "riser": patch_riser(),
}

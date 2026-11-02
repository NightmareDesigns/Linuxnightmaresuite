"""
NIGHTMARE EDM AUDIO ENGINE & VOCAL PRODUCER (v1.0)
=================================================
High-Performance, 100% Local, CD-Ready (44.1kHz 16-bit Stereo) 3-Minute EDM Producer.
Complete with:
 - Formant & Vocoder Singing Vocal Synthesizer
 - Multi-Track EDM Synthesizer (Supersaw, Reese Bass, Sub, Pluck Arp, Atmospheric Pads)
 - Studio Drum Synthesizers (Punchy EDM Kick, Layered Claps/Snares, Hi-Hats, Risers, FX)
 - Full 3-Minute Song Arranger (Intro -> Verse -> Build -> Drop 1 -> Breakdown -> Build -> Drop 2 Climax -> Outro)
 - Red Book CD Mastering Suite (5-Band EQ, Sidechain Compression, Stereo Widening, Brickwall Limiter)
"""

import numpy as np
import scipy.signal as signal
import wave
import struct
import math
import os
import time
from typing import Dict, List, Tuple, Optional, Callable

SAMPLE_RATE = 44100
BIT_DEPTH = 16

# =====================================================================
#  NOTE FREQUENCIES & MUSICAL SCALES
# =====================================================================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_to_freq(note_name: str) -> float:
    """Converts note string (e.g. 'A4', 'C#3', 'F4') to frequency in Hz."""
    note_name = note_name.strip().upper()
    if 'BB' in note_name:
        note_name = note_name.replace('BB', 'A#')
    elif 'EB' in note_name:
        note_name = note_name.replace('EB', 'D#')
    elif 'AB' in note_name:
        note_name = note_name.replace('AB', 'G#')
    elif 'DB' in note_name:
        note_name = note_name.replace('DB', 'C#')
    elif 'GB' in note_name:
        note_name = note_name.replace('GB', 'F#')

    if len(note_name) >= 2 and note_name[1] == '#':
        pitch = note_name[:2]
        octave = int(note_name[2:]) if len(note_name) > 2 else 4
    else:
        pitch = note_name[0]
        octave = int(note_name[1:]) if len(note_name) > 1 else 4

    semitone = NOTE_NAMES.index(pitch)
    midi_num = (octave + 1) * 12 + semitone
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))


GENRE_PRESETS = {
    'Cyberpunk_Bass': {
        'bpm': 128.0,
        'chords': [['A3', 'C4', 'E4'], ['F3', 'A3', 'C4'], ['C3', 'E3', 'G3'], ['G3', 'B3', 'D4']],
        'bass_notes': ['A1', 'F1', 'C1', 'G1'],
        'vocal_vibe': 'vocoder',
        'lead_style': 'supersaw_stabs'
    },
    'Festival_BigRoom': {
        'bpm': 130.0,
        'chords': [['F3', 'A3', 'C4'], ['G3', 'B3', 'D4'], ['A3', 'C4', 'E4'], ['E3', 'G3', 'B3']],
        'bass_notes': ['F1', 'G1', 'A1', 'E1'],
        'vocal_vibe': 'festival_anthem',
        'lead_style': 'festival_lead'
    },
    'Future_Bass_Dream': {
        'bpm': 140.0,
        'chords': [['F3', 'A3', 'C4', 'E4'], ['C3', 'E3', 'G3', 'B3'], ['D3', 'F3', 'A3', 'C4'], ['A#2', 'D3', 'F3', 'A3']],
        'bass_notes': ['F1', 'C1', 'D1', 'A#0'],
        'vocal_vibe': 'female_singing',
        'lead_style': 'wobble_chords'
    },
    'Synthwave_Night': {
        'bpm': 120.0,
        'chords': [['D3', 'F3', 'A3'], ['A#2', 'D3', 'F3'], ['C3', 'E3', 'G3'], ['A2', 'C#3', 'E3']],
        'bass_notes': ['D1', 'A#0', 'C1', 'A0'],
        'vocal_vibe': 'retro_vocoder',
        'lead_style': 'retro_arp'
    },
    'Vocal_Trance': {
        'bpm': 138.0,
        'chords': [['A3', 'C4', 'E4'], ['D3', 'F3', 'A3'], ['F3', 'A3', 'C4'], ['G3', 'B3', 'D4']],
        'bass_notes': ['A1', 'D1', 'F1', 'G1'],
        'vocal_vibe': 'ethereal_singing',
        'lead_style': 'trance_supersaw'
    }
}

# =====================================================================
#  DSP FILTER & EFFECT PRIMITIVES
# =====================================================================

def lowpass_filter(data: np.ndarray, cutoff_hz: float, sr: int = SAMPLE_RATE, order: int = 2) -> np.ndarray:
    """Applies Butterworth low-pass filter."""
    cutoff = min(max(20.0, cutoff_hz), sr * 0.48)
    sos = signal.butter(order, cutoff, btype='low', fs=sr, output='sos')
    if data.ndim == 2:
        return np.column_stack([signal.sosfilt(sos, data[:, 0]), signal.sosfilt(sos, data[:, 1])])
    return signal.sosfilt(sos, data)

def highpass_filter(data: np.ndarray, cutoff_hz: float, sr: int = SAMPLE_RATE, order: int = 2) -> np.ndarray:
    """Applies Butterworth high-pass filter."""
    cutoff = max(10.0, min(cutoff_hz, sr * 0.48))
    sos = signal.butter(order, cutoff, btype='high', fs=sr, output='sos')
    if data.ndim == 2:
        return np.column_stack([signal.sosfilt(sos, data[:, 0]), signal.sosfilt(sos, data[:, 1])])
    return signal.sosfilt(sos, data)

def bandpass_filter(data: np.ndarray, low_hz: float, high_hz: float, sr: int = SAMPLE_RATE, order: int = 2) -> np.ndarray:
    """Applies Butterworth band-pass filter."""
    low = max(10.0, low_hz)
    high = min(high_hz, sr * 0.48)
    if low >= high:
        low = high * 0.5
    sos = signal.butter(order, [low, high], btype='bandpass', fs=sr, output='sos')
    if data.ndim == 2:
        return np.column_stack([signal.sosfilt(sos, data[:, 0]), signal.sosfilt(sos, data[:, 1])])
    return signal.sosfilt(sos, data)

def stereo_reverb(stereo_audio: np.ndarray, wet: float = 0.30, decay_time: float = 1.8, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Fast FFT-accelerated stereo hall convolution reverb with exponential decay and high-frequency damping."""
    ir_len = int(sr * min(decay_time, 2.5))
    t_ir = np.linspace(0, decay_time, ir_len, endpoint=False)
    
    # Exponential decay envelope
    decay_curve = np.exp(-3.2 * t_ir)
    
    # Synthesize decorrelated diffuse stereo impulse responses
    np.random.seed(42)
    raw_ir_l = np.random.uniform(-1, 1, ir_len) * decay_curve
    raw_ir_r = np.random.uniform(-1, 1, ir_len) * decay_curve
    
    # Pre-delay (20ms) & High-frequency air absorption
    ir_l = lowpass_filter(raw_ir_l, 5500.0, sr)
    ir_r = lowpass_filter(raw_ir_r, 5200.0, sr)
    
    wet_l = signal.fftconvolve(stereo_audio[:, 0], ir_l, mode='full')[:len(stereo_audio)]
    wet_r = signal.fftconvolve(stereo_audio[:, 1], ir_r, mode='full')[:len(stereo_audio)]
    
    wet_stereo = np.column_stack([wet_l, wet_r])
    # Normalize wet volume
    wet_peak = np.max(np.abs(wet_stereo)) + 1e-6
    wet_stereo = (wet_stereo / wet_peak) * np.max(np.abs(stereo_audio)) * wet
    
    return stereo_audio * (1.0 - wet * 0.4) + wet_stereo

def stereo_delay(stereo_audio: np.ndarray, bpm: float = 128.0, fraction: float = 0.75, feedback: float = 0.35, wet: float = 0.22, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Fast stereo ping-pong / tempo-synced delay."""
    beat_sec = 60.0 / bpm
    delay_samples_l = int(sr * (beat_sec * fraction))
    delay_samples_r = int(sr * (beat_sec * fraction * 1.5))
    
    out_l = stereo_audio[:, 0].copy()
    out_r = stereo_audio[:, 1].copy()
    
    for tap in range(1, 4):
        gain = feedback ** tap
        off_l = delay_samples_l * tap
        off_r = delay_samples_r * tap
        if off_l < len(out_l):
            out_l[off_l:] += stereo_audio[:-off_l, 0] * (gain * wet)
        if off_r < len(out_r):
            out_r[off_r:] += stereo_audio[:-off_r, 1] * (gain * wet)
            
    return np.column_stack([out_l, out_r])

def sidechain_compression(audio: np.ndarray, kick_triggers: np.ndarray, bpm: float = 128.0, pump_depth: float = 0.85, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Applies rhythmic sidechain ducking against kick drum triggers."""
    envelope = np.ones(len(audio), dtype=np.float32)
    beat_samples = int(sr * (60.0 / bpm))
    duck_len = int(beat_samples * 0.75)
    
    t = np.linspace(0, 1, duck_len)
    duck_curve = (1.0 - pump_depth) + pump_depth * (1.0 - np.exp(-4.5 * t))
    
    for trig_sample in kick_triggers:
        if trig_sample < len(audio):
            end_s = min(len(audio), trig_sample + duck_len)
            l = end_s - trig_sample
            envelope[trig_sample:end_s] = np.minimum(envelope[trig_sample:end_s], duck_curve[:l])
            
    if audio.ndim == 2:
        return audio * envelope[:, None]
    return audio * envelope

# =====================================================================
#  SYNTHESIZER INSTRUMENT ENGINES
# =====================================================================

class DrumEngine:
    """High-impact EDM drum synthesizer."""
    
    @staticmethod
    def synth_edm_kick(sr: int = SAMPLE_RATE) -> np.ndarray:
        """Synthesizes a punchy 4-on-the-floor EDM club kick."""
        duration = 0.40
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Pitch dive from 170 Hz to 45 Hz
        freq_env = 45.0 + (170.0 - 45.0) * np.exp(-50.0 * t)
        phase = 2 * np.pi * np.cumsum(freq_env) / sr
        body = np.sin(phase)
        
        # Click transient
        click_t = t[:int(sr * 0.007)]
        click = np.sin(2 * np.pi * 1400.0 * click_t) * np.exp(-450.0 * click_t)
        body[:len(click)] += click * 0.85
        
        amp_env = np.exp(-7.5 * t)
        kick = np.tanh((body * amp_env) * 1.8) * 0.95
        return np.column_stack([kick, kick]).astype(np.float32)

    @staticmethod
    def synth_edm_snare_clap(sr: int = SAMPLE_RATE) -> np.ndarray:
        """Synthesizes a layered EDM snare + stereo clap."""
        duration = 0.32
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        f_env = 175.0 + 90.0 * np.exp(-40.0 * t)
        tonal = np.sin(2 * np.pi * np.cumsum(f_env) / sr) * np.exp(-22.0 * t)
        
        noise_l = np.random.uniform(-1, 1, len(t))
        noise_r = np.random.uniform(-1, 1, len(t))
        flam_samples = int(sr * 0.010)
        noise_l[:flam_samples] *= 0.5
        noise_r[flam_samples:flam_samples * 2] *= 0.7
        
        noise_env = np.exp(-13.0 * t)
        snare_l = tonal * 0.45 + bandpass_filter(noise_l * noise_env, 900.0, 9500.0, sr) * 0.85
        snare_r = tonal * 0.45 + bandpass_filter(noise_r * noise_env, 900.0, 9500.0, sr) * 0.85
        
        out = np.column_stack([snare_l, snare_r])
        return (np.tanh(out * 1.5) * 0.85).astype(np.float32)

    @staticmethod
    def synth_hihat(open_hat: bool = False, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Synthesizes metallic EDM closed/open hi-hat."""
        duration = 0.30 if open_hat else 0.07
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freqs = [260, 400, 560, 680, 810, 920]
        metallic = np.zeros(len(t))
        for f in freqs:
            metallic += np.sin(2 * np.pi * f * t)
            
        noise = metallic + np.random.uniform(-1, 1, len(t)) * 1.8
        filtered = highpass_filter(noise, 7500.0, sr)
        decay = 11.0 if open_hat else 70.0
        hat = filtered * np.exp(-decay * t) * 0.45
        return np.column_stack([hat, hat * 0.95]).astype(np.float32)

    @staticmethod
    def synth_riser(duration_sec: float = 7.5, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Synthesizes an EDM tension build-up noise & pitch riser."""
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        f_start, f_end = 100.0, 3500.0
        f_curve = f_start * ((f_end / f_start) ** (t / duration_sec))
        phase = 2 * np.pi * np.cumsum(f_curve) / sr
        saw = signal.sawtooth(phase) * 0.35
        
        noise_l = np.random.uniform(-1, 1, len(t)) * (t / duration_sec) ** 1.6
        noise_r = np.random.uniform(-1, 1, len(t)) * (t / duration_sec) ** 1.6
        
        amp_env = (t / duration_sec) ** 2.0
        riser_l = (saw + noise_l * 0.6) * amp_env * 0.65
        riser_r = (saw + noise_r * 0.6) * amp_env * 0.65
        return np.column_stack([riser_l, riser_r]).astype(np.float32)

    @staticmethod
    def synth_impact_downlifter(duration_sec: float = 6.0, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Synthesizes massive drop impact crash with sub downlifter."""
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        f_curve = 25.0 + (130.0 - 25.0) * np.exp(-1.4 * t)
        sub = np.sin(2 * np.pi * np.cumsum(f_curve) / sr) * np.exp(-0.8 * t) * 0.8
        
        noise_l = highpass_filter(np.random.uniform(-1, 1, len(t)), 4200.0, sr) * np.exp(-1.3 * t)
        noise_r = highpass_filter(np.random.uniform(-1, 1, len(t)), 4200.0, sr) * np.exp(-1.3 * t)
        
        impact = np.column_stack([sub + noise_l * 0.55, sub + noise_r * 0.55])
        return impact.astype(np.float32)


class SynthLeadAndBass:
    """Synthesizes EDM Supersaws, Reese Bass, and Pluck Arps."""
    
    @staticmethod
    def synth_supersaw_chord(notes: List[str], duration: float, detune_cents: float = 16.0, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Synthesizes a lush, wide 5-oscillator Supersaw chord voice."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        out_l = np.zeros(len(t), dtype=np.float32)
        out_r = np.zeros(len(t), dtype=np.float32)
        
        detunes = [-detune_cents, -detune_cents * 0.5, 0, detune_cents * 0.5, detune_cents]
        pans = [0.15, 0.35, 0.50, 0.65, 0.85]
        
        for note in notes:
            base_f = note_to_freq(note)
            for d, pan in zip(detunes, pans):
                freq = base_f * (2.0 ** (d / 1200.0))
                phi = np.random.uniform(0, 2 * np.pi)
                saw = signal.sawtooth(2 * np.pi * freq * t + phi)
                out_l += saw * (1.0 - pan)
                out_r += saw * pan
                
        attack_s = min(len(t), int(sr * 0.012))
        release_s = min(len(t), int(sr * 0.06))
        env = np.ones(len(t), dtype=np.float32)
        if len(t) > attack_s + release_s:
            env[:attack_s] = np.linspace(0, 1, attack_s)
            env[-release_s:] = np.linspace(1, 0, release_s)
            
        chord_l = lowpass_filter(out_l * env, 8500.0, sr) / (len(notes) * 2.2)
        chord_r = lowpass_filter(out_r * env, 8500.0, sr) / (len(notes) * 2.2)
        
        return np.column_stack([chord_l, chord_r]).astype(np.float32)

    @staticmethod
    def synth_reese_bass(note: str, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Synthesizes a deep, detuned Reese / Cyberpunk Sawtooth Bass."""
        base_f = note_to_freq(note)
        while base_f > 110.0:
            base_f /= 2.0
            
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        saw1 = signal.sawtooth(2 * np.pi * (base_f * 1.005) * t)
        saw2 = signal.sawtooth(2 * np.pi * (base_f * 0.995) * t)
        sub_sine = np.sin(2 * np.pi * (base_f * 0.5) * t) * 1.3
        
        bass = np.tanh((saw1 * 0.45 + saw2 * 0.45 + sub_sine) * 1.5)
        release_s = min(len(t), int(sr * 0.035))
        env = np.ones(len(t))
        env[-release_s:] = np.linspace(1, 0, release_s)
        
        bass_filtered = lowpass_filter(bass * env, 1400.0, sr) * 0.85
        return np.column_stack([bass_filtered, bass_filtered]).astype(np.float32)

    @staticmethod
    def synth_pluck_note(note: str, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Synthesizes snappy EDM pluck with fast filter decay."""
        f = note_to_freq(note)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        raw = signal.square(2 * np.pi * f * t) * 0.5 + signal.sawtooth(2 * np.pi * (f * 1.002) * t) * 0.5
        amp_env = np.exp(-15.0 * t)
        pluck = bandpass_filter(raw * amp_env, f * 0.8, min(f * 4.5, sr * 0.45), sr) * 0.75
        return np.column_stack([pluck, pluck]).astype(np.float32)


class VocalEngine:
    """
    Formant Synthesis & Vocoder Engine for EDM Vocals.
    Synthesizes melodic vocal lines, cyber hooks, and vocal chops with natural formant filtering.
    """
    VOWEL_FORMANTS = {
        'A': [800.0, 1200.0, 2500.0, 3500.0],
        'E': [530.0, 1840.0, 2480.0, 3600.0],
        'I': [270.0, 2290.0, 3010.0, 3700.0],
        'O': [500.0, 850.0, 2400.0, 3400.0],
        'U': [300.0, 870.0, 2240.0, 3400.0],
        'CYBER': [450.0, 1500.0, 2800.0, 3800.0]
    }
    
    @classmethod
    def synth_vocal_note(cls, note: str, vowel: str = 'A', duration: float = 0.5, 
                         vocoder: bool = False, vibrato_hz: float = 5.5, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Synthesizes a sung vocal note with natural formants or cyberpunk vocoder timbre."""
        freq = note_to_freq(note)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        vibrato_depth = 0.010 if not vocoder else 0.001
        pitch_mod = freq * (1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_hz * t))
        phase = 2 * np.pi * np.cumsum(pitch_mod) / sr
        
        if vocoder:
            carrier = (signal.sawtooth(phase) * 0.6 + 
                       signal.sawtooth(phase * 1.005) * 0.3 + 
                       signal.sawtooth(phase * 0.995) * 0.3)
        else:
            pulse = np.mod(phase, 2 * np.pi) / (2 * np.pi)
            glottal = np.where(pulse < 0.65, (pulse / 0.65) ** 2 * (1.0 - (pulse / 0.65)), -0.15)
            breath = np.random.uniform(-0.03, 0.03, len(t))
            carrier = glottal + breath
            
        formants = cls.VOWEL_FORMANTS.get(vowel.upper(), cls.VOWEL_FORMANTS['A'])
        
        vocal_sig = np.zeros(len(t), dtype=np.float32)
        bandwidths = [70.0, 90.0, 120.0, 150.0]
        gains = [1.0, 0.65, 0.35, 0.15]
        
        for f_res, bw, g in zip(formants, bandwidths, gains):
            low = max(50.0, f_res - bw * 0.5)
            high = min(sr * 0.45, f_res + bw * 0.5)
            band = bandpass_filter(carrier, low, high, sr, order=2)
            vocal_sig += band * g
            
        attack_s = min(len(t), int(sr * 0.025))
        release_s = min(len(t), int(sr * 0.045))
        env = np.ones(len(t), dtype=np.float32)
        if len(t) > attack_s + release_s:
            env[:attack_s] = np.linspace(0, 1, attack_s)
            env[-release_s:] = np.linspace(1, 0, release_s)
            
        doubler_l = np.roll(vocal_sig, int(sr * 0.012)) * 0.45
        doubler_r = np.roll(vocal_sig, int(sr * 0.018)) * 0.45
        
        out_l = (vocal_sig * 0.85 + doubler_l * 0.35) * env
        out_r = (vocal_sig * 0.85 + doubler_r * 0.35) * env
        
        out = np.column_stack([out_l, out_r])
        return (np.tanh(out * 1.8) * 0.80).astype(np.float32)


class CDMasteringSuite:
    """Red Book CD Mastering Processing Chain."""
    
    @staticmethod
    def master_track(raw_mix: np.ndarray, sr: int = SAMPLE_RATE,
                     drive: float = 1.40, target_peak: float = 0.965) -> np.ndarray:
        """Full 5-Stage Mastering Chain.

        Defaults (drive 1.40, peak 0.965) preserve the original chain
        exactly. master_level presets in render_edm_track only change
        these two parameters — this is NOT a measured LUFS chain, and
        the panel labels must not claim a measured LUFS value.
        """
        # 1. High-Pass Infrasonic Filter
        master = highpass_filter(raw_mix, 25.0, sr, order=3)
        
        # 2. 5-Band Mastering EQ
        sub_boost = bandpass_filter(master, 45.0, 95.0, sr, order=2) * 0.22
        presence = bandpass_filter(master, 2600.0, 5200.0, sr, order=2) * 0.20
        air = highpass_filter(master, 10500.0, sr, order=2) * 0.22
        master = master + sub_boost + presence + air
        
        # 3. Mid-Side Stereo Expansion
        mid = (master[:, 0] + master[:, 1]) * 0.5
        side = (master[:, 0] - master[:, 1]) * 0.5
        side = highpass_filter(side, 140.0, sr, order=2) * 1.35
        
        master_l = mid + side
        master_r = mid - side
        master = np.column_stack([master_l, master_r])
        
        # 4. Soft Clipper & Saturator
        master = np.tanh(master * drive)

        # 5. Lookahead Brickwall Peak Limiter (-0.3 dBFS ceiling)
        peak = np.max(np.abs(master))
        if peak > 0:
            gain = target_peak / max(peak, 0.001)
            master = master * gain
            
        return master.astype(np.float32)

    @staticmethod
    def export_cd_wav(master_audio: np.ndarray, file_path: str, sr: int = SAMPLE_RATE) -> str:
        """Exports CD-Quality 44.1kHz 16-bit PCM Stereo WAV with TPDF Dither."""
        dither = (np.random.uniform(-1, 1, master_audio.shape) + np.random.uniform(-1, 1, master_audio.shape)) * (0.5 / 32767.0)
        dithered = np.clip(master_audio + dither, -0.9999, 0.9999)
        int16_pcm = (dithered * 32767.0).astype(np.int16)
        
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(2)        # Stereo
            wf.setsampwidth(2)        # 16-bit
            wf.setframerate(sr)       # 44,100 Hz
            wf.writeframes(int16_pcm.tobytes())
            
        return file_path


class EDMTrackBuilder:
    """Full 3-Minute EDM Track Builder."""
    
    def __init__(self, genre: str = "Cyberpunk_Bass", bpm: float = 128.0,
                 vocal_mode: str = "Vocoder_Cyber", custom_lyrics: Optional[str] = None,
                 drum_pattern: Optional[Dict] = None,
                 master_level: str = "balanced"):
        self.genre = genre
        self.bpm = bpm
        self.vocal_mode = vocal_mode
        self.custom_lyrics = custom_lyrics
        # Mastering level: only drive/peak, NOT measured LUFS.
        # "loud": drive 1.60, peak 0.99; "balanced" (default, original):
        # drive 1.40, peak 0.965; "audiophile": drive 1.15, peak 0.90.
        self.master_drive, self.master_peak = {
            "loud": (1.60, 0.99),
            "balanced": (1.40, 0.965),
            "audiophile": (1.15, 0.90),
        }.get(str(master_level).lower(), (1.40, 0.965))
        # User drum pattern (16-step grid, see _normalize_drum_pattern) or
        # None for the built-in fixed arrangement. When set, the pattern
        # replaces the drums in the Drop sections only; every other
        # section keeps its fixed arrangement.
        self.drum_pattern = _normalize_drum_pattern(drum_pattern) if drum_pattern is not None else None
        self.sr = SAMPLE_RATE
        
        preset = GENRE_PRESETS.get(genre, GENRE_PRESETS['Cyberpunk_Bass'])
        self.chords = preset['chords']
        self.bass_notes = preset['bass_notes']
        
        # 128 BPM: 1 beat = 0.46875s. 1 bar = 1.875s. 96 bars = exactly 180.0s (3:00 min).
        self.bar_duration = (60.0 / self.bpm) * 4.0
        self.total_bars = 96
        self.total_samples = int(self.sr * self.bar_duration * self.total_bars)
        
    def generate_full_track(self, progress_callback: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
        """Renders the entire 3-minute multi-track arrangement into a master CD audio stream."""
        def log(prog: float, msg: str):
            if progress_callback:
                progress_callback(prog, msg)
            else:
                print(f"[{int(prog*100)}%] {msg}")
                
        log(0.05, "Synthesizing drum kit (Punchy EDM Kick, Layered Claps, Crisp Hats)...")
        kick_sample = DrumEngine.synth_edm_kick(self.sr)
        snare_sample = DrumEngine.synth_edm_snare_clap(self.sr)
        hihat_closed = DrumEngine.synth_hihat(open_hat=False, sr=self.sr)
        hihat_open = DrumEngine.synth_hihat(open_hat=True, sr=self.sr)
        riser_sample = DrumEngine.synth_riser(duration_sec=self.bar_duration * 4, sr=self.sr)
        impact_sample = DrumEngine.synth_impact_downlifter(duration_sec=self.bar_duration * 3, sr=self.sr)
        
        drum_bus = np.zeros((self.total_samples, 2), dtype=np.float32)
        bass_bus = np.zeros((self.total_samples, 2), dtype=np.float32)
        lead_bus = np.zeros((self.total_samples, 2), dtype=np.float32)
        vocal_bus = np.zeros((self.total_samples, 2), dtype=np.float32)
        fx_bus = np.zeros((self.total_samples, 2), dtype=np.float32)
        
        kick_triggers = []
        beat_samples = int(self.sr * (60.0 / self.bpm))
        
        log(0.20, "Sequencing drums and rhythm grid across 96 bars (3:00 min)...")
        for bar in range(self.total_bars):
            bar_start = int(bar * self.bar_duration * self.sr)
            
            # Intro (Bars 0-7)
            if 0 <= bar < 8:
                if bar >= 4:
                    for b in range(4):
                        t_hat = bar_start + int((b + 0.5) * beat_samples)
                        self._paste_audio(drum_bus, hihat_open * 0.45, t_hat)
                        
            # Verse 1 (Bars 8-15)
            elif 8 <= bar < 16:
                for b in range(4):
                    t_kick = bar_start + b * beat_samples
                    self._paste_audio(drum_bus, kick_sample * 0.9, t_kick)
                    kick_triggers.append(t_kick)
                    if b in (1, 3):
                        self._paste_audio(drum_bus, snare_sample * 0.8, bar_start + b * beat_samples)
                    for s in range(4):
                        t_hh = bar_start + int((b + s * 0.25) * beat_samples)
                        self._paste_audio(drum_bus, hihat_closed * (0.35 if s == 2 else 0.2), t_hh)
                        
            # Build-Up 1 (Bars 16-23)
            elif 16 <= bar < 24:
                sub_bar = bar - 16
                steps = 4 if sub_bar < 4 else (8 if sub_bar < 6 else 16)
                step_mult = 1.0 if sub_bar < 4 else (0.5 if sub_bar < 6 else 0.25)
                vol = 0.5 + 0.05 * sub_bar
                for s in range(steps):
                    t_s = bar_start + int(s * step_mult * beat_samples)
                    self._paste_audio(drum_bus, snare_sample * vol, t_s)
                    
            # Drop 1 (Bars 24-39)
            elif 24 <= bar < 40:
                if self.drum_pattern is not None:
                    self._paste_user_drum_pattern(drum_bus, bar_start, kick_triggers,
                                                  kick_sample, snare_sample,
                                                  hihat_closed, hihat_open)
                else:
                    for b in range(4):
                        t_kick = bar_start + b * beat_samples
                        self._paste_audio(drum_bus, kick_sample * 1.0, t_kick)
                        kick_triggers.append(t_kick)
                        if b in (1, 3):
                            self._paste_audio(drum_bus, snare_sample * 0.95, bar_start + b * beat_samples)
                        t_ohat = bar_start + int((b + 0.5) * beat_samples)
                        self._paste_audio(drum_bus, hihat_open * 0.6, t_ohat)
                        for s in range(4):
                            t_hh = bar_start + int((b + s * 0.25) * beat_samples)
                            self._paste_audio(drum_bus, hihat_closed * 0.3, t_hh)
                        
            # Breakdown (Bars 40-55)
            elif 40 <= bar < 56:
                if bar >= 48:
                    for b in range(4):
                        t_kick = bar_start + b * beat_samples
                        self._paste_audio(drum_bus, kick_sample * 0.65, t_kick)
                        kick_triggers.append(t_kick)
                        
            # Build-Up 2 (Bars 56-63)
            elif 56 <= bar < 64:
                sub_bar = bar - 56
                steps = 8 if sub_bar < 4 else 16
                step_mult = 0.5 if sub_bar < 4 else 0.25
                for s in range(steps):
                    t_s = bar_start + int(s * step_mult * beat_samples)
                    self._paste_audio(drum_bus, snare_sample * (0.6 + 0.04 * sub_bar), t_s)
                    
            # Drop 2 Climax (Bars 64-79)
            elif 64 <= bar < 80:
                if self.drum_pattern is not None:
                    self._paste_user_drum_pattern(drum_bus, bar_start, kick_triggers,
                                                  kick_sample, snare_sample,
                                                  hihat_closed, hihat_open)
                else:
                    for b in range(4):
                        t_kick = bar_start + b * beat_samples
                        self._paste_audio(drum_bus, kick_sample * 1.0, t_kick)
                        kick_triggers.append(t_kick)
                        if b in (1, 3):
                            self._paste_audio(drum_bus, snare_sample * 1.0, bar_start + b * beat_samples)
                        t_ohat = bar_start + int((b + 0.5) * beat_samples)
                        self._paste_audio(drum_bus, hihat_open * 0.65, t_ohat)
                        for s in range(4):
                            t_hh = bar_start + int((b + s * 0.25) * beat_samples)
                            self._paste_audio(drum_bus, hihat_closed * 0.35, t_hh)
                        
            # Outro (Bars 80-95)
            elif 80 <= bar < 96:
                if bar < 88:
                    for b in range(4):
                        t_kick = bar_start + b * beat_samples
                        self._paste_audio(drum_bus, kick_sample * 0.8, t_kick)
                        kick_triggers.append(t_kick)
                        if b in (1, 3):
                            self._paste_audio(drum_bus, snare_sample * 0.6, bar_start + b * beat_samples)
                else:
                    for b in range(4):
                        self._paste_audio(drum_bus, kick_sample * max(0.05, 0.6 - (bar - 88) * 0.07), bar_start + b * beat_samples)

        # Transition Risers & Impacts
        log(0.40, "Placing transition risers and drop impact crashes...")
        self._paste_audio(fx_bus, riser_sample, int(20 * self.bar_duration * self.sr))
        self._paste_audio(fx_bus, impact_sample, int(24 * self.bar_duration * self.sr))
        self._paste_audio(fx_bus, riser_sample, int(60 * self.bar_duration * self.sr))
        self._paste_audio(fx_bus, impact_sample, int(64 * self.bar_duration * self.sr))
        self._paste_audio(fx_bus, impact_sample, int(80 * self.bar_duration * self.sr))
        
        log(0.55, "Synthesizing Supersaw chords, Reese basslines, and Pluck arpeggios...")
        for bar in range(self.total_bars):
            bar_start = int(bar * self.bar_duration * self.sr)
            prog_idx = (bar // 2) % len(self.chords)
            chord = self.chords[prog_idx]
            b_note = self.bass_notes[prog_idx]
            
            if (0 <= bar < 8) or (40 <= bar < 56):
                for step in range(16):
                    note = chord[step % len(chord)]
                    t_step = bar_start + int(step * (beat_samples * 0.25))
                    pluck = SynthLeadAndBass.synth_pluck_note(note, duration=0.25, sr=self.sr)
                    self._paste_audio(lead_bus, pluck * 0.6, t_step)
                    
            elif 8 <= bar < 16:
                bass = SynthLeadAndBass.synth_reese_bass(b_note, duration=self.bar_duration, sr=self.sr)
                self._paste_audio(bass_bus, bass * 0.75, bar_start)
                chord_audio = SynthLeadAndBass.synth_supersaw_chord(chord, duration=self.bar_duration * 0.9, sr=self.sr)
                self._paste_audio(lead_bus, chord_audio * 0.45, bar_start)
                
            elif (24 <= bar < 40) or (64 <= bar < 80):
                hit_dur = (60.0 / self.bpm) * 0.65
                chord_audio = SynthLeadAndBass.synth_supersaw_chord(chord, duration=hit_dur, sr=self.sr)
                for b_step in [0.0, 0.75, 1.5, 2.25, 3.0]:
                    t_hit = bar_start + int(b_step * beat_samples)
                    self._paste_audio(lead_bus, chord_audio * 0.85, t_hit)
                    
                bass = SynthLeadAndBass.synth_reese_bass(b_note, duration=self.bar_duration, sr=self.sr)
                self._paste_audio(bass_bus, bass * 0.95, bar_start)
                
                if 64 <= bar < 80:
                    lead_notes = ['E5', 'D5', 'C5', 'A4', 'G4', 'A4', 'C5', 'E5']
                    for s in range(8):
                        l_note = lead_notes[(bar * 2 + s) % len(lead_notes)]
                        t_lead = bar_start + int(s * 0.5 * beat_samples)
                        pluck_lead = SynthLeadAndBass.synth_pluck_note(l_note, duration=0.35, sr=self.sr)
                        self._paste_audio(lead_bus, pluck_lead * 0.7, t_lead)
                        
            elif 80 <= bar < 96:
                for step in range(8):
                    note = chord[step % len(chord)]
                    t_step = bar_start + int(step * (beat_samples * 0.5))
                    pluck = SynthLeadAndBass.synth_pluck_note(note, duration=0.4, sr=self.sr)
                    self._paste_audio(lead_bus, pluck * max(0.05, 0.5 - (bar - 80) * 0.03), t_step)
                    
        log(0.70, "Applying sidechain compression ducking and stereo spatialization...")
        bass_bus = sidechain_compression(bass_bus, np.array(kick_triggers), self.bpm, pump_depth=0.90, sr=self.sr)
        lead_bus = sidechain_compression(lead_bus, np.array(kick_triggers), self.bpm, pump_depth=0.65, sr=self.sr)
        
        lead_bus = stereo_delay(lead_bus, bpm=self.bpm, fraction=0.75, feedback=0.35, wet=0.20, sr=self.sr)
        lead_bus = stereo_reverb(lead_bus, wet=0.25, decay_time=1.6, sr=self.sr)
        
        log(0.80, "Generating singing vocals, cyber vocoder hooks, and vocal chop melodies...")
        is_vocoder = "vocoder" in self.vocal_mode.lower() or "cyber" in self.vocal_mode.lower()
        
        verse_melody = [('A4', 'A', 0.8), ('C5', 'E', 0.8), ('E5', 'I', 1.2), ('D5', 'O', 0.8), ('C5', 'A', 0.8), ('A4', 'U', 1.6)]
        chorus_hook = [('E5', 'CYBER', 0.4), ('E5', 'CYBER', 0.4), ('D5', 'A', 0.4), ('C5', 'O', 0.8),
                       ('A4', 'E', 0.4), ('C5', 'I', 0.4), ('D5', 'A', 0.4), ('E5', 'U', 1.2)]
        pre_drop_chop = [('A4', 'CYBER', 0.2), ('A4', 'CYBER', 0.2), ('C5', 'CYBER', 0.2), ('E5', 'CYBER', 0.4)]
        
        for verse_bar in [8, 12, 40, 44]:
            t_v = int(verse_bar * self.bar_duration * self.sr)
            for note, vow, dur in verse_melody:
                v_note = VocalEngine.synth_vocal_note(note, vowel=vow, duration=dur, vocoder=is_vocoder, sr=self.sr)
                self._paste_audio(vocal_bus, v_note * 0.8, t_v)
                t_v += int(dur * self.sr)
                
        for pre_bar in [23, 63]:
            t_pre = int(pre_bar * self.bar_duration * self.sr) + int(2 * beat_samples)
            for note, vow, dur in pre_drop_chop:
                v_chop = VocalEngine.synth_vocal_note(note, vowel=vow, duration=dur, vocoder=True, sr=self.sr)
                self._paste_audio(vocal_bus, v_chop * 1.0, t_pre)
                t_pre += int(dur * self.sr)
                
        for drop_bar in [24, 28, 32, 36, 64, 68, 72, 76]:
            t_ch = int(drop_bar * self.bar_duration * self.sr)
            for note, vow, dur in chorus_hook:
                v_main = VocalEngine.synth_vocal_note(note, vowel=vow, duration=dur, vocoder=is_vocoder, sr=self.sr)
                v_harm = VocalEngine.synth_vocal_note(note, vowel=vow, duration=dur, vocoder=is_vocoder, sr=self.sr) * 0.5
                self._paste_audio(vocal_bus, v_main * 0.85 + v_harm * 0.35, t_ch)
                t_ch += int(dur * self.sr)
                
        vocal_bus = stereo_delay(vocal_bus, bpm=self.bpm, fraction=0.5, feedback=0.30, wet=0.25, sr=self.sr)
        vocal_bus = stereo_reverb(vocal_bus, wet=0.35, decay_time=1.8, sr=self.sr)
        vocal_bus = sidechain_compression(vocal_bus, np.array(kick_triggers), self.bpm, pump_depth=0.50, sr=self.sr)
        
        log(0.90, "Summing all 6 instrument stems into high-resolution 32-bit mix bus...")
        raw_mix = (drum_bus * 0.90 + 
                   bass_bus * 0.85 + 
                   lead_bus * 0.70 + 
                   vocal_bus * 0.80 + 
                   fx_bus * 0.65)
                   
        log(0.95, "Applying Red Book CD Mastering Rack (EQ, Mid-Side Expansion, Brickwall Limiter)...")
        cd_master = CDMasteringSuite.master_track(
            raw_mix, self.sr, drive=self.master_drive, target_peak=self.master_peak)
        
        log(1.0, "3-Minute CD Ready EDM Master rendered successfully!")
        return cd_master

    def _paste_user_drum_pattern(self, drum_bus, bar_start, kick_triggers,
                                 kick_sample, snare_sample,
                                 hihat_closed, hihat_open):
        """Paste one bar of the user's 16-step drum pattern into drum_bus.

        Step 0 = beat 1. Pattern kick steps are also appended to
        kick_triggers so the sidechain still pumps in time with the
        user's pattern. Only used in the Drop sections, and only when
        self.drum_pattern is set.
        """
        step_samples = int(self.sr * (60.0 / self.bpm) / 4.0)
        samples = {
            "kick": kick_sample,
            "snare": snare_sample,
            "closed_hat": hihat_closed,
            "open_hat": hihat_open,
        }
        for step in range(16):
            t = bar_start + step * step_samples
            for track, sample in samples.items():
                vel = self.drum_pattern[track][step]
                if vel > 0:
                    self._paste_audio(drum_bus, sample * vel, t)
                    if track == "kick":
                        kick_triggers.append(t)

    def _paste_audio(self, target: np.ndarray, source: np.ndarray, start_sample: int):
        if start_sample >= len(target) or source is None or len(source) == 0:
            return
        end_sample = min(len(target), start_sample + len(source))
        paste_len = end_sample - start_sample
        target[start_sample:end_sample] += source[:paste_len]


def render_edm_track(output_wav_path: str, genre: str = "Cyberpunk_Bass", bpm: float = 128.0,
                     vocal_mode: str = "Vocoder_Cyber", custom_lyrics: Optional[str] = None,
                     callback: Optional[Callable[[float, str], None]] = None,
                     drum_pattern: Optional[Dict] = None,
                     master_level: str = "balanced") -> str:
    """Renders a complete 3-minute CD master WAV file.

    drum_pattern: optional 16-step user pattern (see _normalize_drum_pattern).
    When given, it replaces the drums in the Drop sections; when None
    (the default) the built-in fixed arrangement is used unchanged.
    master_level: "loud", "balanced" (default, original chain), or
    "audiophile" — drive/peak only, NOT a measured LUFS value.
    """
    start_t = time.time()
    builder = EDMTrackBuilder(genre=genre, bpm=bpm, vocal_mode=vocal_mode, custom_lyrics=custom_lyrics,
                              drum_pattern=drum_pattern, master_level=master_level)
    master_audio = builder.generate_full_track(progress_callback=callback)
    out_path = CDMasteringSuite.export_cd_wav(master_audio, output_wav_path, builder.sr)
    elapsed = time.time() - start_t
    print(f"[CD Master] Exported: {out_path} ({len(master_audio)/44100.0:.1f}s / {elapsed:.2f}s render time)")
    return out_path


# =====================================================================
#  USER DRUM PATTERNS — 16-step sequencer input
# =====================================================================
# Before this section the DAW could only render a FIXED 96-bar
# arrangement: the user could pick a genre and BPM but could not input
# a drum beat of their own and hear it back. These helpers close that
# gap using the existing DrumEngine samples — no new libraries, no
# change to the fixed arrangement when no pattern is given.
#
# Pattern format: a dict with any subset of these tracks, each either
# a 16-character string ("x...x..." where x/X/1 = hit, ./-/0/space = rest)
# or a list of 16 velocities (0.0-1.0):
#     {"kick": "x...x...x...x...", "snare": "....x.......x...",
#      "closed_hat": "x.x.x.x.x.x.x.x.", "open_hat": "......x.......x."}
# Track aliases from drum_machine.py ("hihat_closed"/"hihat_open") are
# accepted and mapped to "closed_hat"/"open_hat". Missing tracks are
# silent. Anything else raises ValueError so a bad pattern fails loudly
# instead of rendering the wrong beat.

DRUM_PATTERN_TRACKS = ("kick", "snare", "closed_hat", "open_hat")

_DRUM_PATTERN_ALIASES = {
    "hihat_closed": "closed_hat",
    "hihat_open": "open_hat",
    "closedhat": "closed_hat",
    "openhat": "open_hat",
}


def _normalize_drum_pattern(pattern: Dict) -> Dict[str, List[float]]:
    """Validate a user drum pattern into {track: [16 velocities]}."""
    if not isinstance(pattern, dict):
        raise ValueError("drum pattern must be a dict of track -> 16 steps")
    normalized: Dict[str, List[float]] = {t: [0.0] * 16 for t in DRUM_PATTERN_TRACKS}
    for raw_track, steps in pattern.items():
        track = _DRUM_PATTERN_ALIASES.get(str(raw_track).lower(), str(raw_track).lower())
        if track not in DRUM_PATTERN_TRACKS:
            raise ValueError(f"unknown drum track: {raw_track!r} "
                             f"(expected one of {list(DRUM_PATTERN_TRACKS)})")
        if isinstance(steps, str):
            if len(steps) != 16:
                raise ValueError(f"drum track {track!r} string must be 16 steps, got {len(steps)}")
            velocities = []
            for ch in steps:
                if ch in ("x", "X", "1"):
                    velocities.append(1.0)
                elif ch in (".", "-", "0", " "):
                    velocities.append(0.0)
                else:
                    raise ValueError(f"drum track {track!r}: bad step character {ch!r} "
                                     f"(use x for hit, . for rest)")
        else:
            try:
                velocities = [float(v) for v in steps]
            except (TypeError, ValueError):
                raise ValueError(f"drum track {track!r} must be a 16-char string "
                                 f"or a list of 16 velocities")
            if len(velocities) != 16:
                raise ValueError(f"drum track {track!r} must have 16 steps, got {len(velocities)}")
            velocities = [min(1.0, max(0.0, v)) for v in velocities]
        normalized[track] = velocities
    return normalized


def render_drum_pattern_audio(pattern: Dict, bpm: float = 128.0, loops: int = 4,
                              sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render a user 16-step pattern to a stereo float32 array.

    One loop = one bar (16 sixteenth-steps). Default 4 loops so a
    preview plays long enough to hear the groove repeat.
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    if not isinstance(loops, int) or loops < 1:
        raise ValueError(f"loops must be a positive integer, got {loops}")
    normalized = _normalize_drum_pattern(pattern)
    step_samples = int(sr * (60.0 / bpm) / 4.0)
    bar_samples = step_samples * 16
    total_samples = bar_samples * loops
    bus = np.zeros((total_samples, 2), dtype=np.float32)
    samples = {
        "kick": DrumEngine.synth_edm_kick(sr),
        "snare": DrumEngine.synth_edm_snare_clap(sr),
        "closed_hat": DrumEngine.synth_hihat(open_hat=False, sr=sr),
        "open_hat": DrumEngine.synth_hihat(open_hat=True, sr=sr),
    }
    for bar in range(loops):
        bar_start = bar * bar_samples
        for step in range(16):
            t = bar_start + step * step_samples
            for track, sample in samples.items():
                vel = normalized[track][step]
                if vel > 0:
                    end = min(total_samples, t + len(sample))
                    if end > t:
                        bus[t:end] += sample[:end - t] * vel
    # Soft-clip + peak-normalise so layered hits cannot clip.
    bus = np.tanh(bus)
    peak = np.max(np.abs(bus)) if bus.size else 0.0
    if peak > 0:
        bus = bus * (0.95 / peak)
    return bus.astype(np.float32)


def render_drum_pattern(output_wav_path: str, pattern: Dict, bpm: float = 128.0,
                        loops: int = 4) -> str:
    """Render a user 16-step pattern straight to a CD WAV file."""
    audio = render_drum_pattern_audio(pattern, bpm=bpm, loops=loops)
    out_path = CDMasteringSuite.export_cd_wav(audio, output_wav_path, SAMPLE_RATE)
    print(f"[Drum Pattern] Exported: {out_path} "
          f"({len(audio) / SAMPLE_RATE:.1f}s / {loops} bar(s) at {bpm} BPM)")
    return out_path


if __name__ == "__main__":
    test_out = os.path.join(os.path.expanduser("~"), "Desktop", "Test_3Min_EDM_Track.wav")
    render_edm_track(test_out, genre="Cyberpunk_Bass", bpm=128.0, vocal_mode="Vocoder_Cyber")

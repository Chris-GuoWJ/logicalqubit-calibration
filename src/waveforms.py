"""Waveform primitives shared by all three calibration modules."""

import numpy as np


def align8(value):
    return (int(value) + 7) // 8 * 8


def raised_cosine(length):
    if length < 4:
        raise ValueError("Pulse duration must be at least 4 ns")
    t = np.arange(length)
    envelope = .5 * (1 - np.cos(2 * np.pi * t / (length - 1)))
    return envelope / envelope.max()


def square(total, start, duration, amplitude):
    wave = np.zeros(total, dtype=np.float32)
    wave[start:start + duration] = amplitude
    return wave


def flat_top(total, start, duration, amplitude, edge):
    wave = np.zeros(total, dtype=np.float32)
    if duration == 0 or amplitude == 0:
        return wave
    if duration < 2 * edge:
        raise ValueError("Interaction duration must be at least twice the edge")
    ramp = .5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    wave[start:start + edge] = amplitude * ramp
    wave[start + edge:start + duration - edge] = amplitude
    wave[start + duration - edge:start + duration] = amplitude * ramp[::-1]
    return wave


def xy_wave(total, start, duration, amplitude, drive_if_ghz, envelope="raised_cosine"):
    if envelope not in ("raised_cosine", "square"):
        raise ValueError("Unsupported envelope")
    wave = np.zeros(total, dtype=np.complex64)
    shape = raised_cosine(duration) if envelope == "raised_cosine" else np.ones(duration)
    t = start + np.arange(duration)
    wave[start:start + duration] = amplitude * shape * np.exp(-2j * np.pi * drive_if_ghz * t)
    return wave


def readout(qid, qconf, total, start):
    length = int(qconf["read_length"])
    if length < 8 or start < 0 or start + length > total:
        raise ValueError("Readout exceeds waveform duration")
    carrier = float(qconf["df_read"])
    wave = np.zeros(total, dtype=np.complex64)
    t = start + np.arange(length)
    wave[start:start + length] = float(qconf["read_amp"]) * np.exp(-2j * np.pi * carrier * t)
    phase = 2 * np.pi * carrier * np.arange(min(length // 8 * 8, 4000))
    weights = np.array([[np.cos(phase), -np.sin(phase)]])
    return {f"q{qid}.readin": [wave], f"q{qid}.readout": [np.array([start]), weights]}

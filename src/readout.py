"""Validated raw IQ extraction and basic statistics."""

import numpy as np


def extract_iq(result, qid, shots):
    data = result.get_iq_data()
    if not isinstance(data, dict) or f"q{qid}" not in data:
        raise ValueError(f"Missing raw IQ for Q{qid}")
    iq = np.asarray(data[f"q{qid}"], dtype=float)
    if iq.shape != (shots, 1, 2) or not np.isfinite(iq).all():
        raise ValueError(f"Q{qid}: expected finite IQ shape {(shots, 1, 2)}, got {iq.shape}")
    return iq[:, 0, :].copy()


def extract_pair(result, q_a, q_b, shots):
    return extract_iq(result, q_a, shots), extract_iq(result, q_b, shots)


def center_distance(iq, reference):
    center = iq.mean(axis=0)
    zero = reference.mean(axis=0)
    distance = float(np.linalg.norm(center - zero))
    covariance = np.cov(iq, rowvar=False) / len(iq) + np.cov(reference, rowvar=False) / len(reference)
    noise = float(np.sqrt(max(float(np.trace(covariance)), 0)))
    return {"I0": float(zero[0]), "Q0": float(zero[1]),
            "I_mean": float(center[0]), "Q_mean": float(center[1]),
            "center_distance": distance, "S": distance, "mean_displacement_noise": noise}

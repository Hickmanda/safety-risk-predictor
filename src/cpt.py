"""
Cumulative Prospect Theory (CPT) — full implementation.
Based on Zhang et al. (2025, ASCE JCEM).
"""

import numpy as np

# ============================================================
# Default parameters (Table 7)
# ============================================================
ALPHA = 0.88
BETA = 0.88
GAMMA = 0.61
DELTA = 0.65
LAMBDA = 1.18

# ============================================================
# Outcome values (Eq. 21)
# ============================================================
X_1 = 1.00
X_2 = 0.80
X_M1 = 0.40
X_M2 = 0.20
X_M3 = 0.10
X_M4 = 0.05

REFERENCE_POINT = 0.6

# ============================================================
# Probabilities (Table 7)
# ============================================================
R_M4 = 0.01
R_M3 = 0.10
R_M2 = 0.50
R_M1 = 1.00
R_1 = 1.00
R_2 = 0.50


def value_function(x, reference_point, alpha=ALPHA, beta=BETA, lam=LAMBDA):
    """CPT value function — Eq. 10."""
    x = np.asarray(x, dtype=float)
    deviation = x - reference_point
    v = np.zeros_like(deviation)
    gains = deviation > 0
    losses = ~gains
    v[gains] = deviation[gains] ** alpha
    v[losses] = -lam * (np.abs(deviation[losses]) ** beta)
    return v


def weight_gain(P, gamma=GAMMA):
    """Probability weight for gains — Eq. 12."""
    P = np.asarray(P, dtype=float)
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return (P ** gamma) / ((P ** gamma + (1 - P) ** gamma) ** (1 / gamma))


def weight_loss(P, delta=DELTA):
    """Probability weight for losses — Eq. 13."""
    P = np.asarray(P, dtype=float)
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return (P ** delta) / ((P ** delta + (1 - P) ** delta) ** (1 / delta))


def prospect(outcomes, probabilities, reference_point,
             alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA, lam=LAMBDA):
    """Generic CPT for a set of outcomes — Eq. 9, 11."""
    outcomes = np.asarray(outcomes, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    order = np.argsort(outcomes)
    outcomes = outcomes[order]
    probabilities = probabilities[order]

    gains_mask = outcomes > 0
    losses_mask = ~gains_mask

    cpt = 0.0

    if gains_mask.any():
        go = outcomes[gains_mask]
        gp = probabilities[gains_mask]
        cum = np.cumsum(gp[::-1])[::-1]
        w = weight_gain(cum, gamma)
        cpt += np.sum(value_function(go, reference_point, alpha, beta, lam) * w)

    if losses_mask.any():
        lo = outcomes[losses_mask]
        lp = probabilities[losses_mask]
        cum = np.cumsum(lp)
        w = weight_loss(cum, delta)
        cpt += np.sum(value_function(lo, reference_point, alpha, beta, lam) * w)

    return cpt


def compute_cpt_unsafe(reference_point=REFERENCE_POINT,
                       alpha=ALPHA, beta=BETA, gamma=GAMMA,
                       delta=DELTA, lam=LAMBDA,
                       r_m4=R_M4, r_m3=R_M3, r_m2=R_M2,
                       x_1=X_1, x_m2=X_M2, x_m3=X_M3, x_m4=X_M4):
    """Cumulative prospect of UNSAFE behavior — Eq. 14."""
    p_1 = (1 - r_m4 - r_m3) * (1 - r_m2)
    p_m2 = (1 - r_m4 - r_m3) * r_m2
    p_m3 = r_m3
    p_m4 = r_m4

    outcomes = [x_1, x_m2, x_m3, x_m4]
    probabilities = [p_1, p_m2, p_m3, p_m4]

    return prospect(outcomes, probabilities, reference_point,
                    alpha, beta, gamma, delta, lam)


def compute_cpt_safe(reference_point=REFERENCE_POINT,
                     alpha=ALPHA, beta=BETA, gamma=GAMMA,
                     delta=DELTA, lam=LAMBDA,
                     r_m1=R_M1, r_2=R_2,
                     x_2=X_2, x_m1=X_M1):
    """Cumulative prospect of SAFE behavior — Eq. 15."""
    p_2 = r_2
    p_m1 = r_m1 - r_2

    outcomes = [x_2, x_m1]
    probabilities = [p_2, p_m1]

    return prospect(outcomes, probabilities, reference_point,
                    alpha, beta, gamma, delta, lam)


def decide(reference_point=REFERENCE_POINT):
    """Decide SAFE vs UNSAFE."""
    cpt_s = compute_cpt_safe(reference_point)
    cpt_u = compute_cpt_unsafe(reference_point)
    decision = "SAFE" if cpt_s >= cpt_u else "UNSAFE"
    return {"cpt_safe": cpt_s, "cpt_unsafe": cpt_u, "decision": decision}


if __name__ == '__main__':
    print("=" * 60)
    print("CPT full test — based on Zhang et al. (2025)")
    print("=" * 60)
    result = decide()
    print(f"CPT (safe)   = {result['cpt_safe']:.4f}")
    print(f"CPT (unsafe) = {result['cpt_unsafe']:.4f}")
    print(f"Decision: {result['decision']}")
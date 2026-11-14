"""Numeric check of Theorem 1 (conformal risk control of accepted-but-violating slates).

Synthetic exchangeable data: score s ~ N(0,1), violation y ~ Bernoulli(sigmoid(-a*s - b)).
For each (n, alpha, a) we draw many calibration sets plus one test point and compare the
empirical P(accepted and violating) with the bound alpha and the lower bound alpha - 2/(n+1).
Writes results/sim_crc.csv.
"""
import os

import numpy as np
import pandas as pd

from verifier import crc_threshold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(7)
rows = []
TRIALS = 100000
for n in (50, 200, 1000):
    for alpha in (0.05, 0.10):
        for a in (0.0, 1.0, 3.0):          # a = 0: uninformative score
            losses, acc = [], []
            for _ in range(TRIALS):
                s = rng.normal(size=n + 1)
                y = (rng.random(n + 1) < 1 / (1 + np.exp(a * s + 0.5))).astype(float)
                lam = crc_threshold(s[:n], y[:n], alpha)
                accept = s[n] > lam
                losses.append(y[n] * accept); acc.append(accept)
            risk = float(np.mean(losses))
            rows.append({"n": n, "alpha": alpha, "informativeness_a": a, "risk": risk,
                         "risk_se": float(np.std(losses) / np.sqrt(TRIALS)), "upper": alpha,
                         "lower": alpha - 2 / (n + 1), "accept_rate": float(np.mean(acc)),
                         "holds": risk <= alpha + 3 * np.std(losses) / np.sqrt(TRIALS)})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "sim_crc.csv"), index=False)
print(df.round(4).to_string())

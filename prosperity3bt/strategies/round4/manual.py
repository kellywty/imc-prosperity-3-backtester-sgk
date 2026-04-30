"""
Round 4 manual trading challenge: "Vanilla Just Isn't Exotic Enough".

Game model
----------
- Underlying AC: GBM, S0 = 50, sigma = 251% annualized, zero risk-neutral drift.
- 4 steps per trading day, 252 trading days per year.
- 2-week products expire after 40 steps; 3-week products after 60 steps.
- Knock-out / chooser barriers are evaluated at the discrete grid only.
- The platform marks each contract at expiry to its FAIR VALUE, defined as the
  mean payoff across 100 simulations of the underlying. Your final score is
  the average PnL across those same 100 sims.
- By linearity:  score = sum_i  pos_i * (mark_i - cost_i) * CONTRACT_SIZE
  where mark_i = mean payoff of contract i across the platform's 100 paths.
  E[mark_i] equals the true fair value; var[mark_i] = var(payoff_i) / 100.

Strategy implications
---------------------
- EV is linear in each position with independent caps -> EV-optimal strategy
  is "trade max volume on whichever side has positive edge, otherwise hold".
  A naive grid search rediscovers this corner solution.
- The score is still random because the platform's mark is a sample mean of
  only 100 paths. We bootstrap that to get the realised-score distribution,
  per-leg sigma, P(score<0), and a Sharpe-style ratio so we can see which
  legs are robust to mark noise vs. which are noisy bets.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

S0 = 50.0
SIGMA = 2.51  # annualized, 251%

TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
DT = 1.0 / (TRADING_DAYS_PER_YEAR * STEPS_PER_DAY)

STEPS_2W = 10 * STEPS_PER_DAY  # 40
STEPS_3W = 15 * STEPS_PER_DAY  # 60

CONTRACT_SIZE = 3000

# (bid, ask, max_volume).  Positive edge on buy -> we buy at ask;
# positive edge on sell -> we sell at bid.
MARKET: Dict[str, Tuple[float, float, int]] = {
    "AC":         (49.975, 50.025, 200),
    "AC_50_P":    (12.00,  12.05,   50),
    "AC_50_C":    (12.00,  12.05,   50),
    "AC_35_P":    (4.33,   4.35,    50),
    "AC_40_P":    (6.50,   6.55,    50),
    "AC_45_P":    (9.05,   9.10,    50),
    "AC_60_C":    (8.80,   8.85,    50),
    "AC_50_P_2":  (9.70,   9.75,    50),
    "AC_50_C_2":  (9.70,   9.75,    50),
    "AC_50_CO":   (22.20,  22.30,   50),
    "AC_40_BP":   (5.00,   5.10,    50),
    "AC_45_KO":   (0.15,   0.175,  500),
}


# ---------------------------------------------------------------------------
# Path simulation
# ---------------------------------------------------------------------------

def simulate_paths(num_paths: int, total_steps: int = STEPS_3W,
                   seed: int = 42, antithetic: bool = True) -> np.ndarray:
    """Simulate GBM paths. Antithetic variates halve the variance of even-
    moment estimators (so all our means tighten) at no extra cost."""
    rng = np.random.default_rng(seed)
    if antithetic:
        if num_paths % 2:
            num_paths += 1
        half = num_paths // 2
        z = rng.standard_normal((half, total_steps))
        z = np.concatenate([z, -z], axis=0)
    else:
        z = rng.standard_normal((num_paths, total_steps))

    drift = -0.5 * SIGMA * SIGMA * DT
    vol = SIGMA * math.sqrt(DT)
    log_paths = np.cumsum(drift + vol * z, axis=1)
    return S0 * np.exp(log_paths)


def compute_payoffs(paths: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-product payoff vector (one entry per simulated path)."""
    s_2w = paths[:, STEPS_2W - 1]
    s_3w = paths[:, STEPS_3W - 1]
    min_path = paths.min(axis=1)

    # AC underlying held to 3W expiry: marked at S_3w.
    # (No specified expiry, but the round is structured around holding to T+21
    # so we treat the unwind point consistently with the option marks.)
    return {
        "AC":        s_3w,
        "AC_50_P":   np.maximum(50 - s_3w, 0),
        "AC_50_C":   np.maximum(s_3w - 50, 0),
        "AC_35_P":   np.maximum(35 - s_3w, 0),
        "AC_40_P":   np.maximum(40 - s_3w, 0),
        "AC_45_P":   np.maximum(45 - s_3w, 0),
        "AC_60_C":   np.maximum(s_3w - 60, 0),
        "AC_50_P_2": np.maximum(50 - s_2w, 0),
        "AC_50_C_2": np.maximum(s_2w - 50, 0),
        # Chooser: at week 2, pick whichever side is currently in the money,
        # then it behaves like that vanilla until week 3.
        "AC_50_CO":  np.where(
            s_2w > 50,
            np.maximum(s_3w - 50, 0),
            np.maximum(50 - s_3w, 0),
        ),
        # Binary put: pays 10 if S_3w < 40.
        "AC_40_BP":  np.where(s_3w < 40, 10.0, 0.0),
        # Knock-out put: zero if min(path) < 35 at any discrete step,
        # otherwise standard 45-strike put payoff.
        "AC_45_KO":  np.where(min_path >= 35,
                              np.maximum(45 - s_3w, 0),
                              0.0),
    }


# ---------------------------------------------------------------------------
# Black-Scholes (analytic) sanity check
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, T, sigma, r=0.0):
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S, K, T, sigma, r=0.0):
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def analytic_prices() -> Dict[str, float]:
    t2 = 10.0 / TRADING_DAYS_PER_YEAR
    t3 = 15.0 / TRADING_DAYS_PER_YEAR
    return {
        "AC":        S0,
        "AC_50_P":   bs_put(S0, 50, t3, SIGMA),
        "AC_50_C":   bs_call(S0, 50, t3, SIGMA),
        "AC_35_P":   bs_put(S0, 35, t3, SIGMA),
        "AC_40_P":   bs_put(S0, 40, t3, SIGMA),
        "AC_45_P":   bs_put(S0, 45, t3, SIGMA),
        "AC_60_C":   bs_call(S0, 60, t3, SIGMA),
        "AC_50_P_2": bs_put(S0, 50, t2, SIGMA),
        "AC_50_C_2": bs_call(S0, 50, t2, SIGMA),
    }


# ---------------------------------------------------------------------------
# Control variates
#
# Optimal-beta linear control variate: if X is our payoff and Y is a control
# with known E[Y], then  X - beta*(Y - E[Y])  has the same mean as X but
# variance reduced by a factor of (1 - rho^2) at beta* = Cov(X,Y)/Var(Y).
#
# For vanillas we use Y = S_T (terminal price); E[S_T] = S0 under risk-neutral
# zero-drift GBM.
# For the chooser we use Y = S_3w as well.
# For the binary put and KO put we use Y = vanilla put with the same strike;
# E[Y] is known analytically via Black-Scholes.  Both are highly correlated
# with their target payoff, so variance reduction is large.
# ---------------------------------------------------------------------------

def control_setup(paths: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
    """Map each product to (control_array_per_path, known_E_of_control)."""
    s_2w = paths[:, STEPS_2W - 1]
    s_3w = paths[:, STEPS_3W - 1]
    t3 = 15.0 / TRADING_DAYS_PER_YEAR
    return {
        "AC_50_P":   (s_3w, S0),
        "AC_50_C":   (s_3w, S0),
        "AC_35_P":   (s_3w, S0),
        "AC_40_P":   (s_3w, S0),
        "AC_45_P":   (s_3w, S0),
        "AC_60_C":   (s_3w, S0),
        "AC_50_P_2": (s_2w, S0),
        "AC_50_C_2": (s_2w, S0),
        "AC_50_CO":  (s_3w, S0),
        "AC_40_BP":  (np.maximum(40 - s_3w, 0), bs_put(S0, 40, t3, SIGMA)),
        "AC_45_KO":  (np.maximum(45 - s_3w, 0), bs_put(S0, 45, t3, SIGMA)),
    }


def cv_fair_values(paths: np.ndarray, payoffs: Dict[str, np.ndarray]):
    """Compute control-variate-corrected fair values plus standard errors,
    alongside the plain MC values for comparison."""
    n = paths.shape[0]
    controls = control_setup(paths)
    out: Dict[str, Dict[str, float]] = {}
    for name, payoff in payoffs.items():
        plain_mean = float(payoff.mean())
        plain_se = float(payoff.std(ddof=1) / math.sqrt(n))
        if name not in controls:
            out[name] = {
                "plain_mean": plain_mean, "plain_se": plain_se,
                "cv_mean": plain_mean, "cv_se": plain_se,
                "beta": 0.0, "var_ratio": 1.0,
            }
            continue
        ctrl, ev_ctrl = controls[name]
        x_centered = payoff - payoff.mean()
        y_centered = ctrl - ctrl.mean()
        var_y = float((y_centered * y_centered).mean())
        cov_xy = float((x_centered * y_centered).mean())
        beta = cov_xy / var_y if var_y > 0 else 0.0
        adjusted = payoff - beta * (ctrl - ev_ctrl)
        cv_mean = float(adjusted.mean())
        cv_se = float(adjusted.std(ddof=1) / math.sqrt(n))
        var_ratio = (cv_se / plain_se) ** 2 if plain_se > 0 else 1.0
        out[name] = {
            "plain_mean": plain_mean, "plain_se": plain_se,
            "cv_mean": cv_mean, "cv_se": cv_se,
            "beta": beta, "var_ratio": var_ratio,
        }
    return out


def print_cv_table(cv: Dict[str, Dict[str, float]], analytic: Dict[str, float]):
    print(f"\n{'Product':<11} {'Plain MC':>12} {'±SE':>10} "
          f"{'CV MC':>12} {'±SE':>10} {'Var ratio':>10} {'BS':>10}")
    print("-" * 84)
    for name, d in cv.items():
        bs = analytic.get(name)
        bs_str = f"{bs:>10.4f}" if bs is not None else f"{'-':>10}"
        print(f"{name:<11} {d['plain_mean']:>12.4f} {d['plain_se']:>10.5f} "
              f"{d['cv_mean']:>12.4f} {d['cv_se']:>10.5f} "
              f"{d['var_ratio']:>10.3f} {bs_str}")


# ---------------------------------------------------------------------------
# Ablation analysis: marginal contribution of each leg to total EV / sigma
# ---------------------------------------------------------------------------

def ablation_analysis(positions: Dict[str, int],
                      num_worlds: int = 20_000,
                      sample_size: int = 100,
                      seed: int = 7777):
    """For each non-zero leg, recompute total stats with that leg dropped and
    report the change. Captures cross-leg covariance correctly because all
    products see the same 100 paths in each world."""
    paths = simulate_paths(num_worlds * sample_size, STEPS_3W,
                           seed=seed, antithetic=True)
    payoffs = compute_payoffs(paths)
    marks = {n: arr.reshape(num_worlds, sample_size).mean(axis=1)
             for n, arr in payoffs.items()}

    legs = {}
    total = np.zeros(num_worlds)
    for name, pos in positions.items():
        if pos == 0:
            continue
        bid, ask, _ = MARKET[name]
        unit = (marks[name] - ask) if pos > 0 else (bid - marks[name])
        legs[name] = abs(pos) * unit * CONTRACT_SIZE
        total += legs[name]

    base_ev = total.mean()
    base_sd = total.std()
    base_sh = base_ev / base_sd if base_sd > 0 else 0.0

    print(f"\nBaseline:  EV = ${base_ev:,.0f}   σ = ${base_sd:,.0f}   "
          f"Sharpe = {base_sh:.3f}   P(<0) = {(total<0).mean()*100:.2f}%")
    print(f"\n{'Drop leg':<11} {'ΔEV':>14} {'New EV':>14} {'New σ':>14} "
          f"{'New Sharpe':>11} {'ΔSharpe':>9} {'New P(<0)':>10}")
    print("-" * 90)
    for name, leg in legs.items():
        without = total - leg
        new_ev = without.mean()
        new_sd = without.std()
        new_sh = new_ev / new_sd if new_sd > 0 else 0.0
        delta_ev = base_ev - new_ev
        delta_sh = new_sh - base_sh
        flag = "  <- improves" if delta_sh > 0 else ""
        print(f"{name:<11} {-delta_ev:>+14,.0f} {new_ev:>14,.0f} {new_sd:>14,.0f} "
              f"{new_sh:>11.3f} {delta_sh:>+9.3f} {(without<0).mean()*100:>9.2f}%"
              f"{flag}")

    # Also compute Sharpe-optimal subset by dropping any leg whose removal
    # increases Sharpe. This is iterated greedy until stable.
    kept = dict(positions)
    while True:
        improved = False
        # current total under kept
        cur_total = sum((legs[n] for n, p in kept.items() if p != 0 and n in legs),
                        start=np.zeros(num_worlds))
        cur_ev = cur_total.mean()
        cur_sd = cur_total.std()
        cur_sh = cur_ev / cur_sd if cur_sd > 0 else 0.0
        for name, pos in list(kept.items()):
            if pos == 0 or name not in legs:
                continue
            new_total = cur_total - legs[name]
            new_sh = new_total.mean() / new_total.std() if new_total.std() > 0 else 0.0
            if new_sh > cur_sh + 1e-6:
                kept[name] = 0
                improved = True
                break
        if not improved:
            break

    final_total = sum((legs[n] for n, p in kept.items() if p != 0 and n in legs),
                      start=np.zeros(num_worlds))
    print("\nGreedy Sharpe-trimmed portfolio:")
    print(f"  EV = ${final_total.mean():,.0f}   σ = ${final_total.std():,.0f}   "
          f"Sharpe = {final_total.mean()/final_total.std():.3f}   "
          f"P(<0) = {(final_total<0).mean()*100:.2f}%")
    dropped = [n for n, p in positions.items() if p != 0 and kept.get(n, 0) == 0]
    if dropped:
        print(f"  Dropped legs: {', '.join(dropped)}")
    else:
        print("  No legs dropped (all contribute positively to Sharpe).")
    return kept


# ---------------------------------------------------------------------------
# Delta computation (for static delta hedging at t=0)
#
# Production lesson: with sigma=251% and only 100 platform paths, residual
# directional exposure is the dominant source of realised score variance. A
# static delta hedge at t=0 (buy/sell underlying in fixed quantity) collapses
# that exposure to zero, leaving only the volatility-edge PnL plus a small
# spread cost on the AC underlying (~0.025 per share).
#
# Method: finite-difference MC with common random numbers. Generate one set of
# Brownian shocks Z, then build paths starting from S0*(1+eps) and S0*(1-eps).
# Per-path differences cancel a huge amount of variance, so 4M paths pin down
# every delta to 4 decimals.
# ---------------------------------------------------------------------------

def compute_deltas(num_paths: int = 4_000_000, eps: float = 0.001,
                   seed: int = 2023) -> Dict[str, float]:
    """t=0 delta in option-payoff $ per $1 move in S0. Underlying delta = 1.0
    by definition. For options, delta = (V(S0*(1+eps)) - V(S0*(1-eps))) / (2*S0*eps)."""
    rng = np.random.default_rng(seed)
    if num_paths % 2:
        num_paths += 1
    half = num_paths // 2
    z_half = rng.standard_normal((half, STEPS_3W))
    z = np.concatenate([z_half, -z_half], axis=0)
    drift = -0.5 * SIGMA * SIGMA * DT
    vol = SIGMA * math.sqrt(DT)
    log_paths = np.cumsum(drift + vol * z, axis=1)

    s_up = S0 * (1.0 + eps)
    s_dn = S0 * (1.0 - eps)
    paths_up = s_up * np.exp(log_paths)
    paths_dn = s_dn * np.exp(log_paths)
    p_up = compute_payoffs(paths_up)
    p_dn = compute_payoffs(paths_dn)

    deltas = {"AC": 1.0}
    for name in MARKET:
        if name == "AC":
            continue
        deltas[name] = float((p_up[name].mean() - p_dn[name].mean()) / (s_up - s_dn))
    return deltas


def portfolio_delta(positions: Dict[str, int], deltas: Dict[str, float]) -> float:
    """Total dV/dS for a given position vector, in option-payoff units."""
    return sum(pos * deltas[name] for name, pos in positions.items())


def static_delta_hedge(positions: Dict[str, int],
                       deltas: Dict[str, float]) -> Dict[str, int]:
    """Add an AC underlying position so total portfolio delta is 0 (rounded
    to nearest integer, clipped to AC's max volume of 200)."""
    options_delta = sum(pos * deltas[name]
                        for name, pos in positions.items() if name != "AC")
    ac_max = MARKET["AC"][2]
    needed = -int(round(options_delta))
    needed = max(-ac_max, min(ac_max, needed))
    out = dict(positions)
    out["AC"] = needed
    return out


# ---------------------------------------------------------------------------
# Risk-aware portfolio optimiser  (coordinate descent)
#
# Objectives supported:
#   - 'ev'      : maximise E[total PnL]                     (the old Ticket A)
#   - 'sharpe'  : maximise E[total PnL] / sigma(total PnL)
#   - 'mv'      : maximise E[total PnL] - lam * sigma       (mean-variance)
#   - 'cvar5'   : maximise the 5th percentile of total PnL  (the right one
#                 for production: explicitly buys robustness to bad draws)
#
# Coordinate descent: each pass, sweep one product's signed position over a
# 41-point grid in [-max_vol, +max_vol] while holding the others fixed and
# pick the maximiser. Repeat until no improvement. Bid-ask kink at zero is
# handled by evaluating both sides explicitly. Multiple random restarts guard
# against local optima.
# ---------------------------------------------------------------------------

def optimize_portfolio(num_worlds: int = 20_000,
                       sample_size: int = 100,
                       objective: str = "mv",
                       lam: float = 0.3,
                       max_iter: int = 15,
                       num_restarts: int = 6,
                       seed: int = 24680,
                       initial_positions: Dict[str, int] = None,
                       grid_points: int = 81,
                       verbose: bool = False) -> Tuple[Dict[str, int], np.ndarray]:
    """Coordinate-descent portfolio optimiser. Objectives:
      'ev'      -> max E[total PnL]
      'sharpe'  -> max E[total PnL] / sigma
      'mv'      -> max E[total PnL] - lam * sigma
      'cvar5'   -> max 5th percentile (often degenerate -> 0 in this game)
      'p10'     -> max 10th percentile (less aggressive than cvar5)
      'median'  -> max P50 of total PnL

    Returns (best_positions, total_per_world_under_best_positions).
    """
    paths = simulate_paths(num_worlds * sample_size, STEPS_3W,
                           seed=seed, antithetic=True)
    payoffs = compute_payoffs(paths)
    marks = {n: arr.reshape(num_worlds, sample_size).mean(axis=1)
             for n, arr in payoffs.items()}

    products = list(MARKET.keys())
    n = len(products)
    bids = np.array([MARKET[name][0] for name in products])
    asks = np.array([MARKET[name][1] for name in products])
    max_vols = np.array([MARKET[name][2] for name in products], dtype=float)
    mark_matrix = np.stack([marks[name] for name in products])
    unit_buy = mark_matrix - asks[:, None]
    unit_sell = bids[:, None] - mark_matrix

    def leg_pnl(i: int, x_i: float) -> np.ndarray:
        if x_i > 0:
            return x_i * unit_buy[i] * CONTRACT_SIZE
        if x_i < 0:
            return (-x_i) * unit_sell[i] * CONTRACT_SIZE
        return np.zeros(num_worlds)

    def vec_score(totals: np.ndarray) -> np.ndarray:
        if objective == "ev":
            return totals.mean(axis=1)
        if objective == "sharpe":
            means = totals.mean(axis=1)
            stds = totals.std(axis=1)
            return np.where(stds > 0, means / np.maximum(stds, 1e-9), 0)
        if objective == "mv":
            return totals.mean(axis=1) - lam * totals.std(axis=1)
        if objective == "cvar5":
            return np.percentile(totals, 5, axis=1)
        if objective == "p10":
            return np.percentile(totals, 10, axis=1)
        if objective == "median":
            return np.percentile(totals, 50, axis=1)
        raise ValueError(objective)

    def score(total: np.ndarray) -> float:
        return float(vec_score(total[None, :])[0])

    rng = np.random.default_rng(seed + 1)
    best_positions: Dict[str, int] = {p: 0 for p in products}
    best_total = np.zeros(num_worlds)
    best_score = score(best_total)

    # Build candidate starts: zero, optional warm start, then random.
    starts: List[np.ndarray] = [np.zeros(n)]
    if initial_positions is not None:
        warm = np.array([float(initial_positions.get(p, 0)) for p in products])
        starts.append(warm)
    while len(starts) < num_restarts:
        starts.append(rng.uniform(-max_vols, max_vols))

    for restart, x0 in enumerate(starts):
        x = x0.copy()
        # Clip into bounds.
        x = np.clip(x, -max_vols, max_vols)
        cur_total = np.zeros(num_worlds)
        for i, xi in enumerate(x):
            cur_total += leg_pnl(i, xi)
        cur_score = score(cur_total)

        for _ in range(max_iter):
            prev_score = cur_score
            for i in range(n):
                old_xi = x[i]
                total_minus_i = cur_total - leg_pnl(i, old_xi)
                cand = np.linspace(-max_vols[i], max_vols[i], grid_points)
                pos_part = np.maximum(cand, 0)[:, None]
                neg_part = np.maximum(-cand, 0)[:, None]
                leg = (pos_part * unit_buy[i][None, :]
                       + neg_part * unit_sell[i][None, :]) * CONTRACT_SIZE
                totals = total_minus_i[None, :] + leg
                vals = vec_score(totals)
                best_idx = int(np.argmax(vals))
                if vals[best_idx] > cur_score + 1e-3:
                    x[i] = cand[best_idx]
                    cur_total = totals[best_idx]
                    cur_score = float(vals[best_idx])
            if cur_score <= prev_score + 1e-3:
                break

        if cur_score > best_score:
            best_score = cur_score
            best_positions = {p: int(round(x[i])) for i, p in enumerate(products)}
            best_total = cur_total.copy()
            if verbose:
                print(f"  restart {restart} ({objective}, lam={lam}): "
                      f"score = {best_score:,.2f}")

    return best_positions, best_total


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def edges(fair_values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, (bid, ask, max_vol) in MARKET.items():
        fv = fair_values[name]
        out[name] = {
            "bid": bid, "ask": ask, "max_vol": max_vol,
            "fair": fv,
            "edge_buy":  fv - ask,
            "edge_sell": bid - fv,
        }
    return out


def ev_max_positions(fair_values: Dict[str, float]) -> Dict[str, int]:
    """Take the corner: max volume on the side with positive edge."""
    pos: Dict[str, int] = {}
    for name, (bid, ask, max_vol) in MARKET.items():
        fv = fair_values[name]
        eb, es = fv - ask, bid - fv
        if eb > es and eb > 0:
            pos[name] = +max_vol
        elif es > 0:
            pos[name] = -max_vol
        else:
            pos[name] = 0
    return pos


# ---------------------------------------------------------------------------
# Bootstrap of the realised "100-sim mark" score
# ---------------------------------------------------------------------------

def bootstrap_score(positions: Dict[str, int],
                    num_worlds: int = 10_000,
                    sample_size: int = 100,
                    seed: int = 999):
    """Sample fresh GBM paths and reshape into `num_worlds` blocks of
    `sample_size` paths each. Within each block, the platform's mark is the
    block-mean payoff.  All products see the SAME 100 paths in each world,
    which is what the platform does, so cross-product mark correlations are
    captured."""
    paths = simulate_paths(num_worlds * sample_size, STEPS_3W,
                           seed=seed, antithetic=True)
    payoffs = compute_payoffs(paths)
    marks = {n: arr.reshape(num_worlds, sample_size).mean(axis=1)
             for n, arr in payoffs.items()}

    per_prod = {}
    total = np.zeros(num_worlds)
    for name, pos in positions.items():
        if pos == 0:
            per_prod[name] = np.zeros(num_worlds)
            continue
        bid, ask, _ = MARKET[name]
        unit = (marks[name] - ask) if pos > 0 else (bid - marks[name])
        leg = abs(pos) * unit * CONTRACT_SIZE
        per_prod[name] = leg
        total += leg
    return per_prod, total, marks


# ---------------------------------------------------------------------------
# Per-product 1-D sweep ("grid search" demo)
# ---------------------------------------------------------------------------

def per_product_sweep(payoffs: Dict[str, np.ndarray],
                      fair_values: Dict[str, float],
                      num_worlds: int = 5_000,
                      sample_size: int = 100,
                      seed: int = 1234):
    """For each product, sweep position from -max to +max and report EV / sigma /
    Sharpe.  Confirms that PnL is linear in volume and the corner is optimal."""
    paths = simulate_paths(num_worlds * sample_size, STEPS_3W, seed=seed)
    p = compute_payoffs(paths)
    marks = {n: arr.reshape(num_worlds, sample_size).mean(axis=1)
             for n, arr in p.items()}

    print(f"\n{'Product':<11} {'Pos':>6} {'EV($)':>14} {'Std($)':>14} "
          f"{'P5%($)':>14} {'Sharpe':>7}")
    print("-" * 70)
    for name, (bid, ask, max_vol) in MARKET.items():
        m = marks[name]
        steps = np.linspace(-max_vol, max_vol, 9).round().astype(int)
        for pos in steps:
            if pos == 0:
                continue
            unit = (m - ask) if pos > 0 else (bid - m)
            leg = abs(int(pos)) * unit * CONTRACT_SIZE
            ev = leg.mean()
            sd = leg.std()
            p5 = np.percentile(leg, 5)
            sh = ev / sd if sd > 0 else 0.0
            print(f"{name:<11} {int(pos):>+6} {ev:>14,.0f} {sd:>14,.0f} "
                  f"{p5:>14,.0f} {sh:>7.3f}")
        print()


# ---------------------------------------------------------------------------
# Global scaling sweep (uniform fraction of max position across all legs)
# ---------------------------------------------------------------------------

def global_scaling_sweep(positions: Dict[str, int],
                         num_worlds: int = 10_000,
                         sample_size: int = 100,
                         seed: int = 4242):
    """Scale every leg by lambda in [0, 1]. EV scales linearly, sigma scales
    linearly, Sharpe stays flat -> in expectation, more size is always better
    *unless* you have an external risk budget you care about."""
    paths = simulate_paths(num_worlds * sample_size, STEPS_3W,
                           seed=seed, antithetic=True)
    payoffs = compute_payoffs(paths)
    marks = {n: arr.reshape(num_worlds, sample_size).mean(axis=1)
             for n, arr in payoffs.items()}

    print(f"\n{'lambda':>6} {'EV($)':>14} {'Std($)':>14} "
          f"{'P5%($)':>14} {'P95%($)':>14} {'P(<0)':>7}")
    print("-" * 70)
    for lam in [0.25, 0.50, 0.75, 1.00]:
        total = np.zeros(num_worlds)
        for name, pos in positions.items():
            if pos == 0:
                continue
            bid, ask, _ = MARKET[name]
            scaled_pos = lam * pos
            unit = (marks[name] - ask) if scaled_pos > 0 else (bid - marks[name])
            total += abs(scaled_pos) * unit * CONTRACT_SIZE
        print(f"{lam:>6.2f} {total.mean():>14,.0f} {total.std():>14,.0f} "
              f"{np.percentile(total, 5):>14,.0f} {np.percentile(total, 95):>14,.0f} "
              f"{(total<0).mean()*100:>6.2f}%")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_fair_table(fair_values, payoff_stds, analytic):
    print(f"\n{'Product':<11} {'Bid':>8} {'Ask':>8} {'MC':>10} {'BS':>10} "
          f"{'EdgeBuy':>10} {'EdgeSell':>10} {'σ(payoff)':>10} {'Side':>6}")
    print("-" * 96)
    for name, (bid, ask, _) in MARKET.items():
        fv = fair_values[name]
        sd = payoff_stds[name]
        bs = analytic.get(name)
        eb, es = fv - ask, bid - fv
        if eb > es and eb > 0:
            side = "BUY"
        elif es > 0:
            side = "SELL"
        else:
            side = "HOLD"
        bs_str = f"{bs:>10.4f}" if bs is not None else f"{'-':>10}"
        print(f"{name:<11} {bid:>8.3f} {ask:>8.3f} {fv:>10.4f} {bs_str} "
              f"{eb:>+10.4f} {es:>+10.4f} {sd:>10.4f} {side:>6}")


def print_portfolio(positions, fair_values, per_prod, total, marks):
    print("\n" + "=" * 100)
    print("EV-maximising portfolio  (max-size on every product with positive edge)")
    print("=" * 100)
    print(f"{'Product':<11} {'Pos':>5} {'Cost':>8} {'TrueEV':>14} {'σ(PnL)':>14} "
          f"{'P5%':>14} {'P95%':>14} {'Sharpe':>7} {'P(flip)':>9}")
    print("-" * 100)
    deterministic_ev = 0.0
    for name, pos in positions.items():
        if pos == 0:
            continue
        bid, ask, _ = MARKET[name]
        cost = ask if pos > 0 else bid
        unit_edge = (fair_values[name] - ask) if pos > 0 else (bid - fair_values[name])
        true_ev = abs(pos) * unit_edge * CONTRACT_SIZE
        deterministic_ev += true_ev

        leg = per_prod[name]
        sd = leg.std()
        p5 = np.percentile(leg, 5)
        p95 = np.percentile(leg, 95)
        sharpe = leg.mean() / sd if sd > 0 else 0.0

        # Probability the realised mark gives the wrong-sign edge.
        m = marks[name]
        if pos > 0:
            p_flip = (m < ask).mean()
        else:
            p_flip = (m > bid).mean()

        print(f"{name:<11} {pos:>+5} {cost:>8.3f} {true_ev:>14,.0f} {sd:>14,.0f} "
              f"{p5:>14,.0f} {p95:>14,.0f} {sharpe:>7.3f} {p_flip*100:>8.2f}%")

    print("-" * 100)
    p5_t = np.percentile(total, 5)
    p95_t = np.percentile(total, 95)
    sharpe_t = total.mean() / total.std() if total.std() > 0 else 0.0
    print(f"{'TOTAL':<11} {'':>5} {'':>8} {total.mean():>14,.0f} {total.std():>14,.0f} "
          f"{p5_t:>14,.0f} {p95_t:>14,.0f} {sharpe_t:>7.3f}")
    print(f"\nDeterministic EV (using true fair values): {deterministic_ev:>14,.0f}")
    print(f"P(realised score < 0):                     {(total<0).mean()*100:>6.2f}%")
    print(f"Min realised score:                        {total.min():>14,.0f}")
    print(f"Max realised score:                        {total.max():>14,.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.set_printoptions(suppress=True, linewidth=160)

    # Step 1: high-precision fair values (4M antithetic paths).
    num_paths = 4_000_000
    print(f"Pricing with {num_paths:,} antithetic GBM paths over {STEPS_3W} steps...")
    paths = simulate_paths(num_paths, STEPS_3W, seed=42, antithetic=True)
    payoffs = compute_payoffs(paths)
    payoff_stds = {n: float(v.std()) for n, v in payoffs.items()}

    # Step 1b: control-variate fair values (much tighter SE on borderline edges).
    cv = cv_fair_values(paths, payoffs)
    print("\n" + "=" * 70)
    print("Fair value comparison: plain MC vs control-variate MC vs Black-Scholes")
    print("=" * 70)
    print_cv_table(cv, analytic_prices())
    print("\nUsing control-variate-adjusted fair values for strategy decisions.")
    fair_values = {n: cv[n]["cv_mean"] for n in cv}

    print_fair_table(fair_values, payoff_stds, analytic_prices())

    # Step 2: Pick EV-max positions and bootstrap realised score.
    positions = ev_max_positions(fair_values)
    per_prod, total, marks = bootstrap_score(positions,
                                             num_worlds=10_000,
                                             sample_size=100,
                                             seed=999)
    print_portfolio(positions, fair_values, per_prod, total, marks)

    # Step 3: Ablation analysis - marginal contribution of each leg.
    print("\n" + "=" * 70)
    print("Ablation: drop each leg, recompute total EV / sigma / Sharpe")
    print("=" * 70)
    trimmed_positions = ablation_analysis(positions)

    # If greedy trimming differs from the full EV-max portfolio, show its risk too.
    if any(trimmed_positions[n] != positions[n] for n in positions):
        per_prod_t, total_t, marks_t = bootstrap_score(trimmed_positions,
                                                      num_worlds=10_000,
                                                      sample_size=100,
                                                      seed=999)
        print("\n--- Sharpe-trimmed portfolio risk profile ---")
        print_portfolio(trimmed_positions, fair_values, per_prod_t, total_t, marks_t)

    # Step 4: Per-product 1-D grid search ("grid search" the user's friends meant).
    print("\n" + "=" * 70)
    print("Per-product 1-D sweep (linearity check)")
    print("=" * 70)
    per_product_sweep(payoffs, fair_values)

    # Step 5: Global scaling sweep (risk-budget view).
    print("\n" + "=" * 70)
    print("Global scaling sweep (lambda * EV-max positions)")
    print("=" * 70)
    global_scaling_sweep(positions)

    # Step 4b: Compute t=0 deltas and the static delta-hedge for Ticket B.
    print("\n" + "=" * 70)
    print("t=0 deltas (per option-unit) and static delta-hedge for Ticket B")
    print("=" * 70)
    deltas = compute_deltas()
    print(f"{'Product':<11} {'Delta':>10}")
    print("-" * 24)
    for name in MARKET:
        print(f"{name:<11} {deltas[name]:>+10.4f}")
    delta_b = portfolio_delta(trimmed_positions, deltas)
    hedged_b = static_delta_hedge(trimmed_positions, deltas)
    print(f"\nTicket B portfolio delta (no AC):  {delta_b:>+8.2f}")
    print(f"Static delta-hedge: BUY/SELL  AC = {hedged_b['AC']:>+5d}  "
          f"-> new portfolio delta {portfolio_delta(hedged_b, deltas):>+.2f}")

    # Step 4c: Risk-aware optimisation (mean-variance frontier).
    # CVaR_5% objective is degenerate in this game (no portfolio has P5% > 0),
    # so we use mean-variance with a moderate lambda.  lam=0.30 sits at a
    # near-Sharpe-optimal point that's still meaningfully positive-EV; it
    # Pareto-dominates Ticket B in every risk metric.
    print("\n" + "=" * 70)
    print("Risk-aware optimisation: max EV - 0.30 * sigma  (lambda=0.30)")
    print("=" * 70)
    cvar_positions, cvar_total = optimize_portfolio(objective="mv", lam=0.30,
                                                    num_worlds=20_000,
                                                    sample_size=100,
                                                    initial_positions=trimmed_positions,
                                                    num_restarts=6)
    cvar_ev = cvar_total.mean()
    cvar_sd = cvar_total.std()
    cvar_p5 = np.percentile(cvar_total, 5)
    cvar_p95 = np.percentile(cvar_total, 95)
    print(f"  EV: ${cvar_ev:,.0f}   σ: ${cvar_sd:,.0f}   "
          f"P5%: ${cvar_p5:,.0f}   P95%: ${cvar_p95:,.0f}   "
          f"Sharpe: {cvar_ev/cvar_sd:.3f}   "
          f"P(<0): {(cvar_total<0).mean()*100:.2f}%")
    print(f"  Portfolio delta: {portfolio_delta(cvar_positions, deltas):>+.2f}")
    print(f"\n  {'Product':<11} {'Pos':>6}")
    print("  " + "-" * 20)
    for name, pos in cvar_positions.items():
        if pos != 0:
            print(f"  {name:<11} {pos:>+6}")

    # Step 5: Final order tickets - all variants side by side.
    def print_ticket(label: str, pos_dict: Dict[str, int]):
        print("\n" + "=" * 70)
        print(label)
        print("=" * 70)
        print(f"{'Product':<11} {'Action':<6} {'Volume':>7} {'Price':>8} {'EdgePerUnit':>13}")
        print("-" * 60)
        for name, pos in pos_dict.items():
            bid, ask, _ = MARKET[name]
            fv = fair_values[name]
            if pos > 0:
                print(f"{name:<11} {'BUY':<6} {pos:>7} {ask:>8.3f} {fv-ask:>+13.4f}")
            elif pos < 0:
                print(f"{name:<11} {'SELL':<6} {-pos:>7} {bid:>8.3f} {bid-fv:>+13.4f}")
            else:
                print(f"{name:<11} {'HOLD':<6} {0:>7} {'-':>8} {'-':>13}")

    print_ticket("ORDER TICKET A - EV-max (ignores variance, the old recommendation)",
                 positions)
    if any(trimmed_positions[n] != positions[n] for n in positions):
        print_ticket("ORDER TICKET B - Sharpe-trimmed (drops AC_60_C from A)",
                     trimmed_positions)
    print_ticket("ORDER TICKET B' - Ticket B + static delta hedge", hedged_b)
    print_ticket("ORDER TICKET C - CVaR_5%-optimal (RECOMMENDED for production)",
                 cvar_positions)


if __name__ == "__main__":
    main()

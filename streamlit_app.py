import io
import math
import os
import subprocess
import sys
from pathlib import Path
import time
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import tensorflow as tf
import torch
import yfinance as yf
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats
from scipy.interpolate import griddata
from scipy.linalg import lu_factor, lu_solve
from scipy.stats import norm
from typing import Callable

from Longstaff.option import Option
from Longstaff.pricing import (
    black_scholes_merton,
    crr_pricing,
    monte_carlo_simulation,
)
from Longstaff.process import GeometricBrownianMotion, HestonProcess
from Lookback.european_call import european_call_option
from Lookback.lookback_call import lookback_call_option
from Heston.heston_torch import HestonParams, carr_madan_call_torch

torch.set_default_dtype(torch.float64)
HES_DEVICE = torch.device("cpu")
MIN_IV_MATURITY = 0.1


def simulate_gbm_paths(S0, r, q, sigma, T, M, N_paths, seed=42):
    """
    Simulate GBM paths under the risk-neutral measure:
        dS_t = (r - q) S_t dt + sigma S_t dW_t
    Returns:
        S : array of shape (M+1, N_paths)
        dt: time step
    """
    dt = T / M
    rng = np.random.default_rng(seed)
    S = np.empty((M + 1, N_paths))
    S[0, :] = S0
    Z = rng.normal(size=(M, N_paths))
    drift = (r - q - 0.5 * sigma**2) * dt
    vol_term = sigma * np.sqrt(dt)
    for t in range(1, M + 1):
        S[t, :] = S[t - 1, :] * np.exp(drift + vol_term * Z[t - 1, :])
    return S, dt


def price_bermudan_lsmc(
    S0,
    K,
    T,
    r,
    q,
    sigma,
    cpflag="p",
    M=50,
    N_paths=100_000,
    degree=3,
    n_ex_dates=6,
    seed: int = 42,
):
    """
    Longstaff–Schwartz Monte Carlo pricing for a Bermudan option
    under risk-neutral GBM (Black–Scholes dynamics).
    """
    S, dt = simulate_gbm_paths(S0, r, q, sigma, T, M, N_paths, seed=seed)
    disc = np.exp(-r * dt)

    if cpflag == "c":
        Y = np.maximum(S - K, 0.0)
    elif cpflag == "p":
        Y = np.maximum(K - S, 0.0)
    else:
        raise ValueError("cpflag must be 'c' or 'p'")

    C = Y[-1, :].copy()
    ex_indices = np.linspace(1, M - 1, max(1, n_ex_dates - 1), dtype=int)
    ex_set = set(ex_indices.tolist())

    for j in range(M - 1, 0, -1):
        C *= disc
        if j in ex_set:
            S_j = S[j, :]
            Y_j = Y[j, :]
            itm = Y_j > 0.0
            if np.any(itm):
                X = np.vstack([S_j[itm] ** k for k in range(degree + 1)]).T
                y_reg = C[itm]
                beta, *_ = np.linalg.lstsq(X, y_reg, rcond=None)
                X_all = np.vstack([S_j**k for k in range(degree + 1)]).T
                C_hat = X_all @ beta
                exercise = (Y_j > C_hat) & itm
                C[exercise] = Y_j[exercise]

    C *= disc
    price = np.mean(C)
    return float(price)


# ---------------------------------------------------------------------------
#  Bermudan / European / American + Barrier (Crank–Nicolson)
# ---------------------------------------------------------------------------

class CrankNicolsonBS:
    """
    Solveur Crank–Nicolson pour la PDE de Black–Scholes en log(S).

    Typeflag:
        'Eu'  : option européenne
        'Am'  : option américaine (exercice possible à chaque date de grille)
        'Bmd' : option bermudéenne (exercice possible à certaines dates)
    cpflag:
        'c' : call
        'p' : put
    """

    def __init__(
        self,
        Typeflag: str,
        cpflag: str,
        S0: float,
        K: float,
        T: float,
        vol: float,
        r: float,
        d: float,
        n_spatial: int = 500,
        n_time: int = 600,
        exercise_step: int | None = None,
        n_exercise_dates: int | None = None,
        **_,
    ) -> None:
        self.Typeflag = Typeflag
        self.cpflag = cpflag
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.vol = float(vol)
        self.r = float(r)
        self.d = float(d)

        self.n_spatial = max(50, int(n_spatial))
        self.n_time = max(50, int(n_time))

        # Deux modes possibles pour la Bermudane :
        # - exercise_step       : exercice tous les 'exercise_step' pas
        # - n_exercise_dates    : nb de dates d'exercice (incluant T)
        # Si les deux sont donnés -> erreur, c'est ambigu.
        if exercise_step is not None and n_exercise_dates is not None:
            raise ValueError(
                "Spécifie soit exercise_step, soit n_exercise_dates, pas les deux."
            )

        self.exercise_step = int(exercise_step) if exercise_step is not None else None
        self.n_exercise_dates = (
            int(n_exercise_dates) if n_exercise_dates is not None else None
        )

    # -------------------- utils --------------------

    def _resolve_params(
        self,
        Typeflag: str | None,
        cpflag: str | None,
        S0: float | None,
        K: float | None,
        T: float | None,
        vol: float | None,
        r: float | None,
        d: float | None,
    ):
        """Résout les paramètres effectifs sans casser les valeurs 0 éventuelles."""

        Typeflag = self.Typeflag if Typeflag is None else Typeflag
        cpflag = self.cpflag if cpflag is None else cpflag
        S0 = self.S0 if S0 is None else float(S0)
        K = self.K if K is None else float(K)
        T = self.T if T is None else float(T)
        vol = self.vol if vol is None else float(vol)
        r = self.r if r is None else float(r)
        d = self.d if d is None else float(d)
        return Typeflag, cpflag, S0, K, T, vol, r, d

    # -------------------- solveur CN --------------------

    def CN_option_info(
        self,
        Typeflag: str | None = None,
        cpflag: str | None = None,
        S0: float | None = None,
        K: float | None = None,
        T: float | None = None,
        vol: float | None = None,
        r: float | None = None,
        d: float | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Résout la PDE et retourne (Price, Delta, Gamma, Theta).
        """

        Typeflag, cpflag, S0, K, T, vol, r, d = self._resolve_params(
            Typeflag, cpflag, S0, K, T, vol, r, d
        )

        Typeflag = Typeflag.strip()
        cpflag = cpflag.strip()
        if Typeflag not in {"Eu", "Am", "Bmd"}:
            raise ValueError("Typeflag doit être 'Eu', 'Am' ou 'Bmd'.")
        if cpflag not in {"c", "p"}:
            raise ValueError("cpflag doit être 'c' ou 'p'.")

        # Cas trivial T=0
        if T <= 0.0 or self.n_time <= 0:
            payoff0 = max(S0 - K, 0.0) if cpflag == "c" else max(K - S0, 0.0)
            return float(payoff0), 0.0, 0.0, 0.0

        if Typeflag == "Bmd":
            M_lsmc = max(1, min(self.n_time, 50))
            N_paths = 50_000
            n_ex_dates = self.n_exercise_dates or 6
            seed_base = 12345

            def _lsmc_price(s0_val: float, t_val: float) -> float:
                return price_bermudan_lsmc(
                    S0=s0_val,
                    K=K,
                    T=max(t_val, 1e-6),
                    r=r,
                    q=d,
                    sigma=vol,
                    cpflag=cpflag,
                    M=M_lsmc,
                    N_paths=N_paths,
                    degree=3,
                    n_ex_dates=n_ex_dates,
                    seed=seed_base,
                )

            price_bmd = _lsmc_price(S0, T)

            bump_s = max(1e-4, 0.01 * S0)
            price_up = _lsmc_price(S0 + bump_s, T)
            price_down = _lsmc_price(max(S0 - bump_s, 1e-6), T)
            delta = (price_up - price_down) / (2.0 * bump_s)
            gamma = (price_up - 2.0 * price_bmd + price_down) / (bump_s**2)

            theta = 0.0
            theta_h = min(max(1.0 / 365.0, 0.01 * T), max(T / 2.0, 1e-6))
            if T > theta_h:
                price_short = _lsmc_price(S0, T - theta_h)
                theta = (price_short - price_bmd) / theta_h

            return float(price_bmd), float(delta), float(gamma), float(theta)

        # ----- Grille en log(S) -----
        mu = r - d - 0.5 * vol * vol
        x_max = vol * np.sqrt(max(T, 1e-8)) * 5.0
        n_points = self.n_spatial
        dx = 2.0 * x_max / n_points

        X = np.linspace(-x_max, x_max, n_points + 1)
        max_log = np.log(np.finfo(float).max / max(S0, 1e-12))
        X_clipped = np.clip(X, -max_log, max_log)
        s_grid = S0 * np.exp(X_clipped)

        n_index = np.arange(0, n_points + 1)

        n_time = self.n_time
        dt = T / n_time

        a = 0.25 * dt * ((vol**2) * (n_index**2) - mu * n_index)
        b = -0.5 * dt * ((vol**2) * (n_index**2) + r)
        c = 0.25 * dt * ((vol**2) * (n_index**2) + mu * n_index)

        main_diag_A = 1.0 - b - 2.0 * a
        upper_A = a + c
        lower_A = a - c

        main_diag_B = 1.0 + b + 2.0 * a
        upper_B = -a - c
        lower_B = -a + c

        A = np.zeros((n_points + 1, n_points + 1))
        B = np.zeros((n_points + 1, n_points + 1))

        np.fill_diagonal(A, main_diag_A)
        np.fill_diagonal(A[1:], lower_A[:-1])
        np.fill_diagonal(A[:, 1:], upper_A[:-1])
        A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)

        np.fill_diagonal(B, main_diag_B)
        np.fill_diagonal(B[1:], lower_B[:-1])
        np.fill_diagonal(B[:, 1:], upper_B[:-1])
        B = np.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)

        lu_factor_A = lu_factor(A)

        # Payoff terminal
        if cpflag == "c":
            values = np.maximum(s_grid - K, 0.0)
        else:
            values = np.maximum(K - s_grid, 0.0)

        payoff = values.copy()
        values_prev_time = values.copy()

        S_max = s_grid[-1]
        S_min = s_grid[0]  # pas utilisé mais dispo si besoin

        # ----- Boucle backward -----
        for time_index in range(n_time):
            # Sauvegarde pour theta (un seul pas suffit)
            if time_index == n_time - 1:
                values_prev_time = values.copy()

            rhs = B.dot(values)
            values = lu_solve(lu_factor_A, rhs)

            t_now = T - (time_index + 1) * dt
            tau = T - t_now  # temps restant à maturité

            # Conditions aux bords
            if cpflag == "c":
                values[0] = 0.0
                values[-1] = S_max - K * np.exp(-r * tau)
            else:
                values[0] = K * np.exp(-r * tau)
                values[-1] = 0.0

            # Gestion du style
            if Typeflag == "Am":
                values = np.maximum(values, payoff)
            elif Typeflag == "Eu":
                pass

        # ----- Grecs par différences finies -----
        middle_index = n_points // 2
        price = values[middle_index]

        s_plus = S0 * np.exp(dx)
        s_minus = S0 * np.exp(-dx)

        v_plus = values[middle_index + 1]
        v_0 = values[middle_index]
        v_minus = values[middle_index - 1]

        delta = (v_plus - v_minus) / (s_plus - s_minus)

        dVdS_plus = (v_plus - v_0) / (s_plus - S0)
        dVdS_minus = (v_0 - v_minus) / (S0 - s_minus)
        gamma = (dVdS_plus - dVdS_minus) / ((s_plus - s_minus) / 2.0)

        theta = -(values[middle_index] - values_prev_time[middle_index]) / dt

        return float(price), float(delta), float(gamma), float(theta)


def CN_Barrier_option(Typeflag, cpflag, S0, K, Hu, Hd, T, vol, r, d):
    """
    Pricing d'une option barrière par Crank–Nicolson.
    """

    mu = r - d - 0.5 * vol * vol
    x_max = vol * np.sqrt(T) * 5
    n_points = 500
    dx = 2 * x_max / n_points
    X = np.linspace(-x_max, x_max, n_points + 1)
    n_index = np.arange(0, n_points + 1)

    n_time = 600
    dt = T / n_time

    a = 0.25 * dt * ((vol**2) * (n_index**2) - mu * n_index)
    b = -0.5 * dt * ((vol**2) * (n_index**2) + r)
    c = 0.25 * dt * ((vol**2) * (n_index**2) + mu * n_index)

    main_diag_A = 1 - b - 2 * a
    upper_A = a + c
    lower_A = a - c

    main_diag_B = 1 + b + 2 * a
    upper_B = -a - c
    lower_B = -a + c

    A = np.zeros((n_points + 1, n_points + 1))
    B = np.zeros((n_points + 1, n_points + 1))

    np.fill_diagonal(A, main_diag_A)
    np.fill_diagonal(A[1:], lower_A[:-1])
    np.fill_diagonal(A[:, 1:], upper_A[:-1])

    np.fill_diagonal(B, main_diag_B)
    np.fill_diagonal(B[1:], lower_B[:-1])
    np.fill_diagonal(B[:, 1:], upper_B[:-1])

    Ainv = np.linalg.inv(A)

    s_grid = S0 * np.exp(X)
    if cpflag == "c":
        values = np.clip(s_grid - K, 0, 1e10)
    elif cpflag == "p":
        values = np.clip(K - s_grid, 0, 1e10)
    else:
        raise ValueError("cpflag doit être 'c' ou 'p'.")

    typeflag = Typeflag.upper()
    if typeflag in {"UNO", "UO"}:
        values = np.where(s_grid < Hu, values, 0.0)
    elif typeflag == "DNO":
        values = np.where((s_grid > Hd) & (s_grid < Hu), values, 0.0)
    elif typeflag in {"DO"}:
        values = np.where(s_grid > Hd, values, 0.0)
    else:
        raise ValueError("Typeflag doit être 'UNO', 'UO', 'DO' ou 'DNO'.")

    values_prev_time = values.copy()

    for time_index in range(n_time):
        if time_index == n_time - 1:
            values_prev_time = values.copy()

        values = B.dot(values)
        values = Ainv.dot(values)

        s_grid = S0 * np.exp(X)
        if typeflag in {"UNO", "UO"}:
            values = np.where(s_grid < Hu, values, 0.0)
        elif typeflag == "DNO":
            values = np.where((s_grid > Hd) & (s_grid < Hu), values, 0.0)
        elif typeflag == "DO":
            values = np.where(s_grid > Hd, values, 0.0)

    middle_index = n_points // 2
    price = values[middle_index]

    s_plus = S0 * np.exp(dx)
    s_minus = S0 * np.exp(-dx)

    delta = (values[middle_index + 1] - values[middle_index - 1]) / (s_plus - s_minus)

    d_value_d_s_plus = (values[middle_index + 1] - values[middle_index]) / (s_plus - S0)
    d_value_d_s_minus = (values[middle_index] - values[middle_index - 1]) / (S0 - s_minus)
    gamma = (d_value_d_s_plus - d_value_d_s_minus) / ((s_plus - s_minus) / 2.0)

    theta = -(values[middle_index] - values_prev_time[middle_index]) / dt

    return float(price), float(delta), float(gamma), float(theta)


# ---------------------------------------------------------------------------
#  Helper Longstaff–Schwartz qui retourne le prix (version locale)
# ---------------------------------------------------------------------------


def longstaff_schwartz_price(option: Option, process, n_paths: int, n_steps: int) -> float:
    """
    Implémentation locale de l'algorithme LS, basée sur Longstaff/pricing.py,
    mais qui renvoie le prix comme float.
    """
    from numpy.polynomial import Polynomial

    simulated_paths = process.simulate(s0=option.s0, v0=option.v0, T=option.T, n=n_paths, m=n_steps)
    payoffs = option.payoff(s=simulated_paths)

    continuation_values = np.zeros_like(payoffs)
    continuation_values[-1] = payoffs[-1]

    dt = option.T / n_steps
    discount = np.exp(-process.mu * dt)

    for time_index in range(n_steps - 1, 0, -1):
        polynomial = Polynomial.fit(simulated_paths[time_index], discount * continuation_values[time_index + 1], 5)
        continuation = polynomial(simulated_paths[time_index])
        continuation_values[time_index] = np.where(
            payoffs[time_index] > continuation,
            payoffs[time_index],
            discount * continuation_values[time_index + 1],
        )

    price = discount * np.mean(continuation_values[1])
    return float(price)


# ---------------------------------------------------------------------------
#  Outils pour les heatmaps européennes
# ---------------------------------------------------------------------------

HEATMAP_GRID_SIZE = 11


def _heatmap_axis(center: float, span: float, n_points: int = HEATMAP_GRID_SIZE) -> np.ndarray:
    lower = max(0.01, center - span)
    upper = max(lower, center + span)
    if np.isclose(lower, upper) or n_points == 1:
        return np.array([lower])
    return np.linspace(lower, upper, n_points)


def _render_heatmap(
    matrix: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    xlabel: str = "Spot",
    ylabel: str = "Strike",
) -> None:
    fig, ax = plt.subplots()
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
        cmap="viridis",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)


def _render_call_put_heatmaps(
    label: str, call_matrix: np.ndarray, put_matrix: np.ndarray, x_values: np.ndarray, y_values: np.ndarray
) -> None:
    col_call, col_put = st.columns(2)
    with col_call:
        st.write(f"Heatmap Call ({label})")
        _render_heatmap(call_matrix, x_values, y_values, f"Call ({label})")
    with col_put:
        st.write(f"Heatmap Put ({label})")
        _render_heatmap(put_matrix, x_values, y_values, f"Put ({label})")


def _compute_bsm_heatmaps(
    s_values: np.ndarray, k_values: np.ndarray, maturity: float, rate: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    call_matrix = np.zeros((len(k_values), len(s_values)))
    put_matrix = np.zeros_like(call_matrix)
    for i, strike in enumerate(k_values):
        for j, spot in enumerate(s_values):
            option_call = Option(s0=spot, T=maturity, K=strike, call=True)
            option_put = Option(s0=spot, T=maturity, K=strike, call=False)
            call_matrix[i, j] = black_scholes_merton(r=rate, sigma=sigma, option=option_call)
            put_matrix[i, j] = black_scholes_merton(r=rate, sigma=sigma, option=option_put)
    return call_matrix, put_matrix


def _compute_mc_heatmaps(
    s_values: np.ndarray,
    k_values: np.ndarray,
    maturity: float,
    mu: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    process = GeometricBrownianMotion(mu=mu, sigma=sigma)
    discount = np.exp(-mu * maturity)
    call_matrix = np.zeros((len(k_values), len(s_values)))
    put_matrix = np.zeros_like(call_matrix)

    for j, spot in enumerate(s_values):
        simulated_paths = process.simulate(s0=spot, T=maturity, n=n_paths, m=n_steps, v0=None)
        terminal_prices = simulated_paths[-1]
        for i, strike in enumerate(k_values):
            call_matrix[i, j] = np.mean(np.maximum(terminal_prices - strike, 0)) * discount
            put_matrix[i, j] = np.mean(np.maximum(strike - terminal_prices, 0)) * discount

    return call_matrix, put_matrix


def _vanilla_price_with_dividend(
    option_type: str,
    S0: float,
    K: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> float:
    if T <= 0 or sigma <= 0 or K <= 0 or S0 <= 0:
        intrinsic = max(S0 - K, 0.0) if option_type.lower() in {"call", "c"} else max(K - S0, 0.0)
        return float(intrinsic)
    sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - dividend + 0.5 * sigma * sigma) * T) / sqrt_T
    d2 = d1 - sqrt_T
    if option_type.lower() in {"call", "c"}:
        price = S0 * np.exp(-dividend * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-dividend * T) * norm.cdf(-d1)
    return float(max(price, 0.0))


def _barrier_closed_form_price(
    option_type: str,
    barrier_type: str,
    S0: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> float:
    if barrier <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Paramètres invalides pour la formule fermée barrière.")
    if barrier_type == "up" and S0 >= barrier:
        return 0.0
    if barrier_type == "down" and S0 <= barrier:
        return 0.0

    option_flag = option_type.lower()
    phi = 1.0 if option_flag in {"call", "c"} else -1.0
    eta = 1.0 if barrier_type == "down" else -1.0
    mu = (r - dividend - 0.5 * sigma * sigma) / (sigma * sigma)
    sigma_sqrt_T = sigma * np.sqrt(T)
    if sigma_sqrt_T == 0:
        return 0.0
    x1 = (np.log(S0 / K) / sigma_sqrt_T) + (1.0 + mu) * sigma_sqrt_T
    y1 = (np.log((barrier * barrier) / (S0 * K)) / sigma_sqrt_T) + (1.0 + mu) * sigma_sqrt_T
    power1 = (barrier / S0) ** (2.0 * (mu + eta))
    power2 = (barrier / S0) ** (2.0 * mu)
    term1 = phi * S0 * np.exp(-dividend * T) * (norm.cdf(phi * x1) - power1 * norm.cdf(eta * y1))
    term2 = phi * K * np.exp(-r * T) * (norm.cdf(phi * x1 - phi * sigma_sqrt_T) - power2 * norm.cdf(eta * y1 - eta * sigma_sqrt_T))
    price = term1 - term2
    return max(float(price), 0.0)
 

def _knock_in_closed_form_price(
    option_type: str,
    barrier_type: str,
    S0: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> float:
    vanilla = _vanilla_price_with_dividend(
        option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma
    )
    barrier_out_price = _barrier_closed_form_price(
        option_type=option_type,
        barrier_type=barrier_type,
        S0=S0,
        K=K,
        barrier=barrier,
        T=T,
        r=r,
        dividend=dividend,
        sigma=sigma,
    )
    return max(vanilla - barrier_out_price, 0.0)


def _barrier_monte_carlo_price(
    option_type: str,
    barrier_type: str,
    S0: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    knock_in: bool = False,
) -> float:
    if barrier <= 0 or n_paths <= 0 or n_steps <= 0:
        raise ValueError("Paramètres invalides pour le Monte Carlo barrière.")
    option_type_lower = option_type.lower()
    if barrier_type == "up" and S0 >= barrier:
        if knock_in:
            return _vanilla_price_with_dividend(option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma)
        return 0.0
    if barrier_type == "down" and S0 <= barrier:
        if knock_in:
            return _vanilla_price_with_dividend(option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma)
        return 0.0
    dt = T / n_steps
    drift = (r - dividend - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)
    payoffs = []
    for _ in range(n_paths):
        s = S0
        barrier_hit = False
        for _ in range(n_steps):
            z = np.random.normal()
            s *= np.exp(drift + diffusion * z)
            if barrier_type == "up" and s >= barrier:
                barrier_hit = True
                if not knock_in:
                    break
            elif barrier_type == "down" and s <= barrier:
                barrier_hit = True
                if not knock_in:
                    break
        if knock_in and not barrier_hit:
            payoffs.append(0.0)
            continue
        if not knock_in and barrier_hit:
            payoffs.append(0.0)
            continue
        if option_type_lower in {"call", "c"}:
            payoff = max(s - K, 0.0)
        else:
            payoff = max(K - s, 0.0)
        payoffs.append(payoff)
    return discount * (float(np.mean(payoffs)) if payoffs else 0.0)


def _compute_barrier_heatmap_matrix(
    option_type: str,
    barrier_type: str,
    strike_values: np.ndarray,
    offset_values: np.ndarray,
    S0: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(strike_values), len(offset_values)))
    ratio_axis = np.zeros(len(offset_values))

    for j, offset in enumerate(offset_values):
        if barrier_type == "up":
            ratio = 1.1 + offset
        else:
            ratio = max(0.01, 0.9 - offset)
        ratio_axis[j] = ratio

        for i, strike in enumerate(strike_values):
            barrier = max(strike * ratio, 0.01)
            try:
                price = _barrier_closed_form_price(
                    option_type=option_type,
                    barrier_type=barrier_type,
                    S0=S0,
                    K=float(strike),
                    barrier=float(barrier),
                    T=T,
                    r=r,
                    dividend=dividend,
                    sigma=sigma,
                )
            except ValueError:
                price = 0.0
            matrix[i, j] = price

    if np.any(np.diff(ratio_axis) < 0):
        order = np.argsort(ratio_axis)
        ratio_axis = ratio_axis[order]
        matrix = matrix[:, order]

    return matrix, ratio_axis


def _compute_up_and_out_strike_heatmap(
    option_type: str,
    barrier: float,
    strike_values: np.ndarray,
    maturity_values: np.ndarray,
    spot: float,
    r: float,
    dividend: float,
    sigma: float,
) -> np.ndarray:
    """
    Construit une matrice de prix up-and-out selon (T, K) pour un spot fixe.
    """
    matrix = np.zeros((len(maturity_values), len(strike_values)))
    for i, maturity in enumerate(maturity_values):
        for j, strike in enumerate(strike_values):
            if strike <= 0.0:
                matrix[i, j] = 0.0
                continue
            try:
                price = _barrier_closed_form_price(
                    option_type=option_type,
                    barrier_type="up",
                    S0=float(spot),
                    K=float(strike),
                    barrier=float(barrier),
                    T=float(maturity),
                    r=r,
                    dividend=dividend,
                    sigma=sigma,
                )
            except ValueError:
                price = 0.0
            matrix[i, j] = price
    return matrix


def _compute_up_and_in_strike_heatmap(
    option_type: str,
    barrier: float,
    strike_values: np.ndarray,
    maturity_values: np.ndarray,
    spot: float,
    r: float,
    dividend: float,
    sigma: float,
) -> np.ndarray:
    matrix = np.zeros((len(maturity_values), len(strike_values)))
    for i, maturity in enumerate(maturity_values):
        for j, strike in enumerate(strike_values):
            if strike <= 0.0:
                matrix[i, j] = 0.0
                continue
            vanilla = _vanilla_price_with_dividend(option_type, spot, float(strike), float(maturity), r, dividend, sigma)
            try:
                barrier_out = _barrier_closed_form_price(
                    option_type=option_type,
                    barrier_type="up",
                    S0=float(spot),
                    K=float(strike),
                    barrier=float(barrier),
                    T=float(maturity),
                    r=r,
                    dividend=dividend,
                    sigma=sigma,
                )
            except ValueError:
                matrix[i, j] = 0.0
                continue
            matrix[i, j] = max(vanilla - barrier_out, 0.0)
    return matrix


def _compute_lookback_exact_heatmap(
    s_values: np.ndarray,
    t_values: np.ndarray,
    t_current: float,
    rate: float,
    sigma: float,
) -> np.ndarray:
    matrix = np.zeros((len(t_values), len(s_values)))
    for i, maturity in enumerate(t_values):
        for j, spot in enumerate(s_values):
            lookback_opt = lookback_call_option(
                T=float(maturity), t=float(t_current), S0=float(spot), r=float(rate), sigma=float(sigma)
            )
            matrix[i, j] = lookback_opt.price_exact()
    return matrix


def _compute_lookback_mc_heatmap(
    s_values: np.ndarray,
    t_values: np.ndarray,
    t_current: float,
    rate: float,
    sigma: float,
    n_iters: int,
) -> np.ndarray:
    matrix = np.zeros((len(t_values), len(s_values)))
    for i, maturity in enumerate(t_values):
        for j, spot in enumerate(s_values):
            lookback_opt = lookback_call_option(
                T=float(maturity), t=float(t_current), S0=float(spot), r=float(rate), sigma=float(sigma)
            )
            matrix[i, j] = lookback_opt.price_monte_carlo(n_iters)
    return matrix


def _compute_down_in_heatmap(
    option_type: str,
    strike_values: np.ndarray,
    offset_values: np.ndarray,
    S0: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_out, ratio_axis = _compute_barrier_heatmap_matrix(
        option_type=option_type,
        barrier_type="down",
        strike_values=strike_values,
        offset_values=offset_values,
        S0=S0,
        T=T,
        r=r,
        dividend=dividend,
        sigma=sigma,
    )
    matrix_in = np.zeros_like(matrix_out)
    for i, strike in enumerate(strike_values):
        vanilla = _vanilla_price_with_dividend(
            option_type=option_type, S0=S0, K=float(strike), T=T, r=r, dividend=dividend, sigma=sigma
        )
        matrix_in[i, :] = np.maximum(vanilla - matrix_out[i, :], 0.0)
    return matrix_in, ratio_axis


def _compute_american_ls_heatmaps(
    s_values: np.ndarray,
    k_values: np.ndarray,
    maturity: float,
    process,
    n_paths: int,
    n_steps: int,
    v0=None,
) -> tuple[np.ndarray, np.ndarray]:
    call_matrix = np.zeros((len(k_values), len(s_values)))
    put_matrix = np.zeros_like(call_matrix)
    for i, strike in enumerate(k_values):
        for j, spot in enumerate(s_values):
            option_call = Option(s0=spot, T=maturity, K=strike, v0=v0, call=True)
            option_put = Option(s0=spot, T=maturity, K=strike, v0=v0, call=False)
            call_matrix[i, j] = longstaff_schwartz_price(option_call, process, n_paths, n_steps)
            put_matrix[i, j] = longstaff_schwartz_price(option_put, process, n_paths, n_steps)
    return call_matrix, put_matrix


def _compute_american_crr_heatmaps(
    s_values: np.ndarray,
    k_values: np.ndarray,
    maturity: float,
    rate: float,
    sigma: float,
    n_tree: int,
) -> tuple[np.ndarray, np.ndarray]:
    call_matrix = np.zeros((len(k_values), len(s_values)))
    put_matrix = np.zeros_like(call_matrix)
    for i, strike in enumerate(k_values):
        for j, spot in enumerate(s_values):
            option_call = Option(s0=spot, T=maturity, K=strike, call=True)
            option_put = Option(s0=spot, T=maturity, K=strike, call=False)
            call_matrix[i, j] = crr_pricing(r=rate, sigma=sigma, option=option_call, n=n_tree)
            put_matrix[i, j] = crr_pricing(r=rate, sigma=sigma, option=option_put, n=n_tree)
    return call_matrix, put_matrix


def _build_crr_tree(option: Option, r: float, sigma: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    if n_steps <= 0:
        raise ValueError("n_steps doit être supérieur à 0.")
    dt = option.T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    a = np.exp(r * dt)
    p = (a - d) / (u - d)
    q = 1 - p

    spot_tree = np.full((n_steps + 1, n_steps + 1), np.nan)
    value_tree = np.full_like(spot_tree, np.nan)

    for level in range(n_steps + 1):
        for up_moves in range(level + 1):
            spot_tree[level, up_moves] = option.s0 * (u**up_moves) * (d ** (level - up_moves))

    payoff_last = option.payoff(spot_tree[n_steps, : n_steps + 1])
    value_tree[n_steps, : n_steps + 1] = payoff_last
    discount = np.exp(-r * dt)

    for level in range(n_steps - 1, -1, -1):
        for up_moves in range(level + 1):
            continuation = discount * (
                p * value_tree[level + 1, up_moves + 1] + q * value_tree[level + 1, up_moves]
            )
            exercise = option.payoff(np.array([spot_tree[level, up_moves]]))[0]
            value_tree[level, up_moves] = max(exercise, continuation)

    return spot_tree, value_tree


def _format_tree_matrix(matrix: np.ndarray, precision: int = 4) -> np.ndarray:
    fmt = f"{{:.{precision}f}}"
    formatted = []
    for row in matrix:
        formatted.append([fmt.format(value) if not np.isnan(value) else "" for value in row])
    return np.array(formatted)


def _plot_crr_tree(spots: np.ndarray, values: np.ndarray) -> plt.Figure:
    n_levels = spots.shape[0]
    fig_width = min(12, 4 + n_levels * 0.25)
    fig_height = min(10, 3 + n_levels * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()

    def _node_coords(level: int, index: int) -> tuple[float, float]:
        x = index - level / 2
        y = n_levels - 1 - level
        return x, y

    for level in range(n_levels - 1):
        for index in range(level + 1):
            if np.isnan(spots[level, index]):
                continue
            x_curr, y_curr = _node_coords(level, index)
            x_down, y_down = _node_coords(level + 1, index)
            x_up, y_up = _node_coords(level + 1, index + 1)
            ax.plot([x_curr, x_down], [y_curr, y_down], color="lightgray", linewidth=0.8)
            ax.plot([x_curr, x_up], [y_curr, y_up], color="lightgray", linewidth=0.8)

    x_coords = []
    y_coords = []
    color_values = []
    spots_list = []
    option_list = []

    for level in range(n_levels):
        for index in range(level + 1):
            value = spots[level, index]
            option_value = values[level, index]
            if np.isnan(value) or np.isnan(option_value):
                continue
            x, y = _node_coords(level, index)
            x_coords.append(x)
            y_coords.append(y)
            color_values.append(option_value)
            spots_list.append(value)
            option_list.append(option_value)

    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=color_values,
        cmap="viridis",
        s=120,
        edgecolors="black",
        linewidths=0.5,
    )
    display_labels = n_levels - 1 <= 10
    if display_labels:
        for x, y, spot_value, option_value in zip(x_coords, y_coords, spots_list, option_list):
            ax.text(x, y + 0.25, f"S={spot_value:.2f}", ha="center", va="bottom", fontsize=7)
            ax.text(x, y - 0.25, f"V={option_value:.2f}", ha="center", va="top", fontsize=7)

    ax.set_ylim(-0.5, n_levels - 0.5)
    ax.set_xlim(min(x_coords, default=-1) - 1, max(x_coords, default=1) + 1)
    ax.set_title("Arbre CRR (couleur = valeur de l'option)")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Valeur de l'option")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
#  Modules Basket & Asian – helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def get_option_expiries(ticker: str):
    tk = yf.Ticker(ticker)
    return tk.options or []


@st.cache_data(show_spinner=False)
def get_option_surface_from_yf(ticker: str, expiry: str):
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)

    frames = []
    for frame in [chain.calls, chain.puts]:
        tmp = frame[["strike", "impliedVolatility"]].rename(columns={"strike": "K", "impliedVolatility": "iv"})
        tmp["T"] = 0.0
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["K", "iv"])
    return df


@st.cache_data(show_spinner=False)
def get_spot_and_hist_vol(ticker: str, period: str = "6mo", interval: str = "1d"):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError("Aucune donnée téléchargée.")
    close = data["Close"]
    spot = float(close.iloc[-1])
    log_returns = np.log(close / close.shift(1)).dropna()
    sigma = float(log_returns.std() * np.sqrt(252))
    hist_df = data.reset_index()
    hist_df["Date"] = pd.to_datetime(hist_df["Date"])
    return spot, sigma, hist_df


def fetch_closing_prices(tickers, period="1mo", interval="1d"):
    if isinstance(tickers, str):
        tickers = [tickers]
    for var in ["YF_IMPERSONATE", "YF_SCRAPER_IMPERSONATE"]:
        try:
            os.environ.pop(var, None)
        except Exception:
            pass
    try:
        yf.set_config(proxy=None)
    except Exception:
        pass

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"Aucune donnée récupérée pour {tickers} sur {period}.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
    else:
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].copy()
        elif "Close" in data.columns:
            prices = data[["Close"]].copy()
        else:
            raise RuntimeError("Colonnes de prix introuvables dans les données yfinance.")
        prices.columns = tickers

    prices = prices.reset_index()
    return prices


def compute_corr_from_prices(prices_df: pd.DataFrame):
    price_cols = [c for c in prices_df.columns if c.lower() != "date"]
    returns = np.log(prices_df[price_cols] / prices_df[price_cols].shift(1)).dropna(how="any")
    if returns.empty:
        raise RuntimeError("Pas assez de données pour calculer la corrélation.")
    return returns.corr()


def load_closing_prices_with_tickers(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not path.exists():
        return None, []
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, []
    ticker_cols: list[str] = []
    for col in df.columns:
        col_str = str(col).strip()
        if not col_str or col_str.lower() == "date":
            continue
        ticker_cols.append(col_str)
    return df, ticker_cols


class BasketOption:
    def __init__(self, weights, prices, volatility, corr, strike, maturity, rate):
        self.weights = weights
        self.vol = volatility
        self.strike = strike
        self.mat = maturity
        self.rate = rate
        self.corr = corr
        self.prices = prices

    def get_mc(self, m_paths: int = 10000):
        b_ts = stats.multivariate_normal(np.zeros(len(self.weights)), cov=self.corr).rvs(size=m_paths)
        s_ts = self.prices * np.exp((self.rate - 0.5 * self.vol**2) * self.mat + self.vol * b_ts)
        if len(self.weights) > 1:
            payoffs = (np.sum(self.weights * s_ts, axis=1) - self.strike).clip(0)
        else:
            payoffs = np.maximum(s_ts - self.strike, np.zeros(m_paths))
        return float(np.exp(-self.rate * self.mat) * np.mean(payoffs))

    def get_bs_price(self):
        d1 = (np.log(self.prices / self.strike) + (self.rate + 0.5 * self.vol**2) * self.mat) / (
            self.vol * np.sqrt(self.mat)
        )
        d2 = d1 - self.vol * np.sqrt(self.mat)
        bs_price = stats.norm.cdf(d1) * self.prices - stats.norm.cdf(d2) * self.strike * np.exp(-self.rate * self.mat)
        return float(bs_price)


class DataGen:
    def __init__(self, n_assets: int, n_samples: int):
        if n_samples <= 0:
            raise ValueError("n_samples needs to be positive")
        if n_assets <= 0:
            raise ValueError("n_assets needs to be positive")
        self.n_assets = n_assets
        self.n_samples = n_samples

    def generate(self, corr, strike_price: float, base_price: float, method: str = "bs"):
        mats = np.random.uniform(0.2, 1.1, size=self.n_samples)
        vols = np.random.uniform(0.01, 1.0, size=self.n_samples)
        rates = np.random.uniform(0.02, 0.1, size=self.n_samples)

        strikes = np.random.randn(self.n_samples) + strike_price
        prices = np.random.randn(self.n_samples) + base_price

        if self.n_assets > 1:
            weights = np.random.rand(self.n_samples * self.n_assets).reshape((self.n_samples, self.n_assets))
            weights /= np.sum(weights, axis=1)[:, np.newaxis]
        else:
            weights = np.ones((self.n_samples, self.n_assets))

        labels = []
        for i in range(self.n_samples):
            basket = BasketOption(
                weights[i],
                prices[i],
                vols[i],
                corr,
                strikes[i],
                mats[i],
                rates[i],
            )
            if method == "bs":
                labels.append(basket.get_bs_price())
            else:
                labels.append(basket.get_mc())

        data = pd.DataFrame(
            {
                "S/K": prices / strikes,
                "Maturity": mats,
                "Volatility": vols,
                "Rate": rates,
                "Labels": labels,
                "Prices": prices,
                "Strikes": strikes,
            }
        )
        for i in range(self.n_assets):
            data[f"Weight_{i}"] = weights[:, i]
        return data


def simulate_dataset_notebook(n_assets: int, n_samples: int, method: str, corr: np.ndarray, base_price: float, base_strike: float):
    generator = DataGen(n_assets=n_assets, n_samples=n_samples)
    return generator.generate(corr=corr, strike_price=base_strike, base_price=base_price, method=method)


@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def split_data_nn(data: pd.DataFrame, split_ratio: float = 0.7):
    feature_cols = ["S/K", "Maturity", "Volatility", "Rate"]
    target_col = "Labels"
    train = data.iloc[: int(split_ratio * len(data)), :]
    test = data.iloc[int(split_ratio * len(data)) :, :]
    x_train, y_train = train[feature_cols], train[target_col]
    x_test, y_test = test[feature_cols], test[target_col]
    return x_train, y_train, x_test, y_test


def build_model_nn(input_dim: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(32, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Dense(1, activation="relu")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=["mean_squared_error"],
    )
    return model


def price_basket_nn(model: tf.keras.Model, S: float, K: float, maturity: float, volatility: float, rate: float) -> float:
    S_over_K = S / K
    x = np.array([[S_over_K, maturity, volatility, rate]], dtype=float)
    return float(model.predict(x, verbose=0)[0, 0])


def plot_heatmap_nn(
    model: tf.keras.Model,
    data: pd.DataFrame,
    spot_ref: float | None = None,
    strike_ref: float | None = None,
    maturity_fixed: float = 1.0,
):
    df = data.copy()
    if "Prices" not in df.columns and spot_ref is not None:
        df["Prices"] = spot_ref
    if "Strikes" not in df.columns and strike_ref is not None:
        df["Strikes"] = strike_ref

    if not {"Prices", "Strikes"}.issubset(df.columns):
        raise ValueError("Colonnes Prices et Strikes requises pour reproduire la heatmap du notebook.")

    s_min, s_max = df["Prices"].quantile([0.01, 0.99])
    k_min, k_max = df["Strikes"].quantile([0.01, 0.99])
    n_S, n_K = 50, 50
    s_vals = np.linspace(s_min, s_max, n_S)
    k_vals = np.linspace(k_min, k_max, n_K)

    K_grid, S_grid = np.meshgrid(k_vals, s_vals)
    s_over_k_grid = S_grid / K_grid

    sigma_ref = float(df["Volatility"].median())
    rate_ref = float(df["Rate"].median())

    X = np.stack(
        [
            s_over_k_grid.ravel(),
            np.full(s_over_k_grid.size, maturity_fixed),
            np.full(s_over_k_grid.size, sigma_ref),
            np.full(s_over_k_grid.size, rate_ref),
        ],
        axis=1,
    )
    prices_grid = model.predict(X, verbose=0).reshape(n_S, n_K)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        prices_grid,
        origin="lower",
        extent=[k_vals.min(), k_vals.max(), s_vals.min(), s_vals.max()],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Spot S")
    ax.set_title("Heatmap du prix NN en fonction de S et K (T=1 an)")
    fig.colorbar(im, ax=ax, label="Prix NN")
    plt.tight_layout()
    return fig


def build_grid(
    df: pd.DataFrame,
    spot: float,
    n_k: int = 200,
    n_t: int = 200,
    k_min: float | None = None,
    k_max: float | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    k_span: float | None = None,
    margin_frac: float = 0.02,
):
    if k_min is None or k_max is None:
        if k_span is not None:
            k_min = spot - k_span
            k_max = spot + k_span
        else:
            data_k_min = float(df["K"].min())
            data_k_max = float(df["K"].max())
            delta_k = data_k_max - data_k_min
            pad = delta_k * margin_frac
            k_min = data_k_min - pad
            k_max = data_k_max + pad

    if t_min is None:
        t_min = float(df["T"].min())
    if t_max is None:
        t_max = float(df["T"].max())

    if k_min >= k_max:
        raise ValueError("k_min doit être inférieur à k_max.")
    if t_min >= t_max:
        raise ValueError("t_min doit être inférieur à t_max.")

    k_vals = np.linspace(k_min, k_max, n_k)
    t_vals = np.linspace(t_min, t_max, n_t)

    df = df[(df["K"] >= k_min) & (df["K"] <= k_max)].copy()
    df = df[(df["T"] >= t_min) & (df["T"] <= t_max)]

    if df.empty:
        raise ValueError("Aucun point n'appartient au domaine défini par la grille.")

    df["K_idx"] = np.searchsorted(k_vals, df["K"], side="left").clip(0, n_k - 1)
    df["T_idx"] = np.searchsorted(t_vals, df["T"], side="left").clip(0, n_t - 1)

    grouped = df.groupby(["T_idx", "K_idx"])["iv"].mean().reset_index()

    iv_grid = np.full((n_t, n_k), np.nan, dtype=float)
    for _, row in grouped.iterrows():
        iv_grid[int(row["T_idx"]), int(row["K_idx"])] = row["iv"]

    k_grid, t_grid = np.meshgrid(k_vals, t_vals)
    return k_grid, t_grid, iv_grid


def make_iv_surface_figure(k_grid, t_grid, iv_grid, title_suffix=""):
    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    iv_flat = iv_grid[~np.isnan(iv_grid)]
    if iv_flat.size == 0:
        raise ValueError("La grille iv_grid ne contient aucune valeur non-NaN.")
    iv_mean = iv_flat.mean()
    iv_grid_filled = np.where(np.isnan(iv_grid), iv_mean, iv_grid)

    surf = ax3d.plot_surface(
        k_grid,
        t_grid,
        iv_grid_filled,
        rstride=1,
        cstride=1,
        linewidth=0.2,
        antialiased=True,
        cmap="viridis",
    )

    ax3d.set_xlabel("Strike K")
    ax3d.set_ylabel("Maturité T (années)")
    ax3d.set_zlabel("Implied vol")
    ax3d.set_title(f"Surface 3D de volatilité implicite{title_suffix}")

    fig.colorbar(surf, shrink=0.5, aspect=10, ax=ax3d, label="iv")

    ax2d = fig.add_subplot(1, 2, 2)
    im = ax2d.imshow(
        iv_grid_filled,
        extent=[k_grid.min(), k_grid.max(), t_grid.min(), t_grid.max()],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax2d.set_xlabel("Strike K")
    ax2d.set_ylabel("Maturité T (années)")
    ax2d.set_title(f"Heatmap IV{title_suffix}")
    fig.colorbar(im, ax=ax2d, label="iv")

    plt.tight_layout()
    return fig


def btm_asian(strike_type, option_type, spot, strike, rate, sigma, maturity, steps):
    delta_t = maturity / steps
    up = np.exp(sigma * np.sqrt(delta_t))
    down = 1.0 / up
    prob = (np.exp(rate * delta_t) - down) / (up - down)

    spot_paths = [spot]
    avg_paths = [spot]
    strike_paths = [strike]

    for _ in range(steps):
        spot_paths = [s * up for s in spot_paths] + [s * down for s in spot_paths]
        avg_paths = avg_paths + avg_paths
        strike_paths = strike_paths + strike_paths
        for index in range(len(avg_paths)):
            avg_paths[index] = avg_paths[index] + spot_paths[index]

    avg_paths = np.array(avg_paths) / (steps + 1)
    spot_paths = np.array(spot_paths)
    strike_paths = np.array(strike_paths)

    if strike_type == "fixed":
        if option_type == "C":
            payoff = np.maximum(avg_paths - strike_paths, 0.0)
        else:
            payoff = np.maximum(strike_paths - avg_paths, 0.0)
    else:
        if option_type == "C":
            payoff = np.maximum(spot_paths - avg_paths, 0.0)
        else:
            payoff = np.maximum(avg_paths - spot_paths, 0.0)

    option_price = payoff.copy()
    for _ in range(steps):
        length = len(option_price) // 2
        option_price = prob * option_price[:length] + (1 - prob) * option_price[length:]

    return float(option_price[0])


def hw_btm_asian(strike_type, option_type, spot, strike, rate, sigma, maturity, steps, m_points):
    n_steps = steps
    delta_t = maturity / n_steps
    up = np.exp(sigma * np.sqrt(delta_t))
    down = 1.0 / up
    prob = (np.exp(rate * delta_t) - down) / (up - down)

    avg_grid = []
    strike_vec = np.array([strike] * m_points)

    for j_index in range(n_steps + 1):
        path_up_then_down = np.array(
            [spot * up**j * down**0 for j in range(n_steps - j_index)]
            + [spot * up**(n_steps - j_index) * down**j for j in range(j_index + 1)]
        )
        avg_max = path_up_then_down.mean()

        path_down_then_up = np.array(
            [spot * down**j * up**0 for j in range(j_index + 1)]
            + [spot * down**j_index * up**(j + 1) for j in range(n_steps - j_index)]
        )
        avg_min = path_down_then_up.mean()

        diff = avg_max - avg_min
        avg_vals = [avg_max - diff * k_index / (m_points - 1) for k_index in range(m_points)]
        avg_grid.append(avg_vals)

    avg_grid = np.round(avg_grid, 4)

    payoff = []
    for j_index in range(n_steps + 1):
        avg_vals = np.array(avg_grid[j_index])
        spot_vals = np.array([spot * up**(n_steps - j_index) * down**j_index] * m_points)

        if strike_type == "fixed":
            if option_type == "C":
                pay = np.maximum(avg_vals - strike_vec, 0.0)
            else:
                pay = np.maximum(strike_vec - avg_vals, 0.0)
        else:
            if option_type == "C":
                pay = np.maximum(spot_vals - avg_vals, 0.0)
            else:
                pay = np.maximum(avg_vals - spot_vals, 0.0)

        payoff.append(pay)

    payoff = np.round(np.array(payoff), 4)

    for n_index in range(n_steps - 1, -1, -1):
        avg_backward = []
        payoff_backward = []

        for j_index in range(n_index + 1):
            path_up_then_down = np.array(
                [spot * up**j * down**0 for j in range(n_index - j_index)]
                + [spot * up**(n_index - j_index) * down**j for j in range(j_index + 1)]
            )
            avg_max = path_up_then_down.mean()

            path_down_then_up = np.array(
                [spot * down**j * up**0 for j in range(j_index + 1)]
                + [spot * down**j_index * up**(j + 1) for j in range(n_index - j_index)]
            )
            avg_min = path_down_then_up.mean()

            diff = avg_max - avg_min
            avg_vals = np.array([avg_max - diff * k_index / (m_points - 1) for k_index in range(m_points)])
            avg_backward.append(avg_vals)

        avg_backward = np.round(np.array(avg_backward), 4)

        payoff_new = []
        for j_index in range(n_index + 1):
            avg_vals = avg_backward[j_index]
            pay_vals = np.zeros_like(avg_vals)

            avg_up = np.array(avg_grid[j_index])
            avg_down = np.array(avg_grid[j_index + 1])
            pay_up = payoff[j_index]
            pay_down = payoff[j_index + 1]

            for k_index, avg_k in enumerate(avg_vals):
                if avg_k <= avg_up[0]:
                    fu = pay_up[0]
                elif avg_k >= avg_up[-1]:
                    fu = pay_up[-1]
                else:
                    idx = np.searchsorted(avg_up, avg_k) - 1
                    x0, x1 = avg_up[idx], avg_up[idx + 1]
                    y0, y1 = pay_up[idx], pay_up[idx + 1]
                    fu = y0 + (y1 - y0) * (avg_k - x0) / (x1 - x0)

                if avg_k <= avg_down[0]:
                    fd = pay_down[0]
                elif avg_k >= avg_down[-1]:
                    fd = pay_down[-1]
                else:
                    idx = np.searchsorted(avg_down, avg_k) - 1
                    x0, x1 = avg_down[idx], avg_down[idx + 1]
                    y0, y1 = pay_down[idx], pay_down[idx + 1]
                    fd = y0 + (y1 - y0) * (avg_k - x0) / (x1 - x0)

                pay_vals[k_index] = (prob * fu + (1 - prob) * fd) * np.exp(-rate * delta_t)

            payoff_backward.append(pay_vals)

        avg_grid = avg_backward
        payoff = np.round(np.array(payoff_backward), 4)

    option_price = payoff[0].mean()
    return float(option_price)


def bs_option_price(time, spot, strike, maturity, rate, sigma, option_kind):
    tau = maturity - time
    if tau <= 0:
        if option_kind == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if option_kind == "call":
        price = spot * norm.cdf(d1) - strike * np.exp(-rate * tau) * norm.cdf(d2)
    else:
        price = strike * np.exp(-rate * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return float(price)


def asian_geometric_closed_form(spot, strike, rate, sigma, maturity, n_obs, option_type):
    if n_obs < 1:
        return 0.0
    dt = maturity / n_obs
    nu = rate - 0.5 * sigma**2
    sigma_g_sq = (sigma**2) * (n_obs + 1) * (2 * n_obs + 1) / (6 * n_obs**2)
    sigma_g = np.sqrt(sigma_g_sq)
    mu_g = (nu * (n_obs + 1) / (2 * n_obs) + 0.5 * sigma_g_sq) * maturity
    d1 = (np.log(spot / strike) + mu_g + 0.5 * sigma_g_sq * maturity) / (sigma_g * np.sqrt(maturity))
    d2 = d1 - sigma_g * np.sqrt(maturity)
    df = np.exp(-rate * maturity)
    if option_type == "call":
        return float(df * (spot * np.exp(mu_g) * norm.cdf(d1) - strike * norm.cdf(d2)))
    else:
        return float(df * (strike * norm.cdf(-d2) - spot * np.exp(mu_g) * norm.cdf(-d1)))


def asian_mc_control_variate(
    spot,
    strike,
    rate,
    sigma,
    maturity,
    n_obs,
    n_paths,
    option_type,
    antithetic=True,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
    dt = maturity / n_obs
    drift = (rate - 0.5 * sigma**2) * dt
    vol_step = sigma * np.sqrt(dt)

    if antithetic:
        n_base = max(1, n_paths // 2)
        z_base = np.random.randn(n_obs, n_base)
        z = np.concatenate([z_base, -z_base], axis=1)
        n_eff = z.shape[1]
    else:
        z = np.random.randn(n_obs, n_paths)
        n_eff = n_paths

    log_s = np.log(spot) + np.cumsum(drift + vol_step * z, axis=0)
    s_paths = np.exp(log_s)

    arith_mean = s_paths.mean(axis=0)
    geom_mean = np.exp(np.log(s_paths).mean(axis=0))
    if option_type == "call":
        arith_payoff = np.maximum(arith_mean - strike, 0.0)
        geom_payoff = np.maximum(geom_mean - strike, 0.0)
    else:
        arith_payoff = np.maximum(strike - arith_mean, 0.0)
        geom_payoff = np.maximum(strike - geom_mean, 0.0)
    closed_geom = asian_geometric_closed_form(spot, strike, rate, sigma, maturity, n_obs, option_type)
    cov = np.cov(arith_payoff, geom_payoff)[0, 1]
    var_geom = np.var(geom_payoff)
    c = cov / var_geom if var_geom > 0 else 0.0
    control_estimator = arith_payoff - c * (geom_payoff - closed_geom)
    disc = np.exp(-rate * maturity)
    disc_payoff = disc * control_estimator
    price = np.mean(disc_payoff)
    stderr = np.std(disc_payoff, ddof=1) / np.sqrt(n_eff)
    return float(price), float(stderr), float(c)


def compute_asian_price(
    strike_type: str,
    option_type: str,
    model: str,
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    steps: int,
    m_points: int | None,
):
    if model == "BTM naïf":
        return btm_asian(
            strike_type=strike_type,
            option_type=option_type,
            spot=spot,
            strike=strike,
            rate=rate,
            sigma=sigma,
            maturity=maturity,
            steps=int(steps),
        )
    m_points_val = int(m_points) if m_points is not None else 10
    return hw_btm_asian(
        strike_type=strike_type,
        option_type=option_type,
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        steps=int(steps),
        m_points=m_points_val,
    )


def ui_basket_surface(spot_common, maturity_common, rate_common, strike_common):
    st.header("Basket – Pricing NN + corrélation (3 actifs)")

    min_assets, max_assets = 2, 10
    closing_path = Path("data/closing_prices.csv")
    prices_df_cached, csv_tickers = load_closing_prices_with_tickers(closing_path)

    def _normalize_tickers(candidates: list[str]) -> list[str]:
        cleaned = [str(tk).strip().upper() for tk in candidates if str(tk).strip()]
        trimmed = cleaned[:max_assets]
        if len(trimmed) < min_assets:
            trimmed += ["SPY"] * (min_assets - len(trimmed))
        return trimmed

    if "basket_tickers" not in st.session_state:
        default_list = csv_tickers if csv_tickers else ["AAPL", "SPY", "MSFT"]
        st.session_state["basket_tickers"] = _normalize_tickers(default_list)

    with st.container():
        st.subheader("Sélection des assets (2 à 10)")
        btn_col_add, btn_col_remove = st.columns(2)
        with btn_col_add:
            if st.button("Ajouter un asset", disabled=len(st.session_state["basket_tickers"]) >= max_assets):
                st.session_state["basket_tickers"].append(
                    f"TICKER{len(st.session_state['basket_tickers']) + 1}"
                )
        with btn_col_remove:
            if st.button("Retirer un asset", disabled=len(st.session_state["basket_tickers"]) <= min_assets):
                st.session_state["basket_tickers"].pop()

        tickers = []
        for i, default_tk in enumerate(st.session_state["basket_tickers"]):
            if i % 3 == 0:
                cols = st.columns(3)
            col = cols[i % 3]
            with col:
                tick = st.text_input(f"Ticker {i + 1}", value=default_tk, key=f"corr_tk_dynamic_{i}")
                tickers.append(tick.strip().upper() or default_tk)
        tickers = tickers[:max_assets]
        if len(tickers) < min_assets:
            tickers += ["SPY"] * (min_assets - len(tickers))
        st.session_state["basket_tickers"] = tickers
    tickers = st.session_state["basket_tickers"]

    period = st.selectbox("Période yfinance", ["1mo", "3mo", "6mo", "1y"], index=0, key="corr_period")
    interval = st.selectbox("Intervalle", ["1d", "1h"], index=0, key="corr_interval")

    st.caption(
        "Le calcul de corrélation utilise les prix de clôture présents dans data/closing_prices.csv (générés via le script). "
        "En cas d'échec, une matrice de corrélation inventée sera utilisée."
    )
    regen_csv = st.button("Mettre à jour la Matrice de Corrélation", key="btn_regen_closing")
    try:
        if regen_csv or not closing_path.exists():
            cmd = [sys.executable, "fetch_closing_prices.py", "--tickers", *tickers, "--output", "data/closing_prices.csv"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.info(f"data/closing_prices.csv généré via le script ({res.stdout.strip()})")
            prices_df_cached, csv_tickers = load_closing_prices_with_tickers(closing_path)
            if csv_tickers:
                st.session_state["basket_tickers"] = _normalize_tickers(csv_tickers)
                tickers = st.session_state["basket_tickers"]
    except Exception as exc:
        st.warning(f"Impossible d'exécuter fetch_closing_prices.py : {exc}")

    corr_df = None
    try:
        if prices_df_cached is None:
            prices_df_cached, _ = load_closing_prices_with_tickers(closing_path)
        if prices_df_cached is None:
            raise FileNotFoundError("Impossible de charger data/closing_prices.csv.")
        corr_df = compute_corr_from_prices(prices_df_cached)
        st.success(f"Corrélation calculée à partir de {closing_path.name}")
        st.dataframe(corr_df)
    except Exception as exc:
        st.warning(f"Impossible de calculer la corrélation depuis data/closing_prices.csv : {exc}")
        corr_df = pd.DataFrame(
            [
                [1.0, 0.6, 0.4],
                [0.6, 1.0, 0.7],
                [0.4, 0.7, 1.0],
            ],
            columns=tickers,
            index=tickers,
        )
        st.info("Utilisation d'une matrice de corrélation inventée pour la suite des calculs.")
        st.dataframe(corr_df)

    st.subheader("Dataset Basket pour NN")
    st.caption("Dataset généré automatiquement via DataGen (comme dans le notebook).")
    n_samples = st.slider("Taille du dataset simulé", 1000, 20000, 10000, 1000)
    method = st.selectbox("Méthode de pricing pour les labels", ["bs", "mc"], index=0)

    df = simulate_dataset_notebook(
        n_assets=len(tickers),
        n_samples=int(n_samples),
        method=method,
        corr=corr_df.values,
        base_price=float(spot_common),
        base_strike=float(strike_common),
    )

    st.write("Aperçu :", df.head())
    st.write("Shape :", df.shape)

    split_ratio = st.slider("Train ratio", 0.5, 0.9, 0.7, 0.05)
    epochs = st.slider("Epochs d'entraînement", 5, 200, 20, 5)

    x_train, y_train, x_test, y_test = split_data_nn(df, split_ratio=split_ratio)
    Path("data").mkdir(parents=True, exist_ok=True)
    pd.concat([x_train, y_train], axis=1).to_csv("data/train.csv", index=False)
    pd.concat([x_test, y_test], axis=1).to_csv("data/test.csv", index=False)
    st.info("train.csv et test.csv régénérés pour la surface IV.")

    st.write(f"Train size: {x_train.shape[0]} | Test size: {x_test.shape[0]}")

    train_button = st.button("Entraîner le modèle NN", key="btn_train_nn")
    if not train_button:
        st.info("Clique sur 'Entraîner le modèle NN' pour lancer l'apprentissage.")
        return

    tf.keras.backend.clear_session()
    model = build_model_nn(input_dim=x_train.shape[1])
    train_logs: list[str] = []
    log_box = st.empty()

    class StreamlitLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            msg = (
                f"Epoch {epoch + 1}/{epochs} - loss: {logs.get('loss', float('nan')):.4f} - "
                f"mse: {logs.get('mean_squared_error', float('nan')):.4f}"
            )
            if "val_loss" in logs or "val_mean_squared_error" in logs:
                msg += (
                    f" - val_loss: {logs.get('val_loss', float('nan')):.4f} - "
                    f"val_mse: {logs.get('val_mean_squared_error', float('nan')):.4f}"
                )
            train_logs.append(msg)
            log_box.text("\n".join(train_logs))

    with st.spinner("Entraînement du NN en cours…"):
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=0,
            callbacks=[StreamlitLogger()],
        )
    st.success("Entraînement terminé.")

    st.subheader("Courbe MSE NN")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(history.history["mean_squared_error"], label="train")
    ax.plot(history.history["val_mean_squared_error"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

    st.subheader("Heatmap prix NN (S vs K)")
    try:
        with st.spinner("Calcul de la heatmap…"):
            heatmap_fig = plot_heatmap_nn(
                model=model,
                data=df,
                spot_ref=float(spot_common),
                strike_ref=float(strike_common),
                maturity_fixed=1.0,
            )
        st.pyplot(heatmap_fig)
    except Exception as exc:
        st.warning(f"Impossible d'afficher la heatmap : {exc}")

    st.subheader("Surface IV (Strike, Maturité)")
    try:
        with st.spinner("Calcul de la surface IV…"):
            iv_df = df.copy()
            if "Strikes" in iv_df.columns:
                iv_df["K"] = iv_df["Strikes"]
            else:
                iv_df["K"] = spot_common / iv_df["S/K"].replace(0.0, np.nan)
            iv_df = iv_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["K", "Maturity", "Volatility"])

            if iv_df.empty:
                raise ValueError("Pas de données IV exploitables (S/K nuls ou manquants).")

            spot_ref_for_grid = float(iv_df["Prices"].mean()) if "Prices" in iv_df.columns else float(spot_common)

            grid_k, grid_t, grid_iv = build_grid(
                df=iv_df.rename(columns={"Maturity": "T", "Volatility": "iv"}),
                spot=spot_ref_for_grid,
            )
            iv_fig = make_iv_surface_figure(grid_k, grid_t, grid_iv, title_suffix=" (dataset NN)")
        st.pyplot(iv_fig)
    except Exception as exc:
        st.warning(f"Impossible d'afficher la surface IV : {exc}")


def ui_asian_options(
    spot_default,
    sigma_common,
    maturity_common,
    strike_common,
    rate_common,
):
    st.header("Options asiatiques (module Asian)")

    if spot_default is None:
        st.warning("Aucun téléchargement yfinance : utilisez le spot commun.")
        spot_default = 57830.0
    if sigma_common is None:
        sigma_common = 0.05

    col1, col2 = st.columns(2)
    with col1:
        spot_common = st.session_state.get("common_spot", spot_default)
        strike_common_local = st.session_state.get("common_strike", strike_common)
        st.info(f"Spot commun S0 = {spot_common:.4f}")
        st.info(f"Strike commun K = {strike_common_local:.4f}")
        st.info(f"Taux sans risque commun r = {rate_common:.4f}")
    with col2:
        sigma = sigma_common
        st.info(f"Volatilité commune σ = {sigma:.4f}")
        st.info("Pricing asiatique via Monte Carlo + control variate (méthode notebook).")

    st.subheader("Heatmaps prix asiatiques (K vs T)")
    col_k, col_t = st.columns(2)
    with col_k:
        k_center = st.session_state.get("common_strike", strike_common)
        default_k_min = st.session_state.get("asian_k_min", max(0.01, k_center - 40.0))
        default_k_max = st.session_state.get("asian_k_max", k_center + 40.0)
        k_min = st.number_input(
            "K min",
            value=float(default_k_min),
            min_value=0.01,
            step=1.0,
            key="asian_k_min",
        )
        k_max = st.number_input(
            "K max",
            value=float(default_k_max),
            min_value=k_min + 1.0,
            step=1.0,
            key="asian_k_max",
        )
        st.caption(f"Domaine K: [{k_min:.2f}, {k_max:.2f}]")
    with col_t:
        t_center = st.session_state.get("common_maturity", maturity_common)
        default_t_min = st.session_state.get("asian_t_min", max(0.05, t_center / 2.0))
        default_t_max = st.session_state.get("asian_t_max", t_center * 2.0)
        t_min = st.number_input(
            "T min (années)",
            value=float(default_t_min),
            min_value=0.01,
            step=0.05,
            key="asian_t_min",
        )
        t_max = st.number_input(
            "T max (années)",
            value=float(default_t_max),
            min_value=t_min + 0.01,
            step=0.05,
            key="asian_t_max",
        )
        st.caption(f"Domaine T: [{t_min:.2f}, {t_max:.2f}]")

    n_k = st.slider("Résolution en K", 10, 40, 20, 2, key="asian_n_k")
    n_t = st.slider("Résolution en T", 10, 40, 20, 2, key="asian_n_t")
    n_paths_surface = st.slider("Nombre de trajectoires Monte Carlo", 5_000, 50_000, 20_000, 5_000, key="asian_n_paths")

    k_vals = np.linspace(k_min, k_max, n_k)
    t_vals = np.linspace(t_min, t_max, n_t)

    prices_call = np.zeros((n_t, n_k), dtype=float)
    prices_put = np.zeros((n_t, n_k), dtype=float)

    with st.spinner("Calcul des surfaces de prix (MC asiatique)…"):
        for i_t, t_val in enumerate(t_vals):
            n_obs = max(2, int(50 * t_val))
            for i_k, k_val in enumerate(k_vals):
                call_price, _, _ = asian_mc_control_variate(
                    spot=float(spot_common),
                    strike=float(k_val),
                    rate=float(rate_common),
                    sigma=float(sigma),
                    maturity=float(t_val),
                    n_obs=int(n_obs),
                    n_paths=int(n_paths_surface),
                    option_type="call",
                    antithetic=True,
                    seed=None,
                )
                put_price, _, _ = asian_mc_control_variate(
                    spot=float(spot_common),
                    strike=float(k_val),
                    rate=float(rate_common),
                    sigma=float(sigma),
                    maturity=float(t_val),
                    n_obs=int(n_obs),
                    n_paths=int(n_paths_surface),
                    option_type="put",
                    antithetic=True,
                    seed=None,
                )
                prices_call[i_t, i_k] = call_price
                prices_put[i_t, i_k] = put_price

    fig_call, ax_call = plt.subplots(figsize=(7, 4))
    im0 = ax_call.imshow(
        prices_call,
        origin="lower",
        extent=[k_vals.min(), k_vals.max(), t_vals.min(), t_vals.max()],
        aspect="auto",
        cmap="viridis",
    )
    ax_call.set_xlabel("Strike K")
    ax_call.set_ylabel("Maturité T (années)")
    ax_call.set_title("Call asiatique arithmétique (MC + control variate)")
    fig_call.colorbar(im0, ax=ax_call, label="Prix")
    fig_call.tight_layout()
    st.pyplot(fig_call)

    fig_put, ax_put = plt.subplots(figsize=(7, 4))
    im1 = ax_put.imshow(
        prices_put,
        origin="lower",
        extent=[k_vals.min(), k_vals.max(), t_vals.min(), t_vals.max()],
        aspect="auto",
        cmap="viridis",
    )
    ax_put.set_xlabel("Strike K")
    ax_put.set_ylabel("Maturité T (années)")
    ax_put.set_title("Put asiatique arithmétique (MC + control variate)")
    fig_put.colorbar(im1, ax=ax_put, label="Prix")
    fig_put.tight_layout()
    st.pyplot(fig_put)


# ---------------------------------------------------------------------------
#  Module Heston – pipeline complet
# ---------------------------------------------------------------------------


def heston_mc_pricer(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    theta: float,
    kappa: float,
    sigma_v: float,
    rho: float,
    n_paths: int = 50_000,
    n_steps: int = 100,
    option_type: str = "call",
) -> float:
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    S = np.full(n_paths, S0)
    v = np.full(n_paths, v0)
    for _ in range(n_steps):
        z1 = np.random.randn(n_paths)
        z2 = np.random.randn(n_paths)
        z_s = z1
        z_v = rho * z1 + math.sqrt(1 - rho**2) * z2
        v_pos = np.maximum(v, 0)
        S = S * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z_s)
        v = v + kappa * (theta - v_pos) * dt + sigma_v * np.sqrt(v_pos) * sqrt_dt * z_v
        v = np.maximum(v, 0)
    payoff = np.maximum(S - K, 0) if option_type == "call" else np.maximum(K - S, 0)
    return float(math.exp(-r * T) * np.mean(payoff))


def download_options_cboe(symbol: str, option_type: str) -> tuple[pd.DataFrame, float]:
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol.upper()}.json"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", {})
    options = data.get("options", [])
    spot = float(data.get("current_price") or data.get("close") or np.nan)
    now = pd.Timestamp.utcnow().tz_localize(None)
    pattern = re.compile(rf"^{symbol.upper()}(?P<expiry>\d{{6}})(?P<cp>[CP])(?P<strike>\d+)$")

    rows: list[dict] = []
    for opt in options:
        match = pattern.match(opt.get("option", ""))
        if not match:
            continue
        cp = match.group("cp")
        if (option_type == "call" and cp != "C") or (option_type == "put" and cp != "P"):
            continue
        expiry_dt = pd.to_datetime(match.group("expiry"), format="%y%m%d")
        T = (expiry_dt - now).total_seconds() / (365.0 * 24 * 3600)
        if T <= 0:
            continue
        T = round(T, 2)
        if T <= MIN_IV_MATURITY:
            continue
        strike = int(match.group("strike")) / 1000.0
        bid = float(opt.get("bid") or 0.0)
        ask = float(opt.get("ask") or 0.0)
        last = float(opt.get("last_trade_price") or 0.0)
        if bid > 0 and ask > 0:
            mid = 0.5 * (bid + ask)
        elif last > 0:
            mid = last
        else:
            mid = np.nan
        if np.isnan(mid) or mid <= 0:
            continue
        iv_val = opt.get("iv", np.nan)
        iv_val = float(iv_val) if iv_val not in (None, "") else np.nan
        rows.append(
            {
                "S0": spot,
                "K": strike,
                "T": T,
                ("C_mkt" if option_type == "call" else "P_mkt"): round(mid, 2),
                "iv_market": iv_val,
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["T"] > MIN_IV_MATURITY]
    return df, spot


@st.cache_data(show_spinner=False)
def load_cboe_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    calls_df, spot_calls = download_options_cboe(symbol, "call")
    puts_df, spot_puts = download_options_cboe(symbol, "put")
    S0_ref = float(np.nanmean([spot_calls, spot_puts]))
    return calls_df, puts_df, S0_ref


def prices_from_unconstrained(u: torch.Tensor, S0_t: torch.Tensor, K_t: torch.Tensor, T_t: torch.Tensor, r: float, q: float):
    params = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
    prices = []
    for S0_i, K_i, T_i in zip(S0_t, K_t, T_t):
        price_i = carr_madan_call_torch(S0_i, r, q, T_i, params, K_i)
        prices.append(price_i)
    return torch.stack(prices)


def heston_nn_loss(
    u: torch.Tensor,
    S0_t: torch.Tensor,
    K_t: torch.Tensor,
    T_t: torch.Tensor,
    C_mkt_t: torch.Tensor,
    r: float,
    q: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    model_prices = prices_from_unconstrained(u, S0_t, K_t, T_t, r, q)
    diff = model_prices - C_mkt_t
    if weights is not None:
        return 0.5 * (weights * diff**2).mean()
    return 0.5 * (diff**2).mean()


def calibrate_heston_nn(
    df: pd.DataFrame,
    r: float,
    q: float,
    max_iters: int,
    lr: float,
    spot_override: float | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> HestonParams:
    if df.empty:
        raise ValueError("DataFrame vide.")
    df_clean = df.dropna(subset=["S0", "K", "T", "C_mkt"])
    df_clean = df_clean[(df_clean["T"] > MIN_IV_MATURITY) & (df_clean["C_mkt"] > 0.05)]
    df_clean = df_clean[df_clean.get("iv_market", 0) > 0]
    if df_clean.empty:
        raise ValueError("Pas de points pour la calibration")

    S0_ref = spot_override if spot_override is not None else float(df_clean["S0"].median())
    moneyness = df_clean["K"].values / S0_ref

    S0_t = torch.tensor(df_clean["S0"].values, dtype=torch.float64, device=HES_DEVICE)
    K_t = torch.tensor(df_clean["K"].values, dtype=torch.float64, device=HES_DEVICE)
    T_t = torch.tensor(df_clean["T"].values, dtype=torch.float64, device=HES_DEVICE)
    C_mkt_t = torch.tensor(df_clean["C_mkt"].values, dtype=torch.float64, device=HES_DEVICE)

    weights_np = 1.0 / (np.abs(moneyness - 1.0) + 1e-3)
    weights_np = np.clip(weights_np / weights_np.mean(), 0.5, 5.0)
    weights_t = torch.tensor(weights_np, dtype=torch.float64, device=HES_DEVICE)

    u = torch.zeros(5, dtype=torch.float64, device=HES_DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=lr)

    for iteration in range(max_iters):
        optimizer.zero_grad()
        loss_val = heston_nn_loss(u, S0_t, K_t, T_t, C_mkt_t, r, q, weights=weights_t)
        loss_val.backward()
        optimizer.step()
        if progress_callback:
            progress_callback(iteration + 1, max_iters, float(loss_val.detach().cpu()))

    return HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol_option(price: float, S: float, K: float, T: float, r: float, option_type: str = "call", tol: float = 1e-6, max_iter: int = 100) -> float:
    if T < MIN_IV_MATURITY:
        return np.nan
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if price <= intrinsic:
        return np.nan
    sigma = 0.3
    for _ in range(max_iter):
        price_est = bs_call(S, K, T, r, sigma) if option_type == "call" else bs_put(S, K, T, r, sigma)
        diff = price_est - price
        if abs(diff) < tol:
            return sigma
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-10:
            return np.nan
        sigma = sigma - diff / vega
        if sigma <= 0:
            return np.nan
    return np.nan


def build_market_surface(
    df: pd.DataFrame,
    price_col: str,
    option_type: str,
    kk_grid: np.ndarray,
    tt_grid: np.ndarray,
    rf_rate: float,
) -> np.ndarray | None:
    df = df.dropna(subset=[price_col]).copy()
    df = df[(df["T"] >= MIN_IV_MATURITY) & (df[price_col] > 0)]
    if len(df) < 5:
        return None
    df["iv_calc"] = df.apply(
        lambda row: implied_vol_option(
            row[price_col], row["S0"], row["K"], row["T"], rf_rate, option_type
        ),
        axis=1,
    )
    df = df.dropna(subset=["iv_calc"])
    if df.empty:
        return None
    pts = df[["K", "T"]].to_numpy()
    vals = df["iv_calc"].to_numpy()
    surf = griddata(pts, vals, (kk_grid, tt_grid), method="linear")
    if surf is None or np.all(np.isnan(surf)):
        surf = griddata(pts, vals, (kk_grid, tt_grid), method="nearest")
    else:
        mask = np.isnan(surf)
        if mask.any():
            surf[mask] = griddata(pts, vals, (kk_grid[mask], tt_grid[mask]), method="nearest")
    return surf


def build_market_price_grid(
    df: pd.DataFrame,
    price_col: str,
    kk_grid: np.ndarray,
    tt_grid: np.ndarray,
) -> np.ndarray | None:
    df = df.dropna(subset=[price_col]).copy()
    df = df[(df["T"] >= MIN_IV_MATURITY) & (df[price_col] > 0)]
    if len(df) < 5:
        return None
    pts = df[["K", "T"]].to_numpy()
    vals = df[price_col].to_numpy()
    grid = griddata(pts, vals, (kk_grid, tt_grid), method="linear")
    if grid is None or np.all(np.isnan(grid)):
        grid = griddata(pts, vals, (kk_grid, tt_grid), method="nearest")
    else:
        mask = np.isnan(grid)
        if mask.any():
            grid[mask] = griddata(pts, vals, (kk_grid[mask], tt_grid[mask]), method="nearest")
    return grid



def ui_heston_full_pipeline():
    st.header("🚀 Surface IV Heston : CBOE → Calibration NN → Carr-Madan")
    st.write(
        "**Pipeline issu du notebook `heston_iv_surfaces.ipynb`** :\n"
        "1️⃣ Téléchargement des options CBOE (données retardées)\n"
        "2️⃣ Calibration Heston (NN Carr-Madan) ciblée sur la zone d'analyse\n"
        "3️⃣ Surfaces IV Carr-Madan vs Marché + heatmaps de prix\n"
    )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        ticker = st.text_input("Ticker (sous-jacent)", value="SPY", key="heston_cboe_ticker").strip().upper()
        rf_rate = st.number_input("Taux sans risque (r)", value=0.02, step=0.01, format="%.3f", key="heston_cboe_r")
        div_yield = st.number_input("Dividende (q)", value=0.00, step=0.01, format="%.3f", key="heston_cboe_q")
    with col_cfg2:
        span_mc = st.number_input(
            "Span autour de S0 pour les grilles K",
            value=20.0,
            min_value=5.0,
            max_value=100.0,
            step=5.0,
            key="heston_cboe_span",
        )
        n_maturities = 40
        st.caption(f"Points maturité (Carr-Madan) : {n_maturities}")

    state = st.session_state
    if "heston_calls_df" not in state:
        state.heston_calls_df = None
        state.heston_puts_df = None
        state.heston_S0_ref = None
        state.heston_calib_T_target = None

    fetch_btn = st.button("Récupérer les données du ticker", use_container_width=True, key="heston_cboe_fetch")
    st.divider()

    if fetch_btn:
        try:
            calls_df, puts_df, S0_ref = load_cboe_data(ticker)
            state.heston_calls_df = calls_df
            state.heston_puts_df = puts_df
            state.heston_S0_ref = S0_ref
            st.info(f"📡 Données CBOE chargées pour {ticker} (cache)")
            st.success(f"{len(calls_df)} calls, {len(puts_df)} puts | S0 ≈ {S0_ref:.2f}")
        except Exception as exc:
            st.error(f"❌ Erreur lors du téléchargement des données CBOE : {exc}")

    calls_df = state.heston_calls_df
    puts_df = state.heston_puts_df
    S0_ref = state.heston_S0_ref
    calib_T_target = state.heston_calib_T_target

    calib_band_range: tuple[float, float] | None = None
    calib_T_band = 0.04
    max_iters = 1000
    learning_rate = 0.005

    if calls_df is not None and puts_df is not None and S0_ref is not None:
        col_nn, col_modes = st.columns(2)
        with col_nn:
            st.subheader("🎯 Calibration NN Carr-Madan")
            calib_T_band = st.number_input(
                "Largeur bande T (±)",
                value=0.04,
                min_value=0.01,
                max_value=0.5,
                step=0.01,
                format="%.2f",
                key="heston_cboe_calib_band",
            )

            unique_T = sorted(calls_df["T"].round(2).unique().tolist())
            if unique_T:
                if calib_T_target is None:
                    target_guess = max(MIN_IV_MATURITY, unique_T[0] + calib_T_band + 0.1)
                    idx_default = int(np.argmin(np.abs(np.array(unique_T) - target_guess)))
                else:
                    try:
                        idx_default = unique_T.index(calib_T_target)
                    except ValueError:
                        idx_default = 0

                calib_T_target = st.selectbox(
                    "Maturité T cible pour la calibration (Time to Maturity)",
                    unique_T,
                    index=idx_default,
                    format_func=lambda x: f"{x:.2f}",
                    key="heston_cboe_calib_target",
                )
                state.heston_calib_T_target = calib_T_target
            else:
                st.warning("Pas de maturités disponibles dans les données CBOE.")
                calib_T_target = None

        with col_modes:
            st.subheader("⚙️ Modes de calibration NN")
            mode = st.radio(
                "Choisir un mode",
                ["Rapide", "Bonne", "Excellente"],
                index=1,
                horizontal=True,
                key="heston_cboe_mode",
            )
            if mode == "Rapide":
                max_iters = 300
                learning_rate = 0.01
            elif mode == "Bonne":
                max_iters = 1000
                learning_rate = 0.005
            else:
                max_iters = 2000
                learning_rate = 0.001
            st.markdown(
                f"**Itérations NN** : `{max_iters}`  \n"
                f"**Learning rate** : `{learning_rate}`"
            )

        if calib_T_target is not None:
            calib_band_range = (
                max(MIN_IV_MATURITY, calib_T_target - calib_T_band),
                calib_T_target + calib_T_band,
            )
        else:
            calib_band_range = None

    run_button = False
    if calls_df is not None and puts_df is not None and S0_ref is not None:
        run_button = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True, key="heston_cboe_run")
        st.divider()

    if run_button:
        if calls_df is None or puts_df is None or S0_ref is None:
            st.error("Veuillez d'abord cliquer sur 'Récupérer les données du ticker'.")
            return
        if calib_band_range is None or calib_T_target is None:
            st.error("Veuillez choisir une maturité T cible après avoir chargé les données.")
            return

        try:
            st.info(f"📡 Données CBOE chargées pour {ticker} (cache)")
            st.success(f"{len(calls_df)} calls, {len(puts_df)} puts | S0 ≈ {S0_ref:.2f}")
            st.write(f"Maturité T cible pour la calibration : {calib_T_target:.2f} ans")

            st.info("🧠 Calibration ciblée...")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            loss_log: list[float] = []

            def progress_cb(current: int, total: int, loss_val: float) -> None:
                progress_bar.progress(current / total)
                status_text.text(f"⏳ Iter {current}/{total} | Loss = {loss_val:.6f}")
                loss_log.append(loss_val)

            calib_slice = calls_df[
                (calls_df["T"].round(2).between(*calib_band_range))
                & (calls_df["K"].between(S0_ref - span_mc, S0_ref + span_mc))
                & (calls_df["C_mkt"] > 0.05)
                & (calls_df["iv_market"] > 0)
            ]
            if len(calib_slice) < 5:
                calib_slice = calls_df.copy()

            params_cm = calibrate_heston_nn(
                calib_slice,
                r=rf_rate,
                q=div_yield,
                max_iters=int(max_iters),
                lr=learning_rate,
                spot_override=S0_ref,
                progress_callback=progress_cb,
            )
            progress_bar.empty()
            status_text.empty()

            params_dict = {
                "kappa": float(params_cm.kappa.detach()),
                "theta": float(params_cm.theta.detach()),
                "sigma": float(params_cm.sigma.detach()),
                "rho": float(params_cm.rho.detach()),
                "v0": float(params_cm.v0.detach()),
            }
            st.success("✓ Calibration terminée")
            st.dataframe(pd.Series(params_dict, name="Paramètre").to_frame())

            st.info("📐 Surfaces analytiques Carr-Madan")
            K_grid = np.arange(S0_ref - span_mc, S0_ref + span_mc + 1, 1)
            t_min = max(MIN_IV_MATURITY, calib_band_range[0])
            t_max = max(t_min + 0.05, min(2.0, calib_band_range[1]))
            T_grid = np.linspace(t_min, t_max, n_maturities)
            K_grid = np.unique(K_grid)
            T_grid = np.unique(T_grid)
            Ks_t = torch.tensor(K_grid, dtype=torch.float64)

            call_prices_cm = np.zeros((len(T_grid), len(K_grid)))
            put_prices_cm = np.zeros_like(call_prices_cm)
            for i, T_val in enumerate(T_grid):
                call_vals = carr_madan_call_torch(S0_ref, rf_rate, div_yield, float(T_val), params_cm, Ks_t)
                discount = torch.exp(-torch.tensor(rf_rate * T_val, dtype=torch.float64))
                forward = torch.exp(-torch.tensor(div_yield * T_val, dtype=torch.float64))
                put_vals = call_vals - S0_ref * forward + Ks_t * discount
                call_prices_cm[i, :] = call_vals.detach().cpu().numpy()
                put_prices_cm[i, :] = put_vals.detach().cpu().numpy()

            call_iv_cm = np.zeros_like(call_prices_cm)
            put_iv_cm = np.zeros_like(put_prices_cm)
            for i, T_val in enumerate(T_grid):
                for j, K_val in enumerate(K_grid):
                    call_iv_cm[i, j] = implied_vol_option(call_prices_cm[i, j], S0_ref, K_val, T_val, rf_rate, "call")
                    put_iv_cm[i, j] = implied_vol_option(put_prices_cm[i, j], S0_ref, K_val, T_val, rf_rate, "put")

            KK_cm, TT_cm = np.meshgrid(K_grid, T_grid, indexing="xy")
            call_iv_max = float(np.nanmax(call_iv_cm)) if call_iv_cm.size > 0 else 0.0
            put_iv_max = float(np.nanmax(put_iv_cm)) if put_iv_cm.size > 0 else 0.0

            surf_call_market = build_market_surface(calls_df, "C_mkt", "call", KK_cm, TT_cm, rf_rate)
            surf_put_market = build_market_surface(puts_df, "P_mkt", "put", KK_cm, TT_cm, rf_rate)

            if surf_call_market is not None:
                call_iv_max = max(call_iv_max, float(np.nanmax(surf_call_market)))
            if surf_put_market is not None:
                put_iv_max = max(put_iv_max, float(np.nanmax(surf_put_market)))

            call_iv_zmax = call_iv_max + 0.1
            put_iv_zmax = put_iv_max + 0.1

            fig_call_cm = go.Figure(
                data=[
                    go.Surface(
                        x=KK_cm,
                        y=TT_cm,
                        z=call_iv_cm,
                        colorscale="Viridis",
                        cmin=0.0,
                        cmax=call_iv_zmax,
                    )
                ]
            )
            fig_call_cm.update_layout(
                title=f"IV Surface Calls (Carr-Madan) - {ticker}",
                scene=dict(
                    xaxis_title="K",
                    yaxis_title="T",
                    zaxis_title="IV",
                    zaxis=dict(range=[0.0, call_iv_zmax]),
                ),
                height=600,
            )

            fig_put_cm = go.Figure(
                data=[
                    go.Surface(
                        x=KK_cm,
                        y=TT_cm,
                        z=put_iv_cm,
                        colorscale="Viridis",
                        cmin=0.0,
                        cmax=put_iv_zmax,
                    )
                ]
            )
            fig_put_cm.update_layout(
                title=f"IV Surface Puts (Carr-Madan) - {ticker}",
                scene=dict(
                    xaxis_title="K",
                    yaxis_title="T",
                    zaxis_title="IV",
                    zaxis=dict(range=[0.0, put_iv_zmax]),
                ),
                height=600,
            )

            fig_call_market = None
            fig_put_market = None
            if surf_call_market is not None:
                fig_call_market = go.Figure(
                    data=[
                        go.Surface(
                            x=KK_cm,
                            y=TT_cm,
                            z=surf_call_market,
                            colorscale="Plasma",
                            cmin=0.0,
                            cmax=call_iv_zmax,
                        )
                    ]
                )
                fig_call_market.update_layout(
                    title=f"IV Surface Calls (Marché) - {ticker}",
                    scene=dict(
                        xaxis_title="K",
                        yaxis_title="T",
                        zaxis_title="IV",
                        zaxis=dict(range=[0.0, call_iv_zmax]),
                    ),
                    height=600,
                )
            if surf_put_market is not None:
                fig_put_market = go.Figure(
                    data=[
                        go.Surface(
                            x=KK_cm,
                            y=TT_cm,
                            z=surf_put_market,
                            colorscale="Plasma",
                            cmin=0.0,
                            cmax=put_iv_zmax,
                        )
                    ]
                )
                fig_put_market.update_layout(
                    title=f"IV Surface Puts (Marché) - {ticker}",
                    scene=dict(
                        xaxis_title="K",
                        yaxis_title="T",
                        zaxis_title="IV",
                        zaxis=dict(range=[0.0, put_iv_zmax]),
                    ),
                    height=600,
                )

            market_call_grid = build_market_price_grid(calls_df, "C_mkt", KK_cm, TT_cm)
            market_put_grid = build_market_price_grid(puts_df, "P_mkt", KK_cm, TT_cm)

            tab_calls, tab_puts = st.tabs(["📈 Calls", "📉 Puts"])

            pad = 10.0
            call_min = float(np.nanmin(call_prices_cm)) if call_prices_cm.size > 0 else 0.0
            call_max = float(np.nanmax(call_prices_cm)) if call_prices_cm.size > 0 else 0.0
            call_zmin = call_min - pad
            call_zmax = call_max + pad

            put_min = float(np.nanmin(put_prices_cm)) if put_prices_cm.size > 0 else 0.0
            put_max = float(np.nanmax(put_prices_cm)) if put_prices_cm.size > 0 else 0.0
            put_zmin = put_min - pad
            put_zmax = put_max + pad

            fig_heat_call_cm = go.Figure(
                data=[
                    go.Heatmap(
                        z=call_prices_cm,
                        x=K_grid,
                        y=T_grid,
                        colorscale="Viridis",
                        colorbar=dict(title="Prix des Call Heston"),
                        zmin=call_zmin,
                        zmax=call_zmax,
                    )
                ]
            )
            fig_heat_put_cm = go.Figure(
                data=[
                    go.Heatmap(
                        z=put_prices_cm,
                        x=K_grid,
                        y=T_grid,
                        colorscale="Viridis",
                        colorbar=dict(title="Prix des Put Heston"),
                        zmin=put_zmin,
                        zmax=put_zmax,
                    )
                ]
            )

            with tab_calls:
                st.subheader("Carr-Madan : IV & prix (Calls)")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig_call_cm, use_container_width=True)
                with c2:
                    st.plotly_chart(fig_heat_call_cm, use_container_width=True)

                st.subheader("Marché : IV & prix (Calls)")
                c3, c4 = st.columns(2)
                with c3:
                    if fig_call_market:
                        st.plotly_chart(fig_call_market, use_container_width=True)
                    else:
                        st.info("Pas assez de points marché pour la surface call.")
                with c4:
                    if market_call_grid is not None:
                        fig_heat_call_mkt = go.Figure(
                            data=[
                                go.Heatmap(
                                    z=market_call_grid,
                                    x=K_grid,
                                    y=T_grid,
                                    colorscale="Plasma",
                                    colorbar=dict(title="Prix des Call Marché"),
                                    zmin=call_zmin,
                                    zmax=call_zmax,
                                )
                            ]
                        )
                        st.plotly_chart(fig_heat_call_mkt, use_container_width=True)
                    else:
                        st.info("Pas assez de points marché pour la heatmap call.")

                st.markdown(
                    f"""
**Lecture des heatmaps (Calls)**
- Axe des abscisses **K** : strikes autour de S0 ≈ `{S0_ref:.2f}`
- Axe des ordonnées **T** : maturité en années
- Couleur : niveau de **prix du call** pour chaque couple (K, T)
- Paramètres utilisés : `S0 = {S0_ref:.2f}`, `r = {rf_rate:.3f}`, `q = {div_yield:.3f}`
"""
                )

            with tab_puts:
                st.subheader("Carr-Madan : IV & prix (Puts)")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig_put_cm, use_container_width=True)
                with c2:
                    st.plotly_chart(fig_heat_put_cm, use_container_width=True)

                st.subheader("Marché : IV & prix (Puts)")
                c3, c4 = st.columns(2)
                with c3:
                    if fig_put_market:
                        st.plotly_chart(fig_put_market, use_container_width=True)
                    else:
                        st.info("Pas assez de points marché pour la surface put.")
                with c4:
                    if market_put_grid is not None:
                        fig_heat_put_mkt = go.Figure(
                            data=[
                                go.Heatmap(
                                    z=market_put_grid,
                                    x=K_grid,
                                    y=T_grid,
                                    colorscale="Plasma",
                                    colorbar=dict(title="Prix des Put Marché"),
                                    zmin=put_zmin,
                                    zmax=put_zmax,
                                )
                            ]
                        )
                        st.plotly_chart(fig_heat_put_mkt, use_container_width=True)
                    else:
                        st.info("Pas assez de points marché pour la heatmap put.")

                st.markdown(
                    f"""
**Lecture des heatmaps (Puts)**
- Axe des abscisses **K** : strikes autour de S0 ≈ `{S0_ref:.2f}`
- Axe des ordonnées **T** : maturité en années
- Couleur : niveau de **prix du put** pour chaque couple (K, T)
- Paramètres utilisés : `S0 = {S0_ref:.2f}`, `r = {rf_rate:.3f}`, `q = {div_yield:.3f}`
"""
                )

            st.balloons()
            st.success("🎉 Analyse terminée")

        except Exception as exc:
            st.error(f"❌ Erreur : {exc}")
            import traceback

            st.code(traceback.format_exc())
# ---------------------------------------------------------------------------
#  Application Streamlit unifiée
# ---------------------------------------------------------------------------


st.title("Application unifiée de pricing d'options")

st.sidebar.header("Paramètres communs")
S0_common = st.sidebar.number_input("S0 (spot)", value=100.0, min_value=0.01, key="S0_common")
K_common = st.sidebar.number_input("K (strike)", value=100.0, min_value=0.01, key="K_common")
T_common = st.sidebar.number_input("T (maturité, années)", value=1.0, min_value=0.01, key="T_common")
sigma_common = st.sidebar.number_input("Volatilité σ", value=0.2, min_value=0.0001, key="sigma_common")
r_common = st.sidebar.number_input("Taux sans risque r", value=0.05, key="r_common")
d_common = st.sidebar.number_input("Dividende continu d", value=0.0, key="d_common")
heatmap_span = st.sidebar.number_input(
    "Span autour du spot (heatmaps)",
    value=25.0,
    min_value=0.1,
    help="Définit l'écart symétrique autour du spot utilisé pour les axes Spot / Strike des heatmaps.",
    key="heatmap_span",
)
heatmap_spot_values = _heatmap_axis(S0_common, heatmap_span)
heatmap_strike_values = _heatmap_axis(K_common, heatmap_span)
heatmap_maturity_values = _heatmap_axis(T_common, T_common * 0.5)

common_spot_value = float(S0_common)
common_maturity_value = float(T_common)
common_strike_value = float(K_common)
common_rate_value = float(r_common)
common_sigma_value = float(sigma_common)

st.session_state["common_spot"] = common_spot_value
st.session_state["common_strike"] = common_strike_value
st.session_state["common_maturity"] = common_maturity_value
st.session_state["common_sigma"] = common_sigma_value
st.session_state["common_rate"] = common_rate_value
st.session_state["common_dividend"] = float(d_common)
st.session_state["heatmap_span_value"] = float(heatmap_span)

(
    tab_european,
    tab_american,
    tab_lookback,
    tab_barrier,
    tab_bermudan,
    tab_basket,
    tab_asian,
) = st.tabs(["Européenne", "Américaine", "Lookback", "Barrière", "Bermuda", "Basket", "Asian"])


with tab_european:
    st.header("Option européenne")
    st.caption(
        "Les heatmaps affichent les prix call / put sur un carré Spot × Strike centré autour du spot défini dans la"
        " barre latérale."
    )

    tab_eu_heston, tab_eu_bsm, tab_eu_mc = st.tabs(
        ["Heston", "Black–Scholes–Merton", "Monte Carlo"]
    )

    with tab_eu_heston:
        ui_heston_full_pipeline()

    with tab_eu_bsm:
        st.subheader("Formule fermée BSM")
        call_heatmap_bsm, put_heatmap_bsm = _compute_bsm_heatmaps(
            heatmap_spot_values,
            heatmap_strike_values,
            T_common,
            r_common - d_common,
            sigma_common,
        )
        _render_call_put_heatmaps("BSM", call_heatmap_bsm, put_heatmap_bsm, heatmap_spot_values, heatmap_strike_values)

    with tab_eu_mc:
        st.subheader("Monte Carlo classique")
        n_paths_eu = st.number_input("Trajectoires Monte Carlo", value=10_000, min_value=100, key="n_paths_eu")
        n_steps_eu = st.number_input("Pas de temps", value=50, min_value=1, key="n_steps_eu")
        with st.spinner("Calcul des heatmaps Monte Carlo"):
            call_heatmap_mc, put_heatmap_mc = _compute_mc_heatmaps(
                heatmap_spot_values,
                heatmap_strike_values,
                T_common,
                r_common - d_common,
                sigma_common,
                int(n_paths_eu),
                int(n_steps_eu),
            )
        _render_call_put_heatmaps(
            "Monte Carlo", call_heatmap_mc, put_heatmap_mc, heatmap_spot_values, heatmap_strike_values
        )

with tab_american:
    st.header("Option américaine")
    cpflag_am = st.selectbox("Call / Put (américaine)", ["Call", "Put"], key="cpflag_am")
    cpflag_am_char = "c" if cpflag_am == "Call" else "p"
    st.caption(
        "Les heatmaps affichent les prix call / put sur un carré Spot × Strike centré autour du spot défini dans la barre latérale."
    )

    tab_am_ls, tab_am_crr = st.tabs(
        ["Longstaff–Schwartz", "Arbre CRR"]
    )

    with tab_am_ls:
        st.subheader("Monte Carlo Longstaff–Schwartz")
        process_type_am = st.selectbox(
            "Processus sous-jacent", ["Geometric Brownian Motion", "Heston"], key="process_type_am"
        )
        n_paths_am = st.number_input("Trajectoires Monte Carlo", value=1000, min_value=100, key="n_paths_am")
        n_steps_am = st.number_input("Pas de temps", value=50, min_value=1, key="n_steps_am")

        if process_type_am == "Geometric Brownian Motion":
            process_am = GeometricBrownianMotion(mu=r_common - d_common, sigma=sigma_common)
            v0_am = None
        else:
            kappa_am = st.number_input("κ (vitesse de rappel)", value=2.0, key="kappa_am")
            theta_am = st.number_input("θ (variance long terme)", value=0.04, key="theta_am")
            eta_am = st.number_input("η (vol de la variance)", value=0.5, key="eta_am")
            rho_am = st.number_input("ρ (corrélation)", value=-0.7, min_value=-0.99, max_value=0.99, key="rho_am")
            v0_am = st.number_input("v0 (variance initiale)", value=0.04, min_value=0.0001, key="v0_am")
            process_am = HestonProcess(
                mu=r_common - d_common, kappa=kappa_am, theta=theta_am, eta=eta_am, rho=rho_am
            )

        with st.spinner("Calcul des heatmaps Longstaff–Schwartz"):
            call_heatmap_ls, put_heatmap_ls = _compute_american_ls_heatmaps(
                heatmap_spot_values,
                heatmap_strike_values,
                T_common,
                process_am,
                int(n_paths_am),
                int(n_steps_am),
                v0_am,
            )
        _render_call_put_heatmaps(
            "Longstaff–Schwartz", call_heatmap_ls, put_heatmap_ls, heatmap_spot_values, heatmap_strike_values
        )

    with tab_am_crr:
        st.subheader("Arbre binomial CRR")
        n_tree_am = st.number_input("Nombre de pas de l'arbre", value=10, min_value=5, key="n_tree_am")
        option_am_crr = Option(s0=S0_common, T=T_common, K=K_common, call=cpflag_am == "Call")
        int_n_tree = int(n_tree_am)
        if int_n_tree > 10:
            st.info("L'affichage peut devenir difficile à lire pour un nombre de pas supérieur à 10.")
        with st.spinner("Construction de l'arbre CRR"):
            spot_tree, value_tree = _build_crr_tree(
                option=option_am_crr, r=r_common, sigma=sigma_common, n_steps=int_n_tree
            )
        st.write("**Représentation graphique**")
        fig_tree = _plot_crr_tree(spot_tree, value_tree)
        st.pyplot(fig_tree)
        plt.close(fig_tree)
        
        with st.spinner("Calcul de la heatmap CRR"):
            call_heatmap_crr, put_heatmap_crr = _compute_american_crr_heatmaps(
                heatmap_spot_values,
                heatmap_strike_values,
                T_common,
                r_common,
                sigma_common,
                int_n_tree,
            )
        if cpflag_am == "Call":
            st.write(f"Heatmap {cpflag_am} (CRR)")
            _render_heatmap(call_heatmap_crr, heatmap_spot_values, heatmap_strike_values, f"{cpflag_am} (CRR)")
        else:
            st.write(f"Heatmap {cpflag_am} (CRR)")
            _render_heatmap(put_heatmap_crr, heatmap_spot_values, heatmap_strike_values, f"{cpflag_am} (CRR)")


with tab_lookback:
    st.header("Options lookback (floating strike)")
    st.caption(
        "Les heatmaps affichent les prix lookback sur un carré Spot × Maturité centré autour des valeurs définies dans la barre latérale."
    )

    tab_lb_exact, tab_lb_mc = st.tabs(["Exacte", "Monte Carlo"])

    with tab_lb_exact:
        st.subheader("Formule exacte")
        t0_lb = st.number_input("t (temps courant)", value=0.0, min_value=0.0, key="t0_lb_exact")
        with st.spinner("Calcul de la heatmap exacte"):
            heatmap_lb_exact = _compute_lookback_exact_heatmap(
                heatmap_spot_values,
                heatmap_maturity_values,
                t0_lb,
                r_common,
                sigma_common,
            )
        st.write("Heatmap Lookback (formule exacte)")
        _render_heatmap(heatmap_lb_exact, heatmap_spot_values, heatmap_maturity_values, "Prix Lookback (Exact)")

    with tab_lb_mc:
        st.subheader("Monte Carlo lookback")
        t0_lb_mc = st.number_input("t (temps courant) MC", value=0.0, min_value=0.0, key="t0_lb_mc")
        n_iters_lb = st.number_input("Itérations Monte Carlo", value=1000, min_value=100, key="n_iters_lb_mc")
        with st.spinner("Calcul de la heatmap Monte Carlo"):
            heatmap_lb_mc = _compute_lookback_mc_heatmap(
                heatmap_spot_values,
                heatmap_maturity_values,
                t0_lb_mc,
                r_common,
                sigma_common,
                int(n_iters_lb),
            )
        st.write("Heatmap Lookback (Monte Carlo)")
        _render_heatmap(heatmap_lb_mc, heatmap_spot_values, heatmap_maturity_values, "Prix Lookback (MC)")


with tab_barrier:
    st.header("Options barrière")
    (
        tab_barrier_up_out,
        tab_barrier_down_out,
        tab_barrier_up_in,
        tab_barrier_down_in,
    ) = st.tabs(["Up-and-out", "Down-and-out", "Up-and-in", "Down-and-in"])

    with tab_barrier_up_out:
        st.subheader("Up-and-out")
        cpflag_barrier_up = st.selectbox("Call / Put", ["Call", "Put"], key="cpflag_barrier_up")
        cpflag_barrier_up_char = "c" if cpflag_barrier_up == "Call" else "p"
        Hu_up = st.number_input("Barrière haute Hu", value=max(110.0, S0_common * 1.1), min_value=S0_common, key="Hu_up")
        n_paths_up = st.number_input(
            "Trajectoires Monte Carlo", value=1000, min_value=500, step=500, key="n_paths_barrier_up"
        )
        n_steps_up = st.number_input("Pas de temps MC", value=200, min_value=10, key="n_steps_barrier_up")

        if st.button("Calculer (Up-and-out)", key="btn_barrier_up"):
            with st.spinner("Simulation Monte Carlo en cours..."):
                price = _barrier_monte_carlo_price(
                    option_type=cpflag_barrier_up_char,
                    barrier_type="up",
                    S0=S0_common,
                    K=K_common,
                    barrier=Hu_up,
                    T=T_common,
                    r=r_common,
                    dividend=d_common,
                    sigma=sigma_common,
                    n_paths=int(n_paths_up),
                    n_steps=int(n_steps_up),
                )
            st.write(f"**Prix Monte Carlo barrière**: {price:.6f}")

        st.caption(f"Rappel : S0 = {S0_common:.4f}, Hu = {Hu_up:.4f}")

    with tab_barrier_down_out:
        st.subheader("Down-and-out")
        cpflag_barrier_down = st.selectbox("Call / Put", ["Call", "Put"], key="cpflag_barrier_down")
        cpflag_barrier_down_char = "c" if cpflag_barrier_down == "Call" else "p"
        Hd_down = st.number_input(
            "Barrière basse Hd", value=max(1.0, S0_common * 0.8), min_value=0.0001, key="Hd_down"
        )
        n_paths_down = st.number_input(
            "Trajectoires Monte Carlo", value=1000, min_value=500, step=500, key="n_paths_barrier_down"
        )
        n_steps_down = st.number_input("Pas de temps MC", value=200, min_value=10, key="n_steps_barrier_down")

        if st.button("Calculer (Down-and-out)", key="btn_barrier_down"):
            with st.spinner("Simulation Monte Carlo en cours..."):
                price = _barrier_monte_carlo_price(
                    option_type=cpflag_barrier_down_char,
                    barrier_type="down",
                    S0=S0_common,
                    K=K_common,
                    barrier=Hd_down,
                    T=T_common,
                    r=r_common,
                    dividend=d_common,
                    sigma=sigma_common,
                    n_paths=int(n_paths_down),
                    n_steps=int(n_steps_down),
                )
            st.write(f"**Prix Monte Carlo barrière**: {price:.6f}")
        
        st.caption(f"Rappel : S0 = {S0_common:.4f}, Hd = {Hd_down:.4f}")

    with tab_barrier_up_in:
        st.subheader("Up-and-in")
        cpflag_barrier_up_in = st.selectbox("Call / Put", ["Call", "Put"], key="cpflag_barrier_up_in")
        cpflag_barrier_up_in_char = "c" if cpflag_barrier_up_in == "Call" else "p"
        Hu_up_in = st.number_input(
            "Barrière haute Hu (Up-in)", value=max(110.0, S0_common * 1.1), min_value=S0_common, key="Hu_up_in"
        )
        n_paths_up_in = st.number_input(
            "Trajectoires Monte Carlo (Up-in)", value=1000, min_value=500, step=500, key="n_paths_barrier_up_in"
        )
        n_steps_up_in = st.number_input(
            "Pas de temps MC (Up-in)", value=200, min_value=10, key="n_steps_barrier_up_in"
        )

        if st.button("Calculer (Up-and-in)", key="btn_barrier_up_in"):
            with st.spinner("Monte Carlo knock-in (Up)..."):
                price = _barrier_monte_carlo_price(
                    option_type=cpflag_barrier_up_in_char,
                    barrier_type="up",
                    S0=S0_common,
                    K=K_common,
                    barrier=Hu_up_in,
                    T=T_common,
                    r=r_common,
                    dividend=d_common,
                    sigma=sigma_common,
                    n_paths=int(n_paths_up_in),
                    n_steps=int(n_steps_up_in),
                    knock_in=True,
                )
            st.write(f"**Prix Monte Carlo knock-in**: {price:.6f}")

        st.caption(f"Rappel : S0 = {S0_common:.4f}, Hu = {Hu_up_in:.4f}")

    with tab_barrier_down_in:
        st.subheader("Down-and-in")
        cpflag_barrier_down_in = st.selectbox("Call / Put", ["Call", "Put"], key="cpflag_barrier_down_in")
        cpflag_barrier_down_in_char = "c" if cpflag_barrier_down_in == "Call" else "p"
        Hd_down_in = st.number_input(
            "Barrière basse Hd (Down-in)", value=max(1.0, S0_common * 0.8), min_value=0.0001, key="Hd_down_in"
        )
        n_paths_down_in = st.number_input(
            "Trajectoires Monte Carlo (Down-in)",
            value=1000,
            min_value=500,
            step=500,
            key="n_paths_barrier_down_in",
        )
        n_steps_down_in = st.number_input(
            "Pas de temps MC (Down-in)", value=200, min_value=10, key="n_steps_barrier_down_in"
        )

        if st.button("Calculer (Down-and-in)", key="btn_barrier_down_in"):
            with st.spinner("Monte Carlo knock-in (Down)..."):
                price = _barrier_monte_carlo_price(
                    option_type=cpflag_barrier_down_in_char,
                    barrier_type="down",
                    S0=S0_common,
                    K=K_common,
                    barrier=Hd_down_in,
                    T=T_common,
                    r=r_common,
                    dividend=d_common,
                    sigma=sigma_common,
                    n_paths=int(n_paths_down_in),
                    n_steps=int(n_steps_down_in),
                    knock_in=True,
                )
            st.write(f"**Prix Monte Carlo knock-in**: {price:.6f}")


with tab_bermudan:
    st.header("Option bermudéenne")
    cpflag_bmd = st.selectbox("Call / Put (bermuda)", ["Call", "Put"], key="cpflag_bmd")
    cpflag_bmd_char = "c" if cpflag_bmd == "Call" else "p"
    n_ex_dates_bmd = st.number_input(
        "Nombre de dates d'exercice Bermude",
        value=6,
        min_value=2,
        help="Les dates sont réparties uniformément sur la grille PDE (incluant l'échéance).",
        key="n_ex_dates_bmd",
    )

    if st.button("Calculer (PDE Bermuda)", key="btn_bmd_cn"):
        model_bmd = CrankNicolsonBS(
            Typeflag="Bmd",
            cpflag=cpflag_bmd_char,
            S0=S0_common,
            K=K_common,
            T=T_common,
            vol=sigma_common,
            r=r_common,
            d=d_common,
            n_exercise_dates=int(n_ex_dates_bmd),
        )
        price_bmd, delta_bmd, gamma_bmd, theta_bmd = model_bmd.CN_option_info()
        st.write(f"**Prix**: {price_bmd:.4f}")
        st.write(f"**Delta**: {delta_bmd:.4f}")
        st.write(f"**Gamma**: {gamma_bmd:.4f}")
        st.write(f"**Theta**: {theta_bmd:.4f}")


with tab_basket:
    ui_basket_surface(
        spot_common=common_spot_value,
        maturity_common=common_maturity_value,
        rate_common=common_rate_value,
        strike_common=common_strike_value,
    )


with tab_asian:
    ui_asian_options(
        spot_default=common_spot_value,
        sigma_common=common_sigma_value,
        maturity_common=common_maturity_value,
        strike_common=common_strike_value,
        rate_common=common_rate_value,
    )

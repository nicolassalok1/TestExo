import io
import math
import os
import subprocess
import sys
from pathlib import Path
import time
import re
import base64

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


PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["sendDataToCloud"],
}


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
    render_unlock_sidebar_button("tab_basket", "🔓 Réactiver T (onglet Basket)")

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
    render_unlock_sidebar_button("tab_asian", "🔓 Réactiver T (onglet Asian)")
    render_general_definition_explainer(
        "🌏 Comprendre les options asiatiques",
        (
            "- **Spécificité du payoff** : pour une option asiatique arithmétique, le payoff dépend de la moyenne des prix du sous‑jacent observés à différentes dates entre `0` et `T`, plutôt que du seul `S_T`.\n"
            "- **Effet de lissage** : cette moyenne réduit l’impact des pics de volatilité ponctuels et donne un profil de risque plus \"lissé\" que pour une option européenne standard.\n"
            "- **Conséquences sur le prix** : à paramètres identiques, une option asiatique est généralement moins chère que son équivalent européen car elle réagit moins aux extrêmes de la trajectoire.\n"
            "- **Usage pratique** : ces produits sont fréquemment utilisés dans l’énergie, les matières premières ou les produits structurés pour lisser l’exposition à des prix très volatils.\n"
            "- **Objectif du module** : illustrer le pricing d’options asiatiques par simulation Monte Carlo, avec des variates antithétiques et un contrôle par une option de référence."
        ),
    )
    render_method_explainer(
        "🧮 Méthode Monte Carlo + control variate",
        (
            "- **Étape 1 – Paramétrage de la grille** : pour chaque couple `(K, T)` de la grille choisie, on fixe un nombre d’observations `n_obs` le long de `[0, T]` et un nombre de trajectoires Monte Carlo `n_paths_surface`.\n"
            "- **Étape 2 – Simulation des trajectoires de `S_t`** : pour un spot initial donné, on simule sous la mesure neutre au risque `n_paths_surface` trajectoires du sous‑jacent en découpant `[0, T]` en `n_obs` pas. À chaque pas, on applique le schéma d’Euler du GBM.\n"
            "- **Étape 3 – Utilisation des variates antithétiques** : pour chaque suite de chocs gaussiens utilisée pour générer une trajectoire, on génère une trajectoire \"miroir\" avec les chocs opposés. On obtient ainsi des paires de trajectoires fortement corrélées qui réduisent la variance de l’estimateur.\n"
            "- **Étape 4 – Calcul de la moyenne arithmétique** : sur chaque trajectoire, on calcule la moyenne arithmétique des `S_t` observés aux dates de la grille. Cette moyenne est ensuite utilisée pour déterminer le payoff asiatique (call ou put) à l’échéance.\n"
            "- **Étape 5 – Construction d’une variable de contrôle** : en parallèle, on calcule pour chaque trajectoire le payoff d’une option de référence (par exemple une option européenne ou une option asiatique géométrique) dont on connaît une formule de prix fermée.\n"
            "- **Étape 6 – Correction par control variate** : on corrige l’estimation brute du payoff asiatique en soustrayant la composante due à la variable de contrôle, puis en réajoutant l’espérance théorique de cette variable. Cela réduit significativement la variance de l’estimateur final.\n"
            "- **Étape 7 – Actualisation et moyenne** : les payoffs corrigés sont actualisés au taux `rate_common` jusqu’à la date présente et moyennés sur toutes les trajectoires.\n"
            "- **Étape 8 – Remplissage des surfaces** : on répète ce processus pour chaque point `(K, T)` de la grille, ce qui remplit deux matrices de prix (call et put) utilisées pour tracer les surfaces de prix asiatiques."
        ),
    )
    render_inputs_explainer(
        "🔧 Paramètres utilisés – module Asian",
        (
            "- **\"S0 (spot)\"** (via les paramètres communs) : niveau de départ des trajectoires asiatiques.\n"
            "- **\"K (strike)\"** : strike de référence utilisé pour centrer la plage de strikes.\n"
            "- **\"T (maturité, années)\"** : maturité de référence utilisée pour initialiser la plage de maturités.\n"
            "- **\"Taux sans risque r\"** : intervient dans l’actualisation et le drift neutre au risque.\n"
            "- **\"Volatilité σ\"** : volatilité utilisée pour simuler les trajectoires du sous‑jacent.\n"
            "- **\"K min\" / \"K max\"** : bornes de la plage de strikes sur l’axe horizontal des surfaces.\n"
            "- **\"T min (années)\" / \"T max (années)\"** : bornes de la plage de maturités sur l’axe vertical.\n"
            "- **\"Résolution en K\"** et **\"Résolution en T\"** : nombres de points de grille en strike et en maturité.\n"
            "- **\"Nombre de trajectoires Monte Carlo\"** : nombre de trajectoires utilisées pour estimer chaque point de la surface."
        ),
    )
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

    if st.button(
        "Calculer le prix asiatique (Call)",
        key="btn_price_asian",
    ):
        try:
            n_obs_price = max(2, int(50 * float(maturity_common)))
            price_asian_call, _, _ = asian_mc_control_variate(
                spot=float(spot_common),
                strike=float(strike_common_local),
                rate=float(rate_common),
                sigma=float(sigma),
                maturity=float(maturity_common),
                n_obs=int(n_obs_price),
                n_paths=20_000,
                option_type="call",
                antithetic=True,
                seed=None,
            )
            st.success(f"Prix call asiatique arithmétique (MC + control variate) = {price_asian_call:.6f}")
        except Exception as exc:
            st.error(f"Erreur lors du pricing asiatique : {exc}")
    st.caption(
        f"Paramètres utilisés pour le prix asiatique : "
        f"S0={spot_common:.4f}, K={strike_common_local:.4f}, "
        f"T={maturity_common:.4f}, r={rate_common:.4f}, σ={sigma:.4f}"
    )

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
            help="Borne inférieure de la plage de strikes pour les options asiatiques.",
        )
        k_max = st.number_input(
            "K max",
            value=float(default_k_max),
            min_value=k_min + 1.0,
            step=1.0,
            key="asian_k_max",
            help="Borne supérieure de la plage de strikes pour les options asiatiques.",
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
            help="Maturité minimale de la surface asiatique (en années).",
        )
        t_max = st.number_input(
            "T max (années)",
            value=float(default_t_max),
            min_value=t_min + 0.01,
            step=0.05,
            key="asian_t_max",
            help="Maturité maximale de la surface asiatique (en années).",
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


def render_section_explainer(title: str, body: str) -> None:
    """Affiche un menu déroulant descriptif pour guider l'utilisateur."""
    with st.expander(title):
        st.markdown(body)


def render_general_definition_explainer(title: str, body: str) -> None:
    """Affiche la définition générale d'une classe d'options, avec un ton pédagogique."""
    with st.expander(title):
        st.markdown(body)


def render_method_explainer(title: str, body: str) -> None:
    """Affiche une explication détaillée de la méthode de calcul utilisée dans un sous-onglet."""
    with st.expander(title):
        st.markdown(body)


def render_inputs_explainer(title: str, body: str) -> None:
    """Décrit les paramètres d'entrée effectivement utilisés par une méthode de calcul."""
    with st.expander(title):
        st.markdown(body)


def render_unlock_sidebar_button(context_key: str, label: str) -> None:
    """Affiche un bouton permettant de réactiver l'input T lorsque Heston a verrouillé la barre latérale."""
    if st.session_state.get("heston_tab_locked"):
        if st.button(label, key=f"unlock_sidebar_{context_key}"):
            st.session_state["heston_tab_locked"] = False
            st.rerun()


ASIAN_LATEX_DERIVATION = r"""
**Modèle sous la mesure risque-neutre**

Sous la mesure risque-neutre $\mathbb{Q}$, le sous-jacent suit
\[
dS_t = (r-q)\,S_t\,dt + \sigma S_t\,dW_t, \qquad S_0>0,
\]
où $r$ est le taux sans risque, $q$ le dividende continu et $W_t$ un mouvement brownien standard.
La solution explicite s'écrit
\[
S_t = S_0 \exp\Big[(r-q-\tfrac12\sigma^2)t + \sigma W_t\Big].
\]

**Option asiatique géométrique**

On définit la moyenne géométrique continue
\[
G_T = \exp\!\left(\frac{1}{T}\int_0^T \ln S_t\,dt\right),
\]
et le payoff d'un call géométrique $(G_T-K)^+$.
En partant de
\[
\ln S_t = \ln S_0 + (r-q-\tfrac12\sigma^2)t + \sigma W_t,
\]
on montre que
\[
\ln G_T = \frac{1}{T}\int_0^T \ln S_t\,dt
= \ln S_0 + (r-q-\tfrac12\sigma^2)\frac{T}{2}
  + \sigma Y,
\]
avec
\[
Y = \frac{1}{T}\int_0^T W_t\,dt \sim \mathcal{N}\!\Big(0,\tfrac{T}{3}\Big).
\]
Ainsi, $\ln G_T$ est gaussien de moyenne $\mu_G$ et variance $v_G$ :
\[
\mu_G = \ln S_0 + (r-q-\tfrac12\sigma^2)\frac{T}{2},\qquad
v_G = \sigma^2\frac{T}{3},
\]
ce qui implique que $G_T$ est lognormal. On introduit une volatilité effective
\[
\tilde{\sigma} = \frac{\sigma}{\sqrt{3}},
\]
et un niveau initial ajusté $\tilde{S}_0$ (obtenu à partir de la moyenne de $\ln G_T$) de sorte que le pricing du call géométrique s'écrive sous une forme de type Black--Scholes :
\[
C_0^{\mathrm{geom}} = \tilde{S}_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2),
\]
avec
\[
d_1 = \frac{\ln(\tilde{S}_0/K) + (r-q + \tfrac12 \tilde{\sigma}^2)T}{\tilde{\sigma}\sqrt{T}},
\qquad
d_2 = d_1 - \tilde{\sigma}\sqrt{T}.
\]

**Option asiatique arithmétique et PDE associée**

La moyenne arithmétique continue est
\[
A_T = \frac{1}{T}\int_0^T S_t\,dt,
\]
et le payoff du call arithmétique est $(A_T-K)^+$.
Comme ce payoff dépend du chemin complet, on introduit le processus d'intégrale
\[
I_t = \int_0^t S_u\,du,
\]
de sorte que $A_T = I_T/T$.
Le couple $(S_t,I_t)$ est markovien et suit
\[
dS_t = (r-q)S_t\,dt + \sigma S_t\,dW_t, \qquad dI_t = S_t\,dt.
\]

On définit la fonction de valeur
\[
V(t,s,i) = \mathbb{E}^{\mathbb{Q}}\!\left[e^{-r(T-t)}\Big(\tfrac{I_T}{T} - K\Big)^+ \,\big|\, S_t=s,\,I_t=i\right],
\]
avec condition terminale
\[
V(T,s,i) = \Big(\tfrac{i}{T} - K\Big)^+.
\]
Le générateur infinitésimal du couple $(S_t,I_t)$ est
\[
\mathcal{L}V = (r-q)s\,V_s + \tfrac12\sigma^2 s^2\,V_{ss} + s\,V_i,
\]
et, par le théorème de Feynman--Kac, $V$ vérifie la PDE de valorisation
\[
\frac{\partial V}{\partial t}
 + (r-q)s \frac{\partial V}{\partial s}
 + \tfrac12 \sigma^2 s^2 \frac{\partial^2 V}{\partial s^2}
 + s \frac{\partial V}{\partial i}
 - r V = 0,
\]
sur $[0,T)\times (0,\infty)\times (0,\infty)$, avec la condition terminale ci‑dessus.
Cette PDE n'admet pas de solution fermée simple et doit être résolue numériquement (schémas aux différences finies, méthodes spectrales, approches Monte Carlo avancées).
"""


def render_math_derivation(title: str, body_md: str) -> None:
    """Affiche un menu déroulant contenant la dérivation mathématique, rendue avec LaTeX."""
    with st.expander(title):
        st.markdown(body_md)


def render_pdf_derivation(title: str, pdf_path: str, download_name: str | None = None) -> None:
    """
    Affiche, dans un menu déroulant, un PDF (par exemple une dérivation LaTeX compilée).
    Le PDF est encodé en base64 et inclus dans une balise <iframe>.
    """
    from pathlib import Path as _Path

    with st.expander(title):
        path = _Path(pdf_path)
        if not path.exists():
            st.info(
                f"Le fichier PDF '{pdf_path}' n'a pas été trouvé. "
                "Placez le PDF compilé à cet emplacement pour l'afficher ici."
            )
            return

        with path.open("rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        pdf_display = f"""
<iframe
    src="data:application/pdf;base64,{base64_pdf}"
    width="100%"
    height="700"
    type="application/pdf"
></iframe>
"""
        st.markdown(pdf_display, unsafe_allow_html=True)



def ui_heston_full_pipeline():
    st.header("Surface IV Heston")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        ticker = st.text_input(
            "Ticker (sous-jacent)",
            value="SPY",
            key="heston_cboe_ticker",
            help="Code du sous-jacent coté au CBOE utilisé pour la calibration Heston.",
        ).strip().upper()
        rf_rate = float(st.session_state.get("common_rate", 0.02))
        div_yield = float(st.session_state.get("common_dividend", 0.0))

        with col_cfg2:
            span_mc = float(st.session_state.get("heatmap_span_value", 20.0))
            n_maturities = 40


    state = st.session_state
    if "heston_calls_df" not in state:
        state.heston_calls_df = None
        state.heston_puts_df = None
        state.heston_S0_ref = None
        state.heston_calib_T_target = None

    fetch_btn = st.button("Récupérer les données du ticker", width="stretch", key="heston_cboe_fetch")
    st.divider()

    if fetch_btn:
        try:
            calls_df, puts_df, S0_ref = load_cboe_data(ticker)
            state.heston_calls_df = calls_df
            state.heston_puts_df = puts_df
            state.heston_S0_ref = S0_ref
            st.info(f"📡 Données CBOE chargées pour {ticker} (cache)")
            st.success(f"{len(calls_df)} calls, {len(puts_df)} puts | S0 ≈ {S0_ref:.2f}")
            maturity_list = sorted(calls_df["T"].round(2).unique().tolist())
            st.session_state["sidebar_maturity_options"] = maturity_list
            span_sync = float(st.session_state.get("heatmap_span_value", 20.0))
            if maturity_list:
                rnd_T = float(np.random.choice(maturity_list))
            else:
                rnd_T = float(round(calls_df["T"].iloc[0], 2))
            eligible_calls = calls_df[
                (calls_df["T"].round(2) == rnd_T)
                & calls_df["K"].between(S0_ref - span_sync, S0_ref + span_sync)
            ]
            if eligible_calls.empty:
                eligible_calls = calls_df[
                    calls_df["K"].between(S0_ref - span_sync, S0_ref + span_sync)
                ]
            if eligible_calls.empty:
                eligible_calls = calls_df
            chosen_row = eligible_calls.sample(1).iloc[0]
            chosen_K = float(chosen_row["K"])
            chosen_T = float(round(chosen_row["T"], 2))
            sigma_pick = float(chosen_row.get("iv_market") or np.nan)
            if not np.isfinite(sigma_pick):
                sigma_pick = implied_vol_option(
                    float(chosen_row.get("C_mkt", np.nan)),
                    float(chosen_row.get("S0", S0_ref)),
                    chosen_K,
                    float(chosen_row["T"]),
                    rf_rate,
                    "call",
                )
            if not np.isfinite(sigma_pick):
                sigma_pick = float(st.session_state.get("sigma_common", 0.2))
            prefills = {
                "S0_common": float(S0_ref),
                "K_common": chosen_K,
                "sigma_common": float(np.clip(sigma_pick, 0.01, 5.0)),
            }
            st.session_state["heston_sidebar_prefill"] = prefills
            st.session_state["heston_sidebar_placeholders"] = {
                "S0_common": f"{prefills['S0_common']:.2f}",
                "K_common": f"{prefills['K_common']:.2f}",
                "sigma_common": f"{prefills['sigma_common']:.4f}",
            }
            st.rerun()
        except Exception as exc:
            st.error(f"❌ Erreur lors du téléchargement des données CBOE : {exc}")

    calls_df = state.heston_calls_df
    puts_df = state.heston_puts_df
    S0_ref = state.heston_S0_ref
    calib_T_target = state.heston_calib_T_target

    calib_band_range: tuple[float, float] | None = None
    calib_T_band = 0.4
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
                help="Largeur de la bande de maturités autour de la cible utilisée pour la calibration.",
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
                    help="Maturité autour de laquelle la calibration Heston est centrée.",
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
                    help="Choisit un compromis entre vitesse de calibration et précision de l’ajustement.",
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
        run_button = st.button("🚀 Lancer l'analyse", type="primary", width="stretch", key="heston_cboe_run")
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
            state.heston_params_cm = params_cm
            state.heston_params_meta = {
                "r": rf_rate,
                "q": div_yield,
                "S0_ref": float(S0_ref),
            }
            # Met à jour les paramètres Heston globaux, qui alimentent la sidebar
            st.session_state["heston_kappa_common"] = params_dict["kappa"]
            st.session_state["heston_theta_common"] = params_dict["theta"]
            st.session_state["heston_eta_common"] = params_dict["sigma"]
            st.session_state["heston_rho_common"] = params_dict["rho"]
            st.session_state["heston_v0_common"] = params_dict["v0"]
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

            # Paramètres communs utilisés pour le pricing ponctuel Heston
            common_S0 = float(st.session_state.get("common_spot", S0_ref))
            # En absence de strike commun explicite, on prend le spot comme valeur de repli
            common_K = float(st.session_state.get("common_strike", S0_ref))
            # En absence de maturité commune, on prend la maturité cible de calibration ou la première de la grille
            fallback_T = float(calib_T_target if calib_T_target is not None else T_grid[0])
            common_T = float(st.session_state.get("common_maturity", fallback_T))
            common_r = float(st.session_state.get("common_rate", rf_rate))
            common_d = float(st.session_state.get("common_dividend", div_yield))

            cpflag_heston_single = st.selectbox(
                "Call / Put (Heston – prix ponctuel)",
                ["Call", "Put"],
                key="cpflag_heston_single",
                help="Type d’option à pricer avec les paramètres Heston calibrés.",
            )
            if st.button(
                f"Calculer le prix Heston ({cpflag_heston_single})",
                key="btn_price_eu_heston",
            ):
                try:
                    Ks_t_single = torch.tensor([common_K], dtype=torch.float64)
                    call_vals_single = carr_madan_call_torch(
                        float(common_S0),
                        float(common_r),
                        float(common_d),
                        float(common_T),
                        params_cm,
                        Ks_t_single,
                    )
                    call_price_single = float(call_vals_single[0].detach().cpu().numpy())
                    discount_single = math.exp(-common_r * common_T)
                    forward_single = math.exp(-common_d * common_T)
                    put_price_single = call_price_single - common_S0 * forward_single + common_K * discount_single
                    if cpflag_heston_single == "Call":
                        st.success(f"Prix Heston (Call) = {call_price_single:.6f}")
                    else:
                        st.success(f"Prix Heston (Put) = {put_price_single:.6f}")
                except Exception as exc:
                    st.error(f"Erreur lors du calcul Heston : {exc}")

            st.caption(
                f"Paramètres utilisés pour le prix Heston ponctuel : "
                f"S0={common_S0:.4f}, K={common_K:.4f}, T={common_T:.4f}, "
                f"r={common_r:.4f}, d={common_d:.4f}, "
                f"κ={float(st.session_state.get('heston_kappa_common', 0.0)):.4f}, "
                f"θ={float(st.session_state.get('heston_theta_common', 0.0)):.4f}, "
                f"η={float(st.session_state.get('heston_eta_common', 0.0)):.4f}, "
                f"ρ={float(st.session_state.get('heston_rho_common', 0.0)):.4f}, "
                f"v0={float(st.session_state.get('heston_v0_common', 0.0)):.4f}"
            )

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
            fig_heat_call_cm.update_layout(xaxis_title="Strike K", yaxis_title="Maturité T")
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
            fig_heat_put_cm.update_layout(xaxis_title="Strike K", yaxis_title="Maturité T")

            with tab_calls:
                st.subheader("Carr-Madan : IV & prix (Calls)")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig_call_cm, width="stretch", config=PLOTLY_CONFIG)
                with c2:
                    st.plotly_chart(fig_heat_call_cm, width="stretch", config=PLOTLY_CONFIG)

                st.subheader("Marché : IV & prix (Calls)")
                c3, c4 = st.columns(2)
                with c3:
                    if fig_call_market:
                        st.plotly_chart(fig_call_market, width="stretch", config=PLOTLY_CONFIG)
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
                        fig_heat_call_mkt.update_layout(xaxis_title="Strike K", yaxis_title="Maturité T")
                        st.plotly_chart(fig_heat_call_mkt, width="stretch", config=PLOTLY_CONFIG)
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
                    st.plotly_chart(fig_put_cm, width="stretch", config=PLOTLY_CONFIG)
                with c2:
                    st.plotly_chart(fig_heat_put_cm, width="stretch", config=PLOTLY_CONFIG)

                st.subheader("Marché : IV & prix (Puts)")
                c3, c4 = st.columns(2)
                with c3:
                    if fig_put_market:
                        st.plotly_chart(fig_put_market, width="stretch", config=PLOTLY_CONFIG)
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
                        fig_heat_put_mkt.update_layout(xaxis_title="Strike K", yaxis_title="Maturité T")
                        st.plotly_chart(fig_heat_put_mkt, width="stretch", config=PLOTLY_CONFIG)
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


sidebar_prefill = st.session_state.pop("heston_sidebar_prefill", None)
if sidebar_prefill:
    for key, value in sidebar_prefill.items():
        st.session_state[key] = value


st.title("Application unifiée de pricing d'options")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stStatusWidget"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Paramètres de Black-Scholes")
placeholder_vals = st.session_state.get("heston_sidebar_placeholders", {})
heston_tab_locked = st.session_state.get("heston_tab_locked", False)

default_sidebar_values = {
    "S0_common": 100.0,
    "K_common": 100.0,
    "T_common": 1.0,
    "sigma_common": 0.2,
    "r_common": 0.05,
    "d_common": 0.0,
    "heatmap_span": 25.0,
}
for param_key, default_value in default_sidebar_values.items():
    if param_key not in st.session_state:
        st.session_state[param_key] = default_value

S0_common = st.sidebar.number_input(
    "S0 (spot)",
    min_value=0.01,
    step=0.01,
    key="S0_common",
    placeholder=placeholder_vals.get("S0_common"),
    help="Niveau actuel du sous-jacent utilisé comme spot de référence pour les calculs.",
)
K_common = st.sidebar.number_input(
    "K (strike)",
    min_value=0.01,
    step=1.0,
    key="K_common",
    placeholder=placeholder_vals.get("K_common"),
    help="Strike de référence de l’option, au centre des grilles de prix.",
)
T_common = st.sidebar.number_input(
    "T (maturité, années)",
    min_value=0.01,
    step=0.1,
    key="T_common",
    disabled=heston_tab_locked,
    help=(
        "Maturité de l’option en années. "
        "Verrouillée après le téléchargement Heston : cliquez sur un autre onglet pour réactiver."
        if heston_tab_locked
        else "Maturité de l’option en années utilisée pour tous les calculs."
    ),
)
sigma_common = st.sidebar.number_input(
    "Volatilité σ",
    min_value=0.0001,
    key="sigma_common",
    placeholder=placeholder_vals.get("sigma_common"),
    help="Volatilité annualisée utilisée par défaut dans les modèles de diffusion.",
)
r_common = st.sidebar.number_input(
    "Taux sans risque r",
    key="r_common",
    help="Taux sans risque continu utilisé pour actualiser les payoffs.",
)
d_common = st.sidebar.number_input(
    "Dividende continu d",
    key="d_common",
    help="Dividende (ou rendement de portage) continu du sous-jacent.",
)
heatmap_span = st.sidebar.number_input(
    "Span autour du spot (heatmaps)",
    min_value=0.1,
    step=1.0,
    help="Définit l'écart symétrique autour du spot utilisé pour les axes Spot / Strike des heatmaps.",
    key="heatmap_span",
)

st.sidebar.header("Paramètres de Heston")
heston_kappa_common = st.sidebar.number_input(
    "κ",
    value=float(st.session_state.get("heston_kappa_common", 2.0)),
    min_value=0.0,
    key="heston_kappa_input",
    help="Vitesse de rappel de la variance vers son niveau de long terme.",
)
st.session_state["heston_kappa_common"] = float(heston_kappa_common)

heston_theta_common = st.sidebar.number_input(
    "θ",
    value=float(st.session_state.get("heston_theta_common", 0.04)),
    min_value=0.0001,
    key="heston_theta_input",
    help="Niveau de variance de long terme du modèle de Heston.",
)
st.session_state["heston_theta_common"] = float(heston_theta_common)

heston_eta_common = st.sidebar.number_input(
    "η",
    value=float(st.session_state.get("heston_eta_common", 0.5)),
    min_value=0.0001,
    key="heston_eta_input",
    help="Volatilité de la variance, c’est-à-dire l’ampleur des fluctuations de variance.",
)
st.session_state["heston_eta_common"] = float(heston_eta_common)

heston_rho_common = st.sidebar.number_input(
    "ρ",
    value=float(st.session_state.get("heston_rho_common", -0.7)),
    min_value=-0.99,
    max_value=0.99,
    key="heston_rho_input",
    help="Corrélation entre les chocs sur le sous-jacent et sur la variance.",
)
st.session_state["heston_rho_common"] = float(heston_rho_common)

heston_v0_common = st.sidebar.number_input(
    "v0",
    value=float(st.session_state.get("heston_v0_common", 0.04)),
    min_value=0.0001,
    key="heston_v0_input",
    help="Variance initiale au temps 0 dans le modèle de Heston.",
)
st.session_state["heston_v0_common"] = float(heston_v0_common)
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
    render_general_definition_explainer(
        "📘 Comprendre les options européennes",
        (
            "- **Nature du produit** : une option européenne donne le droit, mais pas l'obligation, d'acheter (call) ou de vendre (put) un sous-jacent à une date d'échéance `T` et à un prix fixé à l'avance `K`. L'exercice ne peut avoir lieu **qu'à la maturité**, jamais avant.\n"
            "- **Payoff à l'échéance** :\n"
            "  - Call : `max(S_T - K, 0)` – on exerce seulement si le sous-jacent vaut plus que le strike.\n"
            "  - Put  : `max(K - S_T, 0)` – on exerce seulement si le sous-jacent vaut moins que le strike.\n"
            "- **Mesure neutre au risque** : dans les modèles utilisés ici, on raisonne sous une mesure où le sous-jacent rapporte le taux sans risque ajusté du dividende. Le prix de l'option est alors l'espérance actualisée de ce payoff.\n"
            "- **Variables structurantes** : le prix dépend principalement de `S0` (spot), `K` (strike), `T` (maturité), `r` (taux sans risque), `d` (dividende continu) et `σ` (volatilité implicite ou historique selon le modèle).\n"
            "- **Interprétation des heatmaps** : les cartes de chaleur affichées dans cet onglet montrent comment le prix du call et du put varie lorsque l'on fait bouger `S` et `K` autour des valeurs communes définies dans la barre latérale, pour un `T` et des paramètres donnés.\n"
            "- **Rôle de cet onglet** : il sert de point de départ pour comparer différentes façons de pricer le même produit : modèle de diffusion simple (BSM), simulation Monte Carlo, ou modèle de volatilité stochastique (Heston)."
        ),
    )

    tab_eu_heston, tab_eu_bsm, tab_eu_mc = st.tabs(["Heston", "Black–Scholes–Merton", "Monte Carlo"])

    with tab_eu_heston:
        render_method_explainer(
            "🧮 Méthode Heston pour les options européennes",
            (
                "- **Étape 1 – Choix du cadre probabiliste** : on modélise le sous‑jacent `S_t` et la variance instantanée `v_t` sous la mesure neutre au risque. `S_t` suit une diffusion où le terme de diffusion dépend de `√v_t`, et `v_t` suit un processus de type CIR avec rappel vers `θ`.\n"
                "- **Étape 2 – Spécification des paramètres de Heston** : on travaille avec cinq paramètres structurants : `κ` (vitesse de rappel de la variance), `θ` (variance de long terme), `σ_v` (volatilité de la variance), `ρ` (corrélation entre chocs sur `S_t` et `v_t`) et `v0` (variance initiale).\n"
                "- **Étape 3 – Préparation des données de marché** : les données CBOE sont téléchargées, nettoyées et ramenées sous forme de points `(S0, K, T, C_mkt)` ou `(P_mkt)`, en filtrant les maturités trop courtes et les prix non exploitables.\n"
                "- **Étape 4 – Construction d’un pricer rapide** : pour un jeu de paramètres Heston donné, on évalue les prix de calls européens via la méthode de Carr–Madan (transformée de Fourier) implémentée en `carr_madan_call_torch`, ce qui permet d’avoir un pricer différentiable dans PyTorch.\n"
                "- **Étape 5 – Définition de la fonction de perte** : on compare les prix modèle aux prix de marché sur l’ensemble des points, via une fonction de perte de type somme pondérée des carrés des écarts, éventuellement avec des poids pour privilégier certaines zones du smile.\n"
                "- **Étape 6 – Optimisation / calibration** : à partir d’un vecteur de paramètres non contraints `u`, on reconstruit des paramètres Heston admissibles (positivité, contraintes de Feller) puis on minimise la perte par descente de gradient ou quasi‑Newton (itérations jusqu’à `max_iters` avec un pas `learning_rate`).\n"
                "- **Étape 7 – Exploitation des paramètres calibrés** : une fois les paramètres calibrés obtenus, on peut :\n"
                "  • pricer des options européennes sur une grille `(K, T)` pour construire des surfaces de prix ;\n"
                "  • en déduire des surfaces de volatilité implicite ;\n"
                "  • comparer ces surfaces à celles issues de BSM ou de Monte Carlo simple.\n"
                "- **Étape 8 – Visualisation et diagnostics** : les erreurs de calibration et les surfaces résultantes sont examinées pour vérifier la cohérence du modèle avec les données (forme du smile, term‑structure de volatilité, etc.)."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – Heston européen",
            (
                "- **\"S0 (spot)\"** : niveau actuel du sous‑jacent, utilisé comme référence pour centrer la grille de strikes et interpréter les surfaces de prix.\n"
                "- **\"K (strike)\"** : strike de référence saisi dans la barre latérale, utilisé pour certains graphiques ciblés et pour positionner la grille en moneyness.\n"
                "- **\"T (maturité, années)\"** : maturité commune à laquelle on lit les prix et la surface de volatilité implicite.\n"
                "- **\"Taux sans risque r\"** : taux d’actualisation continu utilisé dans le modèle de Heston pour passer de payoffs futurs aux prix présents.\n"
                "- **\"Dividende continu d\"** : rendement de portage continu, qui vient diminuer le drift du sous‑jacent sous la mesure neutre au risque.\n"
                "- **\"Ticker (sous-jacent)\"** : code CBOE de l’actif (ex. `SPY`) dont on télécharge la chaîne d’options.\n"
                "- **Bouton \"Récupérer les données du ticker\"** : lance la collecte des options marché (calls / puts), qui serviront de base à la calibration.\n"
                "- **\"Largeur bande T (±)\"** : largeur de la bande de maturités autour de `T` sur laquelle la calibration Heston est concentrée.\n"
                "- **\"Maturité T cible pour la calibration\"** : maturité centrale de la bande de calibration, choisie parmi les maturités observées.\n"
                "- **\"Choisir un mode\" (Rapide / Bonne / Excellente)** : règle automatiquement le nombre d’itérations et le pas d’apprentissage de la calibration (compromis précision / temps).\n"
                "- **Paramètres Heston calibrés** : paramètres implicites du modèle `(κ, θ, σ_v, ρ, v0)` que la procédure d’optimisation ajuste pour coller au mieux aux prix observés."
            ),
        )
        ui_heston_full_pipeline()

    with tab_eu_bsm:
        render_unlock_sidebar_button("eu_bsm", "🔓 Réactiver T (onglet BSM)")
        render_method_explainer(
            "🧮 Méthode Black–Scholes–Merton (BSM)",
            (
                "- **Étape 1 – Mise sous la mesure neutre au risque** : on suppose que le sous‑jacent suit un mouvement brownien géométrique avec volatilité constante `σ` et drift neutre au risque `r-d`. Cette hypothèse conduit à une distribution log‑normale de `S_T`.\n"
                "- **Étape 2 – Calcul des quantités intermédiaires** : pour chaque couple `(S, K)` de la grille, on calcule\n"
                "  `d1 = [ln(S/K) + (r-d + 0.5 σ²) T] / (σ√T)` et `d2 = d1 - σ√T`. Ces deux variables normalisées condensent l’information de tous les paramètres du modèle.\n"
                "- **Étape 3 – Utilisation de la loi normale** : on évalue les fonctions de répartition `N(d1)` et `N(d2)` pour obtenir les probabilités risque‑neutres implicites de finir dans la monnaie.\n"
                "- **Étape 4 – Formule de prix** :\n"
                "  • Call : `C = S e^{-dT} N(d1) - K e^{-rT} N(d2)` ;\n"
                "  • Put  : `P = K e^{-rT} N(-d2) - S e^{-dT} N(-d1)`.\n"
                "  On applique ces formules pour chaque point de la grille `(S, K)` afin de remplir les matrices de prix call et put.\n"
                "- **Étape 5 – Construction des heatmaps** : les matrices de prix sont organisées selon les axes `Spot` (valeurs de `S`) et `Strike` (valeurs de `K`) pour produire les cartes de chaleur. On visualise ainsi la structure du prix dans le plan `(S, K)` pour une maturité et des paramètres donnés.\n"
                "- **Étape 6 – Analyse et comparaison** : ces surfaces BSM servent de référence. On peut comparer point par point les niveaux de prix et la forme générale aux surfaces issues du modèle de Heston ou de Monte Carlo afin de mettre en évidence les limites du cadre à volatilité constante."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – BSM",
            (
                "- **\"S0 (spot)\"** : centre de l’axe des spots utilisé pour construire la grille horizontale des heatmaps BSM.\n"
                "- **\"K (strike)\"** : centre de l’axe des strikes autour duquel on génère les valeurs de `K` de la heatmap.\n"
                "- **\"T (maturité, années)\"** : maturité commune appliquée à tous les couples `(S, K)` de la grille.\n"
                "- **\"Taux sans risque r\"** : taux continu utilisé pour actualiser le strike et déterminer le drift neutre au risque.\n"
                "- **\"Dividende continu d\"** : rendement continu soustrait du drift, qui représente le coût de portage du sous‑jacent.\n"
                "- **\"Volatilité σ\"** : volatilité constante utilisée pour tous les points de la grille dans la formule BSM.\n"
                "- **\"Span autour du spot (heatmaps)\"** : amplitude autour de `S0 (spot)` et `K (strike)` qui définit l’étendue de la grille de calcul.\n"
                "- **Résultat** : à partir de ces entrées, l’application construit les matrices de prix call/put utilisées pour les cartes de chaleur."
            ),
        )
        cpflag_eu_bsm = st.selectbox(
            "Call / Put (BSM)",
            ["Call", "Put"],
            key="cpflag_eu_bsm_single",
            help="Type d’option européenne à pricer avec la formule BSM.",
        )
        if st.button(
            f"Calculer le prix BSM ({cpflag_eu_bsm})",
            key="btn_price_eu_bsm",
        ):
            opt_type = "call" if cpflag_eu_bsm == "Call" else "put"
            price_bsm = _vanilla_price_with_dividend(
                option_type=opt_type,
                S0=common_spot_value,
                K=common_strike_value,
                T=common_maturity_value,
                r=common_rate_value,
                dividend=float(d_common),
                sigma=common_sigma_value,
            )
            st.success(f"Prix BSM ({cpflag_eu_bsm}) = {price_bsm:.6f}")
        st.caption(
            f"Paramètres utilisés pour le prix unique BSM : "
            f"S0={common_spot_value:.4f}, K={common_strike_value:.4f}, "
            f"T={common_maturity_value:.4f}, r={common_rate_value:.4f}, "
            f"d={float(d_common):.4f}, σ={common_sigma_value:.4f}"
        )
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
        render_unlock_sidebar_button("eu_mc", "🔓 Réactiver T (onglet Monte Carlo)")
        render_method_explainer(
            "🎲 Méthode Monte Carlo pour options européennes",
            (
                "- **Étape 1 – Fixation du cadre de simulation** : on choisit un nombre de trajectoires `n_paths_eu` et un nombre de pas de temps `n_steps_eu`. La maturité `T_common` est découpée en intervalles `Δt = T_common / n_steps_eu`.\n"
                "- **Étape 2 – Discrétisation de la dynamique** : pour chaque trajectoire, on fait évoluer le sous‑jacent selon le schéma d’Euler pour le GBM neutre au risque :\n"
                "  `S_{t+Δt} = S_t · exp((r-d-0.5 σ²)Δt + σ√Δt · Z)` avec `Z ~ N(0,1)` indépendant.\n"
                "- **Étape 3 – Simulation sur la grille `(S, K)`** : pour chaque valeur de spot de la grille `heatmap_spot_values`, on simule `n_paths_eu` trajectoires jusqu’à `T_common`. On obtient ainsi un vecteur de prix terminaux `S_T` pour ce spot.\n"
                "- **Étape 4 – Calcul des payoffs** : pour chaque strike de `heatmap_strike_values`, on calcule, à partir des `S_T` simulés :\n"
                "  • pour un call : `max(S_T - K, 0)` ;\n"
                "  • pour un put  : `max(K - S_T, 0)`.\n"
                "  On actualise ensuite ces payoffs par le facteur `discount = exp(-r_common·T_common)`.\n"
                "- **Étape 5 – Moyenne Monte Carlo** : pour chaque couple `(S, K)`, on moyenne les payoffs actualisés sur toutes les trajectoires. Cette moyenne est l’estimateur Monte Carlo du prix.\n"
                "- **Étape 6 – Remplissage des matrices** : on stocke les prix estimes dans deux matrices (call et put) indexées par les indices de `S` et `K`, qui serviront à l’affichage des cartes de chaleur.\n"
                "- **Étape 7 – Contrôle de la précision** : en pratique, on compare les surfaces obtenues à celles de BSM pour vérifier la convergence lorsque `n_paths_eu` et `n_steps_eu` augmentent, et on ajuste ces paramètres en fonction du compromis précision / temps de calcul."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – Monte Carlo européen",
            (
                "- **\"S0 (spot)\"** et **\"K (strike)\"** : déterminent le centre de la grille `(S, K)` sur laquelle on lance les simulations Monte Carlo.\n"
                "- **\"T (maturité, années)\"** : définit la durée de chaque trajectoire simulée.\n"
                "- **\"Taux sans risque r\"** : utilisé à la fois dans le drift neutre au risque et dans le facteur d’actualisation des payoffs.\n"
                "- **\"Dividende continu d\"** : réduit le drift du sous‑jacent sous la mesure neutre au risque.\n"
                "- **\"Volatilité σ\"** : volatilité constante utilisée dans la dynamique simulée.\n"
                "- **\"Span autour du spot (heatmaps)\"** : règle l’étendue des valeurs de `S` et `K` explorées dans les heatmaps.\n"
                "- **\"Trajectoires Monte Carlo\"** : nombre de trajectoires simulées pour chaque point de la grille (contrôle la précision statistique).\n"
                "- **\"Pas de temps\"** : nombre de pas de simulation par trajectoire (contrôle la finesse de la discrétisation temporelle)."
            ),
        )
        st.subheader("Monte Carlo classique")
        n_paths_eu = st.number_input(
            "Trajectoires Monte Carlo",
            value=10_000,
            min_value=100,
            key="n_paths_eu",
            help="Nombre de trajectoires simulées pour chaque point de la grille.",
        )
        n_steps_eu = st.number_input(
            "Pas de temps",
            value=50,
            min_value=1,
            key="n_steps_eu",
            help="Nombre de pas de temps utilisés pour discrétiser la maturité.",
        )
        cpflag_eu_mc = st.selectbox(
            "Call / Put (Monte Carlo)",
            ["Call", "Put"],
            key="cpflag_eu_mc_single",
            help="Type d’option européenne à pricer par Monte Carlo.",
        )
        if st.button(
            f"Calculer le prix Monte Carlo ({cpflag_eu_mc})",
            key="btn_price_eu_mc",
        ):
            try:
                paths_eu, _ = simulate_gbm_paths(
                    S0=common_spot_value,
                    r=common_rate_value,
                    q=float(d_common),
                    sigma=common_sigma_value,
                    T=common_maturity_value,
                    M=int(n_steps_eu),
                    N_paths=int(n_paths_eu),
                )
                ST = paths_eu[-1]
                if cpflag_eu_mc == "Call":
                    payoff = np.maximum(ST - common_strike_value, 0.0)
                else:
                    payoff = np.maximum(common_strike_value - ST, 0.0)
                price_mc = float(np.exp(-common_rate_value * common_maturity_value) * payoff.mean())
                st.success(f"Prix Monte Carlo ({cpflag_eu_mc}) = {price_mc:.6f}")
            except Exception as exc:
                st.error(f"Erreur Monte Carlo européen : {exc}")
        st.caption(
            f"Paramètres utilisés pour le prix unique Monte Carlo : "
            f"S0={common_spot_value:.4f}, K={common_strike_value:.4f}, "
            f"T={common_maturity_value:.4f}, r={common_rate_value:.4f}, "
            f"d={float(d_common):.4f}, σ={common_sigma_value:.4f}, "
            f"N_paths={int(n_paths_eu)}, N_steps={int(n_steps_eu)}"
        )
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
    render_unlock_sidebar_button("tab_american", "🔓 Réactiver T (onglet Américain)")
    render_general_definition_explainer(
        "📗 Comprendre les options américaines",
        (
            "- **Droit d'exercice anticipé** : une option américaine peut être exercée à n'importe quel moment entre la date d'émission et la maturité. Elle offre donc plus de flexibilité qu'une option européenne.\n"
            "- **Conséquence sur le prix** : cette flexibilité a une valeur. À paramètres identiques (`S0`, `K`, `T`, `r`, `d`, `σ`), le prix d'une option américaine est **au moins** aussi élevé que celui de l'option européenne correspondante.\n"
            "- **Vision dynamique** : le problème de pricing devient un problème de contrôle optimal : à chaque date de la grille temporelle, l'agent choisit entre exercer immédiatement ou conserver l'option.\n"
            "- **Lien avec les grecs** : pour les puts notamment, la possibilité d'exercer en avance influence fortement `Delta` et `Theta`, en particulier lorsque le sous-jacent est proche ou sous le strike.\n"
            "- **Rôle de cet onglet** : il illustre deux grandes familles d'approches numériques pour ce problème : une méthode Monte Carlo (Longstaff–Schwartz) et une méthode par arbre binomial (CRR)."
        ),
    )
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
        render_method_explainer(
            "🧮 Méthode Longstaff–Schwartz (régression Monte Carlo)",
            (
                "- **Objectif** : approximer la stratégie d'exercice optimale d'une option américaine en combinant simulation Monte Carlo et régressions sur la valeur de continuation.\n"
                "- **Étape 1 – Simulation des trajectoires** : on simule un grand nombre de trajectoires du sous‑jacent (GBM ou Heston) sous la mesure neutre au risque, en discrétisant `[0, T_common]` en `n_steps_am` pas.\n"
                "- **Étape 2 – Calcul des payoffs finaux** : à la dernière date de la grille (≈ maturité), on calcule le payoff européen standard pour chaque trajectoire (call ou put) et on l’utilise comme valeur initiale de continuation.\n"
                "- **Étape 3 – Remontée dans le temps (backward induction)** : pour chaque date de la grille, en partant de l’avant‑dernière jusqu’à la première, on considère les trajectoires où l’option est dans la monnaie à cette date.\n"
                "- **Étape 4 – Régression de la valeur de continuation** : sur l’ensemble des trajectoires in‑the‑money, on ajuste une régression (souvent polynomiale en `S_t`) entre le prix courant `S_t` et la valeur actualisée des payoffs futurs. Cette régression donne une approximation de la valeur de continuation conditionnelle.\n"
                "- **Étape 5 – Décision d’exercice** : pour chaque trajectoire et à chaque date, on compare le payoff d’exercice immédiat à la valeur de continuation régressée. Si le payoff immédiat est plus élevé, on exerce (on fige le payoff sur cette trajectoire et on ignore les valeurs futures) ; sinon, on conserve la valeur de continuation.\n"
                "- **Étape 6 – Agrégation des payoffs** : après avoir remonté toutes les dates, chaque trajectoire porte un payoff actualisé correspondant à la stratégie d’exercice optimale approximée. Le prix de l’option est la moyenne de ces payoffs sur l’ensemble des trajectoires.\n"
                "- **Étape 7 – Utilisation pour les heatmaps** : le schéma précédent est réutilisé sur une grille de `S0` et `K` pour construire des surfaces de prix américains, qui peuvent être comparées aux surfaces européennes ou CRR.\n"
                "- **Intérêt** : la méthode est très flexible (capable de traiter des payoffs complexes) tout en évitant la construction explicite d’un arbre multidimensionnel."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – Longstaff–Schwartz",
            (
                "- **Paramètres communs de la barre latérale** :\n"
                "  - **\"S0 (spot)\"** : niveau de référence du sous‑jacent pour les heatmaps et les simulations.\n"
                "  - **\"K (strike)\"** : strike de l’option américaine utilisée pour le payoff.\n"
                "  - **\"T (maturité, années)\"** : horizon de temps de l’option américaine, donc durée des trajectoires simulées.\n"
                "  - **\"Taux sans risque r\"** et **\"Dividende continu d\"** : entrent dans le drift neutre au risque et dans l’actualisation des payoffs.\n"
                "  - **\"Volatilité σ\"** : volatilité du sous‑jacent lorsque le processus choisi est un GBM.\n"
                "- **\"Processus sous-jacent\"** : menu déroulant qui permet de choisir entre un **Geometric Brownian Motion** et un **processus de Heston** pour simuler `S_t`.\n"
                "- **Si \"Geometric Brownian Motion\" est sélectionné** : seules les entrées ci‑dessus (dont \"Volatilité σ\") pilotent la dynamique.\n"
                "- **Si \"Heston\" est sélectionné** : les champs suivants apparaissent et décrivent la variance stochastique :\n"
                "  - **\"κ (vitesse de rappel)\"**, **\"θ (variance long terme)\"**, **\"η (vol de la variance)\"**, **\"ρ (corrélation)\"**, **\"v0 (variance initiale)\"**.\n"
                "- **\"Trajectoires Monte Carlo\"** : nombre de trajectoires utilisées pour estimer le prix américain.\n"
                "- **\"Pas de temps\"** : nombre de dates intermédiaires sur lesquelles l’algorithme Longstaff–Schwartz peut potentiellement décider d’exercer l’option."
            ),
        )
        process_type_am = st.selectbox(
            "Processus sous-jacent",
            ["Geometric Brownian Motion", "Heston"],
            key="process_type_am",
            help="Choix du modèle utilisé pour simuler le sous-jacent (GBM ou Heston).",
        )
        n_paths_am = st.number_input(
            "Trajectoires Monte Carlo",
            value=1000,
            min_value=100,
            key="n_paths_am",
            help="Nombre de trajectoires Monte Carlo utilisées pour le prix américain.",
        )
        n_steps_am = st.number_input(
            "Pas de temps",
            value=50,
            min_value=1,
            key="n_steps_am",
            help="Nombre de dates intermédiaires possibles d’exercice dans Longstaff–Schwartz.",
        )

        if process_type_am == "Geometric Brownian Motion":
            process_am = GeometricBrownianMotion(mu=r_common - d_common, sigma=sigma_common)
            v0_am = None
        else:
            kappa_am = float(st.session_state.get("heston_kappa_common", 2.0))
            theta_am = float(st.session_state.get("heston_theta_common", 0.04))
            eta_am = float(st.session_state.get("heston_eta_common", 0.5))
            rho_am = float(st.session_state.get("heston_rho_common", -0.7))
            v0_am = float(st.session_state.get("heston_v0_common", 0.04))
            st.caption(
                f"Paramètres de Heston (sidebar) : κ={kappa_am:.4f}, θ={theta_am:.4f}, η={eta_am:.4f}, "
                f"ρ={rho_am:.4f}, v0={v0_am:.4f}"
            )
            process_am = HestonProcess(
                mu=r_common - d_common, kappa=kappa_am, theta=theta_am, eta=eta_am, rho=rho_am
            )

        if st.button(
            f"Calculer le prix américain L-S ({cpflag_am})",
            key="btn_price_am_ls",
        ):
            try:
                option_ls = Option(
                    s0=S0_common,
                    T=T_common,
                    K=K_common,
                    v0=v0_am,
                    call=(cpflag_am == "Call"),
                )
                price_ls = longstaff_schwartz_price(
                    option=option_ls,
                    process=process_am,
                    n_paths=int(n_paths_am),
                    n_steps=int(n_steps_am),
                )
                st.success(f"Prix américain Longstaff–Schwartz ({cpflag_am}) = {price_ls:.6f}")
            except Exception as exc:
                st.error(f"Erreur Longstaff–Schwartz : {exc}")
        st.caption(
            f"Paramètres utilisés pour le prix unique L-S : "
            f"S0={S0_common:.4f}, K={K_common:.4f}, T={T_common:.4f}, "
            f"r={r_common:.4f}, d={d_common:.4f}, σ={sigma_common:.4f}, "
            f"N_paths={int(n_paths_am)}, N_steps={int(n_steps_am)}"
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
        render_method_explainer(
            "🌳 Méthode binomiale CRR pour options américaines",
            (
                "- **Étape 1 – Discrétisation de l’horizon** : la maturité `T_common` est découpée en `n_tree_am` pas de temps de durée `Δt = T_common / n_tree_am`.\n"
                "- **Étape 2 – Paramétrage de l’arbre** : à partir de `σ` et `Δt`, on construit les facteurs de hausse et de baisse, typiquement `u = e^{σ√Δt}` et `d = 1/u`. On en déduit une probabilité neutre au risque `p` telle que `E_Q[S_{t+Δt}] = S_t e^{(r-d)Δt}`.\n"
                "- **Étape 3 – Construction de l’arbre des spots** : en partant de `S0_common`, on génère les valeurs de `S` à chaque nœud du maillage binomial (chaque niveau correspond à un temps, chaque nœud à un nombre de hausses/baisse cumulées).\n"
                "- **Étape 4 – Initialisation des payoffs à maturité** : à la dernière ligne de l’arbre (temps `T_common`), on calcule le payoff européen `max(±(S_T-K_common), 0)` pour chaque nœud et on le stocke dans `value_tree`.\n"
                "- **Étape 5 – Rétro‑propagation (valeur de continuation)** : en remontant niveau par niveau, on calcule à chaque nœud la valeur de continuation comme espérance actualisée des deux nœuds fils : `V_cont = e^{-r_common Δt} [p V_up + (1-p) V_down]`.\n"
                "- **Étape 6 – Prise en compte de l’exercice américain** : pour chaque nœud, on calcule aussi la valeur d’exercice immédiat `V_ex = payoff(S_n)`. La valeur retenue au nœud est `max(V_ex, V_cont)`, ce qui encode la possibilité d’exercer de façon optimale.\n"
                "- **Étape 7 – Prix initial et visualisation** : la valeur à la racine de l’arbre est le prix de l’option américaine. L’arbre des spots et de valeurs (`spot_tree`, `value_tree`) est ensuite représenté graphiquement pour montrer les zones où l’exercice anticipé devient optimal.\n"
                "- **Étape 8 – Lien avec les heatmaps** : en répétant ce calcul pour différents `S0` et `K`, on peut construire une surface de prix CRR comparable à celles obtenues via Longstaff–Schwartz ou BSM."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – CRR",
            (
                "- **\"S0 (spot)\"** : valeur de départ du sous‑jacent à la racine de l’arbre.\n"
                "- **\"K (strike)\"** : strike de l’option américaine modélisée sur l’arbre.\n"
                "- **\"T (maturité, années)\"** : horizon total de l’option, réparti en `Nombre de pas de l'arbre`.\n"
                "- **\"Taux sans risque r\"** : utilisé pour l’actualisation et pour calibrer la probabilité neutre au risque.\n"
                "- **\"Volatilité σ\"** : volatilité reproduite par les facteurs de montée et de descente `u` et `d`.\n"
                "- **\"Nombre de pas de l'arbre\"** : profondeur de l’arbre binomial (résolution temporelle) choisie via le curseur correspondant."
            ),
        )
        if st.button(
            f"Calculer le prix américain CRR ({cpflag_am})",
            key="btn_price_am_crr",
        ):
            try:
                option_am_single = Option(
                    s0=S0_common,
                    T=T_common,
                    K=K_common,
                    call=(cpflag_am == 'Call'),
                )
                # On utilise un arbre de taille moyenne pour le prix ponctuel
                n_steps_single = 50
                price_crr_single = crr_pricing(
                    r=r_common,
                    sigma=sigma_common,
                    option=option_am_single,
                    n=n_steps_single,
                )
                st.success(f"Prix américain CRR ({cpflag_am}) ≈ {price_crr_single:.6f} (avec {n_steps_single} pas)")
            except Exception as exc:
                st.error(f"Erreur CRR : {exc}")
        st.caption(
            f"Paramètres utilisés pour le prix unique CRR : "
            f"S0={S0_common:.4f}, K={K_common:.4f}, T={T_common:.4f}, "
            f"r={r_common:.4f}, σ={sigma_common:.4f}"
        )

        n_tree_am = st.number_input(
            "Nombre de pas de l'arbre",
            value=10,
            min_value=5,
            key="n_tree_am",
            help="Nombre de pas de temps utilisés dans l’arbre binomial CRR.",
        )
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
    render_unlock_sidebar_button("tab_lookback", "🔓 Réactiver T (onglet Lookback)")
    render_general_definition_explainer(
        "🔍 Comprendre les options lookback",
        (
            "- **Payoff dépendant du chemin** : une option lookback ne dépend plus uniquement de `S_T`, mais de l'historique complet de la trajectoire du sous‑jacent (par exemple de son maximum ou de son minimum atteint avant l'échéance).\n"
            "- **Floating strike** : dans cet onglet, on considère des structures où le strike effectif est défini à partir d'un extrême de la trajectoire, par exemple le maximum historique pour un put, ou le minimum pour un call.\n"
            "- **Intérêt intuitif** : ce type d'option permet de \"regarder en arrière\" pour déterminer le niveau de référence du contrat, offrant une protection renforcée contre des mouvements extrêmes défavorables.\n"
            "- **Dimension temporelle** : plus la maturité est longue, plus le sous‑jacent a de chances de visiter des extrêmes éloignés, ce qui impacte directement le niveau du payoff.\n"
            "- **Objectif de cet onglet** : comparer une formule fermée (lorsqu'elle est disponible) à une approche Monte Carlo pour des options lookback, et visualiser l'effet des paramètres via des heatmaps Spot × Maturité."
        ),
    )
    st.caption(
        "Les heatmaps affichent les prix lookback sur un carré Spot × Maturité centré autour des valeurs définies dans la barre latérale."
    )

    tab_lb_exact, tab_lb_mc = st.tabs(["Exacte", "Monte Carlo"])

    with tab_lb_exact:
        st.subheader("Formule exacte")
        render_method_explainer(
            "📗 Méthode analytique pour lookback",
            (
                "- **Étape 1 – Choix du modèle sous‑jacent** : on se place dans le cadre Black–Scholes standard avec volatilité constante `σ`, taux sans risque `r` et éventuellement dividende continu. Le sous‑jacent suit un mouvement brownien géométrique.\n"
                "- **Étape 2 – Caractérisation des extrêmes** : on utilise des résultats de théorie des processus stochastiques sur la distribution du maximum (ou minimum) d’un mouvement brownien géométrique sur un horizon `[0, T]`.\n"
                "- **Étape 3 – Réécriture du payoff** : le payoff lookback (par exemple basé sur `max_t S_t` ou `min_t S_t`) est réécrit de manière à isoler des termes qui ressemblent à des payoffs d’options européennes classiques, plus des termes correctifs dépendant des extrêmes.\n"
                "- **Étape 4 – Intégration analytique** : à partir de cette réécriture, on calcule l’espérance neutre au risque de ce payoff en intégrant par rapport aux densités des extrêmes et du sous‑jacent. On obtient des formules fermées impliquant des fonctions de répartition de la loi normale et des combinaisons exponentielles.\n"
                "- **Étape 5 – Implémentation numérique** : les formules fermées sont implémentées sous forme de fonctions vectorisées qui prennent en entrée `(S0, T, σ, r, …)` et renvoient directement le prix de l’option lookback pour chaque point de la grille Spot × Maturité.\n"
                "- **Étape 6 – Construction de la heatmap** : pour chaque valeur de `S0` et `T` de la grille, la formule analytique est évaluée, ce qui remplit une matrice de prix. Cette matrice est ensuite affichée sous forme de carte de chaleur.\n"
                "- **Étape 7 – Rôle de benchmark** : cette solution analytique sert de référence \"exacte\" pour valider la méthode Monte Carlo : en comparant les deux surfaces, on quantifie l’erreur de simulation et on ajuste le nombre d’itérations ou la granularité temporelle si nécessaire."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – Lookback exact",
            (
                "- **\"S0 (spot)\"** : fixe le centre de l’axe des spots de la heatmap sur lequel la formule exacte est évaluée.\n"
                "- **\"T (maturité, années)\"** : fournit les maturités à partir desquelles on construit l’axe vertical de la heatmap.\n"
                "- **\"t (temps courant)\"** : champ numérique permettant de considérer une option lookback déjà en cours de vie (temps écoulé depuis l’émission).\n"
                "- **\"Taux sans risque r\"** : utilisé pour actualiser l’espérance du payoff dans la formule fermée.\n"
                "- **\"Volatilité σ\"** : volatilité constante supposée par le modèle BSM sous‑jacent."
            ),
        )
        t0_lb = st.number_input(
            "t (temps courant)",
            value=0.0,
            min_value=0.0,
            key="t0_lb_exact",
            help="Temps déjà écoulé depuis l’émission de l’option lookback (en années).",
        )
        if st.button(
            "Calculer le prix lookback exact",
            key="btn_price_lb_exact",
        ):
            try:
                lookback_opt = lookback_call_option(
                    T=float(T_common),
                    t=float(t0_lb),
                    S0=float(common_spot_value),
                    r=float(r_common),
                    sigma=float(sigma_common),
                )
                price_lb_exact = float(lookback_opt.price_exact())
                st.success(f"Prix lookback (formule exacte) = {price_lb_exact:.6f}")
            except Exception as exc:
                st.error(f"Erreur lookback (formule exacte) : {exc}")
        st.caption(
            f"Paramètres utilisés pour le prix lookback exact : "
            f"S0={common_spot_value:.4f}, T={T_common:.4f}, r={r_common:.4f}, σ={sigma_common:.4f}, t={t0_lb:.4f}"
        )
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
        render_method_explainer(
            "🎲 Méthode Monte Carlo pour lookback",
            (
                "- **Étape 1 – Grille temporelle** : on découpe l’horizon `[0, T]` en un certain nombre de pas de temps. Plus la grille est fine, mieux on détecte les extrêmes du sous‑jacent.\n"
                "- **Étape 2 – Simulation des trajectoires** : on simule, sous la mesure neutre au risque, de nombreuses trajectoires `S_t` via un GBM avec volatilité constante `σ`, en appliquant à chaque pas un choc gaussien.\n"
                "- **Étape 3 – Suivi de l’extrême** : pour chaque trajectoire, on met à jour à chaque pas le maximum (ou le minimum) atteint jusqu’alors. Cette valeur représente l’\"historique condensé\" de la trajectoire pour le payoff lookback.\n"
                "- **Étape 4 – Évaluation du payoff** : à la date finale, on calcule le payoff en fonction de cet extrême (par exemple `max(M_T - K, 0)` où `M_T = max_{0≤t≤T} S_t`), ou les variantes floating strike selon le type de contrat.\n"
                "- **Étape 5 – Actualisation** : on actualise le payoff obtenu sur chaque trajectoire au taux sans risque `r_common` jusqu’à la date présente.\n"
                "- **Étape 6 – Moyenne Monte Carlo** : le prix est obtenu en moyennant ces payoffs actualisés sur l’ensemble des trajectoires simulées.\n"
                "- **Étape 7 – Construction de la heatmap** : on répète l’algorithme pour toutes les combinaisons `(S0, T)` de la grille, de sorte à remplir une matrice de prix lookback Monte Carlo comparable à la surface analytique.\n"
                "- **Étape 8 – Analyse d’erreur** : en comparant cette surface MC à la surface exacte, on évalue la qualité de la simulation (variabilité statistique, biais de discretisation des extrêmes) et on ajuste `n_iters_lb` ou la taille des pas de temps si nécessaire."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – Lookback Monte Carlo",
            (
                "- **\"S0 (spot)\"** : centre de l’axe des spots sur lequel les trajectoires lookback sont simulées.\n"
                "- **\"T (maturité, années)\"** : ensemble des maturités pour lesquelles on simule les trajectoires et construit la heatmap.\n"
                "- **\"t (temps courant) MC\"** : temps déjà écoulé avant le début de la période de simulation, pour traiter des options en cours de vie.\n"
                "- **\"Taux sans risque r\"** : intervient dans le drift neutre au risque et l’actualisation des payoffs.\n"
                "- **\"Volatilité σ\"** : volatilité supposée constante dans les trajectoires Monte Carlo.\n"
                "- **\"Itérations Monte Carlo\"** : nombre de trajectoires simulées pour chaque couple `(S0, T)`."
            ),
        )
        t0_lb_mc = st.number_input(
            "t (temps courant) MC",
            value=0.0,
            min_value=0.0,
            key="t0_lb_mc",
            help="Temps déjà écoulé avant la période de simulation Monte Carlo (en années).",
        )
        n_iters_lb = st.number_input(
            "Itérations Monte Carlo",
            value=1000,
            min_value=100,
            key="n_iters_lb_mc",
            help="Nombre de trajectoires lookback simulées pour chaque couple (S0, T).",
        )
        if st.button(
            "Calculer le prix lookback MC",
            key="btn_price_lb_mc",
        ):
            try:
                lookback_opt_mc = lookback_call_option(
                    T=float(T_common),
                    t=float(t0_lb_mc),
                    S0=float(common_spot_value),
                    r=float(r_common),
                    sigma=float(sigma_common),
                )
                price_lb_mc = float(lookback_opt_mc.price_monte_carlo(int(n_iters_lb)))
                st.success(f"Prix lookback (Monte Carlo) = {price_lb_mc:.6f}")
            except Exception as exc:
                st.error(f"Erreur lookback Monte Carlo : {exc}")
        st.caption(
            f"Paramètres utilisés pour le prix lookback MC : "
            f"S0={common_spot_value:.4f}, T={T_common:.4f}, r={r_common:.4f}, σ={sigma_common:.4f}, "
            f"t={t0_lb_mc:.4f}, N_iters={int(n_iters_lb)}"
        )
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
    render_unlock_sidebar_button("tab_barrier", "🔓 Réactiver T (onglet Barrière)")
    render_general_definition_explainer(
        "🚧 Comprendre les options barrière",
        (
            "- **Principe de base** : une option barrière est activée ou désactivée en fonction du franchissement d'un niveau de prix prédéfini (`Hu` ou `Hd`). La trajectoire du sous‑jacent entre `0` et `T` devient donc déterminante.\n"
            "- **Knock-out** : l'option cesse d'exister dès que la barrière est touchée ; le droit d'exercer à l'échéance est alors perdu.\n"
            "- **Knock-in** : à l’inverse, l’option ne \"prend naissance\" que si la barrière a été franchie au moins une fois avant l’échéance.\n"
            "- **Up / Down** : on distingue les barrières **Up** (situées au‑dessus du spot initial) des barrières **Down** (situées en dessous), ce qui permet de modéliser des scénarios de protection ou de conditionnalité différentes.\n"
            "- **Sensibilité au chemin** : ces produits sont très sensibles au maillage temporel : plus les pas sont grossiers, plus on risque de manquer des franchissements de barrière entre deux dates de simulation.\n"
            "- **Objectif de l'onglet** : montrer comment le prix réagit aux combinaisons `S0`, `K`, `T`, `Hu/Hd`, `σ` et au type de barrière (in/out, up/down) via des simulations Monte Carlo."
        ),
    )
    (
        tab_barrier_up_out,
        tab_barrier_down_out,
        tab_barrier_up_in,
        tab_barrier_down_in,
    ) = st.tabs(["Up-and-out", "Down-and-out", "Up-and-in", "Down-and-in"])

    with tab_barrier_up_out:
        st.subheader("Up-and-out")
        render_method_explainer(
            "⬆️ Méthode Monte Carlo – Up-and-out",
            (
                "- **Étape 1 – Définition du niveau de barrière** : on fixe une barrière haute `Hu` strictement au‑dessus du spot `S0_common`. Le contrat stipule qu’en cas de franchissement de `Hu` avant `T`, l’option est annulée.\n"
                "- **Étape 2 – Simulation des trajectoires** : on simule des trajectoires `S_t` sous la mesure neutre au risque (GBM) en discrétisant `[0, T_common]` en `n_steps_up` pas de temps.\n"
                "- **Étape 3 – Détection du knock‑out** : pour chaque trajectoire, on initialise un indicateur `knocked_out = False`. À chaque pas, si `S_t ≥ Hu_up`, on met `knocked_out = True` et on peut considérer que la trajectoire ne contribuera plus au payoff.\n"
                "- **Étape 4 – Calcul du payoff terminal** : à la maturité, pour les trajectoires qui ne sont pas en knock‑out (`knocked_out = False`), on calcule le payoff européen standard `max(±(S_T-K_common), 0)`. Pour les trajectoires en knock‑out, le payoff est `0`.\n"
                "- **Étape 5 – Actualisation et moyenne** : on actualise tous les payoffs par `exp(-r_common T_common)` puis on moyenne sur toutes les trajectoires.\n"
                "- **Étape 6 – Construction de la heatmap barrière** : en répétant ces étapes pour différentes valeurs de `S0_common` ou `Hu`, on peut cartographier l’impact de la position de la barrière sur le prix, et visualiser le compromis entre protection et coût de la prime."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – Up-and-out",
            (
                "- **\"S0 (spot)\"** : niveau de départ du sous‑jacent pour toutes les trajectoires simulées.\n"
                "- **\"K (strike)\"** : strike de l’option barrière (call ou put) utilisée pour le payoff si la barrière n’est jamais touchée.\n"
                "- **\"T (maturité, années)\"** : durée de vie de l’option, donc horizon de simulation.\n"
                "- **\"Taux sans risque r\"** et **\"Dividende continu d\"** : utilisés pour définir le drift neutre au risque et actualiser les payoffs.\n"
                "- **\"Volatilité σ\"** : volatilité constante supposée dans les trajectoires Monte Carlo.\n"
                "- **\"Call / Put\"** : choix du type d’option (call ou put) sur lequel la barrière s’applique.\n"
                "- **\"Barrière haute Hu\"** : niveau de prix au‑dessus du spot à partir duquel le knock‑out se déclenche.\n"
                "- **\"Trajectoires Monte Carlo\"** : nombre de chemins simulés pour estimer le prix.\n"
                "- **\"Pas de temps MC\"** : nombre de pas de temps par trajectoire, qui conditionne la finesse de la détection de la barrière."
            ),
        )
        cpflag_barrier_up = st.selectbox(
            "Call / Put",
            ["Call", "Put"],
            key="cpflag_barrier_up",
            help="Choix du type d’option barrière (call ou put).",
        )
        cpflag_barrier_up_char = "c" if cpflag_barrier_up == "Call" else "p"
        Hu_up = st.number_input("Barrière haute Hu", value=max(110.0, S0_common * 1.1), min_value=S0_common, key="Hu_up")
        n_paths_up = st.number_input(
            "Trajectoires Monte Carlo",
            value=1000,
            min_value=500,
            step=500,
            key="n_paths_barrier_up",
            help="Nombre de trajectoires simulées pour la barrière Up-and-out.",
        )
        n_steps_up = st.number_input(
            "Pas de temps MC",
            value=200,
            min_value=10,
            key="n_steps_barrier_up",
            help="Nombre de pas de temps pour suivre le franchissement de la barrière.",
        )

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
        render_method_explainer(
            "⬇️ Méthode Monte Carlo – Down-and-out",
            (
                "- **Étape 1 – Positionnement de la barrière basse** : on choisit une barrière `Hd` située en dessous du spot `S0_common`. L’option disparaît si `S_t` tombe à ou sous ce niveau avant la maturité.\n"
                "- **Étape 2 – Simulation des trajectoires** : on simule de nombreuses trajectoires `S_t` sous la mesure neutre au risque jusqu’à `T_common`, en `n_steps_down` pas de temps.\n"
                "- **Étape 3 – Suivi du knock‑out** : pour chaque trajectoire, on surveille `S_t`. Dès que `S_t ≤ Hd_down`, on enregistre un état `knocked_out = True`.\n"
                "- **Étape 4 – Payoff terminal** : à l’échéance, si `knocked_out = False`, on calcule le payoff européen standard (call ou put selon `cpflag_barrier_down`). Si `knocked_out = True`, le payoff est nul.\n"
                "- **Étape 5 – Actualisation et moyennage** : on actualise les payoffs et on en prend la moyenne sur toutes les trajectoires pour obtenir le prix Monte Carlo.\n"
                "- **Étape 6 – Étude de sensibilité** : la répétition de ce calcul pour différents `Hd` et `T` permet d’analyser la probabilité de survie de l’option et l’amplitude de la réduction de prime liée à la barrière."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – Down-and-out",
            (
                "- **\"S0 (spot)\"** : valeur initiale utilisée pour les trajectoires.\n"
                "- **\"K (strike)\"** : strike de l’option à barrière.\n"
                "- **\"T (maturité, années)\"** : horizon temporel de l’option.\n"
                "- **\"Taux sans risque r\"** et **\"Dividende continu d\"** : interviennent dans le drift neutre au risque et l’actualisation des payoffs.\n"
                "- **\"Volatilité σ\"** : volatilité constante supposée dans les simulations.\n"
                "- **\"Call / Put\"** : sélection du type d’option (call ou put).\n"
                "- **\"Barrière basse Hd\"** : niveau de prix en dessous du spot à partir duquel le knock‑out est activé.\n"
                "- **\"Trajectoires Monte Carlo\"** : nombre de chemins simulés.\n"
                "- **\"Pas de temps MC\"** : nombre de pas de simulation par trajectoire."
            ),
        )
        cpflag_barrier_down = st.selectbox(
            "Call / Put",
            ["Call", "Put"],
            key="cpflag_barrier_down",
            help="Choix du type d’option barrière (call ou put).",
        )
        cpflag_barrier_down_char = "c" if cpflag_barrier_down == "Call" else "p"
        Hd_down = st.number_input(
            "Barrière basse Hd",
            value=max(1.0, S0_common * 0.8),
            min_value=0.0001,
            key="Hd_down",
            help="Niveau de barrière basse en dessous du spot.",
        )
        n_paths_down = st.number_input(
            "Trajectoires Monte Carlo",
            value=1000,
            min_value=500,
            step=500,
            key="n_paths_barrier_down",
            help="Nombre de trajectoires simulées pour la barrière Down-and-out.",
        )
        n_steps_down = st.number_input(
            "Pas de temps MC",
            value=200,
            min_value=10,
            key="n_steps_barrier_down",
            help="Nombre de pas de temps pour suivre la barrière.",
        )

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
        render_method_explainer(
            "⬆️ Méthode Monte Carlo – Up-and-in",
            (
                "- **Étape 1 – Définition de la condition de knock‑in** : l’option n’a de valeur que si, à un moment entre `0` et `T_common`, le sous‑jacent a franchi la barrière haute `Hu`.\n"
                "- **Étape 2 – Simulation des trajectoires** : on simule un grand nombre de trajectoires `S_t` sous la mesure neutre au risque, sur `n_steps_up_in` pas de temps.\n"
                "- **Étape 3 – Suivi du knock‑in** : pour chaque trajectoire, on initialise un drapeau `knocked_in = False`. À chaque pas, si `S_t ≥ Hu_up_in`, on met `knocked_in = True`.\n"
                "- **Étape 4 – Évaluation à maturité** : à `T_common`, si `knocked_in = True`, on calcule le payoff européen standard (call ou put). Si `knocked_in = False`, le payoff est nul, car la barrière n’a jamais été touchée.\n"
                "- **Étape 5 – Actualisation et moyenne** : on actualise les payoffs et on en prend la moyenne pour obtenir le prix de l’option Up‑and‑in.\n"
                "- **Étape 6 – Lien avec l’Up‑and‑out** : théoriquement, pour un même niveau de barrière, la somme des prix Up‑and‑in et Up‑and‑out (avec même type d’option) s’approche du prix de l’option vanilla, ce qui fournit un contrôle de cohérence."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – Up-and-in",
            (
                "- `S0_common` : spot initial.\n"
                "- `K_common` : strike de l’option conditionnelle.\n"
                "- `T_common` : maturité de l’option.\n"
                "- `r_common` : taux sans risque.\n"
                "- `d_common` : dividende continu.\n"
                "- `sigma_common` : volatilité utilisée pour les simulations.\n"
                "- `cpflag_barrier_up_in` : type d’option (call ou put) pour le scénario Up‑and‑in.\n"
                "- `Hu_up_in` : niveau de barrière haute déclenchant le knock‑in.\n"
                "- `n_paths_up_in` : nombre de trajectoires Monte Carlo.\n"
                "- `n_steps_up_in` : nombre de pas de temps par trajectoire.\n"
                "- `knock_in` : paramètre logique interne positionné à `True` pour spécifier la nature knock‑in du produit.\n"
                "- Variables internes : drapeau de knock‑in par trajectoire, facteur d’actualisation, générateur pseudo‑aléatoire."
            ),
        )
        cpflag_barrier_up_in = st.selectbox(
            "Call / Put",
            ["Call", "Put"],
            key="cpflag_barrier_up_in",
            help="Type d’option (call ou put) pour le scénario Up-and-in.",
        )
        cpflag_barrier_up_in_char = "c" if cpflag_barrier_up_in == "Call" else "p"
        Hu_up_in = st.number_input(
            "Barrière haute Hu (Up-in)",
            value=max(110.0, S0_common * 1.1),
            min_value=S0_common,
            key="Hu_up_in",
            help="Niveau de barrière haute activant l’option Up-and-in.",
        )
        n_paths_up_in = st.number_input(
            "Trajectoires Monte Carlo (Up-in)",
            value=1000,
            min_value=500,
            step=500,
            key="n_paths_barrier_up_in",
            help="Nombre de trajectoires simulées pour l’Up-and-in.",
        )
        n_steps_up_in = st.number_input(
            "Pas de temps MC (Up-in)",
            value=200,
            min_value=10,
            key="n_steps_barrier_up_in",
            help="Nombre de pas de temps par trajectoire pour l’Up-and-in.",
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
        render_method_explainer(
            "⬇️ Méthode Monte Carlo – Down-and-in",
            (
                "- **Étape 1 – Condition de knock‑in** : l’option ne vaut quelque chose que si la barrière basse `Hd` a été touchée ou cassée au moins une fois avant `T_common`.\n"
                "- **Étape 2 – Simulation** : on simule des trajectoires du sous‑jacent et on surveille `S_t` à chaque pas.\n"
                "- **Étape 3 – Suivi du drapeau** : pour chaque trajectoire, on initialise `knocked_in = False`. Dès qu’un `S_t ≤ Hd_down_in` est observé, on met `knocked_in = True`.\n"
                "- **Étape 4 – Payoff terminal** : en fin de trajectoire, si `knocked_in = True`, on évalue le payoff européen (call ou put) ; sinon, le payoff est nul.\n"
                "- **Étape 5 – Actualisation et agrégation** : les payoffs sont actualisés, puis moyennés sur toutes les trajectoires pour obtenir le prix.\n"
                "- **Étape 6 – Sensibilité au niveau de barrière** : plus `Hd` est éloignée sous `S0_common`, moins la barrière a de chances d’être touchée et plus la prime du produit baisse, ce qui se visualise directement dans les résultats numériquement obtenus."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – Down-and-in",
            (
                "- **\"S0 (spot)\"** : spot de départ des trajectoires.\n"
                "- **\"K (strike)\"** : strike de l’option Down‑and‑in.\n"
                "- **\"T (maturité, années)\"** : horizon de l’option.\n"
                "- **\"Taux sans risque r\"** et **\"Dividende continu d\"** : paramètres de taux utilisés dans la simulation et l’actualisation.\n"
                "- **\"Volatilité σ\"** : volatilité utilisée pour la dynamique Monte Carlo.\n"
                "- **\"Call / Put\"** : choix du type d’option.\n"
                "- **\"Barrière basse Hd (Down-in)\"** : niveau de prix sous lequel la barrière est considérée comme touchée.\n"
                "- **\"Trajectoires Monte Carlo (Down-in)\"** : nombre de trajectoires simulées.\n"
                "- **\"Pas de temps MC (Down-in)\"** : nombre de pas de temps par trajectoire."
            ),
        )
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
    render_unlock_sidebar_button("tab_bermudan", "🔓 Réactiver T (onglet Bermuda)")
    render_general_definition_explainer(
        "🏝️ Comprendre les options bermudéennes",
        (
            "- **Positionnement** : une option bermudéenne se situe entre l’option européenne (exercice uniquement à l’échéance) et l’option américaine (exercice possible en continu). Ici, l’exercice est possible sur un ensemble discret de dates prédéfinies.\n"
            "- **Calendrier d'exercice** : l’investisseur dispose d’une série de dates Bermudes (par exemple mensuelles ou trimestrielles) où il peut choisir d’exercer l’option. En dehors de ces dates, l’option reste inerte.\n"
            "- **Impact sur le prix** : plus on multiplie les dates possibles d’exercice, plus le produit se rapproche d’une option américaine en termes de flexibilité et de valorisation.\n"
            "- **Usage pratique** : ces produits apparaissent souvent dans les produits structurés et les options exotiques de marché de taux ou de change, où l’on souhaite offrir une flexibilité encadrée.\n"
            "- **Objectif de l’onglet** : proposer une valorisation cohérente de ces options à l’aide d’un schéma PDE de type Crank–Nicolson adapté au cadre Bermudéen."
        ),
    )
    cpflag_bmd = st.selectbox("Call / Put (bermuda)", ["Call", "Put"], key="cpflag_bmd")
    cpflag_bmd_char = "c" if cpflag_bmd == "Call" else "p"
    n_ex_dates_bmd = st.number_input(
        "Nombre de dates d'exercice Bermude",
        value=6,
        min_value=2,
        help="Les dates sont réparties uniformément sur la grille PDE (incluant l'échéance).",
        key="n_ex_dates_bmd",
    )

    render_method_explainer(
        "🧮 Méthode PDE Crank–Nicolson pour options bermudéennes",
        (
            "- **Étape 1 – Formulation PDE** : on écrit l’équation de Black–Scholes pour le prix `V(t, S)` en fonction du temps et du spot, en supposant volatilité constante `σ_common`, taux `r_common` et dividende `d_common`.\n"
            "- **Étape 2 – Changement de variable en log‑prix** : pour des raisons numériques, on travaille en log‑spot `x = ln(S/S0)` et on construit une grille spatiale régulière en `x` centrée autour de `S0_common`.\n"
            "- **Étape 3 – Discrétisation Crank–Nicolson** : la PDE est discrétisée dans le temps et l’espace en combinant une approche implicite et explicite (50 %–50 %). Cela conduit à des systèmes linéaires tridiagonaux à résoudre à chaque pas de temps.\n"
            "- **Étape 4 – Condition terminale** : à la maturité `T_common`, on initialise `V(T, S)` au payoff européen standard (call ou put) pour toutes les valeurs de `S` sur la grille.\n"
            "- **Étape 5 – Intégration temporelle backward** : on remonte le temps pas à pas en résolvant, à chaque pas, un système linéaire obtenu à partir des matrices `A` et `B` du schéma Crank–Nicolson. On applique en parallèle les conditions aux bornes (comportement pour `S → 0` et `S → +∞`).\n"
            "- **Étape 6 – Traitement des dates Bermudes** : à chaque date d’exercice autorisée, on remplace la valeur obtenue par la PDE par `max(V(t, S), payoff(S))`, de façon à imposer la possibilité d’exercice anticipé discret.\n"
            "- **Étape 7 – Lecture de la solution** : une fois revenue au temps initial, on lit la valeur de `V(0, S0_common)` sur la grille pour obtenir le prix. Les grecs `Delta`, `Gamma` et `Theta` sont ensuite calculés par différences finies à partir des valeurs de la grille dans un voisinage de `S0_common`."
        ),
    )
    render_inputs_explainer(
        "🔧 Paramètres utilisés – Bermuda (PDE)",
        (
            "- **\"S0 (spot)\"** : point de départ sur l’axe des prix pour lequel on lit le résultat de la PDE.\n"
            "- **\"K (strike)\"** : strike de l’option bermudéenne.\n"
            "- **\"T (maturité, années)\"** : échéance finale de l’option.\n"
            "- **\"Volatilité σ\"** : volatilité constante utilisée dans l’équation de Black–Scholes.\n"
            "- **\"Taux sans risque r\"** et **\"Dividende continu d\"** : paramètres de taux du sous‑jacent.\n"
            "- **\"Call / Put (bermuda)\"** : choix du type d’option.\n"
            "- **\"Nombre de dates d'exercice Bermude\"** : nombre de dates intermédiaires où l’exercice anticipé est autorisé (en plus de l’échéance)."
        ),
    )

    if st.button(
        f"Calculer le prix Bermuda (PDE) "
        f"(S0={S0_common:.2f}, K={K_common:.2f}, T={T_common:.2f}, r={r_common:.2f}, d={d_common:.2f}, σ={sigma_common:.2f})",
        key="btn_bmd_cn",
    ):
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
    st.header("Options basket")
    render_general_definition_explainer(
        "🧺 Comprendre les options basket",
        (
            "- **Définition** : une option basket porte sur un panier de plusieurs sous‑jacents (actions, indices, etc.), typiquement via une combinaison pondérée de leurs prix.\n"
            "- **Mécanisme** : le payoff dépend de la valeur de ce panier (par exemple une moyenne pondérée des spots) à l’échéance ou selon une trajectoire donnée.\n"
            "- **Intérêt** : ces produits permettent de mutualiser le risque entre plusieurs actifs et de construire des vues relatives (sur‑/sous‑performance de certains composants du panier).\n"
            "- **Enjeux de modélisation** : la corrélation entre les sous‑jacents et la structure de la volatilité jouent un rôle central dans la forme de la distribution du panier.\n"
            "- **Objectif de cet onglet** : explorer, à travers une surface de prix et éventuellement une calibration, l’impact des paramètres de marché et des pondérations sur le prix du basket."
        ),
    )
    render_method_explainer(
        "🧮 Méthode utilisée dans le module Basket",
        (
            "- **Étape 1 – Chargement des historiques** : on charge les séries de prix de clôture des actifs du panier (ticker par ticker) à partir de fichiers CSV, en s’assurant d’avoir une période historique commune.\n"
            "- **Étape 2 – Construction du dataset** : à partir de ces séries, on construit un jeu de données où chaque ligne correspond à un scénario de marché (niveaux de prix, volatilités implicites, corrélations, strike, maturité, etc.) et à un prix d’option panier associé (label).\n"
            "- **Étape 3 – Séparation train / test** : le dataset est découpé selon `split_ratio` en un ensemble d’entraînement et un ensemble de test, afin de pouvoir évaluer la capacité du modèle à généraliser.\n"
            "- **Étape 4 – Entraînement du réseau de neurones** : un modèle `build_model_nn` est instancié avec une architecture adaptée (couches denses, activations non linéaires). On l’entraîne pendant `epochs` itérations pour minimiser une fonction de perte de type MSE entre prix prédits et prix \"théoriques\" (issus de BSM multi‑actifs ou Monte Carlo).\n"
            "- **Étape 5 – Suivi de l’apprentissage** : pendant l’entraînement, on suit l’évolution de la perte sur le jeu d’entraînement et de validation (MSE train / val) pour détecter surapprentissage ou sous‑apprentissage.\n"
            "- **Étape 6 – Construction des heatmaps de prix** : une fois le modèle entraîné, on le met en production sur une grille de paramètres (par exemple `S` et `K` autour de valeurs communes) pour produire une heatmap des prix d’option basket.\n"
            "- **Étape 7 – Construction de la surface de volatilité implicite** : en inversant éventuellement les prix du modèle sur un ensemble de paramètres, on peut reconstruire une surface de volatilité implicite associée au panier et la comparer aux données de marché.\n"
            "- **Étape 8 – Analyse des résultats** : les heatmaps et les courbes MSE permettent de juger de la qualité de l’approximation et de l’intérêt du modèle pour un pricing rapide en temps réel."
        ),
    )
    render_inputs_explainer(
        "🔧 Paramètres utilisés – Basket",
        (
            "- **\"S0 (spot)\"** : niveau de spot de référence utilisé pour centrer certaines grilles de prix du panier.\n"
            "- **\"K (strike)\"** : strike de référence du basket, autour duquel on définit les domaines de strikes.\n"
            "- **\"T (maturité, années)\"** : maturité de référence utilisée pour les surfaces de prix ou de volatilité.\n"
            "- **\"Taux sans risque r\"** : taux utilisé pour actualiser les flux dans les modèles internes.\n"
            "- **Sélection des actifs du panier** : zone de texte / boutons permettant de choisir les tickers qui composeront le basket.\n"
            "- **\"Train ratio\"** : pourcentage du dataset historique utilisé pour l’apprentissage (le reste servant au test).\n"
            "- **\"Epochs d'entraînement\"** : nombre de passes sur le dataset lors de l’entraînement du réseau de neurones."
        ),
    )
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

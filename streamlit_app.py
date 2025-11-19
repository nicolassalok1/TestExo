import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

from Longstaff.option import Option
from Longstaff.pricing import (
    black_scholes_merton,
    crr_pricing,
    monte_carlo_simulation,
)
from Longstaff.process import GeometricBrownianMotion, HestonProcess
from Lookback.european_call import european_call_option
from Lookback.lookback_call import lookback_call_option


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

    def __init__(self, Typeflag, cpflag, S0, K, T, vol, r, d):
        self.Typeflag = Typeflag
        self.cpflag = cpflag
        self.S0 = S0
        self.K = K
        self.T = T
        self.vol = vol
        self.r = r
        self.d = d

    def CN_option_info(
        self,
        Typeflag=None,
        cpflag=None,
        S0=None,
        K=None,
        T=None,
        vol=None,
        r=None,
        d=None,
    ):
        """
        Résout la PDE et retourne (Price, Delta, Gamma, Theta).
        """

        Typeflag = Typeflag or self.Typeflag
        cpflag = cpflag or self.cpflag
        S0 = S0 or self.S0
        K = K or self.K
        T = T or self.T
        vol = vol or self.vol
        r = r or self.r
        d = d or self.d

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

        if cpflag == "c":
            values = np.clip(S0 * np.exp(X) - K, 0, 1e10)
        elif cpflag == "p":
            values = np.clip(K - S0 * np.exp(X), 0, 1e10)
        else:
            raise ValueError("cpflag doit être 'c' ou 'p'.")

        payoff = values.copy()
        values_prev_time = values.copy()

        if Typeflag == "Am":
            for time_index in range(n_time):
                if time_index == n_time - 1:
                    values_prev_time = values.copy()
                values = B.dot(values)
                values = Ainv.dot(values)
                values = np.where(values > payoff, values, payoff)

        elif Typeflag == "Bmd":
            exercise_step = 10
            for time_index in range(n_time):
                if time_index == n_time - 1:
                    values_prev_time = values.copy()
                values = B.dot(values)
                values = Ainv.dot(values)
                if time_index % exercise_step == 0:
                    values = np.where(values > payoff, values, payoff)

        elif Typeflag == "Eu":
            for time_index in range(n_time):
                if time_index == n_time - 1:
                    values_prev_time = values.copy()
                values = B.dot(values)
                values = Ainv.dot(values)
                if cpflag == "c":
                    values[0] = 0.0
                    values[-1] = S0 * np.exp(x_max) - K * np.exp(-r * dt * time_index)
                else:
                    values[0] = K * np.exp(-r * dt * (n_time - time_index))
                    values[-1] = 0.0
        else:
            raise ValueError("Typeflag doit être 'Eu', 'Am' ou 'Bmd'.")

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

tab_european, tab_american, tab_lookback, tab_barrier, tab_bermudan = st.tabs(
    ["Européenne", "Américaine", "Lookback", "Barrière", "Bermuda"]
)


with tab_european:
    st.header("Option européenne")
    st.caption(
        "Les heatmaps affichent les prix call / put sur un carré Spot × Strike centré autour du spot défini dans la"
        " barre latérale."
    )

    tab_eu_bsm, tab_eu_mc = st.tabs(
        ["Black–Scholes–Merton", "Monte Carlo"]
    )

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
        n_paths_am = st.number_input("Trajectoires Monte Carlo", value=10_000, min_value=100, key="n_paths_am")
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
        n_iters_lb = st.number_input("Itérations Monte Carlo", value=10_000, min_value=100, key="n_iters_lb_mc")
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

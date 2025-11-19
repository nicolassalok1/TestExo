"""Heston model utilities implemented in torch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

# Sépare les types réels/complexes pour éviter les conversions implicites coûteuses.
_REAL_DTYPE = torch.float64
_COMPLEX_DTYPE = torch.complex128


def _as_real(value) -> torch.Tensor:
    # Uniformise les scalaires/listes en tenseurs float64.
    return torch.as_tensor(value, dtype=_REAL_DTYPE)


def _as_complex(value) -> torch.Tensor:
    # Force les paramètres FFT/CF dans l'espace complexe requis par Heston.
    # On évite ainsi les promotions automatiques différentes selon le backend.
    return torch.as_tensor(value, dtype=_COMPLEX_DTYPE)


@dataclass
class HestonParams:
    """Container for Heston parameters."""

    kappa: torch.Tensor
    theta: torch.Tensor
    sigma: torch.Tensor
    rho: torch.Tensor
    v0: torch.Tensor

    @staticmethod
    def from_unconstrained(a, b, c, d, e, eps: float = 1e-4) -> "HestonParams":
        """Map unconstrained tensors to valid Heston parameters."""
        a_t = torch.as_tensor(a, dtype=_REAL_DTYPE)
        b_t = torch.as_tensor(b, dtype=_REAL_DTYPE)
        c_t = torch.as_tensor(c, dtype=_REAL_DTYPE)
        d_t = torch.as_tensor(d, dtype=_REAL_DTYPE)
        e_t = torch.as_tensor(e, dtype=_REAL_DTYPE)
        offset = torch.tensor(eps, dtype=_REAL_DTYPE, device=a_t.device)

        def positive(x: torch.Tensor) -> torch.Tensor:
            # Softplus garantit la positivité; l'offset éloigne des frontières.
            return F.softplus(x) + offset

        # Rho vit dans (-1, 1); on applique un sigmoïde rescalé pour rester borné.
        rho = -0.999 + 1.998 * torch.sigmoid(d_t)
        return HestonParams(
            kappa=positive(a_t),
            theta=positive(b_t),
            sigma=positive(c_t),
            rho=rho,
            v0=positive(e_t),
        )


def heston_cf(u, T, S0, r, q, params: HestonParams) -> torch.Tensor:
    """Little Heston trap characteristic function."""
    # Mise au même format numérique pour tous les arguments (réels ou complexes).
    u_c = _as_complex(u)
    T_t = _as_real(T)
    S0_t = _as_real(S0)
    r_t = _as_real(r)
    q_t = _as_real(q)

    kappa = params.kappa.to(dtype=_COMPLEX_DTYPE)
    theta = params.theta.to(dtype=_COMPLEX_DTYPE)
    sigma = params.sigma.to(dtype=_COMPLEX_DTYPE)
    rho = params.rho.to(dtype=_COMPLEX_DTYPE)
    v0 = params.v0.to(dtype=_COMPLEX_DTYPE)

    i = torch.complex(torch.tensor(0.0, dtype=_REAL_DTYPE), torch.tensor(1.0, dtype=_REAL_DTYPE))
    sigma_sq = sigma ** 2
    d = torch.sqrt((rho * sigma * i * u_c - kappa) ** 2 + sigma_sq * (i * u_c + u_c ** 2))
    denom = kappa - rho * sigma * i * u_c + d
    # Petits epsilons complexes évitent divisions/logs instables quand denom -> 0.
    eps_c = torch.tensor(1e-12, dtype=_COMPLEX_DTYPE)
    g = (kappa - rho * sigma * i * u_c - d) / (denom + eps_c)

    exp_term = torch.exp(-d * T_t)
    one_minus_g_exp = 1.0 - g * exp_term
    one_minus_g = 1.0 - g
    log_term = torch.log(one_minus_g_exp / (one_minus_g + eps_c))
    C = (kappa * theta / sigma_sq) * ((kappa - rho * sigma * i * u_c - d) * T_t - 2.0 * log_term)
    D = ((kappa - rho * sigma * i * u_c - d) / sigma_sq) * ((1.0 - exp_term) / (one_minus_g_exp + eps_c))

    drift = torch.log(S0_t) + (r_t - q_t) * T_t
    phi = torch.exp(i * u_c * drift + C + v0 * D)

    # La CF doit valoir 1 pour u=0 ; correction directe pour compenser les arrondis.
    zero_mask = torch.abs(u_c) < 1e-12
    if torch.any(zero_mask):
        phi = torch.where(zero_mask, torch.ones_like(phi, dtype=_COMPLEX_DTYPE), phi)
    return phi


def _simpson_weights(n: int, dtype=torch.float64) -> torch.Tensor:
    # Poids de Simpson utilisés par la carrée FFT pour intégrer les fréquences.
    weights = torch.ones(n, dtype=dtype)
    if n > 1:
        weights[1:n - 1:2] = 4.0
    if n > 2:
        weights[2:n - 2:2] = 2.0
    weights = weights / 3.0
    return weights


def _linear_interpolate(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Interpolation linéaire maison (torch ne fournit pas d'équivalent 1D natif).
    x_clamped = torch.clamp(x, xp[0], xp[-1])
    indices = torch.searchsorted(xp, x_clamped)
    indices = torch.clamp(indices, min=1, max=xp.numel() - 1)
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]
    denom = torch.clamp(x1 - x0, min=1e-12)
    weight = (x_clamped - x0) / denom
    return y0 + weight * (y1 - y0)


def carr_madan_call_torch(
    S0,
    r,
    q,
    T,
    params: HestonParams,
    K,
    alpha: float = 1.5,
    Nfft: int = 2 ** 12,
    eta: float = 0.25,
) -> torch.Tensor:
    """Carr-Madan FFT-based Heston call price computation with interpolation over strikes."""
    # Tous les scalaires sont aplanis pour vectoriser la FFT dans torch.
    S0_t = _as_real(S0)
    r_t = _as_real(r)
    q_t = _as_real(q)
    T_t = _as_real(T)
    alpha_t = _as_real(alpha)
    eta_t = _as_real(eta)
    N = int(Nfft)

    K_input = _as_real(K)
    original_shape = K_input.shape if K_input.shape != torch.Size([]) else ()
    K_tensor = K_input.reshape(-1)
    j = torch.arange(N, dtype=_REAL_DTYPE)
    u = j * eta_t
    lambda_ = 2.0 * torch.pi / (N * eta_t)
    b = lambda_ * N / 2.0
    k = -b + lambda_ * j

    i = torch.complex(torch.tensor(0.0, dtype=_REAL_DTYPE), torch.tensor(1.0, dtype=_REAL_DTYPE))
    u_complex = u - i * (alpha_t + 1.0)
    phi = heston_cf(u_complex, T_t, S0_t, r_t, q_t, params)

    numerator = torch.exp(-r_t * T_t) * phi
    denominator = (alpha_t ** 2 + alpha_t - u ** 2) + i * u * (2.0 * alpha_t + 1.0)
    # Dénominateur régularisé pour éviter les fréquences où la formule diverge.
    psi = numerator / (denominator + torch.tensor(1e-12, dtype=_COMPLEX_DTYPE))

    weights = _simpson_weights(N, dtype=_REAL_DTYPE)
    exp_term = torch.exp(i * b * u)
    fft_input = psi * exp_term * weights * eta_t
    fft_vals = torch.fft.fft(fft_input)
    # On retire l'exponentielle de damping pour retrouver le prix du call brut.
    calls = torch.exp(-alpha_t * k) / torch.pi * fft_vals.real

    K_grid = torch.exp(k)
    prices = _linear_interpolate(K_grid, calls, K_tensor)
    if original_shape == ():
        return prices.squeeze()
    return prices.reshape(original_shape)


def carr_madan_put_torch(
    S0,
    r,
    q,
    T,
    params: HestonParams,
    K,
    alpha: float = 1.5,
    Nfft: int = 2 ** 12,
    eta: float = 0.25,
) -> torch.Tensor:
    """Carr-Madan FFT-based Heston put price computation with interpolation over strikes."""
    # Tous les scalaires sont aplanis pour vectoriser la FFT dans torch.
    S0_t = _as_real(S0)
    r_t = _as_real(r)
    q_t = _as_real(q)
    T_t = _as_real(T)
    alpha_t = _as_real(alpha)
    eta_t = _as_real(eta)
    N = int(Nfft)

    K_input = _as_real(K)
    original_shape = K_input.shape if K_input.shape != torch.Size([]) else ()
    K_tensor = K_input.reshape(-1)
    j = torch.arange(N, dtype=_REAL_DTYPE)
    u = j * eta_t
    lambda_ = 2.0 * torch.pi / (N * eta_t)
    b = lambda_ * N / 2.0
    k = -b + lambda_ * j

    i = torch.complex(torch.tensor(0.0, dtype=_REAL_DTYPE), torch.tensor(1.0, dtype=_REAL_DTYPE))
    # Pour puts, on utilise u - i * alpha (Carr-Madan 1999)
    u_complex = u - i * alpha_t
    phi = heston_cf(u_complex, T_t, S0_t, r_t, q_t, params)

    numerator = torch.exp(-r_t * T_t) * phi
    denominator = (alpha_t ** 2 + alpha_t - u ** 2) + i * u * (2.0 * alpha_t + 1.0)
    # Dénominateur régularisé pour éviter les fréquences où la formule diverge.
    psi = numerator / (denominator + torch.tensor(1e-12, dtype=_COMPLEX_DTYPE))

    weights = _simpson_weights(N, dtype=_REAL_DTYPE)
    exp_term = torch.exp(i * b * u)
    fft_input = psi * exp_term * weights * eta_t
    fft_vals = torch.fft.fft(fft_input)
    # On retire l'exponentielle de damping pour retrouver le prix du put brut.
    puts = torch.exp(-alpha_t * k) / torch.pi * fft_vals.real

    K_grid = torch.exp(k)
    prices = _linear_interpolate(K_grid, puts, K_tensor)
    if original_shape == ():
        return prices.squeeze()
    return prices.reshape(original_shape)

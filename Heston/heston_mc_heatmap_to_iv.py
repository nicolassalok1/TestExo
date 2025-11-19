#!/usr/bin/env python3
"""
Script pour générer une heatmap de prix Heston via Monte Carlo,
puis calculer la surface IV Black-Scholes à partir de ces prix.
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from heston_torch import HestonParams

# Configuration
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")

# Paramètres Heston calibrés (exemple - à remplacer par vos valeurs)
calib = {
    "kappa": 2.0,
    "theta": 0.04,
    "sigma": 0.3,
    "rho": -0.7,
    "v0": 0.04,
}

# Paramètres de marché
S0 = 100.0  # Prix spot de référence
r = 0.02    # Taux sans risque
q = 0.0     # Dividend yield
T = 1.0     # Maturité (1 an)

# Paramètres de la grille
span = 10.0  # S0 ± 10
points = 21  # 21x21 grille

# Paramètres Monte Carlo
n_paths = 100000  # Nombre de trajectoires
n_steps = 100     # Nombre de pas de temps


def params_from_calib(calib: dict) -> HestonParams:
    """Convert calibration dict to HestonParams."""
    return HestonParams(
        kappa=torch.tensor(calib["kappa"], dtype=torch.float64, device=DEVICE),
        theta=torch.tensor(calib["theta"], dtype=torch.float64, device=DEVICE),
        sigma=torch.tensor(calib["sigma"], dtype=torch.float64, device=DEVICE),
        rho=torch.tensor(calib["rho"], dtype=torch.float64, device=DEVICE),
        v0=torch.tensor(calib["v0"], dtype=torch.float64, device=DEVICE),
    )


def heston_monte_carlo_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    n_paths: int = 50000,
    n_steps: int = 100,
    option_type: str = "call",
    log_steps: bool = False,
    log_prefix: str | None = None,
) -> float:
    """Price option using Monte Carlo simulation with Heston model."""
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Extract Heston parameters
    kappa = float(params.kappa.cpu())
    theta = float(params.theta.cpu())
    sigma = float(params.sigma.cpu())
    rho = float(params.rho.cpu())
    v0 = float(params.v0.cpu())
    
    # Initialize arrays
    S = np.ones(n_paths) * S0
    v = np.ones(n_paths) * v0
    
    # Generate correlated random numbers
    for step in range(1, n_steps + 1):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)
        
        # Update variance (with Milstein scheme)
        v_sqrt = np.sqrt(np.maximum(v, 0))
        v = v + kappa * (theta - v) * dt + sigma * v_sqrt * sqrt_dt * Z2
        v = np.maximum(v, 0)  # Ensure non-negative variance
        
        # Update stock price (with Euler scheme)
        S = S * np.exp((r - q - 0.5 * v) * dt + v_sqrt * sqrt_dt * Z1)

        if log_steps and (step % max(1, n_steps // 10) == 0 or step == n_steps):
            pct = 100 * step / n_steps
            prefix = f"[{log_prefix}] " if log_prefix else ""
            print(f"{prefix}MC step {step}/{n_steps} ({pct:.0f}%)")
    
    # Compute payoff
    if option_type == "call":
        payoff = np.maximum(S - K, 0)
    else:  # put
        payoff = np.maximum(K - S, 0)
    
    # Discount and average
    return np.exp(-r * T) * np.mean(payoff)


def bs_price(S0: float, K: float, T: float, vol: float, r: float, option_type: str = "call") -> float:
    """Black-Scholes option pricing."""
    if T <= 0.0 or vol <= 0.0 or S0 <= 0.0 or K <= 0.0:
        if option_type == "call":
            return max(0.0, S0 - K * math.exp(-r * T))
        else:
            return max(0.0, K * math.exp(-r * T) - S0)
    
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * vol * vol) * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    
    if option_type == "call":
        return S0 * nd1 - K * math.exp(-r * T) * nd2
    else:
        return K * math.exp(-r * T) * (1 - nd2) - S0 * (1 - nd1)


def implied_vol_from_price(
    price: float, S0: float, K: float, T: float, r: float, option_type: str = "call",
    tol: float = 1e-6, max_iter: int = 100
) -> float:
    """Calculate implied volatility using Newton-Raphson."""
    if option_type == "call":
        intrinsic = max(0.0, S0 - K * math.exp(-r * T))
    else:
        intrinsic = max(0.0, K * math.exp(-r * T) - S0)
    
    if price <= intrinsic + 1e-12:
        return 0.0
    
    # Bracket the solution
    low, high = 1e-6, 3.0
    p_high = bs_price(S0, K, T, high, r, option_type)
    
    while p_high < price and high < 10.0:
        high *= 2.0
        p_high = bs_price(S0, K, T, high, r, option_type)
    
    if p_high < price:
        return float("nan")
    
    # Bisection method
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        p_mid = bs_price(S0, K, T, mid, r, option_type)
        
        if abs(p_mid - price) < tol:
            return mid
        
        if p_mid > price:
            high = mid
        else:
            low = mid
    
    return 0.5 * (low + high)


def compute_heston_heatmap(
    S0_ref: float,
    calib: dict,
    r: float,
    q: float,
    T: float,
    span: float,
    points: int,
    n_paths: int,
    n_steps: int,
    option_type: str = "call",
    log_inner_mc: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute heatmap of Heston prices via Monte Carlo."""
    S_min, S_max = S0_ref - span, S0_ref + span
    S_grid = np.linspace(S_min, S_max, points)
    S0_rounded = math.ceil(S0_ref / 10) * 10
    K_grid = np.arange(S0_rounded - 20, S0_rounded + 21, 5)
    
    params = params_from_calib(calib)
    prices = np.zeros((len(S_grid), len(K_grid)))
    
    print(f"Computing {len(S_grid)}x{len(K_grid)} = {len(S_grid)*len(K_grid)} prices...")
    total = len(S_grid) * len(K_grid)
    count = 0
    
    for i, S in enumerate(S_grid):
        for j, K in enumerate(K_grid):
            prices[i, j] = heston_monte_carlo_price(
                S, K, T, r, q, params, n_paths, n_steps, option_type,
                log_steps=log_inner_mc and (i == 0 and j == 0),
                log_prefix=f"S={S:.2f},K={K:.2f}"
            )
            count += 1
            if count % 50 == 0:
                print(f"Progress: {count}/{total} ({100*count/total:.1f}%)")
    
    return S_grid, K_grid, prices


def compute_iv_surface_from_prices(
    S_grid: np.ndarray,
    K_grid: np.ndarray,
    prices: np.ndarray,
    T: float,
    r: float,
    option_type: str = "call",
) -> np.ndarray:
    """Convert Heston prices to Black-Scholes implied volatilities."""
    iv_surface = np.zeros_like(prices)
    
    print(f"\nComputing implied volatilities...")
    total = len(S_grid) * len(K_grid)
    count = 0
    
    for i, S in enumerate(S_grid):
        for j, K in enumerate(K_grid):
            price = prices[i, j]
            iv = implied_vol_from_price(price, S, K, T, r, option_type)
            iv_surface[i, j] = iv
            count += 1
            if count % 50 == 0:
                print(f"IV Progress: {count}/{total} ({100*count/total:.1f}%)")
    
    return iv_surface


def plot_heatmap(data: np.ndarray, K_grid: np.ndarray, S_grid: np.ndarray, title: str, zlabel: str = "Value"):
    """Create plotly heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            x=K_grid,
            y=S_grid,
            colorscale="Viridis",
            colorbar=dict(title=zlabel),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Strike K",
        yaxis_title="Spot S",
        width=700,
        height=600,
    )
    return fig


def plot_iv_surface_3d(S_grid: np.ndarray, K_grid: np.ndarray, iv_surface: np.ndarray, title: str):
    """Create 3D surface plot of implied volatility."""
    # Create meshgrid
    K_mesh, S_mesh = np.meshgrid(K_grid, S_grid)
    
    fig = go.Figure(
        data=go.Surface(
            x=K_mesh,
            y=S_mesh,
            z=iv_surface,
            colorscale="Viridis",
            colorbar=dict(title="Implied Vol"),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Strike K",
            yaxis_title="Spot S",
            zaxis_title="Implied Volatility",
        ),
        width=800,
        height=700,
    )
    return fig


if __name__ == "__main__":
    print("=" * 80)
    print("HESTON MONTE CARLO → HEATMAP → IMPLIED VOLATILITY SURFACE")
    print("=" * 80)
    print(f"\nParamètres Heston calibrés:")
    for k, v in calib.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nParamètres de marché:")
    print(f"  S0: {S0:.2f}")
    print(f"  r: {r:.4f}")
    print(f"  T: {T:.2f} ans")
    print(f"  Grille: S et K de {S0-span:.2f} à {S0+span:.2f}")
    print(f"  Points: {points}x{points}")
    print(f"  Monte Carlo: {n_paths:,} trajectoires, {n_steps} pas")
    
    # Step 1: Compute Heston Call prices via Monte Carlo
    print("\n" + "=" * 80)
    print("STEP 1: Computing Heston CALL prices via Monte Carlo")
    print("=" * 80)
    S_grid, K_grid, call_prices = compute_heston_heatmap(
        S0, calib, r, q, T, span, points, n_paths, n_steps, "call", log_inner_mc=True
    )
    
    # Step 2: Compute Heston Put prices via Monte Carlo
    print("\n" + "=" * 80)
    print("STEP 2: Computing Heston PUT prices via Monte Carlo")
    print("=" * 80)
    S_grid, K_grid, put_prices = compute_heston_heatmap(
        S0, calib, r, q, T, span, points, n_paths, n_steps, "put", log_inner_mc=False
    )
    
    # Step 3: Convert to Implied Volatilities
    print("\n" + "=" * 80)
    print("STEP 3: Computing Black-Scholes Implied Volatilities from Heston prices")
    print("=" * 80)
    call_iv_surface = compute_iv_surface_from_prices(S_grid, K_grid, call_prices, T, r, "call")
    put_iv_surface = compute_iv_surface_from_prices(S_grid, K_grid, put_prices, T, r, "put")
    
    # Statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nCall Prices:")
    print(f"  Min: {np.nanmin(call_prices):.4f}")
    print(f"  Max: {np.nanmax(call_prices):.4f}")
    print(f"  Mean: {np.nanmean(call_prices):.4f}")
    
    print(f"\nPut Prices:")
    print(f"  Min: {np.nanmin(put_prices):.4f}")
    print(f"  Max: {np.nanmax(put_prices):.4f}")
    print(f"  Mean: {np.nanmean(put_prices):.4f}")
    
    print(f"\nCall Implied Volatilities:")
    print(f"  Min: {np.nanmin(call_iv_surface):.4f}")
    print(f"  Max: {np.nanmax(call_iv_surface):.4f}")
    print(f"  Mean: {np.nanmean(call_iv_surface):.4f}")
    
    print(f"\nPut Implied Volatilities:")
    print(f"  Min: {np.nanmin(put_iv_surface):.4f}")
    print(f"  Max: {np.nanmax(put_iv_surface):.4f}")
    print(f"  Mean: {np.nanmean(put_iv_surface):.4f}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Heatmaps of prices
    fig_call_price = plot_heatmap(call_prices, K_grid, S_grid, "Heston Call Prices (Monte Carlo)", "Price")
    fig_put_price = plot_heatmap(put_prices, K_grid, S_grid, "Heston Put Prices (Monte Carlo)", "Price")
    
    # Heatmaps of IVs
    fig_call_iv = plot_heatmap(call_iv_surface, K_grid, S_grid, "Black-Scholes IV from Heston Call Prices", "IV")
    fig_put_iv = plot_heatmap(put_iv_surface, K_grid, S_grid, "Black-Scholes IV from Heston Put Prices", "IV")
    
    # 3D surface plots
    fig_call_iv_3d = plot_iv_surface_3d(S_grid, K_grid, call_iv_surface, "Call IV Surface (3D)")
    fig_put_iv_3d = plot_iv_surface_3d(S_grid, K_grid, put_iv_surface, "Put IV Surface (3D)")
    
    # Show all figures
    print("\nShowing Call Price Heatmap...")
    fig_call_price.show()
    
    print("Showing Put Price Heatmap...")
    fig_put_price.show()
    
    print("Showing Call IV Heatmap...")
    fig_call_iv.show()
    
    print("Showing Put IV Heatmap...")
    fig_put_iv.show()
    
    print("Showing Call IV 3D Surface...")
    fig_call_iv_3d.show()
    
    print("Showing Put IV 3D Surface...")
    fig_put_iv_3d.show()
    
    # Save data to CSV
    print("\n" + "=" * 80)
    print("SAVING DATA")
    print("=" * 80)
    
    # Create DataFrames
    df_results = []
    for i, S in enumerate(S_grid):
        for j, K in enumerate(K_grid):
            df_results.append({
                'S': S,
                'K': K,
                'call_price': call_prices[i, j],
                'put_price': put_prices[i, j],
                'call_iv': call_iv_surface[i, j],
                'put_iv': put_iv_surface[i, j],
            })
    
    df = pd.DataFrame(df_results)
    df.to_csv('heston_mc_iv_surface.csv', index=False)
    print(f"✓ Data saved to: heston_mc_iv_surface.csv")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

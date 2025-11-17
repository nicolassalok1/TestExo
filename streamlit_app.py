import numpy as np
import streamlit as st

from Longstaff.option import Option
from Longstaff.pricing import (
    black_scholes_merton,
    crr_pricing,
    monte_carlo_simulation,
)
from Longstaff.process import GeometricBrownianMotion, HestonProcess
from Lookback.barrier_call import barrier_call_option
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

    if Typeflag == "UNO":
        values = np.where(s_grid < Hu, values, 0.0)
    elif Typeflag == "DNO":
        values = np.where((s_grid > Hd) & (s_grid < Hu), values, 0.0)
    else:
        raise ValueError("Typeflag doit être 'UNO' ou 'DNO'.")

    values_prev_time = values.copy()

    for time_index in range(n_time):
        if time_index == n_time - 1:
            values_prev_time = values.copy()

        values = B.dot(values)
        values = Ainv.dot(values)

        s_grid = S0 * np.exp(X)
        if Typeflag == "UNO":
            values = np.where(s_grid < Hu, values, 0.0)
        elif Typeflag == "DNO":
            values = np.where((s_grid > Hd) & (s_grid < Hu), values, 0.0)

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
#  Application Streamlit unifiée
# ---------------------------------------------------------------------------


st.title("Application unifiée de pricing d'options")

tab_bermuda_barrier, tab_longstaff, tab_lookback = st.tabs(
    [
        "Bermuda / Barrier (Crank–Nicolson)",
        "Longstaff (MC / LS / BSM / CRR)",
        "Lookback / Barrier / European",
    ]
)


with tab_bermuda_barrier:
    st.header("Bermuda / European / American & Barrier options")

    st.sidebar.header("Paramètres Bermuda / Barrier")
    S0_bb = st.sidebar.number_input("S0 (spot) - Bermuda/Barrier", value=100.0, min_value=0.01, key="S0_bb")
    K_bb = st.sidebar.number_input("K (strike) - Bermuda/Barrier", value=100.0, min_value=0.01, key="K_bb")
    T_bb = st.sidebar.number_input("T (maturité) - Bermuda/Barrier", value=1.0, min_value=0.01, key="T_bb")
    vol_bb = st.sidebar.number_input("Volatilité σ - Bermuda/Barrier", value=0.4, min_value=0.0001, key="vol_bb")
    r_bb = st.sidebar.number_input("Taux sans risque r - Bermuda/Barrier", value=0.025, key="r_bb")
    d_bb = st.sidebar.number_input("Dividende continu d - Bermuda/Barrier", value=0.0175, key="d_bb")

    tab_vanilla, tab_barrier = st.tabs(["Bermuda / European / American", "Barrier options"])

    with tab_vanilla:
        st.subheader("Option européenne / américaine / bermudéenne")
        typeflag_bb = st.selectbox("Type d'option", ["Eu", "Am", "Bmd"], key="typeflag_bb")
        cpflag_bb = st.selectbox("Call / Put", ["c", "p"], key="cpflag_bb")

        if st.button("Calculer (Bermuda/Eu/Am)", key="btn_bb_vanilla"):
            model_bb = CrankNicolsonBS(
                Typeflag=typeflag_bb,
                cpflag=cpflag_bb,
                S0=S0_bb,
                K=K_bb,
                T=T_bb,
                vol=vol_bb,
                r=r_bb,
                d=d_bb,
            )
            price_bb, delta_bb, gamma_bb, theta_bb = model_bb.CN_option_info()

            st.write(f"**Prix**: {price_bb:.4f}")
            st.write(f"**Delta**: {delta_bb:.4f}")
            st.write(f"**Gamma**: {gamma_bb:.4f}")
            st.write(f"**Theta**: {theta_bb:.4f}")

    with tab_barrier:
        st.subheader("Options barrière (Up-and-out / Double knock-out)")
        barrier_type_bb = st.selectbox(
            "Type de barrière", ["UNO (Up-and-out)", "DNO (Double knock-out)"], key="barrier_type_bb"
        )
        barrier_flag_bb = "UNO" if barrier_type_bb.startswith("UNO") else "DNO"
        cpflag_barrier_bb = st.selectbox("Call / Put (barrière)", ["c", "p"], key="cpflag_barrier_bb")
        Hu_bb = st.number_input("Barrière haute Hu", value=120.0, min_value=0.01, key="Hu_bb")
        Hd_bb = st.number_input("Barrière basse Hd", value=0.0, min_value=0.0, key="Hd_bb")

        if st.button("Calculer (barrière)", key="btn_bb_barrier"):
            price_b, delta_b, gamma_b, theta_b = CN_Barrier_option(
                Typeflag=barrier_flag_bb,
                cpflag=cpflag_barrier_bb,
                S0=S0_bb,
                K=K_bb,
                Hu=Hu_bb,
                Hd=Hd_bb,
                T=T_bb,
                vol=vol_bb,
                r=r_bb,
                d=d_bb,
            )

            st.write(f"**Prix**: {price_b:.4f}")
            st.write(f"**Delta**: {delta_b:.4f}")
            st.write(f"**Gamma**: {gamma_b:.4f}")
            st.write(f"**Theta**: {theta_b:.4f}")


with tab_longstaff:
    st.header("Longstaff–Schwartz et autres pricers")

    st.sidebar.header("Paramètres Longstaff")
    S0_ls = st.sidebar.number_input("S0 (spot) - Longstaff", value=100.0, min_value=0.01, key="S0_ls")
    K_ls = st.sidebar.number_input("K (strike) - Longstaff", value=100.0, min_value=0.01, key="K_ls")
    T_ls = st.sidebar.number_input("T (maturité) - Longstaff", value=1.0, min_value=0.01, key="T_ls")
    is_call_ls = st.sidebar.selectbox("Type d'option - Longstaff", ["Call", "Put"], key="is_call_ls") == "Call"

    mu_ls = st.sidebar.number_input("μ (drift, MC)", value=0.05, key="mu_ls")
    sigma_ls = st.sidebar.number_input("σ (volatilité, MC/BSM)", value=0.2, min_value=0.0001, key="sigma_ls")
    r_ls = st.sidebar.number_input("Taux sans risque r (BSM / CRR)", value=0.05, key="r_ls")

    n_paths_ls = st.sidebar.number_input(
        "Nombre de trajectoires Monte Carlo (n)", value=10_000, min_value=100, key="n_paths_ls"
    )
    n_steps_ls = st.sidebar.number_input(
        "Nombre de pas de temps (m)", value=50, min_value=1, key="n_steps_ls"
    )
    n_tree_ls = st.sidebar.number_input(
        "Nombre de pas de l'arbre CRR", value=250, min_value=10, key="n_tree_ls"
    )

    process_type_ls = st.sidebar.selectbox(
        "Processus sous-jacent (Longstaff)", ["Geometric Brownian Motion", "Heston"], key="process_type_ls"
    )

    if process_type_ls == "Geometric Brownian Motion":
        process_ls = GeometricBrownianMotion(mu=mu_ls, sigma=sigma_ls)
        v0_ls = None
    else:
        st.sidebar.subheader("Paramètres Heston (Longstaff)")
        kappa_ls = st.sidebar.number_input("κ (vitesse de rappel)", value=2.0, key="kappa_ls")
        theta_ls = st.sidebar.number_input("θ (variance long terme)", value=0.04, key="theta_ls")
        eta_ls = st.sidebar.number_input("η (vol de la variance)", value=0.5, key="eta_ls")
        rho_ls = st.sidebar.number_input("ρ (corrélation)", value=-0.7, min_value=-0.99, max_value=0.99, key="rho_ls")
        v0_ls = st.sidebar.number_input("v0 (variance initiale)", value=0.04, min_value=0.0001, key="v0_ls")
        process_ls = HestonProcess(mu=mu_ls, kappa=kappa_ls, theta=theta_ls, eta=eta_ls, rho=rho_ls)

    option_ls = Option(s0=S0_ls, T=T_ls, K=K_ls, v0=v0_ls, call=is_call_ls)

    tab_mc, tab_ls, tab_bsm, tab_crr = st.tabs(
        ["Monte Carlo classique", "Longstaff–Schwartz (américaine)", "Black–Scholes–Merton", "Arbre CRR (américaine)"]
    )

    with tab_mc:
        st.subheader("Monte Carlo classique")
        if st.button("Calculer (MC)", key="btn_mc_ls"):
            price_mc = monte_carlo_simulation(
                option=option_ls,
                process=process_ls,
                n=int(n_paths_ls),
                m=int(n_steps_ls),
            )
            st.write(f"**Prix Monte Carlo**: {price_mc:.4f}")

    with tab_ls:
        st.subheader("Monte Carlo Longstaff–Schwartz (américaine)")
        if st.button("Calculer (LS)", key="btn_ls_ls"):
            price_ls = longstaff_schwartz_price(
                option=option_ls,
                process=process_ls,
                n_paths=int(n_paths_ls),
                n_steps=int(n_steps_ls),
            )
            st.write(f"**Prix Longstaff–Schwartz**: {price_ls:.4f}")

    with tab_bsm:
        st.subheader("Black–Scholes–Merton (européenne)")
        if st.button("Calculer (BSM)", key="btn_bsm_ls"):
            price_bsm = black_scholes_merton(r=r_ls, sigma=sigma_ls, option=option_ls)
            st.write(f"**Prix BSM**: {price_bsm:.4f}")

    with tab_crr:
        st.subheader("Arbre CRR (américaine)")
        if st.button("Calculer (CRR)", key="btn_crr_ls"):
            price_crr = crr_pricing(r=r_ls, sigma=sigma_ls, option=option_ls, n=int(n_tree_ls))
            st.write(f"**Prix CRR**: {price_crr:.4f}")


with tab_lookback:
    st.header("Lookback, Barrier & European (Lookback module)")

    st.sidebar.header("Paramètres Lookback")
    S0_lb = st.sidebar.number_input("S0 (spot) - Lookback", value=100.0, min_value=0.01, key="S0_lb")
    T_lb = st.sidebar.number_input("T (maturité) - Lookback", value=1.0, min_value=0.01, key="T_lb")
    t0_lb = st.sidebar.number_input("t (temps courant) - Lookback", value=0.0, min_value=0.0, key="t0_lb")
    r_lb = st.sidebar.number_input("Taux sans risque r - Lookback", value=0.05, key="r_lb")
    sigma_lb = st.sidebar.number_input("Volatilité σ - Lookback", value=0.2, min_value=0.0001, key="sigma_lb")

    K_lb = st.sidebar.number_input("K (strike, Euro/Barrière)", value=100.0, min_value=0.01, key="K_lb")
    B_lb = st.sidebar.number_input("B (barrière up-and-out)", value=120.0, min_value=0.01, key="B_lb")

    n_iters_lb = st.sidebar.number_input(
        "Itérations Monte Carlo (Lookback)", value=10_000, min_value=100, key="n_iters_lb"
    )
    n_t_lb = st.sidebar.number_input("Pas de temps PDE n_t (Lookback)", value=200, min_value=10, key="n_t_lb")
    n_s_lb = st.sidebar.number_input("Pas d'espace PDE n_s (Lookback)", value=200, min_value=10, key="n_s_lb")

    tab_eu_lb, tab_barrier_lb, tab_lookback_lb = st.tabs(
        ["European call", "Barrier up-and-out call", "Lookback floating call"]
    )

    with tab_eu_lb:
        st.subheader("European call option (Lookback module)")
        euro_lb = european_call_option(T=T_lb, t=t0_lb, S0=S0_lb, K=K_lb, r=r_lb, sigma=sigma_lb)
        method_eu_lb = st.selectbox(
            "Méthode de pricing (Euro)", ["Exacte (BSM)", "Monte Carlo", "PDE Crank–Nicolson"], key="method_eu_lb"
        )

        if st.button("Calculer (European)", key="btn_eu_lb"):
            if method_eu_lb == "Exacte (BSM)":
                price_eu_exact = euro_lb.price_exact()
                st.write(f"**Prix exact**: {price_eu_exact:.6f}")
            elif method_eu_lb == "Monte Carlo":
                price_eu_mc = euro_lb.price_monte_carlo(int(n_iters_lb))
                st.write(f"**Prix Monte Carlo**: {price_eu_mc:.6f}")
            else:
                euro_lb.price_pde(int(n_t_lb), int(n_s_lb))
                price_eu_pde = euro_lb.get_pde_result(S0_lb)
                st.write(f"**Prix PDE**: {price_eu_pde:.6f}")

    with tab_barrier_lb:
        st.subheader("Barrier up-and-out call option (Lookback module)")
        barrier_lb = barrier_call_option(T=T_lb, t=t0_lb, S0=S0_lb, K=K_lb, B=B_lb, r=r_lb, sigma=sigma_lb)
        method_barrier_lb = st.selectbox(
            "Méthode de pricing (Barrière)",
            ["Exacte (fermée)", "Monte Carlo", "PDE Crank–Nicolson"],
            key="method_barrier_lb",
        )

        if st.button("Calculer (Barrière)", key="btn_barrier_lb"):
            if method_barrier_lb == "Exacte (fermée)":
                price_barrier_exact = barrier_lb.price_exact()
                st.write(f"**Prix exact barrière**: {price_barrier_exact:.6f}")
            elif method_barrier_lb == "Monte Carlo":
                price_barrier_mc = barrier_lb.price_monte_carlo(int(n_iters_lb))
                st.write(f"**Prix Monte Carlo barrière**: {price_barrier_mc:.6f}")
            else:
                barrier_lb.price_pde(int(n_t_lb), int(n_s_lb))
                price_barrier_pde = barrier_lb.get_pde_result(S0_lb)
                st.write(f"**Prix PDE barrière**: {price_barrier_pde:.6f}")

    with tab_lookback_lb:
        st.subheader("Lookback call option (floating strike)")
        lookback_lb = lookback_call_option(T=T_lb, t=t0_lb, S0=S0_lb, r=r_lb, sigma=sigma_lb)
        method_lb_lb = st.selectbox(
            "Méthode de pricing (Lookback)", ["Exacte", "Monte Carlo", "PDE Crank–Nicolson"], key="method_lb_lb"
        )

        if st.button("Calculer (Lookback)", key="btn_lookback_lb"):
            if method_lb_lb == "Exacte":
                price_lb_exact = lookback_lb.price_exact()
                st.write(f"**Prix exact lookback**: {price_lb_exact:.6f}")
            elif method_lb_lb == "Monte Carlo":
                price_lb_mc = lookback_lb.price_monte_carlo(int(n_iters_lb))
                st.write(f"**Prix Monte Carlo lookback**: {price_lb_mc:.6f}")
            else:
                lookback_lb.price_pde(int(n_t_lb), int(n_s_lb))
                price_lb_pde = lookback_lb.get_pde_result(z=1.0)
                st.write(f"**Prix PDE lookback**: {price_lb_pde:.6f}")


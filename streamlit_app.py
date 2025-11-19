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

st.sidebar.header("Paramètres communs")
S0_common = st.sidebar.number_input("S0 (spot)", value=100.0, min_value=0.01, key="S0_common")
K_common = st.sidebar.number_input("K (strike)", value=100.0, min_value=0.01, key="K_common")
T_common = st.sidebar.number_input("T (maturité, années)", value=1.0, min_value=0.01, key="T_common")
sigma_common = st.sidebar.number_input("Volatilité σ", value=0.2, min_value=0.0001, key="sigma_common")
r_common = st.sidebar.number_input("Taux sans risque r", value=0.05, key="r_common")
d_common = st.sidebar.number_input("Dividende continu d", value=0.0, key="d_common")

tab_european, tab_american, tab_lookback, tab_barrier, tab_bermudan = st.tabs(
    ["Européenne", "Américaine", "Lookback", "Barrière", "Bermuda"]
)


with tab_european:
    st.header("Option européenne")
    cpflag_eu = "Call"
    cpflag_eu_char = "c"
    option_eu = Option(s0=S0_common, T=T_common, K=K_common, call=True)

    tab_eu_bsm, tab_eu_cn, tab_eu_mc = st.tabs(
        ["Black–Scholes–Merton", "PDE Crank–Nicolson", "Monte Carlo"]
    )

    with tab_eu_bsm:
        st.subheader("Formule fermée BSM")
        if st.button("Calculer (BSM)", key="btn_eu_bsm"):
            price_bsm = black_scholes_merton(r=r_common - d_common, sigma=sigma_common, option=option_eu)
            st.write(f"**Prix BSM**: {price_bsm:.4f}")

    with tab_eu_cn:
        st.subheader("PDE Crank–Nicolson")
        if st.button("Calculer (PDE)", key="btn_eu_cn"):
            model_cn = CrankNicolsonBS(
                Typeflag="Eu",
                cpflag=cpflag_eu_char,
                S0=S0_common,
                K=K_common,
                T=T_common,
                vol=sigma_common,
                r=r_common,
                d=d_common,
            )
            price_cn, delta_cn, gamma_cn, theta_cn = model_cn.CN_option_info()
            st.write(f"**Prix**: {price_cn:.4f}")
            st.write(f"**Delta**: {delta_cn:.4f}")
            st.write(f"**Gamma**: {gamma_cn:.4f}")
            st.write(f"**Theta**: {theta_cn:.4f}")

    with tab_eu_mc:
        st.subheader("Monte Carlo classique")
        n_paths_eu = st.number_input("Trajectoires Monte Carlo", value=10_000, min_value=100, key="n_paths_eu")
        n_steps_eu = st.number_input("Pas de temps", value=50, min_value=1, key="n_steps_eu")
        if st.button("Calculer (MC)", key="btn_eu_mc"):
            process_eu = GeometricBrownianMotion(mu=r_common - d_common, sigma=sigma_common)
            price_mc_eu = monte_carlo_simulation(
                option=option_eu,
                process=process_eu,
                n=int(n_paths_eu),
                m=int(n_steps_eu),
            )
            st.write(f"**Prix Monte Carlo**: {price_mc_eu:.4f}")


with tab_american:
    st.header("Option américaine")
    cpflag_am = st.selectbox("Call / Put (américaine)", ["Call", "Put"], key="cpflag_am")
    cpflag_am_char = "c" if cpflag_am == "Call" else "p"

    tab_am_ls, tab_am_crr, tab_am_cn = st.tabs(
        ["Longstaff–Schwartz", "Arbre CRR", "PDE Crank–Nicolson"]
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

        if st.button("Calculer (LS)", key="btn_am_ls"):
            option_am_ls = Option(s0=S0_common, T=T_common, K=K_common, v0=v0_am, call=cpflag_am == "Call")
            price_ls = longstaff_schwartz_price(
                option=option_am_ls,
                process=process_am,
                n_paths=int(n_paths_am),
                n_steps=int(n_steps_am),
            )
            st.write(f"**Prix Longstaff–Schwartz**: {price_ls:.4f}")

    with tab_am_crr:
        st.subheader("Arbre binomial CRR")
        n_tree_am = st.number_input("Nombre de pas de l'arbre", value=250, min_value=10, key="n_tree_am")
        if st.button("Calculer (CRR)", key="btn_am_crr"):
            option_am_crr = Option(s0=S0_common, T=T_common, K=K_common, call=cpflag_am == "Call")
            price_crr = crr_pricing(r=r_common, sigma=sigma_common, option=option_am_crr, n=int(n_tree_am))
            st.write(f"**Prix CRR**: {price_crr:.4f}")

    with tab_am_cn:
        st.subheader("PDE Crank–Nicolson (américaine)")
        if st.button("Calculer (PDE Am)", key="btn_am_cn"):
            model_am = CrankNicolsonBS(
                Typeflag="Am",
                cpflag=cpflag_am_char,
                S0=S0_common,
                K=K_common,
                T=T_common,
                vol=sigma_common,
                r=r_common,
                d=d_common,
            )
            price_am, delta_am, gamma_am, theta_am = model_am.CN_option_info()
            st.write(f"**Prix**: {price_am:.4f}")
            st.write(f"**Delta**: {delta_am:.4f}")
            st.write(f"**Gamma**: {gamma_am:.4f}")
            st.write(f"**Theta**: {theta_am:.4f}")


with tab_lookback:
    st.header("Options lookback (floating strike)")

    tab_lb_exact, tab_lb_mc, tab_lb_pde = st.tabs(["Exacte", "Monte Carlo", "PDE Crank–Nicolson"])

    with tab_lb_exact:
        st.subheader("Formule exacte")
        t0_lb = st.number_input("t (temps courant)", value=0.0, min_value=0.0, key="t0_lb_exact")
        if st.button("Calculer (Exact)", key="btn_lb_exact"):
            lookback_exact = lookback_call_option(
                T=T_common, t=t0_lb, S0=S0_common, r=r_common, sigma=sigma_common
            )
            price_lb_exact = lookback_exact.price_exact()
            st.write(f"**Prix exact lookback**: {price_lb_exact:.6f}")

    with tab_lb_mc:
        st.subheader("Monte Carlo lookback")
        t0_lb_mc = st.number_input("t (temps courant) MC", value=0.0, min_value=0.0, key="t0_lb_mc")
        n_iters_lb = st.number_input("Itérations Monte Carlo", value=10_000, min_value=100, key="n_iters_lb_mc")
        if st.button("Calculer (MC Lookback)", key="btn_lb_mc"):
            lookback_mc = lookback_call_option(T=T_common, t=t0_lb_mc, S0=S0_common, r=r_common, sigma=sigma_common)
            price_lb_mc = lookback_mc.price_monte_carlo(int(n_iters_lb))
            st.write(f"**Prix Monte Carlo lookback**: {price_lb_mc:.6f}")

    with tab_lb_pde:
        st.subheader("PDE Crank–Nicolson lookback")
        t0_lb_pde = st.number_input("t (temps courant) PDE", value=0.0, min_value=0.0, key="t0_lb_pde")
        n_t_lb = st.number_input("Pas de temps PDE n_t", value=200, min_value=10, key="n_t_lb")
        n_s_lb = st.number_input("Pas d'espace PDE n_s", value=200, min_value=10, key="n_s_lb")
        if st.button("Calculer (PDE Lookback)", key="btn_lb_pde"):
            lookback_pde = lookback_call_option(
                T=T_common, t=t0_lb_pde, S0=S0_common, r=r_common, sigma=sigma_common
            )
            lookback_pde.price_pde(int(n_t_lb), int(n_s_lb))
            price_lb_pde = lookback_pde.get_pde_result(z=1.0)
            st.write(f"**Prix PDE lookback**: {price_lb_pde:.6f}")


with tab_barrier:
    st.header("Options barrière")

    tab_barrier_cn, tab_barrier_lb = st.tabs(
        ["Crank–Nicolson (UNO / DNO)", "Up-and-out (fermée / MC / PDE)"]
    )

    with tab_barrier_cn:
        st.subheader("Barrières type up-and-out / double knock-out")
        barrier_type = st.selectbox(
            "Type de barrière", ["UNO (Up-and-out)", "DNO (Double knock-out)"], key="barrier_type_cn"
        )
        cpflag_barrier = st.selectbox("Call / Put", ["c", "p"], key="cpflag_barrier_cn")
        Hu_cn = st.number_input("Barrière haute Hu", value=120.0, min_value=0.01, key="Hu_cn")
        Hd_cn = st.number_input("Barrière basse Hd", value=0.0, min_value=0.0, key="Hd_cn")

        if st.button("Calculer (PDE Barrière)", key="btn_barrier_cn"):
            barrier_flag = "UNO" if barrier_type.startswith("UNO") else "DNO"
            price_b, delta_b, gamma_b, theta_b = CN_Barrier_option(
                Typeflag=barrier_flag,
                cpflag=cpflag_barrier,
                S0=S0_common,
                K=K_common,
                Hu=Hu_cn,
                Hd=Hd_cn,
                T=T_common,
                vol=sigma_common,
                r=r_common,
                d=d_common,
            )
            st.write(f"**Prix**: {price_b:.4f}")
            st.write(f"**Delta**: {delta_b:.4f}")
            st.write(f"**Gamma**: {gamma_b:.4f}")
            st.write(f"**Theta**: {theta_b:.4f}")

    with tab_barrier_lb:
        st.subheader("Barrière up-and-out (module Lookback)")
        t_barrier_lb = st.number_input("t (temps courant)", value=0.0, min_value=0.0, key="t_barrier_lb")
        B_lb = st.number_input("B (barrière up-and-out)", value=120.0, min_value=0.01, key="B_lb")
        method_barrier_lb = st.selectbox(
            "Méthode de pricing",
            ["Exacte (fermée)", "Monte Carlo", "PDE Crank–Nicolson"],
            key="method_barrier_lb",
        )
        n_iters_barrier = None
        n_t_barrier = None
        n_s_barrier = None
        if method_barrier_lb == "Monte Carlo":
            n_iters_barrier = st.number_input(
                "Itérations Monte Carlo", value=10_000, min_value=100, key="n_iters_barrier_lb"
            )
        elif method_barrier_lb == "PDE Crank–Nicolson":
            n_t_barrier = st.number_input("Pas de temps PDE n_t", value=200, min_value=10, key="n_t_barrier")
            n_s_barrier = st.number_input("Pas d'espace PDE n_s", value=200, min_value=10, key="n_s_barrier")

        if st.button("Calculer (Barrière UO)", key="btn_barrier_lb"):
            barrier_lb = barrier_call_option(
                T=T_common, t=t_barrier_lb, S0=S0_common, K=K_common, B=B_lb, r=r_common, sigma=sigma_common
            )
            if method_barrier_lb == "Exacte (fermée)":
                price_barrier_exact = barrier_lb.price_exact()
                st.write(f"**Prix exact barrière**: {price_barrier_exact:.6f}")
            elif method_barrier_lb == "Monte Carlo":
                price_barrier_mc = barrier_lb.price_monte_carlo(int(n_iters_barrier))
                st.write(f"**Prix Monte Carlo barrière**: {price_barrier_mc:.6f}")
            else:
                barrier_lb.price_pde(int(n_t_barrier), int(n_s_barrier))
                price_barrier_pde = barrier_lb.get_pde_result(S0_common)
                st.write(f"**Prix PDE barrière**: {price_barrier_pde:.6f}")


with tab_bermudan:
    st.header("Option bermudéenne")
    cpflag_bmd = st.selectbox("Call / Put (bermuda)", ["Call", "Put"], key="cpflag_bmd")
    cpflag_bmd_char = "c" if cpflag_bmd == "Call" else "p"

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
        )
        price_bmd, delta_bmd, gamma_bmd, theta_bmd = model_bmd.CN_option_info()
        st.write(f"**Prix**: {price_bmd:.4f}")
        st.write(f"**Delta**: {delta_bmd:.4f}")
        st.write(f"**Gamma**: {gamma_bmd:.4f}")
        st.write(f"**Theta**: {theta_bmd:.4f}")

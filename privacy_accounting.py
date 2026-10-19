import itertools
import math
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from noise_scheduler import NoiseScheduler
from scipy.integrate import quad
from tqdm import tqdm
from utils.logger import DPLogger, LogLevel
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
import json


def calculate_mu_value(sigmas, sigma_diffs, beta = 1.0):
    mus = []
    # Calculate mu value for each timestep except the last one at t = 0
    for i in range(len(sigmas) - 1):
        variance = (sigmas[i]**3)/(2*sigma_diffs[i])
        mu = np.sqrt(1/variance) * beta
        mus.append(mu)
    return torch.tensor(mus, device=sigmas.device).view(-1)

# KL equation
def Z(x, mu, p):
    return logsumexp([np.log(p) + mu*x - mu**2 / 2, np.log(1 - p)])

def phi(x):
    return 1/(np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

def calculate_kl_divergence(mus, p):
    """
    Calculate the KL divergence for Berry-Essen privacy.
    
    Args:
        sigmas (torch.Tensor): Sigma values for each timestep.
        sigma_diffs (torch.Tensor): Differences between consecutive sigma values.
        batch_size (int): Batch size for the calculations.
    
    Returns:
        torch.Tensor: KL divergence values for each timestep.
    """
    kls = []
    for i in tqdm(range(len(mus)), desc="KL Divergence calculations for all timesteps"):
        def integrand(x):
            return Z(x, mus[i], p) * (phi(x - mus[i]) - phi(x))
        lower = max(0.0, mus[i] - 10.0)
        upper = max(30.0, mus[i] + 10.0)
        kl = p * quad(integrand, lower, upper, limit = 500)[0]
        kls.append(kl)
    return torch.tensor(kls, device=mus.device).view(-1)

def calculate_kappa2(mus, p):
    """
    Calculate the Kappa2 value for Berry-Essen privacy.
    
    Args:
        mus (torch.Tensor): Mu values for each timestep.
        p (float): Probability value.
    
    Returns:
        torch.Tensor: Kappa2 values for each timestep.
    """
    kappa2s = []
    for i in tqdm(range(len(mus)), desc="Kappa2 calculations for all timesteps"):
        def integrand(x):
            return Z(x, mus[i], p)**2 * (p * phi(x - mus[i]) + (2 - p) * phi(x))
        lower = max(0.0, mus[i] - 10.0)
        upper = max(30.0, mus[i] + 10.0)
        kappa2 = quad(integrand,
                      lower,
                      upper,
                      limit = 500,
                      epsabs=1e-6,
                      epsrel=1e-6)[0]
        kappa2s.append(kappa2)
    return torch.tensor(kappa2s, device=mus.device).view(-1)

def calculate_kappa3(mus, kls, p):
    kappa3s = []
    for i in tqdm(range(len(mus)), desc="Kappa3 calculations for all timesteps"):
        def integrand_one(x):
            return (Z(x, mus[i], p) - kls[i])**3 * (p * phi(x - mus[i]) + (1 - p) * phi(x))
        def integrand_two(x):
            return (Z(x, mus[i], p) + kls[i])**3 * phi(x)
        
        lower = max(0.0, mus[i] - 10.0)
        upper = max(30.0, mus[i] + 10.0)
        kappa3_one = quad(integrand_one, lower, upper, limit = 500)[0]
        # integrand_two has only phi(x) peaked near x=0, not x=mu, so use lower=0
        kappa3_two = quad(integrand_two, 0.0, upper, limit = 500)[0]
        kappa3s.append(kappa3_one + kappa3_two)
    return torch.tensor(kappa3s, device=mus.device).view(-1)

def get_mu_tilde_privacy(kls, kappa2s):
    """
    Calculate mu_tilde for Berry-Essen privacy.
    
    Args:
        kls (torch.Tensor): kl divergence values.
        kappa2s (torch.Tensor): Kappa2 values
        kappa3s (torch.Tensor): Kappa3 values
    
    Returns:
        torch.Tensor: Compute the final mu_tilde value.
    """
    return 2 * torch.norm(kls, p = 1, dim = -1)/torch.sqrt(torch.norm(kappa2s, p = 1, dim = -1) - torch.norm(kls, p = 2, dim = -1)**2)

def get_gamma_privacy(kls, kappa2s, kappa3s):
    """
    Calculate gamma for Berry-Essen privacy.
    
    Args:
        kls (torch.Tensor): kl divergence values.
        kappa2s (torch.Tensor): Kappa2 values
        kappa3s (torch.Tensor): Kappa3 values
    
    Returns:
        torch.Tensor: Compute the final gamma value.
    """
    return 0.56 * torch.norm(kappa3s, p = 1, dim = -1) / (torch.norm(kappa2s, p = 1, dim = -1) - torch.norm(kls, p = 2, dim = -1)**2)**(3/2)

# def chi_squared_g_mu(mu):
#     return math.exp(mu**2) * norm.cdf(3 * mu / 2) + 3 * norm.cdf(-mu / 2) - 2

def chi_squared_g_mu(mu):
    # e^{mu^2} * Phi(3mu/2) + 3*Phi(-mu/2) - 2
    log_term1 = mu**2 + norm.logcdf(3 * mu / 2)
    log_term2 = math.log(3) + norm.logcdf(-mu / 2)

    # Avoid overflow: use logaddexp + expm1 instead of exp(...) + exp(...) - 2
    log_AB = np.logaddexp(log_term1, log_term2)
    result = 2.0 * math.expm1(log_AB - math.log(2.0))
    return max(result, 0.0)

def theorem_5_2(c, mu):
    """
    Theorem 5.2: C_p(f)^{⊗T} -> G_{p0 * sqrt(2 * chi^2_+(f))}
    
    c: limit of m*sqrt(T)/n
    mu: per-step GDP parameter
    
    Returns: asymptotic GDP parameter mu_0
    """
    f_div = chi_squared_g_mu(mu)
    mu_0 = math.sqrt(2) * c * math.sqrt(f_div)
    return mu_0

def corollary_5_4(c, sigma):
    """
    Corollary 5.4 from Dong, Roth & Su (2019):
      mu = sqrt(2) * c * sqrt(chi2_plus)
    where
      chi2_plus = exp(s^2) * Phi(1.5*s) + 3*Phi(-0.5*s) - 2,  s = 1/sigma.

    Numerically stable computation using log-space arithmetic
    to avoid overflow from exp(s^2) and cancellation near s=0.
    """
    s = 1.0 / sigma  # mu parameter of the Gaussian mechanism

    # --- Stable computation of chi2_plus(G_s) ---
    # Term A = exp(s^2) * Phi(1.5*s)
    # Compute in log-space: log(A) = s^2 + log(Phi(1.5*s))
    log_A = s**2 + norm.logcdf(1.5 * s)

    # Term B = 3 * Phi(-0.5*s)
    # For small s, Phi(-0.5*s) ≈ 0.5 - 0.5*s/sqrt(2*pi), so B ≈ 1.5 - ...
    # For large s, Phi(-0.5*s) → 0 via the tail, so B → 0.
    log_B = math.log(3.0) + norm.logcdf(-0.5 * s)

    # chi2_plus = A + B - 2
    # When s is small, A ≈ 1*(0.5 + ...) and B ≈ 1.5 - ..., so A+B ≈ 2.
    # Use log-sum-exp to get log(A + B), then expm1 for the subtraction of 2.

    log_AB = np.logaddexp(log_A, log_B)  # log(A + B)

    # chi2_plus = (A + B) - 2 = 2 * (exp(log(A+B) - log(2)) - 1)
    # Using expm1 to avoid cancellation when A+B ≈ 2.
    chi2_plus = 2.0 * math.expm1(log_AB - math.log(2.0))

    # Guard against tiny negative values from floating-point noise
    chi2_plus = max(chi2_plus, 0.0)

    mu = math.sqrt(2.0) * c * math.sqrt(chi2_plus)
    print(f"Corollary 5.4: c={c}, sigma={sigma}, mu={mu}")
    return mu

# Gaussian tradeoff functions
def G_mu(alpha, mu, err=1e-10):
    if (alpha > 1 - err):
        return 0.0
    if (alpha < err):
        return 1.0
    return norm.cdf(norm.ppf(1 - alpha) - mu)

# Define f-DP function
def f_dp(alpha, gamma, mu):
    val = G_mu(alpha + gamma, mu) - gamma
    return max(val, 0)

# Calculates delta for a given epsilon
def delta_eps(eps, gamma, mu):
    def objective(alpha):
        f_val = f_dp(alpha, gamma, mu)
        return -(-np.exp(eps) * alpha - f_val)  # negative for maximization
    res = minimize_scalar(objective, bounds=(1e-10, 1), method='bounded')
    return max(1 + (-res.fun), 0)
    

# To find epsilon for a given delta
def find_eps_for_delta(target_delta, gamma, mu_tilde):
    def obj(eps):
        return delta_eps(eps, gamma, mu_tilde) - target_delta

    return brentq(obj, 1e-6, 1000.0)

def find_eps_for_gdp_delta(target_delta, mu, eps_min=1e-9):
    eps_max = mu * np.sqrt(2 * np.log(1.0 / target_delta)) + mu**2

    def obj(eps):
        return gdp_to_eps_delta(mu, eps) - target_delta

    if obj(eps_max) > 0:
        raise ValueError(f"No solution found: delta={target_delta} may be too small for mu={mu}")

    return brentq(obj, eps_min, eps_max)

# Return the delta for a given mu and epsilon
def gdp_to_eps_delta(mu, eps):
    a = -eps/mu + mu/2
    b = -eps/mu - mu/2
    # Compute Phi(a) - exp(eps)*Phi(b) in log-space to avoid exp(eps) overflow
    # for large eps/mu values.
    log_first = norm.logcdf(a)
    log_second = eps + norm.logcdf(b)
    if log_second >= log_first:
        return 0.0
    return max(float(np.exp(log_first) * (-np.expm1(log_second - log_first))), 0.0)


# Return the total GDP composition of mu values
def composition_gdp(mus):
    return np.sqrt(np.sum(np.array(mus)**2))

# Composition of full batch mu values for multiple samples
def multi_composition_gdp(mus, num_samples, t_switch):
    mus_empirical = mus[:t_switch]
    return np.sqrt(num_samples) * composition_gdp(mus_empirical)

def compute_full_batch_multistep_gdp(mus, N, t_switch, delta):
    mu_N = multi_composition_gdp(mus, N, t_switch)
    # Convert to (eps, delta)-DP for reporting
    eps_full_batch = find_eps_for_gdp_delta(delta, mu_N)
    return {"mu_N": mu_N, "eps_full_batch": eps_full_batch}

# Get asymptotic mu GDP value
def calculate_asymptotic_mu(kls, kappa2s):
    kl_sum = torch.sum(kls)
    kappa2_sum = torch.sum(kappa2s)
    return 2 * kl_sum / torch.sqrt(kappa2_sum)

# Get asymptotic mu GDP value for multiple samples
def calculate_asymptotic_mu_multiple(kls, kappa2s, num_samples):
    kl_sums = torch.sum(kls)
    kappa2_sums = torch.sum(kappa2s)
    return 2 * math.sqrt(num_samples) * kl_sums / torch.sqrt(kappa2_sums)


def compute_fdp_N_samples_multistep(
    mus,
    p,
    N,
    t_switch,
    delta,
    edgeworth=False,
):
    """
    f-DP accounting for N independently generated samples, each running t_switch
    empirical denoiser steps with heterogeneous noise scales.

    Implements Sections 3-8 of the f-DP accounting document (symmetric case:
    same sigma schedule for all samples j).

    Under symmetry (using kl, kappa2, kappa3 as defined by Lemma B.1):
        K^(N)       = N * sum_i kl(f_i)
        (s^(N))^2   = N * sum_i kappa2_i               [direct sum, per Eq. (9)]
        mu^(N)      = 2 * K^(N) / s^(N)
        Gamma^(N)   = N * sum_i kappa3_i               [Berry-Esseen numerator]

    Args:
        mus:        Per-step mu values for the full schedule (torch.Tensor, shape [T]).
                    Computed via calculate_mu_value().
        p:          Poisson subsample rate.
        N:          Number of generated samples.
        t_switch:   Number of empirical denoiser steps (uses mus[:t_switch]).
        delta:      Target delta for (epsilon, delta)-DP conversion.
        edgeworth:  If True, also compute the Edgeworth-corrected epsilon (Section 6).

    Returns:
        dict with keys:
            mu_N          - CLT GDP parameter mu^(N)
            K_N           - Total kl sum K^(N)
            s_N           - Total std s^(N)
            be_error      - Berry-Esseen slack eta = C_BE * Gamma^(N) / (s^(N))^3
            eps_clt       - Epsilon from CLT GDP approximation (Eq. 13)
            eps_rigorous  - Epsilon from rigorous BE-corrected bound (Eq. 14); inf if infeasible
            eps_edgeworth - Epsilon from Edgeworth surrogate (Eq. 11); None if edgeworth=False
            delta, N, t_switch, kls, kappa2s, kappa3s
    """
    mus_empirical = mus[:t_switch]

    # Steps 3-4 of Algorithm (Section 8): compute per-step quantities per Lemma B.1.
    # Lemma B.1 defines kl, kappa2, kappa3 by specific integrals such that they are
    # directly additive across compositions per Eq. (7), so no kl^2 subtraction.
    kls = calculate_kl_divergence(mus_empirical, p)
    kappa2s = calculate_kappa2(mus_empirical, p)
    kappa3s = calculate_kappa3(mus_empirical, kls, p)

    # Section 4, Eq. (9): N-sample aggregation (symmetric case)
    kl_sum = kls.sum()
    kappa2_sum = kappa2s.sum()
    kappa3_sum = kappa3s.sum()

    K_N = N * kl_sum
    s_N_sq = N * kappa2_sum
    s_N = torch.sqrt(s_N_sq)

    # Section 5, Eq. (10): mu^(N) = 2 K^(N) / s^(N)  (Proposition 3: = sqrt(N) * mu^(1))
    mu_N = (2.0 * K_N / s_N).item()

    # Berry-Esseen error: eta = C_BE * Gamma^(N) / (s^(N))^3
    C_BE = 0.56
    Gamma_N = N * kappa3_sum
    be_error = (C_BE * Gamma_N / s_N ** 3).item()

    # Step 5: epsilon from CLT GDP approximation via delta(eps; mu^(N)) = delta (Eq. 13)
    eps_clt = find_eps_for_gdp_delta(delta, mu_N)

    # Rigorous bound (Eq. 14): delta(eps; mu^(N)) + exp(eps) * eta = delta
    def obj_rigorous(eps):
        return gdp_to_eps_delta(mu_N, eps) + np.exp(eps) * be_error - delta

    try:
        if obj_rigorous(1e-9) > 0:
            eps_rigorous = float('inf')
        else:
            eps_rigorous = brentq(obj_rigorous, 1e-9, 100.0)
    except ValueError:
        eps_rigorous = float('inf')

    # Section 6: Edgeworth-corrected epsilon
    eps_edgeworth = None
    if edgeworth:
        kappa3_N = (N * kappa3_sum).item()
        coeff = kappa3_N / (6.0 * s_N.item() ** 3)

        def edgeworth_f(alpha):
            if alpha <= 0 or alpha >= 1:
                return 0.0
            z = norm.ppf(1 - alpha)
            correction = coeff * z ** 2 * norm.pdf(z - mu_N)
            f_val = norm.cdf(z - mu_N) + correction
            
            if (f_val > 1.0 or f_val < 0.0):
                print(f"Warning: Edgeworth correction out of bounds for alpha={alpha:.4e}, z={z:.4f}, mu_N={mu_N:.4f}, coeff={coeff:.4e}, correction={correction:.4e}, f_val={f_val:.4e}")
            return max(min(f_val, 1.0), 0.0)

        def delta_edgeworth(eps):
            def obj(alpha):
                return -(1.0 - alpha - np.exp(eps) * edgeworth_f(alpha))
            res = minimize_scalar(obj, bounds=(1e-10, 1 - 1e-10), method='bounded')
            return max(-res.fun, 0.0)

        def obj_ew(eps):
            return delta_edgeworth(eps) - delta

        try:
            if obj_ew(1e-9) < 0:
                eps_edgeworth = 0.0
            elif obj_ew(100.0) > 0:
                eps_edgeworth = float('inf')
            else:
                eps_edgeworth = brentq(obj_ew, 1e-9, 100.0)
        except ValueError:
            eps_edgeworth = float('inf')

    return {
        "mu_N": mu_N,
        "K_N": K_N.item(),
        "s_N": s_N.item(),
        "be_error": be_error,
        "eps_clt": eps_clt,
        "eps_rigorous": eps_rigorous,
        "eps_edgeworth": eps_edgeworth,
        "delta": delta,
        "N": N,
        "t_switch": t_switch,
        "kls": kls,
        "kappa2s": kappa2s,
        "kappa3s": kappa3s,
    }

def _sgm_cumulants_PQ(mu, p, quad_limit=500, quad_eps=1e-9):
    """
    Per-step cumulants of the sampled-Gaussian privacy loss
        L(x) = log(1 - p + p * exp(mu*x - mu^2/2))
    under both P = N(0,1) and Q = (1-p)*N(0,1) + p*N(mu,1).

    Returns ((k1P, k2P, k3P, k4P), (k1Q, k2Q, k3Q, k4Q)).

    Implements Section 4 (raw moments via 1D quadrature, then conversion to cumulants).
    """
    mu = float(mu)
    p = float(p)

    def L(x):
        return Z(x, mu, p)

    # Integrate over a domain that covers both N(0,1) and the shifted N(mu,1) component.
    lower = min(-10.0, mu - 10.0)
    upper = max(10.0, mu + 10.0)

    mP = [0.0] * 4
    mQ = [0.0] * 4
    for r in range(1, 5):
        def integrand_P(x, r=r):
            return L(x) ** r * phi(x)

        def integrand_Q(x, r=r):
            return L(x) ** r * ((1.0 - p) * phi(x) + p * phi(x - mu))

        mP[r - 1] = quad(integrand_P, lower, upper, limit=quad_limit,
                         epsabs=quad_eps, epsrel=quad_eps)[0]
        mQ[r - 1] = quad(integrand_Q, lower, upper, limit=quad_limit,
                         epsabs=quad_eps, epsrel=quad_eps)[0]

    def raw_to_cum(m1, m2, m3, m4):
        k1 = m1
        k2 = m2 - m1 ** 2
        k3 = m3 - 3.0 * m2 * m1 + 2.0 * m1 ** 3
        k4 = m4 - 4.0 * m3 * m1 - 3.0 * m2 ** 2 + 12.0 * m2 * m1 ** 2 - 6.0 * m1 ** 4
        return k1, k2, k3, k4

    return raw_to_cum(*mP), raw_to_cum(*mQ)


def _stable_directional_delta(s_right, s_left, eps):
    """
    Stable evaluation of max(0, s_right - exp(eps) * s_left) for the
    f-DP directional formula

        delta(eps) = P_A(L > eps) - exp(eps) * P_B(L > eps)

    where s_right = P_A(L > eps) and s_left = P_B(L > eps).  Avoids the
    math.exp(eps) overflow that occurs whenever the brentq bracket is widened
    (e.g. eps_max = 500) by working with log-ratios and expm1.
    """
    if s_right <= 0.0:
        return 0.0
    if s_left <= 0.0:
        return s_right
    log_ratio = eps + math.log(s_left) - math.log(s_right)
    if log_ratio >= 0.0:
        return 0.0
    return s_right * (-math.expm1(log_ratio))


def edgeworth_cdf_deg2(z, V, K3, K4):
    """
    Degree-2 Edgeworth CDF of the standardized sum, Eq. (24):

        F_EW(z) = Phi(z) - phi(z) * [ K3/(6 V^{3/2}) (z^2 - 1)
                                    + K4/(24 V^2)  (z^3 - 3z)
                                    + K3^2/(72 V^3) (z^5 - 10 z^3 + 15 z) ]

    Note this is a numerical approximation; it can leave [0,1] or be non-monotone in tails.
    """
    z = np.asarray(z, dtype=float)
    rho3 = K3 / V ** 1.5
    rho4 = K4 / V ** 2
    bracket = (rho3 / 6.0) * (z ** 2 - 1.0) \
        + (rho4 / 24.0) * (z ** 3 - 3.0 * z) \
        + (rho3 ** 2 / 72.0) * (z ** 5 - 10.0 * z ** 3 + 15.0 * z)
    return norm.cdf(z) - norm.pdf(z) * bracket


def edgeworth_rho_valid(rho3_P, rho4_P, rho3_Q, rho4_Q):
    """
    True if the degree-2 Edgeworth expansion is controlled, i.e. its epsilon can
    be trusted.

    edgeworth_cdf_deg2 is an asymptotic expansion whose correction term carries
    factors rho3, rho4 and rho3^2 (see Eq. 24), so it only means anything while
    those stay below 1.  Once any of them exceeds 1 the "correction" is no
    smaller than the Gaussian term it corrects and the resulting epsilon is not
    a bound.  Non-finite rhos (V collapsing to 0) are invalid too.
    """
    return all(math.isfinite(r) and r <= 1.0
               for r in (rho3_P, rho4_P, rho3_Q, rho4_Q))


def compute_edgeworth_subsampled_gaussian(
    mus,
    p,
    N,
    t_switch,
    delta,
    eps_max=100.0,
):
    """
    Edgeworth accountant for composed Poisson-subsampled Gaussian mechanisms, per
    the Edgeworth/Zheng-style construction in the accompanying note.

    Steps (Sections 2-7 of the note):
      (i)   each step is the binary test P_i = N(0,1) vs Q_i = (1-p)N(0,1) + p N(mu_i,1);
      (ii)  cumulants kappa^P_{r,i}, kappa^Q_{r,i} of L_i(X_i) for r = 1..4;
      (iii) aggregate across the N*t_switch steps (cumulants are additive);
      (iv)  build degree-2 Edgeworth CDFs F_{P,EW}, F_{Q,EW};
      (v-vi) compute symmetric (eps, delta) via the directional formula (31)
             and take the max over both directions.

    Args:
        mus:       per-step mu values (torch.Tensor, shape [T]).
        p:         Poisson sampling probability (scalar).
        N:         number of generated samples (composition multiplier).
        t_switch:  number of empirical denoiser steps used per sample (mus[:t_switch]).
        delta:     target delta for (eps, delta)-DP.
        eps_max:   upper bracket for epsilon root-finding.

    Returns:
        dict with cumulant aggregates, Edgeworth-based eps, and per-direction diagnostics.
    """
    mus_emp = mus[:t_switch]
    p = float(p)

    k1P_list, k2P_list, k3P_list, k4P_list = [], [], [], []
    k1Q_list, k2Q_list, k3Q_list, k4Q_list = [], [], [], []
    for i in tqdm(range(t_switch), desc="Edgeworth per-step cumulants"):
        (k1P, k2P, k3P, k4P), (k1Q, k2Q, k3Q, k4Q) = _sgm_cumulants_PQ(
            mus_emp[i].item(), p
        )
        k1P_list.append(k1P); k2P_list.append(k2P); k3P_list.append(k3P); k4P_list.append(k4P)
        k1Q_list.append(k1Q); k2Q_list.append(k2Q); k3Q_list.append(k3Q); k4Q_list.append(k4Q)

    # Aggregate cumulants across the N * t_switch steps (Eqs. 19-22).
    M_P = N * float(np.sum(k1P_list))
    V_P = N * float(np.sum(k2P_list))
    K3_P = N * float(np.sum(k3P_list))
    K4_P = N * float(np.sum(k4P_list))

    M_Q = N * float(np.sum(k1Q_list))
    V_Q = N * float(np.sum(k2Q_list))
    K3_Q = N * float(np.sum(k3Q_list))
    K4_Q = N * float(np.sum(k4Q_list))

    if V_P <= 0.0 or V_Q <= 0.0:
        raise ValueError(f"Non-positive aggregate variance: V_P={V_P}, V_Q={V_Q}.")

    sP, sQ = math.sqrt(V_P), math.sqrt(V_Q)

    # Unstandardized Edgeworth CDFs of T_m, Eq. (30).
    def F_hat_P(t):
        return float(np.clip(edgeworth_cdf_deg2((t - M_P) / sP, V_P, K3_P, K4_P), 0.0, 1.0))

    def F_hat_Q(t):
        return float(np.clip(edgeworth_cdf_deg2((t - M_Q) / sQ, V_Q, K3_Q, K4_Q), 0.0, 1.0))

    # Directional delta(eps), Eq. (31). Symmetric guarantee = max over the two directions.
    def delta_PQ(eps):
        return _stable_directional_delta(1.0 - F_hat_Q(eps), 1.0 - F_hat_P(eps), eps)

    def delta_QP(eps):
        # Reverse direction uses the lower tail of T_m (privacy loss L' = -L):
        # δ_{Q→P}(ε) = F_P(-ε) - e^ε F_Q(-ε).
        return _stable_directional_delta(F_hat_P(-eps), F_hat_Q(-eps), eps)

    def delta_sym(eps):
        return max(delta_PQ(eps), delta_QP(eps))

    def eps_for_target(delta_fn):
        f = lambda e: delta_fn(e) - delta
        if f(1e-9) < 0:
            return 0.0
        if f(eps_max) > 0:
            return float('inf')
        try:
            return brentq(f, 1e-9, eps_max)
        except ValueError:
            return float('inf')

    eps_PQ = eps_for_target(delta_PQ)
    eps_QP = eps_for_target(delta_QP)
    eps_sym = eps_for_target(delta_sym)

    # Standardized cumulant diagnostics (Section 9, Eqs. 58-59).
    rho3_P = abs(K3_P) / V_P ** 1.5
    rho4_P = abs(K4_P) / V_P ** 2
    rho3_Q = abs(K3_Q) / V_Q ** 1.5
    rho4_Q = abs(K4_Q) / V_Q ** 2
    edgeworth_valid = edgeworth_rho_valid(rho3_P, rho4_P, rho3_Q, rho4_Q)

    return {
        "M_P": M_P, "V_P": V_P, "K3_P": K3_P, "K4_P": K4_P,
        "M_Q": M_Q, "V_Q": V_Q, "K3_Q": K3_Q, "K4_Q": K4_Q,
        "rho3_P": rho3_P, "rho4_P": rho4_P,
        "rho3_Q": rho3_Q, "rho4_Q": rho4_Q,
        "edgeworth_valid": edgeworth_valid,
        "eps_edgeworth_PQ": eps_PQ,
        "eps_edgeworth_QP": eps_QP,
        "eps_edgeworth_sym": eps_sym,
        "delta": delta,
        "N": N,
        "t_switch": t_switch,
    }


def compute_edgeworth_reuse_minibatch(
    mus,
    k,
    n,
    N,
    t_switch,
    delta,
    eps_max=1000.0,
):
    """
    Edgeworth accountant for the "sample once per generated sample, reuse across
    all T denoising steps" mechanism described.

    Regime: a minibatch of k examples is drawn (with replacement) once per
    generated sample and then reused for the full t_switch-step empirical
    trajectory.  The amplification by subsampling is applied to the whole
    trajectory, not per denoising step.

    Implements:
      - q = 1 - (1 - 1/n)^k                        (Section 2, inclusion prob.)
      - mu_bar^2 = sum_{t=0}^{t_switch-1} mu_t^2   (Section 3, trajectory shift)
      - one-sample mixture P = N(0,1),
        Q = (1-q) N(0,1) + q N(mu_bar, 1)         (Section 3 / Section 5)
      - cumulants of L = log(dQ/dP) under P and Q via _sgm_cumulants_PQ at
        sampling rate q and per-trajectory mean mu_bar
      - aggregate over N generated samples (cumulants are additive in N), then
        build degree-2 Edgeworth CDFs and solve the directional delta(eps)
        equations for both directions.

    Also reports a CLT/GDP diagnostic
      mu_eff,reuse approx q * sqrt(N * (exp(mu_bar^2) - 1))
    (Section 4, heterogeneous-mu_t case).

    Args:
        mus:       per-step mu values (torch.Tensor, shape [T]).
        k:         minibatch size (samples drawn with replacement per generated
                   sample).
        n:         training dataset size.
        N:         number of generated samples (composition multiplier).
        t_switch:  number of empirical denoiser steps used per sample
                   (mus[:t_switch]).
        delta:     target delta for (eps, delta)-DP.
        eps_max:   upper bracket for epsilon root-finding.

    Returns:
        dict with q, mu_bar, aggregate cumulants, Edgeworth-based eps, and a
        CLT/GDP eps diagnostic.
    """
    if t_switch <= 0:
        raise ValueError(f"t_switch must be positive, got {t_switch}.")

    mus_emp = mus[:t_switch]
    mu_bar_sq = float((mus_emp.float() ** 2).sum().item())
    mu_bar = math.sqrt(max(mu_bar_sq, 0.0))

    q = 1.0 - (1.0 - 1.0 / float(n)) ** int(k)

    # Per-trajectory cumulants of L = log(dQ/dP) under P and Q,
    # treating the full T-step trajectory as the base Gaussian mechanism with
    # parameter mu_bar and subsampling rate q.
    (k1P, k2P, k3P, k4P), (k1Q, k2Q, k3Q, k4Q) = _sgm_cumulants_PQ(mu_bar, q)

    # Aggregate across N generated samples (cumulants are additive).
    M_P = N * k1P
    V_P = N * k2P
    K3_P = N * k3P
    K4_P = N * k4P

    M_Q = N * k1Q
    V_Q = N * k2Q
    K3_Q = N * k3Q
    K4_Q = N * k4Q

    if V_P <= 0.0 or V_Q <= 0.0:
        raise ValueError(f"Non-positive aggregate variance: V_P={V_P}, V_Q={V_Q}.")

    sP, sQ = math.sqrt(V_P), math.sqrt(V_Q)

    def F_hat_P(t):
        return float(np.clip(edgeworth_cdf_deg2((t - M_P) / sP, V_P, K3_P, K4_P), 0.0, 1.0))

    def F_hat_Q(t):
        return float(np.clip(edgeworth_cdf_deg2((t - M_Q) / sQ, V_Q, K3_Q, K4_Q), 0.0, 1.0))

    def delta_PQ(eps):
        return _stable_directional_delta(1.0 - F_hat_Q(eps), 1.0 - F_hat_P(eps), eps)

    def delta_QP(eps):
        return _stable_directional_delta(F_hat_P(-eps), F_hat_Q(-eps), eps)

    def delta_sym(eps):
        return max(delta_PQ(eps), delta_QP(eps))

    def eps_for_target(delta_fn):
        f = lambda e: delta_fn(e) - delta
        if f(1e-9) < 0:
            return 0.0
        if f(eps_max) > 0:
            return float('inf')
        try:
            return brentq(f, 1e-9, eps_max)
        except ValueError:
            return float('inf')

    eps_PQ = eps_for_target(delta_PQ)
    eps_QP = eps_for_target(delta_QP)
    eps_sym = eps_for_target(delta_sym)

    # CLT/GDP diagnostic for the reuse mechanism (Section 4 of the note):
    #   mu_eff,reuse approx q * sqrt(N * (exp(mu_bar^2) - 1)).
    # math.expm1 overflows for very large mu_bar^2; fall back to inf in that case.
    try:
        mu_eff_reuse = q * math.sqrt(N * math.expm1(mu_bar_sq))
    except OverflowError:
        mu_eff_reuse = float('inf')
    eps_clt_reuse = (find_eps_for_gdp_delta(delta, mu_eff_reuse)
                     if mu_eff_reuse > 0.0 else 0.0)

    rho3_P = abs(K3_P) / V_P ** 1.5
    rho4_P = abs(K4_P) / V_P ** 2
    rho3_Q = abs(K3_Q) / V_Q ** 1.5
    rho4_Q = abs(K4_Q) / V_Q ** 2
    edgeworth_valid = edgeworth_rho_valid(rho3_P, rho4_P, rho3_Q, rho4_Q)

    return {
        "q": q,
        "mu_bar": mu_bar,
        "mu_bar_sq": mu_bar_sq,
        "k": int(k),
        "n": int(n),
        "M_P": M_P, "V_P": V_P, "K3_P": K3_P, "K4_P": K4_P,
        "M_Q": M_Q, "V_Q": V_Q, "K3_Q": K3_Q, "K4_Q": K4_Q,
        "rho3_P": rho3_P, "rho4_P": rho4_P,
        "rho3_Q": rho3_Q, "rho4_Q": rho4_Q,
        "edgeworth_valid": edgeworth_valid,
        "eps_edgeworth_PQ": eps_PQ,
        "eps_edgeworth_QP": eps_QP,
        "eps_edgeworth_sym": eps_sym,
        "mu_eff_reuse": mu_eff_reuse,
        "eps_clt_reuse": eps_clt_reuse,
        "delta": delta,
        "N": N,
        "t_switch": t_switch,
    }


def compute_certified_rowsubsample_bound(
    mus,
    k,
    n,
    N,
    t_switch,
    delta,
    prv_eps_error=0.01,
    prv_delta_error=None,
    pld_discretization=1e-5,
):
    """
    Certified (non-asymptotic) RDP-based (eps, delta) upper bound for the
    row-subsampled version of the mechanism accounted for by
    compute_edgeworth_reuse_minibatch, computed with autodp.

        sigma_eff = 1 / mu_bar,   mu_bar^2 = sum_{t < t_switch} mu_t^2,

    then amplified by sampling at rate prob = k / n and composed over the N
    generated samples via autodp's RDP accountant.  Expect this bound to be
    quite loose relative to the Edgeworth/CLT numbers.

    Additionally reports a certified PLD/PRV bound for the Poisson/add-remove
    dominating pair, which is much tighter than the RDP bounds: a
    (lower, estimate, upper) bracket from prv-accountant, plus the Google
    dp-accounting pessimistic value as a cross-check and fallback.

    Args:
        mus:       per-step mu values (torch.Tensor, shape [T]).
        k:         minibatch size (rows subsampled per generated sample).
        n:         training dataset size.
        N:         number of generated samples (composition multiplier).
        t_switch:  number of empirical denoiser steps used per sample
                   (mus[:t_switch]).
        delta:     target delta for (eps, delta)-DP.

    Returns:
        dict with prob, mu_bar, sigma_eff and certified epsilons:
            eps_rdp_poisson, eps_rdp_poisson_improved,
            eps_rdp_wor, eps_rdp_wor_improved
    """
    if t_switch <= 0:
        raise ValueError(f"t_switch must be positive, got {t_switch}.")

    mus_emp = mus[:t_switch]
    mu_bar_sq = float((mus_emp.float() ** 2).sum().item())
    mu_bar = math.sqrt(max(mu_bar_sq, 0.0))

    result = certified_rowsubsample_eps_from_mu_bar(
        mu_bar, k, n, N, delta,
        prv_eps_error=prv_eps_error,
        prv_delta_error=prv_delta_error,
        pld_discretization=pld_discretization,
    )
    result["mu_bar_sq"] = mu_bar_sq
    result["t_switch"] = t_switch
    return result


def certified_rowsubsample_eps_from_mu_bar(
    mu_bar,
    k,
    n,
    N,
    delta,
    prv_eps_error=0.01,
    prv_delta_error=None,
    pld_discretization=1e-5,
):
    """
    Core of compute_certified_rowsubsample_bound, taking the per-trajectory
    shift mu_bar directly instead of deriving it from a mu schedule.  Useful
    for post-hoc computation from saved privacy results, which store mu_bar
    (as reuse_minibatch_mu_bar in the privacy JSON).
    """
    try:
        from autodp.mechanism_zoo import GaussianMechanism
        from autodp.transformer_zoo import AmplificationBySampling, Composition
    except ImportError as e:
        raise ImportError(
            "certified_rowsubsample_eps_from_mu_bar requires autodp "
            "(pip install autodp)."
        ) from e

    mu_bar = float(mu_bar)
    if mu_bar <= 0.0:
        raise ValueError(f"mu_bar must be positive, got {mu_bar}.")

    sigma_eff = 1.0 / mu_bar
    prob = float(k) / float(n)
    if not 0.0 < prob <= 1.0:
        raise ValueError(f"Sampling rate k/n = {prob} must be in (0, 1].")

    poisson_sample = AmplificationBySampling(PoissonSampling=True)
    sample_wo_replacement = AmplificationBySampling(PoissonSampling=False)
    compose = Composition()

    def certified_eps(poisson, improved):
        gm = GaussianMechanism(sigma_eff, name="trajectory_GM")
        if poisson:
            subsampled = poisson_sample(gm, prob, improved_bound_flag=improved)
        else:
            # autodp's subsampling-without-replacement amplification requires
            # the replace-one neighboring notion on the base mechanism.
            gm.neighboring = "replace_one"
            subsampled = sample_wo_replacement(gm, prob, improved_bound_flag=improved)
        composed = compose([subsampled], [int(N)])
        return float(composed.get_approxDP(delta))

    eps_pld_lower = eps_pld_estimate = eps_pld_upper = None
    try:
        from prv_accountant import PRVAccountant, PoissonSubsampledGaussianMechanism
        accountant = PRVAccountant(
            prvs=PoissonSubsampledGaussianMechanism(
                sampling_probability=prob,
                noise_multiplier=sigma_eff,
            ),
            max_self_compositions=int(N),
            eps_error=prv_eps_error,
            delta_error=prv_delta_error if prv_delta_error is not None else delta / 1000.0,
        )
        lower, estimate, upper = accountant.compute_epsilon(
            delta=delta,
            num_self_compositions=int(N),
        )
        # Epsilon is non-negative, but the accountant's +/- eps_error slack can
        # push the lower end below zero in very private settings.
        eps_pld_lower = max(float(lower), 0.0)
        eps_pld_estimate = max(float(estimate), 0.0)
        eps_pld_upper = max(float(upper), 0.0)
    except ImportError:
        pass
    except (MemoryError, RuntimeError, ValueError) as e:
        print(f"WARNING: PRV accountant failed for sigma_eff={sigma_eff:.4g}, "
              f"prob={prob:.4g}, N={int(N)} ({type(e).__name__}: {e}); "
              f"falling back to the Google PLD upper bound.")

    def _google_eps(pessimistic):
        from dp_accounting.pld import privacy_loss_distribution as pld_lib
        return float(pld_lib.from_gaussian_mechanism(
            standard_deviation=sigma_eff,
            sensitivity=1.0,
            sampling_prob=prob,
            pessimistic_estimate=pessimistic,
            use_connect_dots=pessimistic,
            value_discretization_interval=pld_discretization,
        ).self_compose(int(N)).get_epsilon_for_delta(delta))

    try:
        eps_pld_google = _google_eps(pessimistic=True)
        eps_pld_google_lower = _google_eps(pessimistic=False)
    except ImportError:
        eps_pld_google = eps_pld_google_lower = None

    return {
        "prob": prob,
        "mu_bar": mu_bar,
        "sigma_eff": sigma_eff,
        "k": int(k),
        "n": int(n),
        "eps_rdp_poisson": certified_eps(poisson=True, improved=False),
        "eps_rdp_poisson_improved": certified_eps(poisson=True, improved=True),
        "eps_rdp_wor": certified_eps(poisson=False, improved=False),
        "eps_rdp_wor_improved": certified_eps(poisson=False, improved=True),
        # eps_pld_poisson keeps its old meaning (the value safe to report):
        # the certified PRV upper bound, or the Google value if PRV was
        # unavailable or failed.
        "eps_pld_poisson": eps_pld_upper if eps_pld_upper is not None else eps_pld_google,
        "eps_pld_poisson_lower": eps_pld_lower,
        "eps_pld_poisson_estimate": eps_pld_estimate,
        "eps_pld_poisson_upper": eps_pld_upper,
        # Google bracket: eps_pld_poisson_google is the pessimistic (upper)
        # value, _google_lower the optimistic one; true epsilon lies between.
        "eps_pld_poisson_google": eps_pld_google,
        "eps_pld_poisson_google_lower": eps_pld_google_lower,
        "delta": delta,
        "N": N,
    }


# Graphing functions for debugging purposes
def alpha_fdp_graphs(config, privacydir, mu_tilde, gamma, eps):
    alphas = np.linspace(1e-10, 1 - 1e-10, 100)
    def objective_func(alpha, gamma, mu, eps):
        f_val = f_dp(alpha, gamma, mu)
        return (1 + (-np.exp(eps) * alpha - f_val))
    
    def G_mu_stuff(alpha, mu, gamma):
        return G_mu(alpha + gamma, mu) - gamma    
    
    # Graphs for debugging purposes

    f_dp_vals = [objective_func(alpha, gamma, mu_tilde, eps) for alpha in alphas]
    plt.plot(alphas, f_dp_vals)
    plt.xlabel("α")
    plt.ylabel("objective function value")
    plt.title("Objective Tradeoff")
    plt.grid(True)
    plt.savefig(f"{privacydir}/{config.experiment_name}_f_dp_tradeoff_plot.png")
    print(f"Saved f-DP tradeoff plot to {privacydir}/{config.experiment_name}_f_dp_tradeoff_plot.png")
    plt.close()

    exponent_vals = [-np.exp(eps) * alpha for alpha in alphas]
    plt.plot(alphas, exponent_vals, label='-exp(ε) * α')
    plt.xlabel("α")
    plt.ylabel("-exp(ε) * α")
    plt.title("Exponential Tradeoff")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{privacydir}/{config.experiment_name}_exponential_tradeoff_plot.png")
    print(f"Saved exponential tradeoff plot to {privacydir}/{config.experiment_name}_exponential_tradeoff_plot.png")
    plt.close()

    g_mu_vals = [G_mu_stuff(alpha, mu_tilde, gamma) for alpha in alphas]
    plt.plot(alphas, g_mu_vals)
    plt.xlabel("α")
    plt.ylabel("g_mu_vals function value")
    plt.title("g_mu_vals Tradeoff")
    plt.grid(True)
    plt.savefig(f"{privacydir}/{config.experiment_name}g_mu_vals_tradeoff_plot.png")
    print(f"Saved f-DP tradeoff plot to {privacydir}/{config.experiment_name}g_mu_vals_tradeoff_plot.png")
    plt.close()

    just_g_mu_vals = [G_mu(alpha, 1) for alpha in alphas]
    plt.plot(alphas, just_g_mu_vals)
    plt.xlabel("α")
    plt.ylabel("G_mu function value")
    plt.title("G_mu Tradeoff")
    plt.grid(True)
    plt.savefig(f"{privacydir}/{config.experiment_name}_just_g_mu_tradeoff_plot.png")
    print(f"Saved G_mu tradeoff plot to {privacydir}/{config.experiment_name}_just_g_mu_tradeoff_plot.png")
    plt.close()

def epsilon_delta_varying_sigma_min(config, privacydir, noise_scheduler, sigma_values, p, ls_cycle, eps_vals, sigma_min_grid):
    for sigma_min in tqdm(sigma_min_grid, desc="Sigma min grid search for GDP with subsampling"):
        sigma_values_i = noise_scheduler.get_sigmas_karras(
            config.num_timesteps, 
            sigma_max=config.sigma_max, 
            sigma_min=sigma_min,
            rho=config.rho,
            device=config.device
        )
        sigma_diffs_i = torch.tensor(noise_scheduler.get_sigma_diffs(sigma_values))
        mus_i = calculate_mu_value(sigma_values_i, sigma_diffs_i, beta = config.beta)
        kls_i = calculate_kl_divergence(mus_i, p)
        kappa2s_i = calculate_kappa2(mus_i, p)
        kappa3s_i = calculate_kappa3(mus_i, kls_i, p)
        mu_tilde_i = get_mu_tilde_privacy(kls_i, kappa2s_i)
        gamma_i = get_gamma_privacy(kls_i, kappa2s_i, kappa3s_i)
        delta_vals_i = [delta_eps(e, gamma_i.item(), mu_tilde_i.item()) for e in eps_vals]
        ls = next(ls_cycle)
        plt.plot(eps_vals, delta_vals_i, 
                 label=f"sigma_min={sigma_min}, μ̃={mu_tilde_i:.2f}, γ={gamma_i:.4f}", 
                 alpha = 0.7,
                 linestyle = ls)
        print(f"Explored sigma_min={sigma_min}, μ̃={mu_tilde_i:.2f}, γ={gamma_i:.4f}")


    # Visualize epsilon-delta tradeoff without subsampling
    plt.xlabel("ε")
    plt.ylabel("δ(ε)")
    plt.axhline(y=10**(-5), color="red", linestyle=":", linewidth=2)
    plt.yscale('log')
    plt.ylim(1e-10, 1.2)
    plt.title(f"G-DP to (ε, δ)-DP with subsampling with β {config.beta}")
    plt.legend(loc ='upper right')
    plt.grid(True)
    plt.savefig(f"{privacydir}/{config.experiment_name}_epsilon_delta_subsample_sigma_min.png")
    print(f"Saved epsilon-delta no subsampling plot to {privacydir}/{config.experiment_name}_epsilon_delta_subsample_sigma_min.png")
    plt.close()

def epsilon_delta_varying_stopping_points(config, privacydir, sigma_values, p, kls, kappa2s, kappa3s, ls_cycle, eps_vals):
    proportion_indices = np.linspace(0, config.num_timesteps - 1, config.num_intervals + 1, dtype=int)
    for i in range(1, len(proportion_indices)):
        idx = proportion_indices[i]
        sigma_t = sigma_values[idx]
        kls_interval = kls[:idx]
        kappa2s_interval = kappa2s[:idx]
        kappa3s_interval = kappa3s[:idx]
        proportion = idx / (config.num_timesteps - 1)
        mu_tilde_interval = get_mu_tilde_privacy(kls_interval, kappa2s_interval)
        gamma_interval = get_gamma_privacy(kls_interval, kappa2s_interval, kappa3s_interval)
        delta_vals_i = [delta_eps(e, gamma_interval.item(), mu_tilde_interval.item()) for e in eps_vals]
        ls = next(ls_cycle)

        plt.plot(eps_vals, delta_vals_i, 
                 label=rf"{proportion:.2f} % to σ={sigma_t:.4f}, $\tilde{{μ}}$={mu_tilde_interval:.5f}, $\gamma$={gamma_interval:.5f}", 
                 alpha = 0.7,
                 linestyle = ls)

    plt.xlabel("ε")
    plt.axhline(y=10**(-5), color="red", linestyle=":", linewidth=2)
    plt.ylabel("δ(ε)")
    plt.title(f"Berry Esseen f-DP to (ε, δ)-DP with subsampling at probability {p}, β {config.beta}")
    plt.legend(loc ='upper right', fontsize='x-small')
    # plt.yscale('log')
    plt.ylim(1e-10, 1.1)
    plt.grid(True)
    plt.savefig(f"{privacydir}/{config.experiment_name}_epsilon_delta_subsample_interval.png")
    print(f"Saved epsilon-delta plot to {privacydir}/{config.experiment_name}_epsilon_delta_subsample_interval.png")
    plt.close()

def perform_props_grid_search(theorem_5_2, config, privacydir, sigma_values, p, mus, kls, kappa2s, delta):
    c_grid = p * math.sqrt(config.num_samples)
    grid_rows = []
    grid_rows.append(f"{'prop':>8}  {'idx':>6}  {'sigma_step':>14}  {'cor_5_4_mu':>12}  {'thm_3_5_mu':>12}  {'eps_cor_5_4':>13}  {'eps_thm_3_5':>13}  {'eps_full_batch':>13} {'mu_full_batch':>13} {'eps_full_edgeworth':>13}")
    grid_rows.append("-" * 130)
    for prop_val in config.props_grid:
        idx_g = int(config.num_timesteps * prop_val)
        idx_g = min(idx_g, config.num_timesteps - 2)  # guard against out-of-bounds
        mu_step_g = mus[idx_g + 1].item()
        sigma = sigma_values[idx_g + 1].item()
        cor54_mu = theorem_5_2(c_grid, mu_step_g)
        eps_cor54 = find_eps_for_gdp_delta(delta, cor54_mu)
        final_kl_g = kls[idx_g + 1]
        final_kappa2_g = kappa2s[idx_g + 1]
        thm35_mu = calculate_asymptotic_mu_multiple(final_kl_g, final_kappa2_g, num_samples=config.num_samples).item()
        eps_thm35 = find_eps_for_gdp_delta(delta, thm35_mu)
        full_dataset_privacy = compute_full_batch_multistep_gdp(mus, config.num_samples, idx_g + 1, delta)
        mu_full_batch = full_dataset_privacy["mu_N"]
        eps_full_batch = full_dataset_privacy["eps_full_batch"]
        
        full_batch_results = compute_edgeworth_subsampled_gaussian(
            mus,
            1.0,  # no subsampling for full batch comparison
            config.num_samples,
            idx_g + 1,
            delta,
            eps_max=config.epsilon_max
        )
        full_batch_edgeworth_eps = full_batch_results["eps_edgeworth_sym"]
        
        grid_rows.append(
            f"{prop_val:>8.2f}  {idx_g:>6d}  {sigma:>14.6f}  {cor54_mu:>12.6f}  {thm35_mu:>12.6f}  {eps_cor54:>13.6f}  {eps_thm35:>13.6f}  {eps_full_batch:>13.6f} {mu_full_batch:>13.6f} {full_batch_edgeworth_eps:>13.6f}"
        )

    grid_txt_path = f"{privacydir}/{config.experiment_name}_props_grid_search.txt"
    with open(grid_txt_path, "w") as f:
        f.write(f"Props grid search results\n")
        f.write(f"  num_samples={config.num_samples}, p={p}, c={c_grid:.6f}, delta={delta}, timesteps={config.num_timesteps}\n\n")
        f.write("\n".join(grid_rows) + "\n")
    print(f"Props grid search results saved to {grid_txt_path}")


def perform_full_batch_props_grid_search(config, privacydir, sigma_values, mus, delta):
    """Same grid search as perform_props_grid_search but restricted to the full-batch
    privacy accountants (no subsampling).  Produces mu_N and eps from
    compute_full_batch_multistep_gdp and the Edgeworth accountant at p=1."""
    grid_rows = []
    grid_rows.append(f"{'prop':>8}  {'idx':>6}  {'sigma_step':>14}  {'mu_full_batch':>14}  {'eps_full_batch':>14}  {'eps_edgeworth':>14}  {'eps_edgeworth_PQ':>14}  {'eps_edgeworth_QP':>14}")
    grid_rows.append("-" * 130)
    for prop_val in config.props_grid:
        idx_g = int(config.num_timesteps * prop_val)
        idx_g = min(idx_g, config.num_timesteps - 2)
        sigma = sigma_values[idx_g + 1].item()

        full_batch = compute_full_batch_multistep_gdp(mus, config.num_samples, idx_g + 1, delta)
        mu_full_batch = full_batch["mu_N"]
        eps_full_batch = full_batch["eps_full_batch"]

        edgeworth_result = compute_edgeworth_subsampled_gaussian(
            mus,
            1.0,
            config.num_samples,
            idx_g + 1,
            delta,
            eps_max=config.epsilon_max,
        )
        eps_edgeworth = edgeworth_result["eps_edgeworth_sym"]
        full_batch_edgeworth_P_Q_eps = edgeworth_result["eps_edgeworth_PQ"]
        full_batch_edgeworth_Q_P_eps = edgeworth_result["eps_edgeworth_QP"]

        grid_rows.append(
            f"{prop_val:>8.2f}  {idx_g:>6d}  {sigma:>14.6f}  {mu_full_batch:>14.6f}  {eps_full_batch:>14.6f}  {eps_edgeworth:>14.6f} {full_batch_edgeworth_P_Q_eps:>14.6f} {full_batch_edgeworth_Q_P_eps:>14.6f}"
        )
        

    grid_txt_path = f"{privacydir}/{config.experiment_name}_full_batch_props_grid_search.txt"
    with open(grid_txt_path, "w") as f:
        f.write(f"Full-batch props grid search results\n")
        f.write(f"  num_samples={config.num_samples}, delta={delta}, timesteps={config.num_timesteps}\n\n")
        f.write("\n".join(grid_rows) + "\n")
    print(f"Full-batch props grid search results saved to {grid_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size is m in our write up")
    parser.add_argument("--rho", type=float, default=7.0, help="Rho value for Karras schedule")
    parser.add_argument("--p", type=float, default=0.01, help="Probability value for Berry-Essen privacy calculations")
    parser.add_argument("--verbosity", type=int, default=3, help="Verbosity level: 0=silent, 1=errors, 2=warnings, 3=info, 4=debug, 5=trace")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Target epsilon value")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta value for GDP sensitivity")
    parser.add_argument("--num_intervals", type=int, default=10, help="Number of intervals in the diffusion process to analyze")
    parser.add_argument("--epsilon_max", type = float, default = 500.0, help="Max epsilon value to plot to")
    parser.add_argument("--sigma_min_grid", type=float, nargs='+', default=[0.002, 0.01, 0.1, 0.5, 1, 5], help="Grid of sigma_min values to search over")
    parser.add_argument("--seed", type=int, default=67, help="Random seed for sampling")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples for hybrid privacy calculation")
    parser.add_argument("--dataset_size", type=int, default=50000, help="Size of the training dataset for subsampling calculations")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta for (eps, delta)-DP conversion; defaults to 1/dataset_size")
    parser.add_argument("--prop", type = float, default = 0.6, help = "Proportion of generation before switching")
    parser.add_argument("--props_grid", type=float, nargs='+', default=[0.6, 0.63, 0.65, 0.67, 0.7, 0.73], help="Grid of proportion values to search over for corollary_5_4 and asymptotic mu")
    parser.add_argument("--prv_eps_error", type=float, default=0.01,
                        help="Half-width of the certified PRV epsilon bracket (upper - lower = 2 * this). "
                             "Absolute, so pick it relative to the epsilon being reported: 0.01 is +/-17%% "
                             "at eps~0.06 but +/-0.1%% at eps~10. Cost scales as 1/prv_eps_error "
                             "(0.01 -> ~2s/0.3GB, 0.001 -> ~22s/2.4GB at N=50000)")
    parser.add_argument("--prv_delta_error", type=float, default=None,
                        help="Delta slack for the PRV accountant; defaults to delta/1000. Nearly free to tighten")
    parser.add_argument("--pld_discretization", type=float, default=1e-5,
                        help="Discretization for the Google PLD cross-check/fallback. Accuracy floors at ~1e-6 "
                             "and degrades below that, so keep in [1e-6, 1e-5]")
    config = parser.parse_args()
    
    privacydir = f"{config.outdir}/privacy"
    os.makedirs(privacydir, exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Noise Scheduling
    noise_scheduler = NoiseScheduler(sigma_min = config.sigma_min,
                                     sigma_max = config.sigma_max,
                                     num_timesteps = config.num_timesteps)
    
    sigma_values = noise_scheduler.get_sigmas_karras(
        config.num_timesteps, 
        sigma_max=config.sigma_max, 
        sigma_min=config.sigma_min,
        rho=config.rho,
        device=config.device
    )

    p = config.p

    # Set up logging
    logger = DPLogger(name="berry-esseen", log_to_file=True, verbosity=config.verbosity)

    sigma_diffs = torch.tensor(noise_scheduler.get_sigma_diffs(sigma_values))
    mus = calculate_mu_value(sigma_values, sigma_diffs, beta=config.beta)
    kls = calculate_kl_divergence(mus, p)
    kappa2s = calculate_kappa2(mus, p)
    kappa3s = calculate_kappa3(mus, kls, p)
    mu_tilde = get_mu_tilde_privacy(kls, kappa2s)
    gamma = get_gamma_privacy(kls, kappa2s, kappa3s)
    
    print("Mu values: ", mus)

    # GDP with subsampling for varying sigma_min
    sigma_min_grid = config.sigma_min_grid
    
    # Target delta for (eps, delta)-DP conversion; scales as 1/N with dataset size
    delta = config.delta if config.delta is not None else 1.0 / config.dataset_size
    print(f"Target delta: {delta:.3e} ({'from --delta' if config.delta is not None else f'default 1/dataset_size, N={config.dataset_size}'})")
    idx = int(config.num_timesteps * config.prop)
    
    
    one_kl = kls[idx + 1]
    one_kappa2 = kappa2s[idx + 1]
    asymp_mu_one_theorem_3_5 = calculate_asymptotic_mu_multiple(one_kl, one_kappa2, num_samples=config.num_samples).item()
    eps_at_delta_one_theorem_3_5 = find_eps_for_gdp_delta(delta, asymp_mu_one_theorem_3_5)

    print(f"Asymp-Mu Theorem 3.5 One: {asymp_mu_one_theorem_3_5}")
    print(f"Epsilon Theorem 3.5 One: {eps_at_delta_one_theorem_3_5}")

    # f-DP accounting for N samples across multiple empirical denoiser steps (PDF Sections 3-8)
    # t_switch = idx + 1 empirical steps (indices 0..idx inclusive in the denoising loop)
    print("\n--- compute_fdp_N_samples_multistep ---")
    multistep_result = compute_fdp_N_samples_multistep(
        mus=mus,
        p=p,
        N=config.num_samples,
        t_switch=idx + 1,
        delta=delta,
        edgeworth=True,
    )
    print(f"  mu^(N)          = {multistep_result['mu_N']:.6f}")
    print(f"  K^(N)           = {multistep_result['K_N']:.6f}")
    print(f"  s^(N)           = {multistep_result['s_N']:.6f}")
    print(f"  BE error (eta)  = {multistep_result['be_error']:.6e}")
    print(f"  eps (CLT)       = {multistep_result['eps_clt']:.6f}")
    print(f"  eps (rigorous)  = {multistep_result['eps_rigorous']}")
    print(f"  eps (Edgeworth) = {multistep_result['eps_edgeworth']}")
    print(f"  delta           = {multistep_result['delta']:.1e}")
    print(f"  N={multistep_result['N']}, t_switch={multistep_result['t_switch']}")

    # Full Edgeworth accountant (Sections 2-7 of the Edgeworth note): cumulants of L
    # under both P and Q, degree-2 Edgeworth CDFs, symmetric directional (eps, delta).
    print("\n--- compute_edgeworth_subsampled_gaussian ---")
    edgeworth_result = compute_edgeworth_subsampled_gaussian(
        mus=mus,
        p=p,
        N=config.num_samples,
        t_switch=idx + 1,
        delta=delta,
    )
    print(f"  M_P={edgeworth_result['M_P']:.6f}, V_P={edgeworth_result['V_P']:.6f}")
    print(f"  M_Q={edgeworth_result['M_Q']:.6f}, V_Q={edgeworth_result['V_Q']:.6f}")
    print(f"  rho3_P={edgeworth_result['rho3_P']:.4e}, rho4_P={edgeworth_result['rho4_P']:.4e}")
    print(f"  rho3_Q={edgeworth_result['rho3_Q']:.4e}, rho4_Q={edgeworth_result['rho4_Q']:.4e}")
    print(f"  eps (P->Q)      = {edgeworth_result['eps_edgeworth_PQ']}")
    print(f"  eps (Q->P)      = {edgeworth_result['eps_edgeworth_QP']}")
    print(f"  eps (symmetric) = {edgeworth_result['eps_edgeworth_sym']}")
    print(f"  eps valid       = {edgeworth_result['edgeworth_valid']}")
    
    # Compute the privacy when we reuse a minibatch across all T steps
    edgeworth_reuse_minibatch = compute_edgeworth_reuse_minibatch(
        mus=mus,
        k=int(p * config.dataset_size),  # minibatch size k = p * dataset_size
        n=config.dataset_size,  # dataset size n for subsampling calculations
        N=config.num_samples,
        t_switch=idx + 1,
        delta=delta,
    )
    print("\n--- compute_edgeworth_reuse_minibatch ---")
    print(f"  q (inclusion prob)  = {edgeworth_reuse_minibatch['q']:.6f}")
    print(f"  mu_bar              = {edgeworth_reuse_minibatch['mu_bar']:.6f}")
    print(f"  mu_bar_sq           = {edgeworth_reuse_minibatch['mu_bar_sq']:.6f}")
    print(f"  rho3_P={edgeworth_reuse_minibatch['rho3_P']:.4e}, rho4_P={edgeworth_reuse_minibatch['rho4_P']:.4e}")
    print(f"  rho3_Q={edgeworth_reuse_minibatch['rho3_Q']:.4e}, rho4_Q={edgeworth_reuse_minibatch['rho4_Q']:.4e}")
    print(f"  eps (P->Q)          = {edgeworth_reuse_minibatch['eps_edgeworth_PQ']}")
    print(f"  eps (Q->P)          = {edgeworth_reuse_minibatch['eps_edgeworth_QP']}")
    print(f"  eps (symmetric)     = {edgeworth_reuse_minibatch['eps_edgeworth_sym']}")
    print(f"  mu_eff_reuse (CLT)  = {edgeworth_reuse_minibatch['mu_eff_reuse']:.6f}")
    print(f"  eps_clt_reuse       = {edgeworth_reuse_minibatch['eps_clt_reuse']}")
    print(f"  eps valid           = {edgeworth_reuse_minibatch['edgeworth_valid']}")

    # Certified RDP/PLD bounds (autodp / dp-accounting) for the same
    # subsample-once-and-reuse setting, with row subsampling (without
    # replacement) and Poisson sampling.
    try:
        certified_rowsubsample = compute_certified_rowsubsample_bound(
            mus=mus,
            k=int(p * config.dataset_size),
            n=config.dataset_size,
            N=config.num_samples,
            t_switch=idx + 1,
            delta=delta,
            prv_eps_error=config.prv_eps_error,
            prv_delta_error=config.prv_delta_error,
            pld_discretization=config.pld_discretization,
        )
    except ImportError as e:
        print(f"\nWARNING: skipping certified row-subsample bounds: {e}")
        certified_rowsubsample = None

    if certified_rowsubsample is not None:
        print("\n--- compute_certified_rowsubsample_bound (autodp, certified) ---")
        print(f"  prob (k/n)              = {certified_rowsubsample['prob']:.6f}")
        print(f"  mu_bar                  = {certified_rowsubsample['mu_bar']:.6f}")
        print(f"  sigma_eff               = {certified_rowsubsample['sigma_eff']:.6f}")
        print(f"  eps RDP Poisson         = {certified_rowsubsample['eps_rdp_poisson']}")
        print(f"  eps RDP Poisson (impr.) = {certified_rowsubsample['eps_rdp_poisson_improved']}")
        print(f"  eps RDP WOR             = {certified_rowsubsample['eps_rdp_wor']}")
        print(f"  eps RDP WOR (impr.)     = {certified_rowsubsample['eps_rdp_wor_improved']}")
        print(f"  eps PLD Poisson (upper) = {certified_rowsubsample['eps_pld_poisson_upper']}")
        print(f"  eps PLD Poisson (est.)  = {certified_rowsubsample['eps_pld_poisson_estimate']}")
        print(f"  eps PLD Poisson (lower) = {certified_rowsubsample['eps_pld_poisson_lower']}")
        print(f"  eps PLD Poisson (google)= {certified_rowsubsample['eps_pld_poisson_google']}")
        print(f"  eps PLD Poisson (goog lo)= {certified_rowsubsample['eps_pld_poisson_google_lower']}")


    # Full Dataset Privacy
    full_dataset_privacy = compute_full_batch_multistep_gdp(mus, config.num_samples, idx + 1, delta)
    print(f"Full dataset privacy (mu): {full_dataset_privacy['mu_N']:.6f}")
    print(f"Full dataset privacy (eps): {full_dataset_privacy['eps_full_batch']:.6f}")

    # Save privacy results in pytorch
    privacy_results = {
        "mu_tilde": mu_tilde,
        "gamma": gamma,
        "kls": kls,
        "kappa2s": kappa2s,
        "kappa3s": kappa3s,
        "sigma_values": sigma_values,
        "sigma_diffs": sigma_diffs,
        "asymp_mu_one": asymp_mu_one_theorem_3_5,
        "eps_one_step": eps_at_delta_one_theorem_3_5,
        "delta": delta,
        "multistep_mu_N": multistep_result["mu_N"],
        "multistep_eps_clt": multistep_result["eps_clt"],
        "multistep_eps_rigorous": multistep_result["eps_rigorous"],
        "multistep_eps_edgeworth": multistep_result["eps_edgeworth"],
        "multistep_be_error": multistep_result["be_error"],
        "edgeworth_M_P": edgeworth_result["M_P"],
        "edgeworth_V_P": edgeworth_result["V_P"],
        "edgeworth_K3_P": edgeworth_result["K3_P"],
        "edgeworth_K4_P": edgeworth_result["K4_P"],
        "edgeworth_M_Q": edgeworth_result["M_Q"],
        "edgeworth_V_Q": edgeworth_result["V_Q"],
        "edgeworth_K3_Q": edgeworth_result["K3_Q"],
        "edgeworth_K4_Q": edgeworth_result["K4_Q"],
        "edgeworth_rho3_P": edgeworth_result["rho3_P"],
        "edgeworth_rho4_P": edgeworth_result["rho4_P"],
        "edgeworth_rho3_Q": edgeworth_result["rho3_Q"],
        "edgeworth_rho4_Q": edgeworth_result["rho4_Q"],
        "edgeworth_valid": edgeworth_result["edgeworth_valid"],
        "edgeworth_eps_PQ": edgeworth_result["eps_edgeworth_PQ"],
        "edgeworth_eps_QP": edgeworth_result["eps_edgeworth_QP"],
        "edgeworth_eps_sym": edgeworth_result["eps_edgeworth_sym"],
        "full_dataset_mu": full_dataset_privacy["mu_N"],
        "full_dataset_eps": full_dataset_privacy["eps_full_batch"],
        "q": edgeworth_reuse_minibatch["q"],
        "reuse_minibatch_mu_bar": edgeworth_reuse_minibatch["mu_bar"],
        "reuse_minibatch_mu_bar_sq": edgeworth_reuse_minibatch["mu_bar_sq"],
        "reuse_minibatch_rho3_P": edgeworth_reuse_minibatch["rho3_P"],
        "reuse_minibatch_rho4_P": edgeworth_reuse_minibatch["rho4_P"],
        "reuse_minibatch_rho3_Q": edgeworth_reuse_minibatch["rho3_Q"],
        "reuse_minibatch_rho4_Q": edgeworth_reuse_minibatch["rho4_Q"],
        "reuse_minibatch_valid": edgeworth_reuse_minibatch["edgeworth_valid"],
        "reuse_minibatch_eps_PQ": edgeworth_reuse_minibatch["eps_edgeworth_PQ"],
        "reuse_minibatch_eps_QP": edgeworth_reuse_minibatch["eps_edgeworth_QP"],
        "reuse_minibatch_eps_sym": edgeworth_reuse_minibatch["eps_edgeworth_sym"],
        "reuse_minibatch_mu_eff_reuse": edgeworth_reuse_minibatch["mu_eff_reuse"],
        "reuse_minibatch_eps_clt_reuse": edgeworth_reuse_minibatch["eps_clt_reuse"],
    }
    if certified_rowsubsample is not None:
        privacy_results.update({
            "certified_rowsubsample_prob": certified_rowsubsample["prob"],
            "certified_rowsubsample_mu_bar": certified_rowsubsample["mu_bar"],
            "certified_rowsubsample_sigma_eff": certified_rowsubsample["sigma_eff"],
            "certified_eps_rdp_poisson": certified_rowsubsample["eps_rdp_poisson"],
            "certified_eps_rdp_poisson_improved": certified_rowsubsample["eps_rdp_poisson_improved"],
            "certified_eps_rdp_wor": certified_rowsubsample["eps_rdp_wor"],
            "certified_eps_rdp_wor_improved": certified_rowsubsample["eps_rdp_wor_improved"],
            "certified_eps_pld_poisson": certified_rowsubsample["eps_pld_poisson"],
            "certified_eps_pld_poisson_lower": certified_rowsubsample["eps_pld_poisson_lower"],
            "certified_eps_pld_poisson_estimate": certified_rowsubsample["eps_pld_poisson_estimate"],
            "certified_eps_pld_poisson_upper": certified_rowsubsample["eps_pld_poisson_upper"],
            "certified_eps_pld_poisson_google": certified_rowsubsample["eps_pld_poisson_google"],
            "certified_eps_pld_poisson_google_lower": certified_rowsubsample["eps_pld_poisson_google_lower"],
        })
    
    linestyles = ['-', '--', '-.', ':']
    ls_cycle = itertools.cycle(linestyles)
    epsilon_delta_varying_stopping_points(config, privacydir, sigma_values, p, kls, kappa2s, kappa3s, ls_cycle=ls_cycle, eps_vals=np.linspace(1e-6, 20, 100))

    torch.save(privacy_results, f"{privacydir}/{config.experiment_name}_privacy.pt")
    print(f"Privacy results saved to {privacydir}/{config.experiment_name}_privacy.pt")
    
    privacy_results_json = {k: v.item() if torch.is_tensor(v) and v.numel() == 1 else v.tolist() if torch.is_tensor(v) else v for k, v in privacy_results.items()}
    with open(f"{privacydir}/{config.experiment_name}_privacy.json", "w") as f:
        json.dump(privacy_results_json, f, indent=4)
    print(f"Privacy results saved to {privacydir}/{config.experiment_name}_privacy.json")
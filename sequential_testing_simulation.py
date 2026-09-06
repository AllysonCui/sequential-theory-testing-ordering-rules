"""
Sequential Theory Testing Under Effectuation
Replication code: analytical checks, worked example, Monte Carlo, tables, figures.

Notation follows the paper:
    omega_i : prior that theory i is correct
    V_i     : payoff if theory i is correct and the latent state is created
    x_i     : effectuation effort
    pi_i    : V_i - x_i
    c_i     : cost of one test
    G_i     : false positive rate, Pr(H | theory false)
    N_i     : false negative rate, Pr(L | theory true)

    a_i     = omega_i (1 - N_i)          true-positive mass
    b_i     = (1 - omega_i) G_i          false-positive mass
    P_i     = a_i + b_i                  Pr(good signal)
    Phi_i   = a_i pi_i - b_i x_i         unconditional commitment payoff
    W_i     = Phi_i - c_i                value of testing theory i in isolation
    I_i     = W_i / P_i                  ordering index

Run:  python3 ordering_simulation.py
Outputs: table1.tex, table2.tex, table3.tex,
         fig_boundary.pdf, fig_design.pdf, fig_cost.pdf

Every routine reseeds, so results do not depend on the order in which the
routines are called.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure style matched to the paper: serif text, STIX math, thin rules.
PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.1,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

SEED = 20260904
RNG = np.random.default_rng(SEED)


def _reseed(k):
    global RNG
    RNG = np.random.default_rng(SEED + k)


# ----------------------------------------------------------------------
# 1. Core objects
# ----------------------------------------------------------------------


def masses(omega, N, G):
    a = omega * (1.0 - N)
    b = (1.0 - omega) * G
    return a, b, a + b


def phi(omega, pi, x, N, G):
    a, b, _ = masses(omega, N, G)
    return a * pi - b * x


def index(omega, pi, x, c, N, G):
    """I_i = (Phi_i - c_i)/P_i = omega_i^H V_i - x_i - c_i/P_i."""
    a, b, P = masses(omega, N, G)
    return (a * pi - b * x - c) / P


def payoff(order, om, pi, x, c, N, G):
    """Expected payoff of testing theories in the given order (list of indices)."""
    total, survive = 0.0, 1.0
    for i in order:
        _, _, P = masses(om[i], N[i], G[i])
        total += survive * (phi(om[i], pi[i], x[i], N[i], G[i]) - c[i])
        survive *= 1.0 - P
    return total


def delta_direct(om, pi, x, c, N, G):
    """Pi^{SZ} - Pi^{ZS} from the definition."""
    return payoff([0, 1], om, pi, x, c, N, G) - payoff([1, 0], om, pi, x, c, N, G)


def delta_channels(om, pi, x, c, N, G):
    """Four-channel decomposition, equation (13) in the paper."""
    aS, bS, PS = masses(om[0], N[0], G[0])
    aZ, bZ, PZ = masses(om[1], N[1], G[1])
    return (
        aS * aZ * (pi[0] - pi[1]),                      # net payoff
        aS * c[1] - aZ * c[0],                          # cost saving
        bZ * (aS * pi[0] - c[0] + PS * x[1]),           # false positive on Z
        -bS * (aZ * pi[1] - c[1] + PZ * x[0]),          # false positive on S
    )


def delta_falsification(om, V, x, c, f):
    """Closed form under N=0, G=1-f, allowing x_S != x_Z and c_S != c_Z
    (Proposition 3):

        Delta = f * Delta_P + (1-f) * [Delta_0 + f (1-om_S)(1-om_Z)(x_S - x_Z)]

    Delta_P : perfect-test criterion (Proposition 1)
    Delta_0 : difference in the value of committing on the prior, net of one
              test cost, which is what remains when the test has no power.
    """
    pi = V - x
    delta_P = om[0] * om[1] * (pi[0] - pi[1]) + om[0] * c[1] - om[1] * c[0]
    delta_0 = (om[0] * V[0] - x[0] - c[0]) - (om[1] * V[1] - x[1] - c[1])
    tilt = f * (1.0 - om[0]) * (1.0 - om[1]) * (x[0] - x[1])
    return f * delta_P + (1.0 - f) * (delta_0 + tilt)


# ----------------------------------------------------------------------
# 2. Analytical checks
# ----------------------------------------------------------------------


def check_identities(n=20_000):
    _reseed(1)
    om = RNG.uniform(0.05, 0.95, (n, 2))
    V = RNG.uniform(1.0, 10.0, (n, 2))
    x = V * RNG.uniform(0.1, 0.9, (n, 2))
    pi = V - x
    c = RNG.uniform(0.0, 1.0, (n, 2))
    N = RNG.uniform(0.0, 0.4, (n, 2))
    G = RNG.uniform(0.0, 0.6, (n, 2))

    d_direct = np.array(
        [delta_direct(om[k], pi[k], x[k], c[k], N[k], G[k]) for k in range(n)]
    )
    d_chan = np.array(
        [sum(delta_channels(om[k], pi[k], x[k], c[k], N[k], G[k])) for k in range(n)]
    )

    IS = index(om[:, 0], pi[:, 0], x[:, 0], c[:, 0], N[:, 0], G[:, 0])
    IZ = index(om[:, 1], pi[:, 1], x[:, 1], c[:, 1], N[:, 1], G[:, 1])
    aS, bS, PS = masses(om[:, 0], N[:, 0], G[:, 0])
    err_index = np.max(np.abs(IS - ((aS / PS) * V[:, 0] - x[:, 0] - c[:, 0] / PS)))

    # falsification-only closed form, asymmetric effort and cost
    f = RNG.uniform(0.05, 1.0, n)
    err_fals = 0.0
    for k in range(n):
        Nk, Gk = np.zeros(2), np.full(2, 1.0 - f[k])
        err_fals = max(
            err_fals,
            abs(
                delta_direct(om[k], pi[k], x[k], c[k], Nk, Gk)
                - delta_falsification(om[k], V[k], x[k], c[k], f[k])
            ),
        )

    print("--- analytical checks -------------------------------------")
    print(f"max |Delta_direct - Delta_channels|       : {np.max(np.abs(d_direct-d_chan)):.2e}")
    print(f"max |I_i - (omega^H V - x - c/P)|         : {err_index:.2e}")
    print(f"sign(I_S - I_Z) == sign(Delta), share     : {np.mean(np.sign(IS-IZ)==np.sign(d_direct)):.4f}")
    print(f"max |Delta - falsification closed form|   : {err_fals:.2e}")
    print()


# ----------------------------------------------------------------------
# 3. Experimentation zone (Assumption 1)
# ----------------------------------------------------------------------


def in_zone(om, V, x, c, N, G):
    """
      (i)   no commitment without testing : omega V < x
      (ii)  commitment after H            : omega^H V > x
      (iii) no commitment after L         : omega^L V < x
      (iv)  testing is worthwhile         : W = Phi - c > 0
    Vectorised over draws.
    """
    a, b, P = masses(om, N, G)
    denomL = om * N + (1.0 - om) * (1.0 - G)
    omL = np.where(denomL > 1e-12, om * N / np.maximum(denomL, 1e-12), 0.0)
    return (
        (om * V < x)
        & ((a / np.maximum(P, 1e-12)) * V > x)
        & (omL * V < x)
        & (index(om, V - x, x, c, N, G) > 0)
        & (P > 1e-6)
        & (denomL > 1e-6)
    )


# ----------------------------------------------------------------------
# 4. Worked example (Section 4.1)
# ----------------------------------------------------------------------


def worked_example():
    om = np.array([0.20, 0.60])
    V = np.array([12.5, 4.0])
    x = np.array([10.5, 3.0])
    c = np.array([0.35, 0.35])
    N, G = np.zeros(2), np.zeros(2)
    pi = V - x

    rows = []
    for j, nm in enumerate(["S", "Z"]):
        rows.append(
            dict(
                theory=nm,
                omega=om[j],
                V=V[j],
                x=x[j],
                pi=pi[j],
                omegaV=om[j] * V[j],
                Phi=phi(om[j], pi[j], x[j], N[j], G[j]),
                W=phi(om[j], pi[j], x[j], N[j], G[j]) - c[j],
                I=index(om[j], pi[j], x[j], c[j], N[j], G[j]),
                zone=bool(
                    in_zone(*(np.array([v]) for v in (om[j], V[j], x[j], c[j], N[j], G[j])))[0]
                ),
            )
        )
    df = pd.DataFrame(rows)
    P_SZ = payoff([0, 1], om, pi, x, c, N, G)
    P_ZS = payoff([1, 0], om, pi, x, c, N, G)
    print("--- worked example (perfect tests, c = 0.35) --------------")
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(f"Pi^SZ = {P_SZ:.4f}   Pi^ZS = {P_ZS:.4f}")
    print(f"loss from testing S first = {100*(P_ZS-P_SZ)/P_ZS:.2f}% of the value under the index order")
    print()
    return df


# ----------------------------------------------------------------------
# 5. Monte Carlo machinery
# ----------------------------------------------------------------------


def draw_panel(n, c_level, N_lo, N_hi, G_lo, G_hi, scheme="baseline"):
    if scheme == "correlated priors":
        base = RNG.uniform(0.05, 0.95, (n, 1))
        om = np.clip(base + RNG.normal(0, 0.10, (n, 2)), 0.02, 0.98)
    elif scheme == "beta priors":
        om = np.clip(RNG.beta(2.0, 5.0, (n, 2)), 0.02, 0.98)
    else:
        om = RNG.uniform(0.05, 0.95, (n, 2))

    if scheme == "lognormal payoffs":
        V = np.exp(RNG.normal(1.0, 0.7, (n, 2)))
    else:
        V = RNG.uniform(1.0, 10.0, (n, 2))

    lam = RNG.uniform(0.5, 0.9, (n, 2)) if scheme == "effort-heavy" \
        else RNG.uniform(0.1, 0.9, (n, 2))
    x = lam * V
    pi = V - x

    cbar = c_level * pi.mean(axis=1, keepdims=True)
    c = cbar * RNG.uniform(0.3, 1.7, (n, 2)) if scheme == "asymmetric costs" \
        else np.repeat(cbar, 2, axis=1)

    N = RNG.uniform(N_lo, N_hi, (n, 2))
    G = RNG.uniform(G_lo, G_hi, (n, 2))
    return om, V, x, c, N, G


def run_cell(n, c_level, N_lo, N_hi, G_lo, G_hi, scheme="baseline",
             target=25_000, max_batches=400, min_draws=500):
    """Accept-reject sampling until `target` in-zone draws are collected."""
    pool, attempts = [], 0
    while sum(a.shape[0] for a in pool) < target and attempts < max_batches:
        om, V, x, c, N, G = draw_panel(n, c_level, N_lo, N_hi, G_lo, G_hi, scheme)
        keep = in_zone(om[:, 0], V[:, 0], x[:, 0], c[:, 0], N[:, 0], G[:, 0]) & in_zone(
            om[:, 1], V[:, 1], x[:, 1], c[:, 1], N[:, 1], G[:, 1]
        )
        attempts += 1
        if keep.any():
            pool.append(np.stack([om, V, x, c, N, G], axis=1)[keep])
    if not pool or sum(a.shape[0] for a in pool) < min_draws:
        return None

    block = np.concatenate(pool, axis=0)
    share = block.shape[0] / (attempts * n)
    om, V, x, c, N, G = (block[:, j, :] for j in range(6))
    pi = V - x

    IS = index(om[:, 0], pi[:, 0], x[:, 0], c[:, 0], N[:, 0], G[:, 0])
    IZ = index(om[:, 1], pi[:, 1], x[:, 1], c[:, 1], N[:, 1], G[:, 1])
    S_first = IS > IZ

    PhiS = phi(om[:, 0], pi[:, 0], x[:, 0], N[:, 0], G[:, 0])
    PhiZ = phi(om[:, 1], pi[:, 1], x[:, 1], N[:, 1], G[:, 1])
    _, _, PS = masses(om[:, 0], N[:, 0], G[:, 0])
    _, _, PZ = masses(om[:, 1], N[:, 1], G[:, 1])
    Pi_SZ = (PhiS - c[:, 0]) + (1 - PS) * (PhiZ - c[:, 1])
    Pi_ZS = (PhiZ - c[:, 1]) + (1 - PZ) * (PhiS - c[:, 0])
    opt = np.maximum(Pi_SZ, Pi_ZS)

    hi_pi = pi[:, 0] > pi[:, 1]
    hi_omV = om[:, 0] * V[:, 0] > om[:, 1] * V[:, 1]
    naive_pi = np.where(hi_pi, Pi_SZ, Pi_ZS)
    naive_omV = np.where(hi_omV, Pi_SZ, Pi_ZS)
    rel_pi = (opt - naive_pi) / opt

    return {
        "draws": om.shape[0],
        "share_in_zone": share,
        "surprise_pi": np.mean(S_first != hi_pi),
        "surprise_omV": np.mean(S_first != hi_omV),
        "loss_pi": rel_pi.mean() * 100,
        "loss_pi_p90": np.percentile(rel_pi, 90) * 100,
        "loss_omV": ((opt - naive_omV) / opt).mean() * 100,
    }


REGIMES = {
    "Perfect ($G=N=0$)": dict(N_lo=0.0, N_hi=0.0, G_lo=0.0, G_hi=0.0),
    "Falsification-only, $f=0.8$": dict(N_lo=0.0, N_hi=0.0, G_lo=0.2, G_hi=0.2),
    "Falsification-only, $f=0.5$": dict(N_lo=0.0, N_hi=0.0, G_lo=0.5, G_hi=0.5),
    "General $(G,N)$": dict(N_lo=0.0, N_hi=0.4, G_lo=0.0, G_hi=0.5),
}


def table1(n=50_000):
    _reseed(2)
    c_levels = {"low ($c=0.05\\bar{\\pi}$)": 0.05, "high ($c=0.40\\bar{\\pi}$)": 0.40}
    rows = []
    for rname, rkw in REGIMES.items():
        for cname, clev in c_levels.items():
            res = run_cell(n, clev, **rkw)
            if res is None:
                print(f"  [{rname} | {cname}] experimentation zone empty")
                continue
            rows.append(dict(Regime=rname, Cost=cname, Zone=100 * res["share_in_zone"],
                             **{k: res[k] for k in
                                ("draws", "surprise_pi", "surprise_omV",
                                 "loss_pi", "loss_pi_p90", "loss_omV")}))
    df = pd.DataFrame(rows)
    df["surprise_pi"] *= 100
    df["surprise_omV"] *= 100
    print("--- Table 1 -----------------------------------------------")
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print()

    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Index order differs from} & \multicolumn{3}{c}{Value lost (\%)} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"Test regime & Test cost & Draws & $\pi$-rank & $\omega V$-rank & mean & p90 & $\omega V$ rule \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r.Regime} & {r.Cost} & {int(r.draws):,} & {r.surprise_pi:.1f}\\% & "
                     f"{r.surprise_omV:.1f}\\% & {r.loss_pi:.2f} & {r.loss_pi_p90:.2f} & "
                     f"{r.loss_omV:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("table1.tex", "w").write("\n".join(lines) + "\n")
    return df


def table3(n=50_000):
    """Robustness of the general-regime results to the sampling scheme."""
    _reseed(4)
    schemes = ["baseline", "beta priors", "correlated priors", "lognormal payoffs",
               "effort-heavy", "asymmetric costs"]
    rows = []
    for sc in schemes:
        res = run_cell(n, 0.20, scheme=sc, target=20_000, **REGIMES["General $(G,N)$"])
        rows.append(dict(scheme=sc, draws=res["draws"],
                         surprise_pi=100 * res["surprise_pi"],
                         surprise_omV=100 * res["surprise_omV"],
                         loss_pi=res["loss_pi"], loss_omV=res["loss_omV"]))
    df = pd.DataFrame(rows)
    print("--- Table 3: robustness -----------------------------------")
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print()
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Index order differs from} & \multicolumn{2}{c}{Value lost (\%)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Sampling scheme & Draws & $\pi$-rank & $\omega V$-rank & $\pi$ rule & $\omega V$ rule \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r.scheme.capitalize()} & {int(r.draws):,} & {r.surprise_pi:.1f}\\% & "
                     f"{r.surprise_omV:.1f}\\% & {r.loss_pi:.2f} & {r.loss_omV:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("table3.tex", "w").write("\n".join(lines) + "\n")
    return df


# ----------------------------------------------------------------------
# 6. Joint ordering and test design
# ----------------------------------------------------------------------


def table2(n=60_000, A=0.5, Bslope=1.0, c_level=0.15):
    """Frontier G = A - |B| N. Corners: high-bar (0, A/|B|), low-bar (A, 0)."""
    _reseed(3)
    om = RNG.uniform(0.05, 0.95, (n, 2))
    V = RNG.uniform(1.0, 10.0, (n, 2))
    x = V * RNG.uniform(0.1, 0.9, (n, 2))
    pi = V - x
    c = c_level * pi.mean(axis=1, keepdims=True).repeat(2, axis=1)
    corners = [(0.0, A / Bslope), (A, 0.0)]

    def pay(order, k, design):
        total, survive = 0.0, 1.0
        for slot, i in enumerate(order):
            G, N = corners[design[slot]]
            a, b = om[k, i] * (1 - N), (1 - om[k, i]) * G
            total += survive * (a * pi[k, i] - b * x[k, i] - c[k, i])
            survive *= 1 - (a + b)
        return total

    def myopic(k, i):
        vals = [om[k, i] * (1 - N) * pi[k, i] - (1 - om[k, i]) * G * x[k, i]
                for G, N in corners]
        return int(np.argmax(vals))

    diff_design = diff_position = valid = 0
    gains = []
    for k in range(n):
        best, arg = -np.inf, None
        for order in ([0, 1], [1, 0]):
            for d0 in (0, 1):
                for d1 in (0, 1):
                    v = pay(order, k, (d0, d1))
                    if v > best:
                        best, arg = v, (order, (d0, d1))
        if best <= 0:
            continue
        valid += 1
        order, design = arg
        base = (myopic(k, order[0]), myopic(k, order[1]))
        if design != base:
            diff_design += 1
            gains.append((best - pay(order, k, base)) / best)
        if design[0] != design[1]:
            diff_position += 1

    stats = dict(valid=valid,
                 diff_design=100 * diff_design / valid,
                 diff_position=100 * diff_position / valid,
                 mean_gain=100 * float(np.mean(gains)),
                 median_gain=100 * float(np.median(gains)))
    print("--- Table 2: joint ordering and test accuracy --------------------")
    for k_, v_ in stats.items():
        print(f"  {k_:15s}: {v_:,.2f}")
    print()
    open("table2.tex", "w").write(
        "\\begin{tabular}{lr}\n\\toprule\nStatistic & Value \\\\\n\\midrule\n"
        f"Draws with positive testing value & {valid:,} \\\\\n"
        f"First and second tests call for different accuracy & {stats['diff_position']:.1f}\\% \\\\\n"
        f"Joint solution differs from sequence-independent design & {stats['diff_design']:.1f}\\% \\\\\n"
        f"Mean value gain, conditional on differing & {stats['mean_gain']:.1f}\\% \\\\\n"
        f"Median value gain, conditional on differing & {stats['median_gain']:.1f}\\% \\\\\n"
        "\\bottomrule\n\\end{tabular}\n")
    return stats


# ----------------------------------------------------------------------
# 7. Figures
# ----------------------------------------------------------------------


def fig_boundary(V=(6.0, 4.0), x=2.0, c=0.3, fs=(1.0, 0.8, 0.6)):
    """Sign of Delta over (omega_S, omega_Z) inside the experimentation zone."""
    grid = np.linspace(0.005, 0.70, 500)
    WS, WZ = np.meshgrid(grid, grid, indexing="ij")
    Vv = np.array(V)
    panels, boxes = [], []
    for f in fs:
        D = f * (WS * WZ * ((Vv[0] - x) - (Vv[1] - x)) + c * (WS - WZ)) \
            + (1 - f) * (Vv[0] * WS - Vv[1] * WZ)
        om = np.stack([WS.ravel(), WZ.ravel()], axis=1)
        m = om.shape[0]
        Vg, xg, cg = np.tile(Vv, (m, 1)), np.full((m, 2), x), np.full((m, 2), c)
        Ng, Gg = np.zeros((m, 2)), np.full((m, 2), 1 - f)
        zone = (in_zone(om[:, 0], Vg[:, 0], xg[:, 0], cg[:, 0], Ng[:, 0], Gg[:, 0])
                & in_zone(om[:, 1], Vg[:, 1], xg[:, 1], cg[:, 1], Ng[:, 1], Gg[:, 1])
                ).reshape(WS.shape)
        panels.append((f, D, zone))
        if zone.any():
            boxes.append((WS[zone].min(), WS[zone].max(), WZ[zone].min(), WZ[zone].max()))
    x0, x1 = min(b[0] for b in boxes) - 0.03, max(b[1] for b in boxes) + 0.03
    y0, y1 = min(b[2] for b in boxes) - 0.03, max(b[3] for b in boxes) + 0.03

    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(1, len(fs), figsize=(6.5, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (f, D, zone) in zip(axes, panels):
        Dm = np.where(zone, D, np.nan)
        ax.contourf(WS, WZ, np.sign(Dm), levels=[-1.5, 0, 1.5], colors=["0.62", "0.93"])
        ax.contour(WS, WZ, Dm, levels=[0], colors="k", linewidths=1.3)
        ax.contour(WS, WZ, zone.astype(float), levels=[0.5], colors="k",
                   linewidths=0.7, linestyles=":")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel(r"$\omega_S$")
        ax.set_title(rf"$f={f:.1f}$")
    axes[0].set_ylabel(r"$\omega_Z$")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="0.93", edgecolor="k", lw=0.5, label="test $S$ first"),
                        Patch(facecolor="0.62", edgecolor="k", lw=0.5,
                              label="test $Z$ first (surprise ordering)")],
               frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(rf"$V_S={V[0]:.0f},\; V_Z={V[1]:.0f},\; x_S=x_Z={x:.0f},\; c={c:.1f}$"
                 r" (dotted: experimentation zone)", fontsize=10)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig("fig_boundary.pdf")
    plt.close(fig)
    print("wrote fig_boundary.pdf")


def omega_star(W, pi, x, Bslope):
    """Prior below which the high-bar corner is optimal at continuation value W."""
    num = Bslope * (x + W)
    return num / (num + np.maximum(pi - W, 1e-9))


def fig_design(pi=4.0, x=2.0, Bslope=1.0):
    W = np.linspace(0.0, 0.9 * pi, 300)
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(W, omega_star(W, pi, x, Bslope), "k-", lw=1.4,
            label=r"$\omega^{*}(W)$: first test")
    ax.axhline(omega_star(0.0, pi, x, Bslope), color="0.4", ls="--", lw=1.2,
               label=r"$\omega^{*}(0)$: second test")
    ax.fill_between(W, omega_star(0.0, pi, x, Bslope), omega_star(W, pi, x, Bslope),
                    color="0.85")
    ax.text(0.52 * W[-1], 0.5 * (omega_star(0.0, pi, x, Bslope)
                                 + omega_star(0.62 * W[-1], pi, x, Bslope)),
            "demanding first,\npermissive second", ha="center", va="center", fontsize=8)
    ax.text(0.5 * W[-1], 0.90, "permissive test preferred", ha="center", fontsize=8)
    ax.text(0.5 * W[-1], 0.10, "demanding test preferred", ha="center", fontsize=8)
    ax.set_xlabel(r"value of the theory still in reserve, $W$")
    ax.set_ylabel(r"prior cutoff $\omega^{*}$")
    ax.set_title(rf"$\pi={pi:.0f}$, $x={x:.0f}$, $|B|={Bslope:.0f}$")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("fig_design.pdf")
    plt.close(fig)
    print("wrote fig_design.pdf")


def fig_cost(n=40_000, kappas=np.linspace(0.02, 0.60, 12)):
    """Reversal frequency and welfare loss as the test cost varies."""
    _reseed(5)
    regimes = {"perfect": REGIMES["Perfect ($G=N=0$)"],
               r"falsification-only, $f=0.8$": REGIMES["Falsification-only, $f=0.8$"],
               r"general $(G,N)$": REGIMES["General $(G,N)$"]}
    styles = ["k-o", "k--s", "k:^"]
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
    summary = {}
    for (label, kw), st in zip(regimes.items(), styles):
        ks, freq, loss = [], [], []
        for k in kappas:
            res = run_cell(n, k, target=8_000, max_batches=60, **kw)
            if res is None:
                continue
            ks.append(k)
            freq.append(100 * res["surprise_pi"])
            loss.append(res["loss_pi"])
        summary[label] = (ks[0], ks[-1], freq[0], freq[-1], loss[0], loss[-1])
        print(f"  {label}: kappa {ks[0]:.2f}->{ks[-1]:.2f}, "
              f"reversal {freq[0]:.1f}%->{freq[-1]:.1f}%, loss {loss[0]:.2f}%->{loss[-1]:.2f}%")
        axes[0].plot(ks, freq, st, ms=3.2, label=label)
        axes[1].plot(ks, loss, st, ms=3.2, label=label)
    axes[0].set_xlabel(r"test cost $\kappa = c/\bar{\pi}$")
    axes[0].set_ylabel(r"differs from $\pi$-rank (\%)")
    axes[1].set_xlabel(r"test cost $\kappa = c/\bar{\pi}$")
    axes[1].set_ylabel(r"value lost by $\pi$-rank (\%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig("fig_cost.pdf")
    plt.close(fig)
    print("wrote fig_cost.pdf")
    return summary


if __name__ == "__main__":
    check_identities()
    worked_example()
    table1()
    table2()
    table3()
    fig_boundary()
    fig_design()
    fig_cost()
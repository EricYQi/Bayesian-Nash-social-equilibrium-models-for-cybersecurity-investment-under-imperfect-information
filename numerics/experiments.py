"""Experiment configurations and the common three-regime driver.

The two stylized calibrations of the manuscript are defined here so that the
simulation scripts, the robustness study and the figure scripts all use one
set of parameters.  ``run_regimes`` solves the decentralized, cooperative and
centralized problems and returns every diagnostic reported in the paper.
"""

from __future__ import annotations

import numpy as np

from bnse_core import (Model, bnse_diagnostics, coordinated_diagnostics,
                       certificates, comparison_table, economic_summary,
                       multistart, solve_bnse)
from global_certificate import certify_bargaining_point

SEED = 20260920
RHO = 0.3
BNSB_START_WEIGHTS = (0.25, 0.4, 0.55, 0.7, 0.85, 0.5)


def two_department_model(a_band=-0.15, b_band=0.045):
    """Stylized two-department calibration of Section 5.1."""
    return Model(
        depts=["A", "B"],
        W={"A": [72600.0, 72600.0], "B": [78800.0]},
        D={"A": [145.0, 162.0], "B": [152.0]},
        alpha={"A": [0.625, 0.625], "B": [0.750]},
        prob={"A": [0.5, 0.5], "B": [1.0]},
        delta={"A": 0.8, "B": 0.8},
        a_band={"A": a_band, "B": a_band},
        b_band={"A": b_band, "B": b_band},
    )


def three_department_model(a_band=-0.10, b_band=0.099):
    """Stylized three-department calibration of Section 5.2."""
    return Model(
        depts=["A", "B", "C"],
        W={"A": [72600.0, 72600.0], "B": [78800.0], "C": [45000.0, 45000.0]},
        D={"A": [145.0, 162.0], "B": [152.0], "C": [80.0, 110.0]},
        alpha={"A": [0.50, 0.50], "B": [0.60], "C": [0.40, 0.44]},
        prob={"A": [0.5, 0.5], "B": [1.0], "C": [0.6, 0.4]},
        delta={"A": 0.8, "B": 0.8, "C": 0.8},
        a_band={"A": a_band, "B": a_band, "C": a_band},
        b_band={"A": b_band, "B": b_band, "C": b_band},
    )


# ---------------------------------------------------------------------------
# Multi-start generation (documented in Section 4.3 of the manuscript)
# ---------------------------------------------------------------------------
def welfare_starts(model, x_bnse, seed=SEED):
    """Deterministic and random starts for the centralized problem.

    Start 0 is the uniform profile ``s = 0.1``; start 1 is the BNSE; starts
    2--5 are independent draws from the uniform distribution on the action
    set, produced by ``numpy.random.default_rng(seed)``.
    """
    rng = np.random.default_rng(seed)
    starts = [model.uniform_state(0.1), np.array(x_bnse, float)]
    caps = model.c[model.owner]
    for _ in range(4):
        starts.append(rng.uniform(0.0, caps))
    return starts


def bargaining_starts(model, x_bnse, x_bnso, weights=BNSB_START_WEIGHTS):
    """Convex combinations of the BNSE and the centralized solution."""
    return [(1.0 - w) * np.array(x_bnse, float) + w * np.array(x_bnso, float)
            for w in weights]


# ---------------------------------------------------------------------------
# Second-order (local) check on the frozen bargaining problem
# ---------------------------------------------------------------------------
def reduced_hessian(model, x, disagreement, steps=(1e-4, 1e-5, 1e-6),
                    active_tol=1e-6):
    """Reduced Hessian of the log-Nash objective on the frozen feasible set.

    The Hessian is obtained by central differences of the *analytic*
    gradient and restricted to the coordinates that are not blocked by an
    active bound of the frozen box.
    """
    lo, hi = model.box(x)
    free = [k for k in range(model.N)
            if x[k] - lo[k] > active_tol and hi[k] - x[k] > active_tol]
    report = {"free_coordinates": [model.labels[k] for k in free],
              "blocked_coordinates": [model.labels[k] for k in range(model.N)
                                      if k not in free]}
    for step in steps:
        hessian = np.zeros((model.N, model.N))
        for k in range(model.N):
            forward, backward = x.copy(), x.copy()
            forward[k] += step
            backward[k] -= step
            hessian[:, k] = (model.log_nash_gradient(forward, disagreement)
                             - model.log_nash_gradient(backward, disagreement)
                             ) / (2.0 * step)
        hessian = 0.5 * (hessian + hessian.T)
        if free:
            eigenvalues = np.linalg.eigvalsh(hessian[np.ix_(free, free)])
            report[f"eigenvalues_h={step:g}"] = eigenvalues.tolist()
            report[f"negative_definite_h={step:g}"] = bool(
                np.all(eigenvalues < 0))
        else:
            report[f"eigenvalues_h={step:g}"] = []
            report[f"negative_definite_h={step:g}"] = None
    return report


# ---------------------------------------------------------------------------
# Three-regime driver
# ---------------------------------------------------------------------------
def run_regimes(model, rho=RHO, seed=SEED, certify=True,
                certificate_tolerance=1e-4, certificate_time_limit=600.0,
                verbose=False):
    """Solve the three organizational regimes and collect all diagnostics."""
    bnse = solve_bnse(model, rho=rho, record=True)
    x_bnse = bnse["x"]
    disagreement = model.exante_utility(x_bnse)
    bnse_report = bnse_diagnostics(model, x_bnse, bnse["iterations"],
                                   bnse["history"], rho=rho)

    starts_O = welfare_starts(model, x_bnse, seed=seed)
    x_bnso, iters_O, residual_O, summary_O = multistart(
        model, "welfare", starts_O, label="BNS-O")
    report_O = coordinated_diagnostics(
        model, "welfare", x_bnso, outer_residual=residual_O,
        outer_iterations=iters_O)
    report_O["multistart"] = summary_O
    report_O["start_rule"] = (
        "start 0: uniform s=0.1; start 1: BNSE; starts 2-5: "
        f"uniform random on the action set, numpy default_rng({seed})")

    starts_B = bargaining_starts(model, x_bnse, x_bnso)
    x_bnsb, iters_B, residual_B, summary_B = multistart(
        model, "bargaining", starts_B, disagreement=disagreement,
        label="BNSB")
    report_B = coordinated_diagnostics(
        model, "bargaining", x_bnsb, disagreement=disagreement,
        outer_residual=residual_B, outer_iterations=iters_B)
    report_B["multistart"] = summary_B
    report_B["start_rule"] = (
        "convex combinations (1-w) s^N + w s^O with "
        f"w in {list(BNSB_START_WEIGHTS)}")
    report_B["reduced_hessian_frozen_problem"] = reduced_hessian(
        model, x_bnsb, disagreement)
    if certify:
        report_B["frozen_set_global_certificate"] = certify_bargaining_point(
            model, x_bnsb, disagreement, tolerance=certificate_tolerance,
            time_limit=certificate_time_limit, verbose=verbose)

    results = {
        "BNSE": _regime_block(model, x_bnse, bnse["iterations"]),
        "BNSB": _regime_block(model, x_bnsb, iters_B),
        "BNSO": _regime_block(model, x_bnso, iters_O),
    }
    results["BNSE"]["K"] = _box_block(model, x_bnse)
    results["BNSB"]["K"] = _box_block(model, x_bnsb)
    results["BNSO"]["K"] = _box_block(model, x_bnso)

    return {
        "parameters": _parameter_block(model, rho, seed),
        "results": results,
        "convergence_BNSE": bnse["history"],
        "validation": {"BNSE": bnse_report, "BNSB": report_B,
                       "BNSO": report_O},
        "certificates": certificates(model, rho_test=rho),
        "economics": {
            "BNSE": economic_summary(model, x_bnse),
            "BNSB": economic_summary(model, x_bnsb),
            "BNSO": economic_summary(model, x_bnso),
            "BNSB_vs_BNSE": comparison_table(model, x_bnse, x_bnsb),
            "BNSO_vs_BNSE": comparison_table(model, x_bnse, x_bnso),
        },
        "solutions": {"BNSE": x_bnse.tolist(), "BNSB": x_bnsb.tolist(),
                      "BNSO": x_bnso.tolist()},
    }


def _regime_block(model, x, iterations):
    utilities = model.interim_utility(x)
    state = model.to_dict(x)
    interim = {}
    for k, (i, t) in enumerate(model.coords):
        interim.setdefault(i, []).append(float(utilities[k]))
    exante = model.exante_utility(x)
    return {
        "s": state,
        "U": interim,
        "barU": {model.depts[i]: float(exante[i]) for i in range(model.n)},
        "iters": iterations,
    }


def _box_block(model, x):
    lo, hi = model.box(x)
    return {model.depts[i]: [float(lo[model.masks[i]][0]),
                             float(hi[model.masks[i]][0])]
            for i in range(model.n)}


def _parameter_block(model, rho, seed):
    return {
        "departments": list(model.depts),
        "types": {i: model.ntypes[i] for i in model.depts},
        "W": model.W, "D": model.D, "alpha": model.alpha,
        "prob": model.prob, "delta": model.delta,
        "a": model.a_band, "b": model.b_band,
        "action_cap": {model.depts[i]: float(model.c[i])
                       for i in range(model.n)},
        "budget_translation_span": {
            model.depts[i]: float(model.c[i] - model.b[i] + model.a[i])
            for i in range(model.n)},
        "rho": rho,
        "seed": seed,
        "utility_units": "square root of millions of US dollars",
    }

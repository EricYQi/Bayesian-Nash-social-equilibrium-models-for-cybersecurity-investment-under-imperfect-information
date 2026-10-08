"""Robustness of the results to the size of the endogenous budget movement.

The admissible translation of the reference budget over the whole strategy
domain equals

    tau_i = c_i - b_i + a_i = c_i - (b_i - a_i),

so a wider *feasible interval* ``b_i - a_i`` necessarily produces a smaller
*moving* interval.  The two baseline calibrations of the paper use very wide
intervals and therefore translate by only 0.005 and 0.001.  This script
scales the baseline band by a factor ``kappa``,

    a_i(kappa) = kappa a_i^0,   b_i(kappa) = kappa b_i^0,

which keeps the sign conventions of the baseline intact while increasing
the translation to ``tau_i = c_i - kappa (b_i^0 - a_i^0)``.  For every band it reports the Lipschitz constant ``ell_M`` of the moving set,
the BNSE contraction certificate, the three organizational solutions, their
residuals, and whether the qualitative ordering of the regimes survives.

Run with ``python robustness.py``; the output is ``results_robustness.json``.
"""

import json

import numpy as np

from bnse_core import certificates, solve_bnse
from experiments import (RHO, SEED, run_regimes, three_department_model,
                         two_department_model)

SCALES = [1.0, 0.9, 0.75, 0.6, 0.45, 0.35, 0.25, 0.15, 0.05]
DETAILED_SCALES = [1.0, 0.6, 0.35]


def band_for(builder, scale):
    baseline = builder()
    a0 = baseline.a_band[baseline.depts[0]]
    b0 = baseline.b_band[baseline.depts[0]]
    return builder(a_band=scale * a0, b_band=scale * b0)


def certified_threshold(builder, tolerance=1e-9):
    """Largest band scaling for which the BNSE contraction test still holds.

    The test is monotone in the scaling because ``ell_M`` increases as the
    band narrows, so the threshold is located by bisection.
    """
    def holds(scale):
        model = band_for(builder, scale)
        return certificates(model, rho_test=RHO)["BNSE_contraction_test"][
            "holds"]

    if not holds(1.0):
        return None
    low, high = 0.0, 1.0
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if holds(middle):
            high = middle
        else:
            low = middle
    model = band_for(builder, high)
    certificate = certificates(model, rho_test=RHO)
    return {
        "critical_scale": high,
        "a": model.a_band[model.depts[0]],
        "b": model.b_band[model.depts[0]],
        "feasible_interval_width": float(model.b[0] - model.a[0]),
        "largest_certified_translation": float(
            model.c[0] - model.b[0] + model.a[0]),
        "ell_M": certificate["ell_M"],
        "mu_N": certificate["mu_N"],
        "L_N": certificate["L_N"],
        "rho_interval": certificate["BNSE_contraction_test"]["rho_interval"],
    }


def sweep(builder, name):
    """Certificate and BNSE behaviour along the band-scaling path."""
    rows = []
    for scale in SCALES:
        model = band_for(builder, scale)
        certificate = certificates(model, rho_test=RHO)
        run = solve_bnse(model, rho=RHO, record=True)
        x = run["x"]
        lo, hi = model.box(x)
        rows.append({
            "scale": scale,
            "a": model.a_band[model.depts[0]],
            "b": model.b_band[model.depts[0]],
            "feasible_interval_width": float(model.b[0] - model.a[0]),
            "budget_translation_span": float(
                model.c[0] - model.b[0] + model.a[0]),
            "ell_M": certificate["ell_M"],
            "mu_N": certificate["mu_N"],
            "L_N": certificate["L_N"],
            "contraction_holds": certificate["BNSE_contraction_test"]["holds"],
            "rho_interval": certificate["BNSE_contraction_test"]["rho_interval"],
            f"rho_{RHO}_admissible": certificate["BNSE_contraction_test"][
                f"rho_{RHO}_admissible"],
            "BNSE_iterations": run["iterations"],
            "BNSE_actions": model.to_dict(x),
            "BNSE_mean_actions": model.mean_actions(x).tolist(),
            "BNSE_box": {model.depts[i]: [float(lo[model.masks[i]][0]),
                                          float(hi[model.masks[i]][0])]
                         for i in range(model.n)},
            "BNSE_bounds_active": bool(
                np.any(np.abs(x - lo) < 1e-9) or np.any(np.abs(hi - x) < 1e-9)),
        })
        print(f"  {name} scale={scale:4.2f} "
              f"tau={rows[-1]['budget_translation_span']:.4f} "
              f"ell_M={rows[-1]['ell_M']:.4f} "
              f"contraction={rows[-1]['contraction_holds']}")
    return rows


def detailed(builder, name, certificate_time_limit):
    """Full three-regime comparison at selected band scales."""
    blocks = {}
    for scale in DETAILED_SCALES:
        model = band_for(builder, scale)
        print(f"  {name}: three-regime run at scale {scale}")
        output = run_regimes(model, rho=RHO, seed=SEED,
                             certificate_tolerance=1e-3,
                             certificate_time_limit=certificate_time_limit)
        mean_actions = {
            regime: model.mean_actions(
                np.array(output["solutions"][regime])).tolist()
            for regime in ("BNSE", "BNSB", "BNSO")}
        exante = {regime: output["results"][regime]["barU"]
                  for regime in ("BNSE", "BNSB", "BNSO")}
        ordering = {
            "coordination_raises_mean_investment": bool(
                all(mean_actions["BNSB"][i] >= mean_actions["BNSE"][i] - 1e-9
                    and mean_actions["BNSO"][i] >= mean_actions["BNSE"][i] - 1e-9
                    for i in range(model.n))),
            "bargaining_satisfies_participation": bool(
                all(exante["BNSB"][d] >= exante["BNSE"][d] - 1e-10
                    for d in model.depts)),
            "welfare_ordering_BNSO_best": bool(
                sum(exante["BNSO"].values()) >= sum(exante["BNSB"].values())
                - 1e-10),
            "some_department_loses_under_BNSO": bool(
                any(exante["BNSO"][d] < exante["BNSE"][d] - 1e-12
                    for d in model.depts)),
        }
        blocks[str(scale)] = {
            "parameters": output["parameters"],
            "results": output["results"],
            "certificates": output["certificates"],
            "validation": output["validation"],
            "economics": output["economics"],
            "mean_actions": mean_actions,
            "qualitative_ordering": ordering,
        }
    return blocks


def main():
    payload = {
        "description": (
            "Band-scaling robustness study.  tau_i = c_i - (b_i - a_i) is "
            "the admissible translation of the reference budget; the "
            "baseline scale 1.0 reproduces Sections 5.1 and 5.2."),
        "scales": SCALES,
        "detailed_scales": DETAILED_SCALES,
    }
    print("two-department sweep")
    payload["two_department_sweep"] = sweep(two_department_model, "2-dept")
    payload["two_department_threshold"] = certified_threshold(
        two_department_model)
    print(f"  largest certified translation (2-dept): "
          f"{payload['two_department_threshold']}")
    print("three-department sweep")
    payload["three_department_sweep"] = sweep(three_department_model, "3-dept")
    payload["three_department_threshold"] = certified_threshold(
        three_department_model)
    print(f"  largest certified translation (3-dept): "
          f"{payload['three_department_threshold']}")
    print("two-department detailed runs")
    payload["two_department_detailed"] = detailed(
        two_department_model, "2-dept", certificate_time_limit=300.0)
    print("three-department detailed runs")
    payload["three_department_detailed"] = detailed(
        three_department_model, "3-dept", certificate_time_limit=300.0)

    with open("results_robustness.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    print("results_robustness.json written")


if __name__ == "__main__":
    main()

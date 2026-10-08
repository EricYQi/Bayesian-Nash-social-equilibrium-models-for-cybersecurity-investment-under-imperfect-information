"""Two-department numerical illustration for the revised AOR manuscript.

Run with ``python sim_2dept.py``.  The script writes ``results_2dept.json``,
which contains the reported actions and utilities, every numerical
diagnostic, the full player--type contraction certificates, the deterministic
global certificate for the frozen bargaining problem, the economic
decomposition of the three regimes, and the raw data of Figures 1--4.

The revenue and loss inputs are the public corporate figures described in
the paper.  The cost coefficients and the feasible-band parameters are
illustrative calibrations, not estimates.  All derivatives used by the
optimizers are analytic; finite differences are reported only as an audit.
"""

import json

import numpy as np

from bnse_core import solve_bnse
from experiments import RHO, SEED, run_regimes, two_department_model

# Grids used for the bound-sensitivity figures.  Every plotted pair
# satisfies the admissibility condition c_i - b_i > -a_i.
A_GRID = np.linspace(-0.30, -0.02, 15)
B_FIXED = [0.02, 0.045, 0.09]
B_GRID = np.linspace(0.005, 0.20, 15)
A_FIXED = [-0.15, -0.05]
INITIAL_VALUES = [0.0, 0.02, 0.05, 0.1, 0.15, 0.19]


def band_sensitivity():
    """BNSE response to the band endpoints, with the exact grids recorded."""
    sens_a, sens_b = {}, {}
    admissible_a, admissible_b = {}, {}
    for b_value in B_FIXED:
        rows, admissible = [], []
        for a_value in A_GRID:
            if not 0.2 - b_value > -a_value + 1e-12:
                continue
            model = two_department_model(a_band=float(a_value),
                                         b_band=float(b_value))
            x = solve_bnse(model, rho=RHO)["x"]
            rows.append({"a": float(a_value), "sA1": float(x[0]),
                         "sA2": float(x[1]), "sB": float(x[2])})
            admissible.append(float(a_value))
        sens_a[str(b_value)] = rows
        admissible_a[str(b_value)] = admissible
    for a_value in A_FIXED:
        rows, admissible = [], []
        for b_value in B_GRID:
            if not 0.2 - b_value > -a_value + 1e-12:
                continue
            model = two_department_model(a_band=float(a_value),
                                         b_band=float(b_value))
            x = solve_bnse(model, rho=RHO)["x"]
            rows.append({"b": float(b_value), "sA1": float(x[0]),
                         "sA2": float(x[1]), "sB": float(x[2])})
            admissible.append(float(b_value))
        sens_b[str(a_value)] = rows
        admissible_b[str(a_value)] = admissible
    grids = {
        "a_grid_requested": A_GRID.tolist(),
        "b_values_fixed": B_FIXED,
        "a_values_admissible": admissible_a,
        "b_grid_requested": B_GRID.tolist(),
        "a_values_fixed": A_FIXED,
        "b_values_admissible": admissible_b,
        "admissibility_rule": "c_i - b_i > -a_i with c_i = 0.2",
    }
    return sens_a, sens_b, grids


def initial_value_study(model):
    """Convergence of the BNSE iteration from several starting profiles."""
    effects = []
    for value in INITIAL_VALUES:
        run = solve_bnse(model, rho=RHO, x0=model.uniform_state(value),
                         record=True, record_trajectory=True)
        effects.append({
            "s0": value,
            "s_final": model.to_dict(run["x"]),
            "iters": run["iterations"],
            "history": run["history"],
            "trajectory": [model.to_dict(np.array(row))
                           for row in run["trajectory"]],
        })
    return effects


def main():
    model = two_department_model()
    output = run_regimes(model, rho=RHO, seed=SEED,
                         certificate_tolerance=1e-4,
                         certificate_time_limit=600.0)

    print(json.dumps(output["results"], indent=2))
    print(json.dumps({"validation": output["validation"]}, indent=2))
    print(json.dumps({"certificates": output["certificates"]}, indent=2))

    sens_a, sens_b, grids = band_sensitivity()
    output["sens_a"] = sens_a
    output["sens_b"] = sens_b
    output["sensitivity_grids"] = grids
    output["init_effects"] = initial_value_study(model)

    with open("results_2dept.json", "w") as handle:
        json.dump(output, handle, indent=2)
    print("results_2dept.json written")


if __name__ == "__main__":
    main()

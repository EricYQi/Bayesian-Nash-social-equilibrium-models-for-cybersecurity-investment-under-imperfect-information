"""Three-department numerical illustration for the revised AOR manuscript.

Run with ``python sim.py``.  The script writes ``results.json``, which
contains the reported actions and utilities, every numerical diagnostic, the
full player--type contraction certificates, the deterministic global
certificate for the frozen bargaining problem, the economic decomposition of
the three regimes, and the raw data of Figures 5 and 6.

Departments A and C have two types and department B has one.  Department C,
the cost coefficients and the feasible-band parameters are illustrative
calibrations, not estimates.  All derivatives used by the optimizers are
analytic; finite differences are reported only as an audit.
"""

import json

import numpy as np

from bnse_core import solve_bnse
from experiments import RHO, SEED, run_regimes, three_department_model

PROBABILITY_GRID = np.linspace(0.05, 0.95, 19)


def type_probability_sensitivity():
    """BNSE response to the type distribution of department A.

    The exposure weights ``omega_i`` are recomputed at every probability,
    because they depend on the expected exposure ``E[W_i D_i]``.
    """
    rows = []
    for probability in PROBABILITY_GRID:
        model = three_department_model()
        model.set_probabilities("A", [float(probability),
                                      float(1.0 - probability)])
        x = solve_bnse(model, rho=RHO)["x"]
        rows.append({"p_A1": float(probability), "s_A1": float(x[0]),
                     "s_A2": float(x[1]), "s_B": float(x[2]),
                     "s_C1": float(x[3]), "s_C2": float(x[4])})
    return rows


def main():
    model = three_department_model()
    output = run_regimes(model, rho=RHO, seed=SEED,
                         certificate_tolerance=1e-4,
                         certificate_time_limit=900.0)

    print(json.dumps(output["results"], indent=2))
    print(json.dumps({"validation": output["validation"]}, indent=2))
    print(json.dumps({"certificates": output["certificates"]}, indent=2))

    output["sensitivity"] = type_probability_sensitivity()
    output["sensitivity_grid"] = {
        "q_A(t_A^1)": PROBABILITY_GRID.tolist(),
        "note": "q_A(t_A^2) = 1 - q_A(t_A^1); exposure weights recomputed",
    }

    with open("results.json", "w") as handle:
        json.dump(output, handle, indent=2)
    print("results.json written")


if __name__ == "__main__":
    main()

# Bayesian–Nash Social Equilibrium Experiments

Reproducibility materials for the numerical experiments accompanying
“Bayesian-Nash social equilibrium models for cybersecurity investment under
imperfect information.”

The models compare decentralized Bayesian–Nash social equilibrium (BNSE),
cooperative Bayesian–Nash bargaining (BNSB), and centralized Bayesian system
optimization (BNSO) under finite-type uncertainty and endogenous budget
constraints.

## Contents

- `numerics/`: experiment implementations, pinned dependencies, recorded
  numerical results, and diagnostic plots.
- `Fig1.pdf`–`Fig5.pdf`: manuscript figures generated from the numerical
  results.

The two- and three-department parameterizations are documented in
`numerics/experiments.py`. The repository includes recorded outputs
(`results_2dept.json`, `results.json`, and `results_robustness.json`), including
the diagnostics and data used to create the plots. The public financial and
loss figures used in the two-department example are cited in the manuscript.
The third-department data and rescaled cost parameters are illustrative
calibrations, not empirical estimates.

## Requirements

Use Python and install the pinned packages from the numerics directory:

```bash
cd numerics
python -m pip install -r requirements.txt
```

Dependencies are NumPy 2.3.1, SciPy 1.16.0, and Matplotlib 3.10.3.

## Reproducing the results

Run commands from `numerics/`:

```bash
python sim_2dept.py
python sim.py
python robustness.py
python plot_2dept.py
python plot.py
python report.py
```

The simulation scripts write the JSON result files. Plotting scripts use those
JSON files to recreate the figures in `numerics/` and the corresponding
manuscript figures in the repository root. `report.py` prints the numerical
values reported in the manuscript. The simulation and global-certificate
calculations can take several minutes; see `numerics/README.txt` for further
implementation and runtime details.

All stochastic multi-start calculations use seed `20260920`. Derivatives used
by the optimization routines are analytic; finite differences are used only
for numerical audit diagnostics.

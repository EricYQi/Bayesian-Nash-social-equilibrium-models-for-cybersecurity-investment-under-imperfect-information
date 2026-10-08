Numerical reproduction
=======================

Python dependencies are pinned in requirements.txt
(numpy 2.3.1, scipy 1.16.0, matplotlib 3.10.3).

Layout
------

  bnse_core.py          General finite-type model: interim and ex-ante
                        utilities, ANALYTIC gradients of every objective and
                        constraint, the endogenous budget correspondence, the
                        projected BNSE iteration, SLSQP solvers for the
                        frozen-feasible-set problems, feasibility/fixed-point/
                        KKT diagnostics, and the full player--type
                        monotonicity and contraction certificates
                        (mu_N, L_N, mu_O, L_O, ell_M).

  global_certificate.py Deterministic branch-and-bound certificate for the
                        frozen bargaining problem (Theorem 3.9 of the paper).
                        It bounds the frozen-set global optimum by a concave
                        relaxation indexed by an n-dimensional box of
                        aggregate mean actions, evaluated by one monotone
                        scalar search per coordinate, and returns the
                        certified optimality gap.

  experiments.py        The two calibrations of Sections 5.1 and 5.2, the
                        documented multi-start rules, the reduced-Hessian
                        check, and the three-regime driver.

  sim_2dept.py          Two-department illustration  -> results_2dept.json
  sim.py                Three-department illustration -> results.json
  robustness.py         Band-scaling robustness study -> results_robustness.json
  report.py             Reprints every number quoted in the manuscript
  plot_2dept.py         Figures 1-4 (also copied to ../Fig1.pdf ... ../Fig4.pdf)
  plot.py               Figures 5-7 (also copied to ../Fig5.pdf ... ../Fig7.pdf)

Run order
---------

  python sim_2dept.py
  python sim.py
  python robustness.py
  python plot_2dept.py
  python plot.py
  python report.py

Typical run times on a laptop: sim_2dept.py about 2 minutes, sim.py about
12 minutes and robustness.py about 25 minutes; most of that time is spent
in the global certificate, whose target tolerance and time limit are
arguments of experiments.run_regimes.

Modelling and numerical conventions
-----------------------------------

* The bargaining objective contains one ex-ante expected surplus per
  department, and the logarithmic form is exact: no smooth continuation
  below a positive floor is used anywhere.
* Every derivative used by an optimizer is analytic. Finite differences
  appear only in the audit fields of the JSON output, which report the
  deviation between analytic and central-difference gradients and the KKT
  residual recomputed at several step sizes.
* All random multi-start calculations use seed 20260920.
* The legacy scalar beta_1, beta_2, beta_3 diagnostics of earlier versions
  have been removed: they belonged to a superseded scalar monotonicity
  argument and no longer correspond to any statement in the manuscript. The
  corresponding certificates are now in bnse_core.certificates.

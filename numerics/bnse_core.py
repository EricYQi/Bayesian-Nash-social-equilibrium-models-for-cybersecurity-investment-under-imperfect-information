"""Core implementation of the finite-type Bayesian cybersecurity model.

This module implements, for a general number of departments and types,

* the interim and ex-ante expected utilities of Section 2;
* the endogenous budget correspondence and its Lipschitz constant;
* analytic first derivatives of every objective used in the paper
  (decentralized pseudo-gradient, ex-ante utility gradients, aggregate
  welfare gradient, and the log-bargaining gradient);
* the projected fixed-point iteration for the BNSE;
* frozen-feasible-set SLSQP solvers with exact Jacobians for the
  cooperative and centralized regimes;
* feasibility, fixed-point and KKT diagnostics; and
* the full player--type monotonicity/contraction certificates.

Every derivative used by the optimizer is analytic; finite differences are
used only to *audit* the analytic derivatives and to report the sensitivity
of the reported KKT residuals to the finite-difference step size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, nnls

__all__ = [
    "Model",
    "solve_bnse",
    "bnse_diagnostics",
    "solve_frozen",
    "solve_outer_fixed_point",
    "multistart",
    "coordinated_diagnostics",
    "certificates",
    "kkt_residual",
    "finite_difference_audit",
    "economic_summary",
    "comparison_table",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class Model:
    """Finite-type Bayesian cybersecurity model.

    Parameters
    ----------
    depts   : ordered department labels.
    W, D, alpha, prob : per-department lists indexed by type.
    delta   : per-department parameter with action cap ``c_i = 1 - delta_i``.
    a_band, b_band : per-department band parameters, restricted only by the
        width condition ``0 < b_i - a_i < c_i``; their signs are free.
    """

    depts: list
    W: dict
    D: dict
    alpha: dict
    prob: dict
    delta: dict
    a_band: dict
    b_band: dict

    # derived, filled by __post_init__
    coords: list = field(init=False)
    owner: np.ndarray = field(init=False)
    q: np.ndarray = field(init=False)
    Delta: np.ndarray = field(init=False)
    alpha_vec: np.ndarray = field(init=False)
    fW: np.ndarray = field(init=False)

    def __post_init__(self):
        self.validate()
        self.n = len(self.depts)
        self.ntypes = {i: len(self.W[i]) for i in self.depts}
        self.coords = [(i, t) for i in self.depts for t in range(self.ntypes[i])]
        self.labels = [f"{i}:t{t}" for i, t in self.coords]
        self.N = len(self.coords)
        index = {i: k for k, i in enumerate(self.depts)}
        self.owner = np.array([index[i] for i, _ in self.coords], dtype=int)
        self.q = np.array([self.prob[i][t] for i, t in self.coords])
        self.fW = np.array([np.sqrt(self.W[i][t]) for i, t in self.coords])
        self.Delta = np.array([
            np.sqrt(self.W[i][t]) - np.sqrt(self.W[i][t] - self.D[i][t])
            for i, t in self.coords])
        self.alpha_vec = np.array([self.alpha[i][t] for i, t in self.coords])
        self.c = np.array([1.0 - self.delta[i] for i in self.depts])
        self.a = np.array([self.a_band[i] for i in self.depts])
        self.b = np.array([self.b_band[i] for i in self.depts])
        self.delta_vec = np.array([self.delta[i] for i in self.depts])
        self.masks = [self.owner == k for k in range(self.n)]
        self.refresh()

    # -- parameter validation ------------------------------------------------
    def validate(self):
        for i in self.depts:
            c_i = 1.0 - self.delta[i]
            if not 0.0 < self.delta[i] < 1.0:
                raise ValueError(f"{i}: delta must lie in (0,1)")
            # Equation (2.6): the only restriction on the band.  The left
            # half makes K_{i,t_i} nonempty; the right half is equivalent to
            # c_i - b_i > -a_i and is what keeps K(s) inside [0, c_i] and
            # makes m_i strictly increasing.  The signs of a_i and b_i are
            # deliberately unrestricted (Remark 2.1).
            width = self.b_band[i] - self.a_band[i]
            if not 0.0 < width < c_i:
                raise ValueError(
                    f"{i}: require 0 < b_i - a_i < c_i (got {width})")
            if len(self.prob[i]) != len(self.W[i]):
                raise ValueError(f"{i}: probability/type mismatch")
            if not np.isclose(sum(self.prob[i]), 1.0):
                raise ValueError(f"{i}: type probabilities must sum to one")
            for t, _ in enumerate(self.W[i]):
                if not self.W[i][t] > 0:
                    raise ValueError(f"{i}: W must be positive")
                if not 0 < self.D[i][t] < self.W[i][t]:
                    raise ValueError(f"{i}: require 0 < D < W")
                if not self.alpha[i][t] > 0:
                    raise ValueError(f"{i}: alpha must be positive")

    def refresh(self):
        """Recompute exposure weights (needed after changing probabilities)."""
        self.EWD = np.array([
            sum(self.W[i][t] * self.D[i][t] * self.prob[i][t]
                for t in range(self.ntypes[i]))
            for i in self.depts])
        total = self.EWD.sum()
        self.omega = self.EWD / (total - self.EWD)
        self.cap_others = self.c.sum() - self.c
        self.phi_max = np.exp(self.omega * self.cap_others)
        self.band_span = np.exp(self.c - self.b) - np.exp(-self.a)

    def set_probabilities(self, dept, values):
        self.prob[dept] = list(values)
        for k, (i, t) in enumerate(self.coords):
            if i == dept:
                self.q[k] = self.prob[i][t]
        self.refresh()

    def set_band(self, a_value=None, b_value=None):
        if a_value is not None:
            for i in self.depts:
                self.a_band[i] = float(a_value)
        if b_value is not None:
            for i in self.depts:
                self.b_band[i] = float(b_value)
        self.validate()
        self.a = np.array([self.a_band[i] for i in self.depts])
        self.b = np.array([self.b_band[i] for i in self.depts])
        self.refresh()

    # -- packing helpers -----------------------------------------------------
    def to_dict(self, x):
        out = {}
        for k, (i, t) in enumerate(self.coords):
            out.setdefault(i, []).append(float(x[k]))
        return out

    def to_vector(self, state):
        return np.array([state[i][t] for i, t in self.coords], float)

    def uniform_state(self, value):
        return np.full(self.N, float(value))

    # -- aggregates ----------------------------------------------------------
    def mean_actions(self, x):
        """Ex-ante mean action of every department."""
        return np.array([float(self.q[m] @ x[m]) for m in self.masks])

    def others_mean(self, x):
        bar = self.mean_actions(x)
        return bar.sum() - bar

    # -- utilities -----------------------------------------------------------
    def interim_utility(self, x):
        """Interim expected utility of every department--type coordinate."""
        A = self.others_mean(x)[self.owner]
        breach = (1.0 - (x + A) / self.n) * (1.0 - x)
        cost = self.alpha_vec * (1.0 / np.sqrt(1.0 - x) - 1.0)
        return self.fW - self.Delta * breach - cost

    def exante_utility(self, x):
        """Ex-ante expected utility of every department."""
        u = self.interim_utility(x)
        return np.array([float(self.q[m] @ u[m]) for m in self.masks])

    def own_gradient(self, x):
        """d U_i(t_i; s) / d s_i(t_i) for every coordinate (analytic)."""
        A = self.others_mean(x)[self.owner]
        marginal = self.Delta * (1.0 + (1.0 - 2.0 * x - A) / self.n)
        return marginal - 0.5 * self.alpha_vec * (1.0 - x) ** (-1.5)

    def own_curvature(self, x):
        """d^2 U_i(t_i; s) / d s_i(t_i)^2 (analytic, strictly negative)."""
        return (-2.0 * self.Delta / self.n
                - 0.75 * self.alpha_vec * (1.0 - x) ** (-2.5))

    def pseudo_gradient(self, x):
        """F^N(s), the decentralized pseudo-gradient."""
        return -self.own_gradient(x)

    def exante_gradient(self, x):
        """Matrix G with G[i, k] = d barU_i / d s_k (analytic)."""
        grad = np.zeros((self.n, self.N))
        own = self.own_gradient(x)
        # R_i = sum_{t_i} q_i(t_i) Delta_i(t_i) (1 - s_i(t_i))
        R = np.array([float((self.q[m] * self.Delta[m]) @ (1.0 - x[m]))
                      for m in self.masks])
        for i in range(self.n):
            m = self.masks[i]
            grad[i, m] = self.q[m] * own[m]
            grad[i, ~m] = self.q[~m] * R[i] / self.n
        return grad

    def welfare(self, x):
        return float(self.exante_utility(x).sum())

    def welfare_gradient(self, x):
        return self.exante_gradient(x).sum(axis=0)

    def welfare_F(self, x):
        """F^O(s) = -grad W(s)."""
        return -self.welfare_gradient(x)

    # -- endogenous budget correspondence ------------------------------------
    def reference_budget(self, x):
        A = self.others_mean(x)
        phi = np.exp(self.omega * A)
        ratio = (phi - 1.0) / (self.phi_max - 1.0)
        return np.log(ratio * self.band_span + np.exp(-self.a))

    def box(self, x):
        """Lower/upper bounds of K(s), expanded to coordinates."""
        m = self.reference_budget(x)
        lo = (m + self.a)[self.owner]
        hi = (m + self.b)[self.owner]
        cap = self.c[self.owner]
        if np.any(lo < -1e-10) or np.any(hi > cap + 1e-10) or np.any(lo > hi):
            raise ValueError("endogenous box left the admissible action set")
        return lo, hi

    def project(self, x, reference=None):
        lo, hi = self.box(x if reference is None else reference)
        return np.clip(x, lo, hi)

    # -- bargaining objective ------------------------------------------------
    def surplus(self, x, disagreement):
        return self.exante_utility(x) - disagreement

    def log_nash(self, x, disagreement):
        """Sum_i log g_i(s), the exact objective of the manuscript.

        The function is defined on the strict-surplus region only and
        returns ``-inf`` elsewhere, so the solved problem coincides with the
        displayed one.  No smooth continuation below a positive floor is
        used anywhere in this code base.
        """
        g = self.surplus(x, disagreement)
        if np.any(g <= 0.0):
            return -np.inf
        return float(np.log(g).sum())

    def log_nash_gradient(self, x, disagreement):
        g = self.surplus(x, disagreement)
        if np.any(g <= 0.0):
            raise ValueError("log-bargaining gradient outside the surplus region")
        return (self.exante_gradient(x) / g[:, None]).sum(axis=0)

    def nash_product(self, x, disagreement):
        return float(np.prod(self.surplus(x, disagreement)))


# ---------------------------------------------------------------------------
# Decentralized equilibrium
# ---------------------------------------------------------------------------
def solve_bnse(model, rho=0.3, tol=1e-8, max_iter=20000, x0=None,
               record=False, record_trajectory=False):
    """Projected pseudo-gradient iteration s <- P_{K(s)}(s - rho F^N(s))."""
    x = model.uniform_state(0.1) if x0 is None else np.array(x0, float)
    history = []
    trajectory = [x.copy()] if record_trajectory else None
    residual = np.inf
    for iteration in range(max_iter):
        lo, hi = model.box(x)
        x_new = np.clip(x + rho * model.own_gradient(x), lo, hi)
        residual = float(np.max(np.abs(x_new - x)))
        if record:
            history.append(residual)
        x = x_new
        if record_trajectory:
            trajectory.append(x.copy())
        if residual <= rho * tol:
            break
    else:
        raise RuntimeError(
            f"BNSE iteration did not converge in {max_iter} iterations "
            f"(last step {residual:.3e})")
    out = {"x": x, "iterations": iteration + 1, "history": history}
    if record_trajectory:
        out["trajectory"] = [row.tolist() for row in trajectory]
    return out


def bnse_diagnostics(model, x, iterations, history, rho=0.3, tol=1e-8):
    lo, hi = model.box(x)
    box_violation = float(max(np.max(lo - x), np.max(x - hi), 0.0))
    natural = np.clip(x - model.pseudo_gradient(x), lo, hi)
    qvi_residual = float(np.max(np.abs(x - natural)))
    return {
        "iterations": iterations,
        "rho": rho,
        "natural_QVI_tolerance": tol,
        "final_step_residual_inf": float(history[-1]) if history else None,
        "max_box_violation": box_violation,
        "natural_QVI_residual_inf": qvi_residual,
        "feasible": bool(box_violation <= 1e-7),
        "solver_success": bool(qvi_residual <= tol and box_violation <= 1e-7),
    }


# ---------------------------------------------------------------------------
# Frozen-feasible-set programs (cooperative and centralized regimes)
# ---------------------------------------------------------------------------
def _slsqp_frozen(model, objective, gradient, bounds, constraints, x0,
                  ftol=1e-14, maxiter=1000):
    res = minimize(lambda z: -objective(z), x0, jac=lambda z: -gradient(z),
                   method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": maxiter, "ftol": ftol})
    return res


def _bounds_from(model, x_reference):
    lo, hi = model.box(x_reference)
    return list(zip(lo, hi)), lo, hi


def _ir_constraints(model, disagreement):
    cons = []
    for i in range(model.n):
        cons.append({
            "type": "ineq",
            "fun": (lambda z, i=i: float(
                model.exante_utility(z)[i] - disagreement[i])),
            "jac": (lambda z, i=i: model.exante_gradient(z)[i]),
        })
    return cons


def solve_frozen(model, kind, x_reference, disagreement=None, x0=None,
                 ftol=1e-14, maxiter=1000):
    """Solve one frozen-feasible-set subproblem with analytic derivatives."""
    bounds, lo, hi = _bounds_from(model, x_reference)
    start = np.clip(x_reference if x0 is None else x0, lo, hi)
    if kind == "welfare":
        objective = model.welfare
        gradient = model.welfare_gradient
        constraints = []
    elif kind == "bargaining":
        objective = lambda z: model.log_nash(z, disagreement)
        gradient = lambda z: model.log_nash_gradient(z, disagreement)
        constraints = _ir_constraints(model, disagreement)
    else:
        raise ValueError(f"unknown frozen problem '{kind}'")
    res = _slsqp_frozen(model, objective, gradient, bounds, constraints,
                        start, ftol=ftol, maxiter=maxiter)
    return res, np.array(bounds), objective, gradient, constraints


def solve_outer_fixed_point(model, kind, x_start, disagreement=None,
                            tol=1e-10, max_outer=200, label="inner",
                            ftol=1e-14):
    """Outer iteration on the moving feasible set."""
    x = np.array(x_start, float)
    residual = np.inf
    iterations = 0
    for outer in range(max_outer):
        res, bounds, objective, gradient, constraints = solve_frozen(
            model, kind, x, disagreement=disagreement, x0=x, ftol=ftol)
        _check_solver(model, res, bounds, constraints, label)
        x_new = res.x
        residual = float(np.max(np.abs(x_new - x)))
        x = x_new
        iterations = outer + 1
        if residual <= tol:
            break
    else:
        raise RuntimeError(
            f"{label}: outer fixed-point iteration did not converge in "
            f"{max_outer} iterations (last residual {residual:.3e})")
    return x, iterations, residual


def _check_solver(model, res, bounds, constraints, label):
    violation = _bound_violation(res.x, bounds)
    min_con = min([float(c["fun"](res.x)) for c in constraints], default=None)
    if (not res.success or not np.all(np.isfinite(res.x))
            or not np.isfinite(res.fun) or violation > 1e-8
            or (min_con is not None and min_con < -1e-8)):
        raise RuntimeError(
            f"{label}: SLSQP failed (status={res.status}, "
            f"message={res.message!s}, bound_violation={violation:.3e}, "
            f"min_constraint={min_con})")


def _bound_violation(x, bounds):
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return float(max(np.max(lo - x), np.max(x - hi), 0.0))


# ---------------------------------------------------------------------------
# Stationarity (KKT) diagnostics
# ---------------------------------------------------------------------------
def kkt_residual(model, x, bounds, objective_gradient, constraints,
                 active_tol=1e-5):
    """Infinity norm of the KKT stationarity residual at ``x``.

    The residual is
    ``|| grad(-objective)(x) - sum_a lambda_a grad c_a(x) ||_inf``
    where the sum runs over the constraints active within ``active_tol`` and
    the multipliers are the nonnegative least-squares solution of the
    resulting linear system.  All gradients are analytic.
    """
    grad_min = -objective_gradient(x)
    columns, names = [], []
    for k, (low, high) in enumerate(bounds):
        if x[k] - low <= active_tol:
            column = np.zeros_like(x)
            column[k] = 1.0
            columns.append(column)
            names.append(f"lower[{model.labels[k]}]")
        if high - x[k] <= active_tol:
            column = np.zeros_like(x)
            column[k] = -1.0
            columns.append(column)
            names.append(f"upper[{model.labels[k]}]")
    for number, constraint in enumerate(constraints):
        if float(constraint["fun"](x)) <= active_tol:
            columns.append(np.array(constraint["jac"](x), float))
            names.append(f"IR[{number}]")
    if columns:
        matrix = np.column_stack(columns)
        multipliers, _ = nnls(matrix, grad_min)
        stationarity = grad_min - matrix @ multipliers
    else:
        multipliers = np.array([])
        stationarity = grad_min
    scale = max(float(np.max(np.abs(grad_min))), 1.0e-16)
    absolute = float(np.max(np.abs(stationarity)))
    return {
        "KKT_stationarity_residual_inf": absolute,
        "objective_gradient_norm_inf": scale,
        "KKT_relative_residual": absolute / scale,
        "active_constraints": names,
        "active_multipliers": [float(v) for v in multipliers],
        "active_tolerance": active_tol,
    }


def finite_difference_audit(function, analytic, x, steps=(1e-4, 1e-5, 1e-6, 1e-7)):
    """Maximum deviation between analytic and central-difference gradients."""
    reference = np.array(analytic(x), float)
    report = {}
    for step in steps:
        approx = np.zeros_like(reference)
        for k in range(len(x)):
            forward = x.copy(); forward[k] += step
            backward = x.copy(); backward[k] -= step
            approx[k] = (function(forward) - function(backward)) / (2 * step)
        report[f"{step:g}"] = float(np.max(np.abs(approx - reference)))
    return report


# ---------------------------------------------------------------------------
# Multi-start driver
# ---------------------------------------------------------------------------
def multistart(model, kind, starts, disagreement=None, label="inner",
               tol=1e-10, ftol=1e-14):
    """Run every start, keep the best validated fixed point."""
    runs, candidates = [], []
    for number, start in enumerate(starts):
        try:
            x, iterations, residual = solve_outer_fixed_point(
                model, kind, start, disagreement=disagreement, tol=tol,
                label=f"{label} start {number}", ftol=ftol)
            value = (model.welfare(x) if kind == "welfare"
                     else model.log_nash(x, disagreement))
            runs.append({"start": number, "success": True,
                         "objective": float(value),
                         "outer_iterations": iterations,
                         "outer_residual_inf": residual,
                         "x": x.tolist()})
            candidates.append((float(value), x, iterations, residual))
        except RuntimeError as error:
            runs.append({"start": number, "success": False,
                         "error": str(error)})
    if not candidates:
        raise RuntimeError(f"{label}: every start failed: {runs}")
    best = max(candidates, key=lambda item: item[0])
    spread = max(float(np.max(np.abs(item[1] - best[1])))
                 for item in candidates)
    objective_spread = float(max(item[0] for item in candidates)
                             - min(item[0] for item in candidates))
    summary = {
        "starts": len(starts),
        "successful_starts": len(candidates),
        "solution_spread_inf": spread,
        "objective_spread": objective_spread,
        "runs": runs,
    }
    return best[1], best[2], best[3], summary


def coordinated_diagnostics(model, kind, x, disagreement=None,
                            outer_residual=None, outer_iterations=None,
                            outer_tolerance=1e-6):
    """Feasibility, fixed-point, stationarity and surplus diagnostics."""
    res, bounds, objective, gradient, constraints = solve_frozen(
        model, kind, x, disagreement=disagreement, x0=x)
    _check_solver(model, res, bounds, constraints, kind)
    recomputed = float(np.max(np.abs(res.x - x)))
    report = {
        "outer_iterations": outer_iterations,
        "outer_step_residual_inf": (None if outer_residual is None
                                    else float(outer_residual)),
        "frozen_set_verification_residual_inf": recomputed,
        "outer_fixed_point_residual_inf": (
            recomputed if outer_residual is None
            else max(recomputed, float(outer_residual))),
        "outer_fixed_point_tolerance": outer_tolerance,
        "max_box_violation": _bound_violation(x, bounds),
        "SLSQP_status": int(res.status),
        "SLSQP_message": str(res.message),
        "objective": float(objective(x)),
    }
    report.update(kkt_residual(model, x, bounds, gradient, constraints))
    report["finite_difference_audit_of_analytic_gradient"] = (
        finite_difference_audit(objective, gradient, x))
    report["KKT_residual_finite_difference_variant"] = _fd_kkt_variants(
        model, x, bounds, objective, constraints)
    if disagreement is not None:
        g = model.surplus(x, disagreement)
        report["IR_surplus_by_department"] = {
            model.depts[i]: float(g[i]) for i in range(model.n)}
        report["min_IR_surplus"] = float(np.min(g))
        report["log_objective_is_exact"] = True
        report["feasible"] = bool(report["max_box_violation"] <= 1e-7
                                  and np.min(g) >= -1e-10)
    else:
        report["feasible"] = bool(report["max_box_violation"] <= 1e-7)
    report["outer_fixed_point_converged"] = bool(
        report["outer_fixed_point_residual_inf"] <= outer_tolerance)
    return report


def _fd_kkt_variants(model, x, bounds, objective, constraints,
                     steps=(1e-4, 1e-5, 1e-6, 1e-7)):
    """KKT residual recomputed with finite-difference gradients."""
    out = {}
    for step in steps:
        def fd_gradient(z, step=step):
            grad = np.zeros_like(z)
            for k in range(len(z)):
                forward = z.copy(); forward[k] += step
                backward = z.copy(); backward[k] -= step
                grad[k] = (objective(forward) - objective(backward)) / (2 * step)
            return grad
        fd_constraints = []
        for constraint in constraints:
            fd_constraints.append({
                "fun": constraint["fun"],
                "jac": (lambda z, c=constraint, step=step: _fd_vector(
                    c["fun"], z, step)),
            })
        out[f"{step:g}"] = kkt_residual(
            model, x, bounds, fd_gradient,
            fd_constraints)["KKT_stationarity_residual_inf"]
    return out


def _fd_vector(function, x, step):
    grad = np.zeros_like(x)
    for k in range(len(x)):
        forward = x.copy(); forward[k] += step
        backward = x.copy(); backward[k] -= step
        grad[k] = (function(forward) - function(backward)) / (2 * step)
    return grad


# ---------------------------------------------------------------------------
# Analytical certificates (Section 3)
# ---------------------------------------------------------------------------
def certificates(model, rho_test=0.3):
    """Full player--type monotonicity, Lipschitz and contraction constants."""
    q, Delta, alpha = model.q, model.Delta, model.alpha_vec
    delta_owner = model.delta_vec[model.owner]
    d_minus_N = 2.0 * Delta / model.n + 0.75 * alpha
    d_plus_N = 2.0 * Delta / model.n + 0.75 * alpha * delta_owner ** (-2.5)
    d_minus_O = q * d_minus_N
    d_plus_O = q * d_plus_N

    different = model.owner[:, None] != model.owner[None, :]
    B_N = np.diag(d_plus_N)
    off_N = (Delta[:, None] * q[None, :]) / model.n
    B_N[different] = off_N[different]

    B_O = np.diag(d_plus_O)
    pair = q[:, None] * q[None, :]
    off_O = pair * (Delta[:, None] + Delta[None, :]) / model.n
    B_O[different] = off_O[different]

    radii_N = np.sum((B_N + B_N.T) / 2.0, axis=1) - d_plus_N
    radii_O = np.sum(B_O, axis=1) - d_plus_O
    mu_N = float(np.min(d_minus_N - radii_N))
    mu_O = float(np.min(d_minus_O - radii_O))
    L_N = float(np.linalg.norm(B_N, 2))
    L_O = float(np.linalg.norm(B_O, 2))

    gamma = np.maximum(
        model.band_span * model.omega / ((model.phi_max - 1.0)
                                         * np.exp(-model.a)),
        model.band_span * model.omega * model.phi_max
        / ((model.phi_max - 1.0) * np.exp(model.c - model.b)))
    budget_bound = np.zeros((model.N, model.N))
    gamma_owner = gamma[model.owner]
    budget_bound[different] = (gamma_owner[:, None] * q[None, :])[different]
    ell_M = float(np.linalg.norm(budget_bound, "fro"))

    def contraction(mu, lipschitz):
        discriminant = mu ** 2 - 4.0 * lipschitz ** 2 * ell_M * (1.0 - ell_M)
        holds = bool(mu > 0 and ell_M < 0.5 and discriminant > 0)
        interval = None
        if holds:
            root = np.sqrt(discriminant)
            interval = [(mu - root) / lipschitz ** 2,
                        (mu + root) / lipschitz ** 2]
        best_bound = (2.0 * ell_M
                      + np.sqrt(max(0.0, 1.0 - (mu / lipschitz) ** 2))
                      if mu > 0 and lipschitz > 0 else None)
        bound_at_rho = (2.0 * ell_M + np.sqrt(max(
            0.0, 1.0 - 2.0 * rho_test * mu + rho_test ** 2 * lipschitz ** 2))
            if lipschitz > 0 else None)
        return {
            "mu_positive": bool(mu > 0),
            "ell_M_below_one_half": bool(ell_M < 0.5),
            "discriminant": float(discriminant),
            "holds": holds,
            "rho_interval": interval,
            f"rho_{rho_test}_admissible": bool(
                interval is not None and interval[0] < rho_test < interval[1]),
            f"contraction_bound_at_rho_{rho_test}": (
                float(bound_at_rho) if bound_at_rho is not None else None),
            "best_contraction_bound": (
                float(best_bound) if best_bound is not None else None),
        }

    return {
        "coordinates": model.labels,
        "Delta": Delta.tolist(),
        "d_minus_N": d_minus_N.tolist(),
        "d_plus_N": d_plus_N.tolist(),
        "d_minus_O": d_minus_O.tolist(),
        "d_plus_O": d_plus_O.tolist(),
        "mu_N": mu_N,
        "L_N": L_N,
        "mu_O": mu_O,
        "L_O": L_O,
        "strong_concavity_O": bool(mu_O > 0),
        "omega": model.omega.tolist(),
        "gamma": {model.depts[i]: float(gamma[i]) for i in range(model.n)},
        "band_translation_span": (model.c - model.b + model.a).tolist(),
        "ell_M": ell_M,
        "BNSE_contraction_test": contraction(mu_N, L_N),
        "BNSO_contraction_test": contraction(mu_O, L_O),
    }


# ---------------------------------------------------------------------------
# Economic reporting
# ---------------------------------------------------------------------------
def economic_summary(model, x):
    """Ex-ante utility, expected monetary loss and protection cost."""
    A = model.others_mean(x)[model.owner]
    breach = (1.0 - (x + A) / model.n) * (1.0 - x)
    loss = np.array([model.D[i][t] for i, t in model.coords])
    cost = model.alpha_vec * (1.0 / np.sqrt(1.0 - x) - 1.0)
    utility = model.exante_utility(x)
    out = {}
    for i in range(model.n):
        m = model.masks[i]
        out[model.depts[i]] = {
            "exante_utility": float(utility[i]),
            "expected_breach_probability": float(model.q[m] @ breach[m]),
            "expected_monetary_loss_musd": float(
                model.q[m] @ (breach[m] * loss[m])),
            "expected_protection_cost_utils": float(model.q[m] @ cost[m]),
            "mean_action": float(model.q[m] @ x[m]),
        }
    return out


def comparison_table(model, baseline, alternative):
    """Relative changes of the economic summary between two regimes."""
    base = economic_summary(model, baseline)
    alt = economic_summary(model, alternative)
    out = {}
    for dept in model.depts:
        row = {}
        for key in base[dept]:
            b, a = base[dept][key], alt[dept][key]
            row[key] = {"baseline": b, "value": a, "change": a - b,
                        "percent_change": (100.0 * (a - b) / abs(b)
                                           if abs(b) > 0 else None)}
        out[dept] = row
    return out

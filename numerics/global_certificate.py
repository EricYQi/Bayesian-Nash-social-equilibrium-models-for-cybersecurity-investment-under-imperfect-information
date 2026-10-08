"""Deterministic global certificate for the frozen bargaining problem.

Definition 2.4 of the manuscript requires the bargaining point to be a
*global* maximizer of the Nash product on its own frozen feasible set
``C(s) = K(s) cap R``.  Local nonlinear programming cannot establish that.
This module supplies a deterministic branch-and-bound certificate that
exploits the following structure of the model.

Ex-ante utility depends on the other departments only through the aggregate
mean action ``A_i = sum_{j != i} bar s_j``,

    barU_i(s) = sum_{t_i} q_i(t_i) [ f(W_i(t_i)) - Delta_i(t_i)
                (1 - (s_i(t_i) + A_i)/n)(1 - s_i(t_i))
                - alpha_i(t_i)((1 - s_i(t_i))^{-1/2} - 1) ],

is strictly increasing in ``A_i`` and, for fixed ``A_i``, is strictly
concave and separable in department ``i``'s own coordinates.  Therefore, on
a cell ``V = prod_i [v_i^-, v_i^+]`` of aggregate mean actions,

    barU_i(s) <= tilde U_i^V(s) := barU_i(s)|_{A_i = sum_{j != i} v_j^+},

with ``tilde U_i^V`` concave, so that

    max { Psi(s) : s in C, bar s in V }
        <= max { sum_i log(tilde U_i^V(s) - d_i)
                 : s in K(hat s), bar s in V }  =:  UB(V),

where ``Psi = sum_i log(barU_i - d_i)``.  The right-hand side is a concave
program that, once the aggregates are frozen, *separates completely across
departments*: its objective is a sum of logarithms of disjoint groups of
variables and its constraints (own box, own mean interval) are department
specific.  Its exact optimal value is therefore obtained by maximizing each
``tilde U_i^V`` on its own, which is a strictly concave separable program
with one linear interval constraint and is solved by a monotone scalar
search for the multiplier of that constraint together with one scalar
root-find per coordinate.  Branching on the
``n``-dimensional aggregate box drives the relaxation error to zero, since
the bound becomes exact when the cell degenerates to a point.  The routine
returns the certified optimality gap of the reported point, in log-Nash and
relative Nash-product terms.
"""

from __future__ import annotations

import heapq
import time

import numpy as np

__all__ = ["certify_bargaining_point"]


# ---------------------------------------------------------------------------
# Relaxed (aggregate-frozen) utilities
# ---------------------------------------------------------------------------
def _relaxed_utilities(model, x, A):
    A_coord = A[model.owner]
    breach = (1.0 - (x + A_coord) / model.n) * (1.0 - x)
    cost = model.alpha_vec * (1.0 / np.sqrt(1.0 - x) - 1.0)
    values = model.fW - model.Delta * breach - cost
    return np.array([float(model.q[m] @ values[m]) for m in model.masks])


def _coordinate_maxima(model, lo, hi, A, shadow=0.0):
    """Maximize every separable coordinate term of ``tilde U_i^V``.

    With ``shadow`` equal to a Lagrange multiplier ``lambda`` for the mean
    constraint, this maximizes ``term_k(x) - lambda x`` instead.  The scalar
    first-order conditions are solved simultaneously for all coordinates by
    a bracketed bisection followed by safeguarded Newton steps; the terms
    are strictly concave, so each equation has a unique solution.
    """
    A_coord = A[model.owner]
    Delta, alpha, n = model.Delta, model.alpha_vec, model.n

    def derivative(x):
        return (Delta * (1.0 + (1.0 - 2.0 * x - A_coord) / n)
                - 0.5 * alpha * (1.0 - x) ** (-1.5) - shadow)

    def curvature(x):
        return -2.0 * Delta / n - 0.75 * alpha * (1.0 - x) ** (-2.5)

    at_low = derivative(lo)
    at_high = derivative(hi)
    left, right = lo.copy(), hi.copy()
    for _ in range(12):
        middle = 0.5 * (left + right)
        positive = derivative(middle) > 0.0
        left = np.where(positive, middle, left)
        right = np.where(positive, right, middle)
    x = 0.5 * (left + right)
    for _ in range(3):
        step = derivative(x) / curvature(x)
        x = np.clip(x - step, left, right)
    return np.where(at_low <= 0.0, lo, np.where(at_high >= 0.0, hi, x))


def _department_argmax(model, i, lo, hi, A, cell_low, cell_high):
    """Exact maximizer of ``tilde U_i^V`` on box cap {mean in cell}.

    The department problem is a strictly concave separable maximization with
    one linear interval constraint; it is solved by bisection on the
    multiplier of the mean constraint.
    """
    m = model.masks[i]
    q = model.q[m]

    def mean_at(shadow):
        x = _coordinate_maxima(model, lo, hi, A, shadow=shadow)[m]
        return float(q @ x), x

    value, x = mean_at(0.0)
    if cell_low <= value <= cell_high:
        return x
    target = cell_high if value > cell_high else cell_low
    low_shadow, high_shadow = 0.0, 0.0
    if value > target:
        high_shadow = 1.0
        while mean_at(high_shadow)[0] > target and high_shadow < 1e6:
            high_shadow *= 4.0
    else:
        low_shadow = -1.0
        while mean_at(low_shadow)[0] < target and low_shadow > -1e6:
            low_shadow *= 4.0
    for _ in range(45):
        middle = 0.5 * (low_shadow + high_shadow)
        current, x = mean_at(middle)
        if current > target:
            low_shadow = middle
        else:
            high_shadow = middle
        if abs(current - target) <= 1e-13:
            break
    return x


# ---------------------------------------------------------------------------
# Upper bound on one aggregate cell
# ---------------------------------------------------------------------------
def _cell_bound(model, lo, hi, cell_low, cell_high, disagreement):
    """Exact optimal value of the concave relaxation on one aggregate cell.

    Once the aggregates are frozen at the cell's upper corner, the relaxed
    problem separates completely across departments: the objective is
    ``sum_i log(tilde U_i^V - d_i)`` with disjoint variables, and the
    constraints (own box, own mean interval) are department specific.  The
    exact relaxed optimum is therefore obtained by maximizing every
    ``tilde U_i^V`` separately, which reduces to one monotone scalar search
    for the multiplier of the mean constraint.
    """
    A_upper = cell_high.sum() - cell_high
    x = np.empty(model.N)
    for i in range(model.n):
        x[model.masks[i]] = _department_argmax(
            model, i, lo, hi, A_upper, cell_low[i], cell_high[i])
    surplus = _relaxed_utilities(model, x, A_upper) - disagreement
    if np.any(surplus <= 0.0):
        return -np.inf
    return float(np.log(surplus).sum())


# ---------------------------------------------------------------------------
# Branch and bound
# ---------------------------------------------------------------------------
def certify_bargaining_point(model, x_hat, disagreement, tolerance=1e-3,
                             max_nodes=400000, time_limit=900.0,
                             verbose=False):
    """Certify global optimality of ``x_hat`` on its own frozen feasible set.

    Returns the incumbent value, a certified upper bound for the frozen-set
    global maximum of the Nash product, and the resulting optimality gap.
    """
    lo, hi = model.box(x_hat)
    incumbent_surplus = model.surplus(x_hat, disagreement)
    if np.any(incumbent_surplus <= 0.0):
        raise ValueError("incumbent must have strictly positive surplus")
    incumbent = float(np.log(incumbent_surplus).sum())
    cutoff = incumbent + tolerance

    root_low = np.array([lo[model.masks[i]][0] for i in range(model.n)])
    root_high = np.array([hi[model.masks[i]][0] for i in range(model.n)])

    start_time = time.time()
    counter = 0
    heap = [(-np.inf, counter, root_low.copy(), root_high.copy())]
    nodes = 0
    prunes = 0
    upper_bound = incumbent
    exhausted = True
    smallest_cell = float(np.max(root_high - root_low))

    while heap:
        negative_bound, _, low, high = heapq.heappop(heap)
        bound = -negative_bound
        if bound <= cutoff:
            upper_bound = max(bound, incumbent)
            break
        if nodes >= max_nodes or time.time() - start_time > time_limit:
            upper_bound = bound
            exhausted = False
            break
        nodes += 1

        cell_bound = _cell_bound(model, lo, hi, low, high, disagreement)
        if cell_bound <= cutoff:
            prunes += 1
            continue

        widths = high - low
        axis = int(np.argmax(widths))
        smallest_cell = min(smallest_cell, float(widths[axis]))
        if widths[axis] <= 1e-11:
            upper_bound = cell_bound
            exhausted = False
            break
        middle = 0.5 * (low[axis] + high[axis])
        left_low, left_high = low.copy(), high.copy()
        left_high[axis] = middle
        right_low, right_high = low.copy(), high.copy()
        right_low[axis] = middle
        for child_low, child_high in ((left_low, left_high),
                                      (right_low, right_high)):
            counter += 1
            heapq.heappush(heap, (-cell_bound, counter, child_low, child_high))
        if verbose and nodes % 2000 == 0:
            print(f"  nodes={nodes} open={len(heap)} bound={cell_bound:.6f} "
                  f"incumbent={incumbent:.6f} "
                  f"cell={float(widths[axis]):.2e}")

    if not heap and exhausted:
        upper_bound = cutoff
    upper_bound = max(upper_bound, incumbent)
    gap = float(upper_bound - incumbent)
    return {
        "incumbent_log_nash": incumbent,
        "incumbent_nash_product": float(np.prod(incumbent_surplus)),
        "certified_upper_bound_log_nash": float(upper_bound),
        "certified_gap_log_nash": gap,
        "certified_relative_gap_nash_product": float(np.expm1(gap)),
        "target_tolerance": tolerance,
        "certified": bool(gap <= tolerance * (1.0 + 1e-9)),
        "search_exhausted": bool(exhausted),
        "nodes_explored": nodes,
        "pruned_nodes": prunes,
        "open_nodes": len(heap),
        "smallest_cell_width": smallest_cell,
        "seconds": float(time.time() - start_time),
        "aggregate_box": [root_low.tolist(), root_high.tolist()],
    }

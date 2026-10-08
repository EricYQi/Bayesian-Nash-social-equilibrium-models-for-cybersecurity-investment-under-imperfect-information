"""Figures for the three-department illustration and the robustness study.

Reads ``results.json`` and ``results_robustness.json`` and writes

* ``convergence_extended.pdf`` (BNSE convergence; diagnostic only, not
  included in the manuscript);
* ``robustness.pdf``/``../Fig4.pdf`` (effect of the admissible translation
  on the equilibrium and on the moving-set constant; Figure 4 of the paper);
* ``sensitivity_extended.pdf``/``../Fig5.pdf`` (type-probability
  sensitivity; Figure 5 of the paper).
"""

import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

with open("results.json") as handle:
    data = json.load(handle)


def save(figure, name, figure_number=None):
    figure.savefig(name)
    if figure_number is not None:
        shutil.copyfile(name, f"../Fig{figure_number}.pdf")
    plt.close(figure)


# ---- Diagnostic: convergence of the projected iteration -----------------
history = data["convergence_BNSE"]
figure, axis = plt.subplots(figsize=(5.6, 3.8))
axis.semilogy(range(1, len(history) + 1), history, marker="o", markersize=2,
              linewidth=1)
axis.set_xlabel(r"iteration $k$")
axis.set_ylabel(r"$\|s^{k+1}-s^{k}\|_\infty$ (log scale)")
axis.set_title("Convergence of the projected BNSE iteration")
axis.grid(True, which="both", linestyle=":", linewidth=0.5)
figure.tight_layout()
save(figure, "convergence_extended.pdf")

# ---- Figure 5: sensitivity to the type probabilities --------------------
sensitivity = data["sensitivity"]
probability = [row["p_A1"] for row in sensitivity]
series = [("s_A1", "o", r"$s_A(t_A^1)$"), ("s_A2", "s", r"$s_A(t_A^2)$"),
          ("s_B", "^", r"$s_B(t_B)$"), ("s_C1", "d", r"$s_C(t_C^1)$"),
          ("s_C2", "v", r"$s_C(t_C^2)$")]
figure, axis = plt.subplots(figsize=(5.6, 3.8))
for key, marker, label in series:
    axis.plot(probability, [row[key] for row in sensitivity], marker=marker,
              markersize=3, label=label)
axis.set_xlabel(r"$q_A(t_A^1)$ (probability of the low-loss type)")
axis.set_ylabel("BNSE security level")
axis.set_title("Sensitivity of the BNSE to the type probabilities")
axis.legend(fontsize=8, ncol=2)
axis.grid(True, linestyle=":", linewidth=0.5)
figure.tight_layout()
save(figure, "sensitivity_extended.pdf", 5)

# ---- Figure 4: robustness to the admissible translation -----------------
if os.path.exists("results_robustness.json"):
    with open("results_robustness.json") as handle:
        robust = json.load(handle)

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    panels = [("two_department_sweep", "two_department_threshold",
               ["A", "B"], "Two departments"),
              ("three_department_sweep", "three_department_threshold",
               ["A", "B", "C"], "Three departments")]
    for axis, (sweep_key, threshold_key, names, title) in zip(axes, panels):
        rows = sorted(robust[sweep_key],
                      key=lambda row: row["budget_translation_span"])
        tau = [row["budget_translation_span"] for row in rows]
        for index, name in enumerate(names):
            axis.plot(tau, [row["BNSE_mean_actions"][index] for row in rows],
                      marker="o", markersize=3,
                      label=fr"$\bar s_{name}$ (BNSE)")
        threshold = robust.get(threshold_key)
        if threshold:
            axis.axvline(threshold["largest_certified_translation"],
                         color="k", linestyle="--", linewidth=1,
                         label="uniqueness certificate limit")
        axis.set_xlabel(r"admissible translation $\tau$")
        axis.set_ylabel("ex-ante mean action")
        axis.set_title(title)
        axis.grid(True, linestyle=":", linewidth=0.5)
        twin = axis.twinx()
        twin.plot(tau, [row["ell_M"] for row in rows], color="gray",
                  linestyle=":", linewidth=1.2)
        twin.set_ylabel(r"$\ell_M$", color="gray")
        twin.tick_params(axis="y", colors="gray")
        axis.legend(fontsize=7, loc="center right")
    figure.suptitle(r"Effect of the admissible translation $\tau$ "
                    r"(dotted grey: $\ell_M$)")
    figure.tight_layout()
    save(figure, "robustness.pdf", 4)
    print("three-department and robustness figures written (Fig4, Fig5)")
else:
    print("type-probability figure written (Fig5); "
          "run robustness.py for Fig4")

print("BNSE iterations:", len(history))
for regime in ("BNSE", "BNSB", "BNSO"):
    print(regime, data["results"][regime]["iters"])

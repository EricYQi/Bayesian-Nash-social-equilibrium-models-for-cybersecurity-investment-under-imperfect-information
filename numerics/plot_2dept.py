"""Figures for the two-department illustration.

Reads ``results_2dept.json`` and writes

* ``InitialValuetoBNSE_2dept.pdf`` (convergence from several starting
  profiles; diagnostic only, not included in the manuscript);
* ``x1_2dept.pdf``/``../Fig1.pdf``, ``x2_2dept.pdf``/``../Fig2.pdf`` and
  ``x3_2dept.pdf``/``../Fig3.pdf`` (sensitivity of the BNSE to the band
  endpoints; Figures 1--3 of the paper).

Only band pairs satisfying the admissibility condition ``c_i - b_i > -a_i``
appear in the plots; the exact grids are stored in the JSON file under
``sensitivity_grids``.
"""

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

with open("results_2dept.json") as handle:
    data = json.load(handle)


def save(figure, name, figure_number=None):
    figure.savefig(name)
    if figure_number is not None:
        shutil.copyfile(name, f"../Fig{figure_number}.pdf")
    plt.close(figure)


# ---- Diagnostic: effect of the starting profile on convergence -----------
init_effects = data["init_effects"]
figure, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
labels = [("A", 0, r"$s_A(t_A^1)$"), ("A", 1, r"$s_A(t_A^2)$"),
          ("B", 0, r"$s_B(t_B)$")]
for axis, (dept, t, ylabel) in zip(axes, labels):
    for effect in init_effects:
        path = [row[dept][t] for row in effect["trajectory"]]
        axis.plot(range(len(path)), path, linewidth=1.2,
                  label=f"$s^0={effect['s0']}$")
    axis.set_xlabel(r"iteration $k$")
    axis.set_ylabel(ylabel)
    axis.grid(True, linestyle=":", linewidth=0.5)
axes[0].legend(fontsize=7, ncol=2, loc="lower right")
figure.suptitle("BNSE: effect of the starting profile on the iteration")
figure.tight_layout()
save(figure, "InitialValuetoBNSE_2dept.pdf")

# ---- Figures 1-3: sensitivity of the BNSE to the band endpoints ----------
panels = [("sA1", r"$s_A(t_A^1)$", "x1_2dept.pdf", 1),
          ("sA2", r"$s_A(t_A^2)$", "x2_2dept.pdf", 2),
          ("sB", r"$s_B(t_B)$", "x3_2dept.pdf", 3)]

for key, ylabel, name, number in panels:
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))

    axis = axes[0]
    for b_value, rows in data["sens_a"].items():
        axis.plot([row["a"] for row in rows], [row[key] for row in rows],
                  marker="o", markersize=3, label=f"$b={b_value}$")
    axis.set_xlabel(r"lower band endpoint $a$")
    axis.set_ylabel(ylabel)
    axis.set_title(r"varying $a$ (fixed $b$)")
    axis.legend(fontsize=7)
    axis.grid(True, linestyle=":", linewidth=0.5)

    axis = axes[1]
    for a_value, rows in data["sens_b"].items():
        axis.plot([row["b"] for row in rows], [row[key] for row in rows],
                  marker="s", markersize=3, label=f"$a={a_value}$")
    axis.set_xlabel(r"upper band endpoint $b$")
    axis.set_ylabel(ylabel)
    axis.set_title(r"varying $b$ (fixed $a$)")
    axis.legend(fontsize=7)
    axis.grid(True, linestyle=":", linewidth=0.5)

    figure.suptitle(f"Influence of the band endpoints on {ylabel}")
    figure.tight_layout()
    save(figure, name, number)

print("two-department figures written (Fig1-Fig3)")

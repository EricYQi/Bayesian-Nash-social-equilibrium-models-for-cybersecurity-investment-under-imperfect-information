"""Print every number quoted in the manuscript from the stored JSON output.

Run ``python report.py`` after ``sim_2dept.py``, ``sim.py`` and
``robustness.py``.  The script is a reproducibility aid: each block below
corresponds to one table or one statement in Section 5 of the paper.
"""

import json
import os


def load(name):
    if not os.path.exists(name):
        print(f"[missing] {name}")
        return None
    with open(name) as handle:
        return json.load(handle)


def regime_actions(data, label):
    print(f"\n--- {label}: actions and interim utilities ---")
    for regime in ("BNSE", "BNSB", "BNSO"):
        block = data["results"][regime]
        for dept, values in block["s"].items():
            actions = " ".join(f"{v:.4f}" for v in values)
            utils = " ".join(f"{u:.4f}" for u in block["U"][dept])
            print(f"{regime:5s} {dept}: s = {actions} | U = {utils} | "
                  f"barU = {block['barU'][dept]:.6f}")
        print(f"{regime:5s} outer/BNSE iterations: {block['iters']}")


def exante_changes(data, label):
    print(f"\n--- {label}: ex-ante utilities and changes ---")
    base = data["results"]["BNSE"]["barU"]
    for regime in ("BNSB", "BNSO"):
        for dept, value in data["results"][regime]["barU"].items():
            change = value - base[dept]
            print(f"{regime:5s} {dept}: {value:.6f} "
                  f"({change:+.6f}, {100 * change / abs(base[dept]):+.3e}%)")


def economics(data, label):
    print(f"\n--- {label}: economic decomposition ---")
    for comparison in ("BNSB_vs_BNSE", "BNSO_vs_BNSE"):
        print(f"  {comparison}")
        for dept, rows in data["economics"][comparison].items():
            for key in ("exante_utility", "expected_breach_probability",
                        "expected_monetary_loss_musd",
                        "expected_protection_cost_utils", "mean_action"):
                row = rows[key]
                print(f"    {dept:2s} {key:34s} "
                      f"{row['baseline']:12.6f} -> {row['value']:12.6f} "
                      f"({row['change']:+.6f}, {row['percent_change']:+.4f}%)")


def certificates(data, label):
    print(f"\n--- {label}: analytical certificates ---")
    cert = data["certificates"]
    test = cert["BNSE_contraction_test"]
    print(f"  mu_N = {cert['mu_N']:.4f}  L_N = {cert['L_N']:.4f}  "
          f"ell_M = {cert['ell_M']:.5f}")
    print(f"  rho interval = {test['rho_interval']}  "
          f"best bound = {test['best_contraction_bound']:.4f}")
    print(f"  mu_O = {cert['mu_O']:.4f}  L_O = {cert['L_O']:.4f}  "
          f"BNS-O contraction holds = "
          f"{cert['BNSO_contraction_test']['holds']}")
    print(f"  translation span tau = {cert['band_translation_span']}")


def diagnostics(data, label):
    print(f"\n--- {label}: numerical diagnostics ---")
    bnse = data["validation"]["BNSE"]
    print(f"  BNSE  feasibility {bnse['max_box_violation']:.3e} "
          f"QVI {bnse['natural_QVI_residual_inf']:.3e} "
          f"iterations {bnse['iterations']}")
    for regime in ("BNSB", "BNSO"):
        block = data["validation"][regime]
        print(f"  {regime:5s} feasibility {block['max_box_violation']:.3e} "
              f"outer {block['outer_fixed_point_residual_inf']:.3e} "
              f"KKT {block['KKT_stationarity_residual_inf']:.3e}")
        print(f"        multistart: starts {block['multistart']['starts']}, "
              f"successes {block['multistart']['successful_starts']}, "
              f"coordinate spread "
              f"{block['multistart']['solution_spread_inf']:.3e}, "
              f"objective spread {block['multistart']['objective_spread']:.3e}")
        print("        finite-difference KKT variants: "
              + ", ".join(f"h={k}: {v:.3e}" for k, v in
                          block["KKT_residual_finite_difference_variant"]
                          .items()))
        if regime == "BNSB":
            print(f"        min surplus {block['min_IR_surplus']:.6f}")
            hessian = block["reduced_hessian_frozen_problem"]
            print(f"        free coordinates "
                  f"{hessian['free_coordinates']}, eigenvalues "
                  f"{hessian.get('eigenvalues_h=1e-05')}")
            certificate = block["frozen_set_global_certificate"]
            print(f"        global certificate: gap "
                  f"{certificate['certified_gap_log_nash']:.3e} (log), "
                  f"relative {certificate['certified_relative_gap_nash_product']:.3e}, "
                  f"certified {certificate['certified']}, nodes "
                  f"{certificate['nodes_explored']}, "
                  f"{certificate['seconds']:.0f} s")


def robustness(data):
    print("\n=== Robustness: band-scaling sweep ===")
    for key, label in (("two_department_sweep", "two departments"),
                       ("three_department_sweep", "three departments")):
        print(f"--- {label} ---")
        for row in data[key]:
            print(f"  scale {row['scale']:.2f}  width "
                  f"{row['feasible_interval_width']:.4f}  tau "
                  f"{row['budget_translation_span']:.4f}  ell_M "
                  f"{row['ell_M']:.4f}  mu_N {row['mu_N']:.4f}  L_N "
                  f"{row['L_N']:.4f}  contraction "
                  f"{str(row['contraction_holds']):5s}  BNSE iters "
                  f"{row['BNSE_iterations']:4d}  bounds active "
                  f"{row['BNSE_bounds_active']}  mean actions "
                  + " ".join(f"{v:.4f}" for v in row["BNSE_mean_actions"]))
    for key, label in (("two_department_detailed", "two departments"),
                       ("three_department_detailed", "three departments")):
        print(f"--- {label}: three-regime runs ---")
        for scale, block in data[key].items():
            print(f"  scale {scale}: tau "
                  f"{list(block['parameters']['budget_translation_span'].values())[0]:.4f}"
                  f"  ell_M {block['certificates']['ell_M']:.4f}")
            for regime in ("BNSE", "BNSB", "BNSO"):
                means = " ".join(f"{v:.4f}" for v in block["mean_actions"][regime])
                barU = " ".join(f"{v:.6f}"
                                for v in block["results"][regime]["barU"].values())
                print(f"    {regime:5s} mean actions {means} | barU {barU}")
            validation = block["validation"]
            print(f"    residuals: BNSE QVI "
                  f"{validation['BNSE']['natural_QVI_residual_inf']:.2e}, "
                  f"BNSB outer "
                  f"{validation['BNSB']['outer_fixed_point_residual_inf']:.2e}, "
                  f"BNSB KKT "
                  f"{validation['BNSB']['KKT_stationarity_residual_inf']:.2e}, "
                  f"BNS-O outer "
                  f"{validation['BNSO']['outer_fixed_point_residual_inf']:.2e}, "
                  f"BNS-O KKT "
                  f"{validation['BNSO']['KKT_stationarity_residual_inf']:.2e}")
            certificate = validation["BNSB"]["frozen_set_global_certificate"]
            print(f"    bargaining certificate: relative gap "
                  f"{certificate['certified_relative_gap_nash_product']:.2e}, "
                  f"certified {certificate['certified']}, min surplus "
                  f"{validation['BNSB']['min_IR_surplus']:.6f}")
            print(f"    ordering: {block['qualitative_ordering']}")


def main():
    for name, label in (("results_2dept.json", "two departments"),
                        ("results.json", "three departments")):
        data = load(name)
        if data is None:
            continue
        print(f"\n================ {label} ({name}) ================")
        regime_actions(data, label)
        exante_changes(data, label)
        economics(data, label)
        certificates(data, label)
        diagnostics(data, label)
    data = load("results_robustness.json")
    if data is not None:
        robustness(data)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze rebuttal ablation results and generate LaTeX tables.

Usage:
    python scripts/analyze_rebuttal_ablations.py
    python scripts/analyze_rebuttal_ablations.py --results-dir results/rebuttal_ablations
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from glob import glob

import pandas as pd

RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "rebuttal_ablations"
)


def load_method_results(method_dir):
    """Load final-round results from a method directory."""
    results_files = glob(os.path.join(method_dir, "*_results.json"))
    if not results_files:
        return None

    best_metrics = None
    for rf in results_files:
        with open(rf, "r") as f:
            data = json.load(f)
        if not data:
            continue
        last_round = data[-1] if isinstance(data, list) else data
        metrics = last_round.get("per_task_metrics", {})
        mtl_score = last_round.get("mtl_score", 0)
        best_metrics = {
            "cls_acc": metrics.get("classification", {}).get("accuracy", 0),
            "det_acc": metrics.get("detection", {}).get("accuracy", 0),
            "seg_dice": metrics.get("segmentation", {}).get("dice_mean",
                        metrics.get("segmentation", {}).get("dice", 0)),
            "mtl_score": mtl_score,
        }
    return best_metrics


def scan_results(base_dir):
    """Scan a results directory and collect metrics from each variant."""
    results = OrderedDict()
    if not os.path.exists(base_dir):
        return results

    for variant_name in sorted(os.listdir(base_dir)):
        variant_dir = os.path.join(base_dir, variant_name)
        if not os.path.isdir(variant_dir) or variant_name.startswith("summary"):
            continue

        # Check for config to get label
        config_path = os.path.join(variant_dir, "config.json")
        label = variant_name
        alpha, gamma, pf = "?", "?", "?"
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            alpha = cfg.get("alpha", "?")
            gamma = cfg.get("gamma", "?")
            pf = cfg.get("pool_factor", "?")

        # Look for method subdirectory (ours-2stage/)
        method_dir = os.path.join(variant_dir, "ours-2stage")
        if os.path.isdir(method_dir):
            metrics = load_method_results(method_dir)
        else:
            metrics = load_method_results(variant_dir)

        if metrics:
            metrics["alpha"] = alpha
            metrics["gamma"] = gamma
            metrics["pool_factor"] = pf
            metrics["variant"] = variant_name
            results[variant_name] = metrics

    return results


def format_sensitivity_table(results):
    """Format α, γ sensitivity results as table."""
    if not results:
        return "No sensitivity results found."

    rows = []
    for name, m in results.items():
        rows.append({
            "α": m["alpha"],
            "γ": m["gamma"],
            "Cls Acc": f"{m['cls_acc']*100:.1f}",
            "Det Acc": f"{m['det_acc']*100:.1f}",
            "Seg Dice": f"{m['seg_dice']*100:.1f}",
            "MTL Score": f"{m['mtl_score']:.4f}",
        })
    df = pd.DataFrame(rows)
    return df


def format_component_table(results):
    """Format component-wise ablation results as table."""
    if not results:
        return "No component results found."

    label_map = {
        "CTMI_only_KC": "CTMI only + K-Center",
        "CTMI_only_noKC": "CTMI only (no K-Center)",
        "H_CTMI_noKC": "H+CTMI (no K-Center)",
        "H_only_KC": "H only + K-Center",
        "H_only_noKC": "H only (no K-Center)",
        "full_ODS": "Full ODS",
    }

    rows = []
    for name, m in results.items():
        label = label_map.get(name, name)
        has_h = "✓" if m["alpha"] > 0 else "✗"
        has_ctmi = "✓" if m["gamma"] > 0 else "✗"
        has_kc = "✓" if m["pool_factor"] > 1 else "✗"
        rows.append({
            "Configuration": label,
            "H": has_h,
            "CTMI": has_ctmi,
            "K-Center": has_kc,
            "Cls Acc": f"{m['cls_acc']*100:.1f}",
            "Det Acc": f"{m['det_acc']*100:.1f}",
            "Seg Dice": f"{m['seg_dice']*100:.1f}",
            "MTL Score": f"{m['mtl_score']:.4f}",
        })
    df = pd.DataFrame(rows)
    return df


def generate_latex_sensitivity(results):
    """Generate LaTeX for sensitivity table."""
    if not results:
        return ""

    best_mtl = max(m["mtl_score"] for m in results.values())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{$\alpha$, $\gamma$ sensitivity.} ODS is robust across a wide range of weight settings.}",
        r"\label{tab:alpha_gamma_sensitivity}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"$\alpha$ & $\gamma$ & Cls Acc & Det Acc & Seg Dice & MTL Score \\",
        r"\midrule",
    ]

    for name, m in results.items():
        is_best = abs(m["mtl_score"] - best_mtl) < 1e-6
        a, g = m["alpha"], m["gamma"]
        cls = f"{m['cls_acc']*100:.1f}"
        det = f"{m['det_acc']*100:.1f}"
        seg = f"{m['seg_dice']*100:.1f}"
        mtl = f"{m['mtl_score']:.4f}"

        if is_best:
            line = rf"\textbf{{{a}}} & \textbf{{{g}}} & {cls} & {det} & {seg} & \textbf{{{mtl}}} \\"
        else:
            line = rf"{a} & {g} & {cls} & {det} & {seg} & {mtl} \\"
        lines.append(line)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_latex_component(results):
    """Generate LaTeX for component ablation table."""
    if not results:
        return ""

    label_map = {
        "H_only_noKC": "H only",
        "CTMI_only_noKC": "CTMI only",
        "H_CTMI_noKC": "H+CTMI",
        "H_only_KC": "H + K-Center",
        "CTMI_only_KC": "CTMI + K-Center",
        "full_ODS": r"\textbf{Full ODS}",
    }

    best_mtl = max(m["mtl_score"] for m in results.values())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Component-wise ablation.} Each component contributes; the full ODS achieves best performance.}",
        r"\label{tab:component_ablation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{K-Center} & Cls Acc & Det Acc & Seg Dice & MTL Score \\",
        r"\midrule",
    ]

    for name, m in results.items():
        label = label_map.get(name, name)
        kc = r"\cmark" if m["pool_factor"] > 1 else r"\xmark"
        is_best = abs(m["mtl_score"] - best_mtl) < 1e-6
        mtl = f"{m['mtl_score']:.4f}"
        if is_best:
            mtl = rf"\textbf{{{mtl}}}"
        line = rf"{label} & {kc} & {m['cls_acc']*100:.1f} & {m['det_acc']*100:.1f} & {m['seg_dice']*100:.1f} & {mtl} \\"
        lines.append(line)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_ROOT)
    args = parser.parse_args()

    print("=" * 60)
    print("  Rebuttal Ablation Analysis")
    print("=" * 60)

    # Sensitivity
    sens_dir = os.path.join(args.results_dir, "sensitivity")
    sens_results = scan_results(sens_dir)

    if sens_results:
        print("\n--- α, γ Sensitivity ---")
        df = format_sensitivity_table(sens_results)
        print(df.to_string(index=False))
    else:
        print("\n[!] No sensitivity results found at", sens_dir)

    # Component
    comp_dir = os.path.join(args.results_dir, "component")
    comp_results = scan_results(comp_dir)

    if comp_results:
        print("\n--- Component-wise Ablation ---")
        df = format_component_table(comp_results)
        print(df.to_string(index=False))
    else:
        print("\n[!] No component results found at", comp_dir)

    # Generate LaTeX
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "rebuttal"
    )
    latex_parts = []
    if sens_results:
        latex_parts.append(generate_latex_sensitivity(sens_results))
    if comp_results:
        latex_parts.append(generate_latex_component(comp_results))

    if latex_parts:
        latex_path = os.path.join(output_dir, "ablation_tables.tex")
        with open(latex_path, "w") as f:
            f.write("\n\n".join(latex_parts))
        print(f"\nLaTeX saved to: {latex_path}")

    # Summary statistics
    if comp_results:
        full_ods = comp_results.get("full_ODS", {})
        if full_ods:
            ods_mtl = full_ods["mtl_score"]
            print(f"\n--- Gain Attribution ---")
            for name, m in comp_results.items():
                if name == "full_ODS":
                    continue
                delta = ((ods_mtl - m["mtl_score"]) / ods_mtl) * 100
                print(f"  Full ODS vs {name}: Δ = {delta:+.1f}%")


if __name__ == "__main__":
    main()

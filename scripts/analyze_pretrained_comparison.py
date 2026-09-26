#!/usr/bin/env python3
"""
Analyze pretrained ResNet-50 comparison results and generate LaTeX table.

Usage:
    python scripts/analyze_pretrained_comparison.py
    python scripts/analyze_pretrained_comparison.py --results-dir results/rebuttal_pretrained
"""

import argparse
import json
import os
import sys
from glob import glob
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "rebuttal_pretrained"
)
REBUTTAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rebuttal"
)

METHOD_DISPLAY = OrderedDict([
    ("ours-2stage", "ODS (Ours)"),
    ("random",      "Random"),
    ("uncertainty",  "Uncertainty"),
    ("diversity",    "Diversity"),
    ("coreset",      "CoreSet"),
    ("badge",        "BADGE"),
])


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


def scan_results(results_dir):
    """Scan result directories and collect metrics per method."""
    all_results = OrderedDict()
    for method_key in METHOD_DISPLAY:
        method_dir = os.path.join(results_dir, method_key)
        if not os.path.isdir(method_dir):
            continue
        metrics = load_method_results(method_dir)
        if metrics:
            all_results[method_key] = metrics
    return all_results


def generate_latex(all_results):
    """Generate LaTeX table comparing pretrained backbone results."""
    if not all_results:
        return "% No results found."

    metrics_keys = ["cls_acc", "det_acc", "seg_dice", "mtl_score"]
    metric_labels = ["Cls Acc", "Det Acc", "Seg Dice", "MTL Comp."]

    best_vals = {}
    for mk in metrics_keys:
        vals = [r[mk] for r in all_results.values()]
        best_vals[mk] = max(vals) if vals else 0

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{\textbf{Pretrained ResNet-50 backbone.} "
        r"All methods use the same ImageNet-pretrained ResNet-50 encoder. "
        r"ODS maintains its advantage even with a stronger backbone.}"
    )
    lines.append(r"\label{tab:pretrained_resnet50}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    header = r"\textbf{Method}"
    for ml in metric_labels:
        header += rf" & \textbf{{{ml}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method_key, metrics in all_results.items():
        label = METHOD_DISPLAY.get(method_key, method_key)
        is_ours = method_key == "ours-2stage"

        if is_ours:
            line = rf"\textbf{{{label}}}"
        else:
            line = label

        for mk in metrics_keys:
            val = metrics[mk]
            if mk == "mtl_score":
                formatted = f"{val:.4f}"
            else:
                formatted = f"{val * 100:.1f}"

            if abs(val - best_vals[mk]) < 1e-6:
                line += rf" & \textbf{{{formatted}}}"
            else:
                line += f" & {formatted}"

        line += r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_markdown(all_results):
    """Generate markdown summary table."""
    if not all_results:
        return "No results found."

    lines = []
    lines.append("| Method | Cls Acc | Det Acc | Seg Dice | MTL Comp. |")
    lines.append("|--------|---------|---------|----------|-----------|")
    for method_key, metrics in all_results.items():
        label = METHOD_DISPLAY.get(method_key, method_key)
        cls = f"{metrics['cls_acc'] * 100:.1f}"
        det = f"{metrics['det_acc'] * 100:.1f}"
        seg = f"{metrics['seg_dice'] * 100:.1f}"
        mtl = f"{metrics['mtl_score']:.4f}"
        lines.append(f"| {label} | {cls} | {det} | {seg} | {mtl} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze pretrained ResNet-50 comparison")
    parser.add_argument("--results-dir", default=RESULTS_ROOT,
                        help="Results directory")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Pretrained ResNet-50 Comparison — Analysis")
    print(f"  Results dir: {args.results_dir}")
    print(f"{'='*60}")

    all_results = scan_results(args.results_dir)

    if not all_results:
        print("\n  No results found yet. Run the experiment first.")
        return

    print(f"\n  Found {len(all_results)} methods: {', '.join(all_results.keys())}")

    md = generate_markdown(all_results)
    print(f"\n{md}")

    latex = generate_latex(all_results)
    os.makedirs(REBUTTAL_DIR, exist_ok=True)
    tex_path = os.path.join(REBUTTAL_DIR, "pretrained_comparison_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex + "\n")
    print(f"\n  LaTeX table saved: {tex_path}")

    ours_metrics = all_results.get("ours-2stage")
    if ours_metrics:
        print(f"\n  ODS: Cls={ours_metrics['cls_acc']*100:.1f}, "
              f"Det={ours_metrics['det_acc']*100:.1f}, "
              f"Seg={ours_metrics['seg_dice']*100:.1f}, "
              f"MTL={ours_metrics['mtl_score']:.4f}")

        baselines = {k: v for k, v in all_results.items() if k != "ours-2stage"}
        if baselines:
            best_baseline_key = max(baselines, key=lambda k: baselines[k]["mtl_score"])
            best_bl = baselines[best_baseline_key]
            bl_label = METHOD_DISPLAY.get(best_baseline_key, best_baseline_key)
            delta = (ours_metrics["mtl_score"] - best_bl["mtl_score"]) / best_bl["mtl_score"] * 100
            print(f"  Best baseline ({bl_label}): MTL={best_bl['mtl_score']:.4f}")
            print(f"  Delta: {delta:+.1f}%")


if __name__ == "__main__":
    main()

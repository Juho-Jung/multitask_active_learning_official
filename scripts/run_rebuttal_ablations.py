#!/usr/bin/env python3
"""
Rebuttal Ablation Experiments

Runs two sets of experiments for the rebuttal:
1. α, γ Sensitivity Ablation: ODS with different weight combinations
2. Component-wise Ablation Matrix: Isolate contribution of each component

Usage:
    python scripts/run_rebuttal_ablations.py                     # Run all
    python scripts/run_rebuttal_ablations.py --only sensitivity   # α,γ only
    python scripts/run_rebuttal_ablations.py --only component     # Component only
    python scripts/run_rebuttal_ablations.py --dry-run            # Print configs only
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from multitask_active_learning.runner import AblationRunner

BASE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "coco_2stage_selection.json"
)
RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "rebuttal_ablations"
)

# ─── α, γ Sensitivity Ablation ───
SENSITIVITY_VARIANTS = OrderedDict([
    ("alpha1.0_gamma0.0", {"alpha": 1.0, "gamma": 0.0, "label": "H only (α=1, γ=0)"}),
    ("alpha0.0_gamma1.0", {"alpha": 0.0, "gamma": 1.0, "label": "CTMI only (α=0, γ=1)"}),
    ("alpha1.0_gamma0.5", {"alpha": 1.0, "gamma": 0.5, "label": "α=1.0, γ=0.5"}),
    ("alpha1.0_gamma1.0", {"alpha": 1.0, "gamma": 1.0, "label": "α=1.0, γ=1.0 (default)"}),
    ("alpha1.0_gamma2.0", {"alpha": 1.0, "gamma": 2.0, "label": "α=1.0, γ=2.0"}),
    ("alpha2.0_gamma1.0", {"alpha": 2.0, "gamma": 1.0, "label": "α=2.0, γ=1.0"}),
    ("alpha0.5_gamma1.0", {"alpha": 0.5, "gamma": 1.0, "label": "α=0.5, γ=1.0"}),
])

# ─── Component-wise Ablation Matrix ───
# pool_factor=1 → Stage 2 (K-Center) is skipped (num_candidates = k)
# pool_factor=5 → Stage 2 active (default)
COMPONENT_VARIANTS = OrderedDict([
    ("H_only_noKC", {
        "alpha": 1.0, "gamma": 0.0, "pool_factor": 1,
        "label": "H only (no K-Center)",
    }),
    ("CTMI_only_noKC", {
        "alpha": 0.0, "gamma": 1.0, "pool_factor": 1,
        "label": "CTMI only (no K-Center)",
    }),
    ("H_CTMI_noKC", {
        "alpha": 1.0, "gamma": 1.0, "pool_factor": 1,
        "label": "H+CTMI (no K-Center)",
    }),
    ("H_only_KC", {
        "alpha": 1.0, "gamma": 0.0, "pool_factor": 5,
        "label": "H only + K-Center",
    }),
    ("CTMI_only_KC", {
        "alpha": 0.0, "gamma": 1.0, "pool_factor": 5,
        "label": "CTMI only + K-Center",
    }),
    ("full_ODS", {
        "alpha": 1.0, "gamma": 1.0, "pool_factor": 5,
        "label": "Full ODS (H+CTMI + K-Center)",
    }),
])


def load_base_config():
    with open(BASE_CONFIG_PATH, "r") as f:
        config = json.load(f)
    # Disable adaptive scheduling for clean ablation comparison
    config["adaptive_hyperparameter_scheduling"] = False
    config["task_aware_weighting"] = False
    return config


def make_variant_config(base, variant_params, save_dir):
    cfg = copy.deepcopy(base)
    for k, v in variant_params.items():
        if k != "label":
            cfg[k] = v
    cfg["methods"] = ["ours-2stage"]
    cfg["save_dir"] = save_dir
    save_model_dir = os.path.join(save_dir, "checkpoints")
    cfg["save_model_path"] = save_model_dir
    return cfg


def run_variant(name, config, device, dry_run=False):
    label = config.pop("_label", name)
    save_dir = config["save_dir"]

    print(f"\n{'='*70}")
    print(f"  Running: {label}")
    print(f"  α={config['alpha']}, γ={config['gamma']}, pool_factor={config.get('pool_factor', 5)}")
    print(f"  Save: {save_dir}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Skipping actual execution.")
        return None

    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    start = time.time()
    try:
        runner = AblationRunner(config, device=device, save_dir=save_dir)
        runner.run_ablation()
        elapsed = time.time() - start
        print(f"  ✅ Completed in {elapsed/60:.1f} min")
        return {"name": name, "label": label, "status": "success", "time_min": elapsed / 60}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ Failed after {elapsed/60:.1f} min: {e}")
        import traceback
        traceback.print_exc()
        return {"name": name, "label": label, "status": "failed", "error": str(e), "time_min": elapsed / 60}


def print_summary(results, experiment_name):
    print(f"\n{'='*70}")
    print(f"  {experiment_name} — Summary")
    print(f"{'='*70}")
    print(f"  {'Name':<25s} {'Status':<10s} {'Time (min)':<10s}")
    print(f"  {'-'*45}")
    for r in results:
        if r is None:
            continue
        print(f"  {r['label']:<25s} {r['status']:<10s} {r['time_min']:<10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Rebuttal ablation experiments")
    parser.add_argument("--only", choices=["sensitivity", "component"],
                        help="Run only one experiment set")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    parser.add_argument("--device", default="auto",
                        help="Device (auto, cpu, cuda, cuda:0, etc.)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    base = load_base_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ─── 1. α, γ Sensitivity ───
    if args.only is None or args.only == "sensitivity":
        print(f"\n{'#'*70}")
        print(f"  EXPERIMENT 1: α, γ Sensitivity Ablation")
        print(f"{'#'*70}")

        results = []
        for name, params in SENSITIVITY_VARIANTS.items():
            save_dir = os.path.join(RESULTS_ROOT, "sensitivity", name)
            cfg = make_variant_config(base, params, save_dir)
            cfg["_label"] = params["label"]
            r = run_variant(name, cfg, device, dry_run=args.dry_run)
            results.append(r)

        if not args.dry_run:
            print_summary(results, "α, γ Sensitivity")
            summary_path = os.path.join(RESULTS_ROOT, "sensitivity", f"summary_{timestamp}.json")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump([r for r in results if r], f, indent=2)

    # ─── 2. Component-wise Ablation ───
    if args.only is None or args.only == "component":
        print(f"\n{'#'*70}")
        print(f"  EXPERIMENT 2: Component-wise Ablation Matrix")
        print(f"{'#'*70}")

        results = []
        for name, params in COMPONENT_VARIANTS.items():
            save_dir = os.path.join(RESULTS_ROOT, "component", name)
            cfg = make_variant_config(base, params, save_dir)
            cfg["_label"] = params["label"]
            r = run_variant(name, cfg, device, dry_run=args.dry_run)
            results.append(r)

        if not args.dry_run:
            print_summary(results, "Component-wise Ablation")
            summary_path = os.path.join(RESULTS_ROOT, "component", f"summary_{timestamp}.json")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump([r for r in results if r], f, indent=2)

    print(f"\n{'='*70}")
    print(f"  All experiments finished!")
    print(f"  Results: {RESULTS_ROOT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Rebuttal: Pretrained ResNet-50 Backbone Comparison

Runs ODS vs. representative baselines with ImageNet-pretrained ResNet-50 encoder
to show that ODS's advantage holds regardless of backbone strength.

Addresses: Reviewer 3 (pretrained model 미사용)

Usage:
    python scripts/run_pretrained_comparison.py                          # Run all methods
    python scripts/run_pretrained_comparison.py --methods ours-2stage random  # Specific methods
    python scripts/run_pretrained_comparison.py --dry-run               # Print configs only
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
    "configs", "coco_pretrained_resnet50.json"
)
RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "rebuttal_pretrained"
)

METHODS = OrderedDict([
    ("ours-2stage", "ODS (Ours)"),
    ("random",      "Random"),
    ("uncertainty",  "Uncertainty"),
    ("diversity",    "Diversity"),
    ("coreset",      "CoreSet"),
    ("badge",        "BADGE"),
])


def load_base_config():
    with open(BASE_CONFIG_PATH, "r") as f:
        config = json.load(f)
    return config


def run_method(method_name, label, base_config, device, dry_run=False):
    cfg = copy.deepcopy(base_config)
    cfg["methods"] = [method_name]
    save_dir = os.path.join(RESULTS_ROOT, method_name)
    cfg["save_dir"] = save_dir
    cfg["save_model_path"] = os.path.join(save_dir, "checkpoints")

    print(f"\n{'='*70}")
    print(f"  [{method_name}] {label}")
    print(f"  Backbone: ResNet-50 (ImageNet pretrained)")
    print(f"  Save: {save_dir}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Skipping execution.")
        print(f"  Config preview: pretrained={cfg.get('pretrained')}, "
              f"lr={cfg.get('learning_rate')}, "
              f"batch={cfg.get('train_batch_size')}")
        return {"name": method_name, "label": label, "status": "dry_run", "time_min": 0}

    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    start = time.time()
    try:
        runner = AblationRunner(cfg, device=device, save_dir=save_dir)
        runner.run_ablation()
        elapsed = time.time() - start
        print(f"  Completed in {elapsed/60:.1f} min")
        return {"name": method_name, "label": label, "status": "success", "time_min": elapsed / 60}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Failed after {elapsed/60:.1f} min: {e}")
        import traceback
        traceback.print_exc()
        return {"name": method_name, "label": label, "status": "failed",
                "error": str(e), "time_min": elapsed / 60}


def print_summary(results):
    print(f"\n{'='*70}")
    print(f"  Pretrained ResNet-50 Comparison — Summary")
    print(f"{'='*70}")
    print(f"  {'Method':<20s} {'Status':<10s} {'Time (min)':<10s}")
    print(f"  {'-'*40}")
    for r in results:
        if r is None:
            continue
        print(f"  {r['label']:<20s} {r['status']:<10s} {r['time_min']:<10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Pretrained ResNet-50 comparison experiment")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Specific methods to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    parser.add_argument("--device", default="auto",
                        help="Device (auto, cpu, cuda, cuda:0, etc.)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"{'#'*70}")
    print(f"  Rebuttal: Pretrained ResNet-50 Backbone Comparison")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"{'#'*70}")

    base = load_base_config()

    methods_to_run = args.methods if args.methods else list(METHODS.keys())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    for method_name in methods_to_run:
        label = METHODS.get(method_name, method_name)
        r = run_method(method_name, label, base, device, dry_run=args.dry_run)
        results.append(r)

    if not args.dry_run:
        print_summary(results)
        summary_path = os.path.join(RESULTS_ROOT, f"summary_{timestamp}.json")
        os.makedirs(RESULTS_ROOT, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Summary saved: {summary_path}")

    print(f"\n{'='*70}")
    print(f"  Done! Results: {RESULTS_ROOT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI script for running Multi-Task Active Learning experiments.

Usage:
    python run_al.py --config configs/cls_seg.json
    python run_al.py --config configs/cls_det_seg.json --save_dir ./my_results
    python run_al.py --tasks classification --criteria hardness --methods random uncertainty ours
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from multitask_active_learning.runner import (AblationRunner,
                                              create_default_config,
                                              load_config)

# Add the package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_list_arg(arg: str) -> List[str]:
    """Parse comma-separated list argument."""
    return [item.strip() for item in arg.split(',')]


def create_config_from_args(args) -> Dict[str, Any]:
    """Create configuration from command line arguments."""
    config = create_default_config()

    # Override with command line arguments
    if args.tasks:
        config['tasks'] = parse_list_arg(args.tasks)

    if args.criteria:
        config['criteria'] = parse_list_arg(args.criteria)

    if args.methods:
        config['methods'] = parse_list_arg(args.methods)

    if args.num_rounds:
        config['num_rounds'] = args.num_rounds

    if args.batch_size:
        config['batch_size'] = args.batch_size

    if args.num_classes:
        config['num_classes'] = args.num_classes

    if args.alpha is not None:
        config['alpha'] = args.alpha

    if args.beta is not None:
        config['beta'] = args.beta

    if args.gamma is not None:
        config['gamma'] = args.gamma

    if args.seed:
        config['seed'] = args.seed

    if args.hardness_mode:
        config['hardness_mode'] = args.hardness_mode

    if args.conflict_mode:
        config['conflict_mode'] = args.conflict_mode

    if args.consistency_mode:
        config['consistency_mode'] = args.consistency_mode

    return config


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Multi-Task Active Learning Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Run with config file
            python run_al.py --config configs/cls_seg.json
            
            # Run with command line arguments
            python run_al.py --tasks classification,segmentation --criteria hardness,conflict,consistency
            
            # Run specific methods only
            python run_al.py --config configs/cls_seg.json --methods random,uncertainty,ours
            
            # Run all baseline methods
            python run_al.py --config configs/coco_cls_det_seg.json --methods random,uncertainty,diversity,coreset,badge,batchbald,clustering,ours
            
            # Custom parameters
            python run_al.py --tasks classification --criteria hardness --alpha 1.0 --beta 0.0 --gamma 0.0
            """
    )

    # Configuration options
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('--save_dir', type=str,
                        default='./results', help='Directory to save results')
    parser.add_argument('--save_model_path', type=str, default='./checkpoints',
                        help='Directory to save best loss models (e.g., ./models). Models saved as round_N/best_loss.pth')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, cuda:0, cuda:1, etc.)')

    # Dataset options
    parser.add_argument('--dataset', type=str, choices=['synthetic', 'coco'], default=None,
                        help='Dataset to use (if not specified, uses value from config file)')
    parser.add_argument('--coco_root', type=str, help='Root directory of COCO dataset (containing images/)')
    parser.add_argument('--coco_ann_file', type=str, help='Path to COCO annotation JSON file')

    # Task and criteria options
    parser.add_argument('--tasks', type=str, choices=['classification', 'detection', 'segmentation'],
                        help='Comma-separated list of tasks (classification,detection,segmentation)')
    parser.add_argument('--criteria', type=str, help='Comma-separated list of criteria (hardness,conflict,consistency)')
    parser.add_argument('--methods', '--method', type=str, dest='methods',
                        help='Comma-separated list of methods (random,uncertainty,diversity,coreset,badge,batchbald,clustering,ours)')

    # Experiment parameters
    parser.add_argument('--num_rounds', type=int, help='Number of AL rounds')
    parser.add_argument('--batch_size', type=int, help='Batch size for selection')
    parser.add_argument('--num_classes', type=int, help='Number of classes')

    # Acquisition function parameters
    parser.add_argument('--alpha', type=float, help='Weight for hardness component')
    parser.add_argument('--beta', type=float, help='Weight for gradient conflict component')
    parser.add_argument('--gamma', type=float, help='Weight for task consistency component')

    # Mode parameters
    parser.add_argument('--hardness_mode', type=str, choices=['entropy', 'margin'], help='Hardness computation mode')
    parser.add_argument('--conflict_mode', type=str,
                        choices=['cosine', 'pcgrad'], help='Gradient conflict computation mode')
    parser.add_argument('--consistency_mode', type=str, choices=['kl', 'mse'], help='Task consistency computation mode')

    # Other options
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # If experiment_name is specified, append it to save_dir and save_model_path as subdirectory
    if args.experiment_name:
        # Use explicitly provided paths or defaults as base directories
        base_save_dir = args.save_dir if args.save_dir else './results'
        base_save_model_path = args.save_model_path if args.save_model_path else './checkpoints'

        # Append experiment_name as subdirectory and datetime
        import datetime
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(base_save_dir, f"{datetime_str}_{args.experiment_name}")
        args.save_model_path = os.path.join(base_save_model_path, f"{datetime_str}_{args.experiment_name}")

        print(f"📁 Experiment name specified: {args.experiment_name}")
        print(f"   Results will be saved to: {args.save_dir}")
        print(f"   Models will be saved to: {args.save_model_path}")

    # Load or create configuration
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)

        config = load_config(args.config)
        print(f"📁 Loaded config from: {args.config}")
    else:
        config = create_config_from_args(args)
        print("📋 Using command line configuration")

    # Override with command line arguments if provided
    if args.tasks:
        config['tasks'] = parse_list_arg(args.tasks)
    if args.criteria:
        config['criteria'] = parse_list_arg(args.criteria)
    if args.methods:
        config['methods'] = parse_list_arg(args.methods)
    if args.num_rounds:
        config['num_rounds'] = args.num_rounds
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_classes:
        config['num_classes'] = args.num_classes
    if args.alpha is not None:
        config['alpha'] = args.alpha
    if args.beta is not None:
        config['beta'] = args.beta
    if args.gamma is not None:
        config['gamma'] = args.gamma
    if args.seed:
        config['seed'] = args.seed
    if args.hardness_mode:
        config['hardness_mode'] = args.hardness_mode
    if args.conflict_mode:
        config['conflict_mode'] = args.conflict_mode
    if args.consistency_mode:
        config['consistency_mode'] = args.consistency_mode

    # Dataset configuration (only override if explicitly provided)
    if args.dataset is not None:
        config['dataset'] = args.dataset
    if args.coco_root:
        config['coco_root'] = args.coco_root
    if args.coco_ann_file:
        config['coco_ann_file'] = args.coco_ann_file

    # Set device
    import torch
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device.startswith('cuda:'):
        # Specific GPU ID (e.g., cuda:0, cuda:1)
        device = torch.device(args.device)
    elif args.device == 'cuda':
        # Use default CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Print GPU info if using CUDA
    if device.type == 'cuda':
        print(f"   GPU ID: {device.index if device.index is not None else 0}")
        print(f"   GPU Name: {torch.cuda.get_device_name(device)}")

    # Print configuration
    if args.verbose:
        print("\n🔧 Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")

    print(f"\n🚀 Starting Multi-Task Active Learning Experiment")
    print(f"   Device: {device}")
    print(f"   Tasks: {config['tasks']}")
    print(f"   Criteria: {config['criteria']}")
    print(f"   Methods: {config['methods']}")
    print(f"   Rounds: {config['num_rounds']}")
    print(f"   Batch size: {config['batch_size']}")

    # Create and run ablation study
    try:
        runner = AblationRunner(config, device=device, save_dir=args.save_dir,
                                save_model_path=args.save_model_path)
        runner.run_ablation()

        print(f"\n✅ Experiment completed successfully!")
        print(f"   Results saved to: {args.save_dir}")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

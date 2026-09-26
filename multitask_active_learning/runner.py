"""
AblationRunner for Multi-Task Active Learning

Provides a unified interface for running ablation studies with different
task combinations, criteria, and baseline methods.
"""

import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .acquisition import MultiTaskActiveLearningAcquisition
from .al_selectors import make_selector
from .datasets import (load_cityscapes_dataset, load_coco_dataset,
                       load_coco_test_dataset, load_nyuv2_dataset,
                       split_coco_for_al)
from .metrics import MetricsLogger, MTLMetrics, PerTaskMetrics
from .models import MultiTaskModel, create_toy_model


def _print_memory_usage(device, prefix="", reset_peak=False):
    """Print GPU memory usage for debugging.

    Args:
        device: torch.device
        prefix: Prefix string for the output
        reset_peak: If True, reset peak memory stats after printing
    """
    if device.type == 'cuda':
        with torch.cuda.device(device):
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
            print(
                f"  💾 {prefix}GPU Memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, max_allocated={max_allocated:.2f}GB")
            if reset_peak:
                torch.cuda.reset_peak_memory_stats(device)


class AblationRunner:
    """Main runner for ablation studies."""

    def __init__(
        self,
        config: Dict,
        device: torch.device = None,
        save_dir: str = './results',
        save_model_path: str = None
    ):
        self.config = config
        # Ensure device is a torch.device object
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        # Use save_dir from config if provided (and not empty), otherwise use the parameter
        # This ensures config['save_dir'] takes precedence if set in main()
        config_save_dir = config.get('save_dir')
        self.save_dir = config_save_dir if (config_save_dir and config_save_dir.strip()) else save_dir
        # Use save_model_path from config if not provided, otherwise use provided value
        self.save_model_path = save_model_path or config.get('save_model_path')

        # Initialize logger (use self.save_dir to ensure consistency)
        self.logger = MetricsLogger(self.save_dir)

        # Set random seeds
        self._set_seeds()

        # Initialize model and data
        self.model = self._create_model()

        # Store full dataset for COCO/NYUv2 (to retrieve labels when selecting)
        # Load this FIRST, then create unlabeled_pool from it
        self.full_dataset_labels = None
        self.full_dataset_indices = None
        self.labeled_indices_set = set()  # Track labeled indices instead of storing data
        if self.config.get('dataset') == 'coco':
            print("  📂 Initializing COCO dataset storage...")
            self._store_full_coco_labels()
            if self.full_dataset_labels is None:
                raise RuntimeError("Failed to initialize full_dataset_labels. Check COCO dataset paths.")
        elif self.config.get('dataset') == 'nyuv2':
            print("  📂 Initializing NYUv2 dataset storage...")
            self._store_full_nyuv2_labels()
            if self.full_dataset_labels is None:
                raise RuntimeError("Failed to initialize full_dataset_labels. Check NYUv2 dataset paths.")
        elif self.config.get('dataset') == 'cityscapes':
            print("  📂 Initializing Cityscapes dataset storage...")
            self._store_full_cityscapes_labels()
            if self.full_dataset_labels is None:
                raise RuntimeError("Failed to initialize full_dataset_labels. Check Cityscapes dataset paths.")

        # Create unlabeled_pool from already-loaded data (for COCO, this is just indices/reference)
        self.unlabeled_pool = self._create_unlabeled_pool()

        # Load test set for evaluation
        self.test_set = None
        if self.config.get('dataset') == 'coco':
            self._load_test_set()
        elif self.config.get('dataset') == 'nyuv2':
            self._load_test_set()

        # Initialize selectors
        self.selectors = self._create_selectors()

        # Store base metrics (Round 0 performance) for MTL score calculation
        self.base_metrics = None
        self.oracle_metrics = None

        # Track previous round performance for dynamic hyperparameter adjustment
        self.previous_round_metrics = {}  # {round_num: {task: metrics}}

        print(f"🚀 AblationRunner initialized")
        print(f"   Device: {self.device}")
        print(f"   Tasks: {self.config['tasks']}")
        print(f"   Criteria: {self.config['criteria']}")
        print(f"   Methods: {list(self.selectors.keys())}")

    def _set_seeds(self, seed=None):
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed to use. If None, uses seed from config.
        """
        if seed is None:
            seed = self.config.get('seed', 42)

        # Set seeds for all random number generators
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Set CUDA random seed if using GPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        tasks = self.config['tasks']

        # Create task configurations
        task_configs = {}
        for task in tasks:
            if task == 'classification':
                task_configs['classification'] = {
                    'task_type': 'classification',
                    'num_classes': self.config.get('num_classes', 10)
                }
            elif task == 'detection':
                task_configs['detection'] = {
                    'task_type': 'detection',
                    'num_classes': self.config.get('num_classes', 10),
                    'num_boxes': self.config.get('num_boxes', 5),
                    'detection_model': self.config.get('model', 'anchor-free'),
                    'anchor_scales': self.config.get('anchor_scales', None),
                    'anchor_ratios': self.config.get('anchor_ratios', None)
                }
            elif task == 'segmentation':
                task_configs['segmentation'] = {
                    'task_type': 'segmentation',
                    'num_classes': self.config.get('num_classes', 10)
                }
            elif task in ['depth', 'normal']:
                # Regression tasks (depth, normal)
                task_configs[task] = {
                    'task_type': task,  # 'depth' or 'normal'
                    'num_classes': 1 if task == 'depth' else 3  # depth: 1 channel, normal: 3 channels
                }
            elif task == 'instance':
                # Instance segmentation: binary foreground/background classification
                task_configs['instance'] = {
                    'task_type': 'instance_segmentation',
                    'num_classes': 2  # Binary: 0=background, 1=foreground (any instance)
                }

        # Create model
        model = MultiTaskModel(
            input_channels=self.config.get('input_channels', 3),
            hidden_dim=self.config.get('hidden_dim', 256),
            num_encoder_layers=self.config.get('num_encoder_layers', 4),
            task_configs=task_configs,
            dropout=self.config.get('dropout', 0.1),
            pretrained=self.config.get('pretrained', False),
            detection_model=self.config.get('model', 'anchor-free'),
            anchor_scales=self.config.get('anchor_scales', None),
            anchor_ratios=self.config.get('anchor_ratios', None)
        )

        model.to(self.device)
        return model

    def _create_unlabeled_pool(self) -> torch.Tensor:
        """Create unlabeled data pool."""
        # Check if using COCO dataset
        if self.config.get('dataset') == 'coco':
            return self._load_coco_unlabeled_pool()
        # Check if using NYUv2 dataset
        elif self.config.get('dataset') == 'nyuv2':
            return self._load_nyuv2_unlabeled_pool()
        # Check if using Cityscapes dataset
        elif self.config.get('dataset') == 'cityscapes':
            return self._load_cityscapes_unlabeled_pool()

        # Default: synthetic data - keep on CPU
        batch_size = self.config.get('unlabeled_pool_size', 1000)
        input_channels = self.config.get('input_channels', 3)
        height = self.config.get('image_height', 64)
        width = self.config.get('image_width', 64)

        return torch.randn(batch_size, input_channels, height, width, device='cpu')

    def _create_labeled_pool(self) -> Dict[str, torch.Tensor]:
        """Create initial labeled data pool."""
        # Check if using COCO dataset
        if self.config.get('dataset') == 'coco':
            return self._load_coco_labeled_pool()

        # Default: synthetic data
        batch_size = self.config.get('initial_labeled_size', 100)
        input_channels = self.config.get('input_channels', 3)
        height = self.config.get('image_height', 64)
        width = self.config.get('image_width', 64)
        num_classes = self.config.get('num_classes', 10)

        labeled_data = {
            'inputs': torch.randn(batch_size, input_channels, height, width, device=self.device)
        }

        # Create targets for each task
        for task in self.config['tasks']:
            if task == 'classification':
                labeled_data['classification'] = torch.randint(0, num_classes, (batch_size,), device=self.device)
            elif task == 'detection':
                # Simplified detection targets
                labeled_data['detection'] = torch.randn(batch_size, self.config.get(
                    'num_boxes', 5), num_classes + 4, device=self.device)
            elif task == 'segmentation':
                labeled_data['segmentation'] = torch.randint(
                    0, num_classes, (batch_size, height, width), device=self.device)

        return labeled_data

    def _load_coco_unlabeled_pool(self) -> torch.Tensor:
        """Load COCO dataset as unlabeled pool.

        For COCO, we already loaded the data in _store_full_coco_labels.
        This function returns only the unlabeled samples (excluding initial labeled ones).
        """
        # Data is already loaded in _store_full_coco_labels
        if self.full_dataset_labels is None or 'inputs' not in self.full_dataset_labels:
            raise RuntimeError("full_dataset_labels not initialized. Call _store_full_coco_labels first.")

        # Get unlabeled indices from _unlabeled_to_full mapping
        # _unlabeled_to_full contains the indices of unlabeled samples in full_dataset_labels
        if hasattr(self, '_unlabeled_to_full') and self._unlabeled_to_full is not None:
            unlabeled_indices = self._unlabeled_to_full.long()
            # Ensure indices are within valid range
            max_valid_idx = len(self.full_dataset_labels['inputs']) - 1
            unlabeled_indices = torch.clamp(unlabeled_indices, 0, max_valid_idx).long()
            unlabeled_pool = self.full_dataset_labels['inputs'][unlabeled_indices]
            print(
                f"  📊 Using {len(unlabeled_pool):,} unlabeled images from {len(self.full_dataset_labels['inputs']):,} total (on CPU)")
        else:
            # Fallback: return all images (should not happen if _store_full_coco_labels was called correctly)
            unlabeled_pool = self.full_dataset_labels['inputs']
            print(
                f"  ⚠️  Warning: _unlabeled_to_full not found, using all {len(unlabeled_pool):,} images as unlabeled pool (on CPU)")

        return unlabeled_pool

    def _load_nyuv2_unlabeled_pool(self) -> torch.Tensor:
        """Load NYUv2 dataset as unlabeled pool.

        For NYUv2, we already loaded the data in _store_full_nyuv2_labels.
        This function returns only the unlabeled samples (excluding initial labeled ones).
        """
        # Data is already loaded in _store_full_nyuv2_labels
        if self.full_dataset_labels is None or 'inputs' not in self.full_dataset_labels:
            raise RuntimeError("full_dataset_labels not initialized. Call _store_full_nyuv2_labels first.")

        # Get unlabeled indices from _unlabeled_to_full mapping
        if hasattr(self, '_unlabeled_to_full') and self._unlabeled_to_full is not None:
            unlabeled_indices = self._unlabeled_to_full.long()
            max_valid_idx = len(self.full_dataset_labels['inputs']) - 1
            unlabeled_indices = torch.clamp(unlabeled_indices, 0, max_valid_idx).long()
            unlabeled_pool = self.full_dataset_labels['inputs'][unlabeled_indices]
            print(
                f"  📊 Using {len(unlabeled_pool):,} unlabeled images from {len(self.full_dataset_labels['inputs']):,} total (on CPU)")
        else:
            unlabeled_pool = self.full_dataset_labels['inputs']
            print(
                f"  ⚠️  Warning: _unlabeled_to_full not found, using all {len(unlabeled_pool):,} images as unlabeled pool (on CPU)")

        return unlabeled_pool

    def _load_coco_labeled_pool(self) -> Dict[str, torch.Tensor]:
        """Load initial labeled pool for COCO dataset.

        Note: This is only used for initialization. After that, we use labeled_indices_set
        and load data from full_dataset_labels on-demand.
        """
        # Return empty dict - we'll use labeled_indices_set instead
        # This method is kept for compatibility but won't store actual data
        return {}

    def _store_full_coco_labels(self):
        """Store full COCO dataset labels for label retrieval.

        This loads the full dataset once and stores it on CPU.
        Only indices are tracked in labeled_indices_set.
        """
        coco_root = self.config.get('coco_root', './data/coco')
        ann_file = self.config.get('coco_ann_file', None)

        if ann_file is None:
            raise ValueError("coco_ann_file must be specified in config for COCO dataset")

        # Load labels for the sampled pool (same samples as unlabeled_pool)
        unlabeled_pool_size = self.config.get('unlabeled_pool_size', 10000)
        seed = self.config.get('seed', 42)

        try:
            print(f"  🔍 _store_full_coco_labels: Starting...")
            print(f"  📂 Loading COCO dataset from {ann_file}...")
            print(f"  📂 Root: {coco_root}, max_samples: {unlabeled_pool_size}, seed: {seed}")
            # Load labels for the sampled pool (same samples as unlabeled_pool) - keep on CPU
            _, full_labeled_pool = load_coco_dataset(
                root=coco_root,
                ann_file=ann_file,
                tasks=self.config['tasks'],
                image_size=(self.config.get('image_height', 64), self.config.get('image_width', 64)),
                max_boxes=self.config.get('num_boxes', 5),
                num_classes=self.config.get('num_classes', 80),
                device='cpu',  # Keep on CPU to save GPU memory
                max_samples=unlabeled_pool_size,
                seed=seed
            )

            print(
                f"  ✅ Loaded dataset, full_labeled_pool keys: {list(full_labeled_pool.keys()) if full_labeled_pool else 'None'}")
            if full_labeled_pool is None or 'inputs' not in full_labeled_pool:
                raise ValueError(f"Failed to load COCO dataset: full_labeled_pool is invalid")

            # Store labels (for label retrieval when selecting samples) - ensure on CPU
            self.full_dataset_labels = {}
            for key, value in full_labeled_pool.items():
                if isinstance(value, torch.Tensor):
                    self.full_dataset_labels[key] = value.cpu() if value.device.type == 'cuda' else value
                else:
                    self.full_dataset_labels[key] = value
            self.full_dataset_indices = torch.arange(len(full_labeled_pool['inputs']), device='cpu')

            # Track which indices are in labeled pool (within the sampled pool)
            # Use _set_seeds to ensure all random generators use the same seed
            self._set_seeds(seed)

            # Split sampled pool into labeled/unlabeled
            n_sampled = len(full_labeled_pool['inputs'])
            initial_labeled_size = self.config.get('initial_labeled_size', 100)
            print(f"  📊 Total samples loaded: {n_sampled}, initial_labeled_size: {initial_labeled_size}")
            if n_sampled == 0:
                raise ValueError(f"Loaded dataset is empty (n_sampled=0)")

            split_indices = torch.randperm(n_sampled, device='cpu', dtype=torch.long)
            labeled_local_indices = split_indices[:initial_labeled_size].long()
            unlabeled_local_indices = split_indices[initial_labeled_size:].long()

            # Ensure all indices are within valid range [0, n_sampled-1]
            max_valid_idx = n_sampled - 1
            labeled_local_indices = torch.clamp(labeled_local_indices, 0, max_valid_idx).long()
            unlabeled_local_indices = torch.clamp(unlabeled_local_indices, 0, max_valid_idx).long()

            self.labeled_indices_set = set(labeled_local_indices.tolist())
            # Mapping from unlabeled pool index (local) to full dataset index (also local since we sampled)
            self._unlabeled_to_full = unlabeled_local_indices.long()

            print(f"  📊 Initialized labeled pool: {len(self.labeled_indices_set)} samples from {n_sampled} total")
            print(f"  ✅ _store_full_coco_labels completed successfully")

        except Exception as e:
            print(f"  ❌ Error loading COCO dataset: {e}")
            import traceback
            traceback.print_exc()
            # Reset to None to indicate failure
            self.full_dataset_labels = None
            self.labeled_indices_set = set()
            raise RuntimeError(
                f"Failed to load COCO dataset. Check if images exist in {coco_root} and annotation file is valid: {ann_file}") from e

    def _store_full_nyuv2_labels(self):
        """Store full NYUv2 dataset labels for label retrieval.

        This loads the full dataset once and stores it on CPU.
        Only indices are tracked in labeled_indices_set.
        """
        nyuv2_root = self.config.get('nyuv2_root')
        nyuv2_split = self.config.get('nyuv2_split', 'train')
        unlabeled_pool_size = self.config.get('unlabeled_pool_size', 795)
        seed = self.config.get('seed', 42)

        try:
            print(f"  🔍 _store_full_nyuv2_labels: Starting...")
            print(f"  📂 Loading NYUv2 dataset from {nyuv2_root}...")
            print(f"  📂 Split: {nyuv2_split}, max_samples: {unlabeled_pool_size}, seed: {seed}")

            # Load labels for the sampled pool (same samples as unlabeled_pool) - keep on CPU
            _, full_labeled_pool = load_nyuv2_dataset(
                root=nyuv2_root,
                split=nyuv2_split,
                tasks=self.config['tasks'],
                image_size=(self.config.get('image_height', 480), self.config.get('image_width', 640)),
                device='cpu',  # Keep on CPU to save GPU memory
                max_samples=unlabeled_pool_size,
                seed=seed
            )

            print(
                f"  ✅ Loaded dataset, full_labeled_pool keys: {list(full_labeled_pool.keys()) if full_labeled_pool else 'None'}")
            if full_labeled_pool is None or 'inputs' not in full_labeled_pool:
                raise ValueError(f"Failed to load NYUv2 dataset: full_labeled_pool is invalid")

            # Store labels (for label retrieval when selecting samples) - ensure on CPU
            self.full_dataset_labels = {}
            for key, value in full_labeled_pool.items():
                if isinstance(value, torch.Tensor):
                    self.full_dataset_labels[key] = value.cpu() if value.device.type == 'cuda' else value
                else:
                    self.full_dataset_labels[key] = value
            self.full_dataset_indices = torch.arange(len(full_labeled_pool['inputs']), device='cpu')

            # Track which indices are in labeled pool (within the sampled pool)
            self._set_seeds(seed)

            # Split sampled pool into labeled/unlabeled
            n_sampled = len(full_labeled_pool['inputs'])
            initial_labeled_size = self.config.get('initial_labeled_size', 8)

            # Randomly select initial labeled samples
            all_indices = torch.randperm(n_sampled, generator=torch.Generator().manual_seed(seed))
            labeled_local_indices = all_indices[:initial_labeled_size]
            unlabeled_local_indices = all_indices[initial_labeled_size:]

            # Store mapping
            self.labeled_indices_set = set(labeled_local_indices.tolist())
            self._unlabeled_to_full = unlabeled_local_indices.long()

            print(f"  📊 Initialized labeled pool: {len(self.labeled_indices_set)} samples from {n_sampled} total")
            print(f"  ✅ _store_full_nyuv2_labels completed successfully")

        except Exception as e:
            print(f"  ❌ Error loading NYUv2 dataset: {e}")
            import traceback
            traceback.print_exc()
            self.full_dataset_labels = None
            self.labeled_indices_set = set()
            raise RuntimeError(f"Failed to initialize full_dataset_labels. Error: {e}")

    def _store_full_cityscapes_labels(self):
        """Store full Cityscapes dataset labels for label retrieval.

        This loads the full dataset once and stores it on CPU.
        Label retrieval happens by indexing into this stored data.
        """
        cityscapes_root = self.config.get('cityscapes_root')
        cityscapes_split = self.config.get('cityscapes_split', 'train')
        unlabeled_pool_size = self.config.get('unlabeled_pool_size', 2975)
        seed = self.config.get('seed', 42)

        try:
            print(f"  🔍 _store_full_cityscapes_labels: Starting...")
            print(f"  📂 Loading Cityscapes dataset from {cityscapes_root}...")
            print(f"  📂 Split: {cityscapes_split}, max_samples: {unlabeled_pool_size}, seed: {seed}")

            # Load labels for the sampled pool (same samples as unlabeled_pool) - keep on CPU
            _, full_labeled_pool = load_cityscapes_dataset(
                root=cityscapes_root,
                split=cityscapes_split,
                tasks=self.config['tasks'],
                image_size=(self.config.get('image_height', 512), self.config.get('image_width', 512)),
                device='cpu',  # Keep on CPU to save GPU memory
                max_samples=unlabeled_pool_size,
                seed=seed
            )

            print(
                f"  ✅ Loaded dataset, full_labeled_pool keys: {list(full_labeled_pool.keys()) if full_labeled_pool else 'None'}")
            if full_labeled_pool is None or 'inputs' not in full_labeled_pool:
                raise ValueError(f"Failed to load Cityscapes dataset: full_labeled_pool is invalid")

            # Store labels (for label retrieval when selecting samples) - ensure on CPU
            self.full_dataset_labels = {}
            for key, value in full_labeled_pool.items():
                if isinstance(value, torch.Tensor):
                    self.full_dataset_labels[key] = value.cpu() if value.device.type == 'cuda' else value
                else:
                    self.full_dataset_labels[key] = value
            self.full_dataset_indices = torch.arange(len(full_labeled_pool['inputs']), device='cpu')

            # Track which indices are in labeled pool (within the sampled pool)
            self._set_seeds(seed)

            # Split sampled pool into labeled/unlabeled
            n_sampled = len(full_labeled_pool['inputs'])
            initial_labeled_size = self.config.get('initial_labeled_size', 30)

            # Randomly select initial labeled samples
            all_indices = torch.randperm(n_sampled, generator=torch.Generator().manual_seed(seed))
            labeled_local_indices = all_indices[:initial_labeled_size]
            unlabeled_local_indices = all_indices[initial_labeled_size:]

            # Store mapping
            self.labeled_indices_set = set(labeled_local_indices.tolist())
            self._unlabeled_to_full = unlabeled_local_indices.long()

            print(f"  📊 Initialized labeled pool: {len(self.labeled_indices_set)} samples from {n_sampled} total")
            print(f"  ✅ _store_full_cityscapes_labels completed successfully")

        except Exception as e:
            print(f"  ❌ Error loading Cityscapes dataset: {e}")
            import traceback
            traceback.print_exc()
            self.full_dataset_labels = None
            self.labeled_indices_set = set()
            raise RuntimeError(f"Failed to initialize full_dataset_labels. Error: {e}")

    def _load_cityscapes_unlabeled_pool(self) -> torch.Tensor:
        """Load Cityscapes dataset as unlabeled pool.

        For Cityscapes, we already loaded the data in _store_full_cityscapes_labels.
        This function returns only the unlabeled samples (excluding initial labeled ones).
        """
        # Data is already loaded in _store_full_cityscapes_labels
        if self.full_dataset_labels is None or 'inputs' not in self.full_dataset_labels:
            raise RuntimeError("full_dataset_labels not initialized. Call _store_full_cityscapes_labels first.")

        # Get unlabeled indices from _unlabeled_to_full mapping
        if hasattr(self, '_unlabeled_to_full') and self._unlabeled_to_full is not None:
            unlabeled_pool = self.full_dataset_labels['inputs'][self._unlabeled_to_full]
            print(
                f"  📊 Using {len(unlabeled_pool):,} unlabeled images from {len(self.full_dataset_labels['inputs']):,} total (on CPU)")
        else:
            # Fallback: return all images
            unlabeled_pool = self.full_dataset_labels['inputs']
            print(
                f"  ⚠️  Warning: _unlabeled_to_full not found, using all {len(unlabeled_pool):,} images as unlabeled pool (on CPU)")

        return unlabeled_pool

    def _load_test_set(self):
        """Load test set for evaluation."""
        if self.config.get('dataset') == 'coco':
            coco_test_root = self.config.get('coco_test_root', None)
            coco_test_ann_file = self.config.get('coco_test_ann_file', None)

            if coco_test_root is None or coco_test_ann_file is None:
                print("⚠️  Warning: Test set not specified. Using labeled pool for evaluation.")
                self.test_set = None
                return

            try:
                print(f"📊 Loading test set from {coco_test_root}...")
                # Load test set on CPU to save GPU memory
                test_set_loaded = load_coco_test_dataset(
                    root=coco_test_root,
                    ann_file=coco_test_ann_file,
                    tasks=self.config['tasks'],
                    image_size=(self.config.get('image_height', 64), self.config.get('image_width', 64)),
                    max_boxes=self.config.get('num_boxes', 5),
                    num_classes=self.config.get('num_classes', 80),
                    device='cpu'  # Keep on CPU, will move to GPU batch-by-batch during evaluation
                )
                # Ensure test_set is on CPU
                self.test_set = {}
                for key, value in test_set_loaded.items():
                    if isinstance(value, torch.Tensor):
                        self.test_set[key] = value.cpu() if value.device.type == 'cuda' else value
                    else:
                        self.test_set[key] = value
                print(f"✅ Test set loaded: {len(self.test_set['inputs'])} samples (on CPU)")
            except Exception as e:
                print(f"⚠️  Warning: Could not load test set: {e}")
                print("   Using labeled pool for evaluation instead.")
                self.test_set = None
        elif self.config.get('dataset') == 'nyuv2':
            nyuv2_root = self.config.get('nyuv2_root')
            nyuv2_test_split = self.config.get('nyuv2_test_split', 'test')

            if nyuv2_root is None:
                print("⚠️  Warning: NYUv2 root not specified. Using labeled pool for evaluation.")
                self.test_set = None
                return

            try:
                print(f"📊 Loading NYUv2 test set from {nyuv2_root}...")
                # Load test set on CPU
                _, test_set_loaded = load_nyuv2_dataset(
                    root=nyuv2_root,
                    split=nyuv2_test_split,
                    tasks=self.config['tasks'],
                    image_size=(self.config.get('image_height', 480), self.config.get('image_width', 640)),
                    device='cpu'  # Keep on CPU
                )
                # Ensure test_set is on CPU
                self.test_set = {}
                for key, value in test_set_loaded.items():
                    if isinstance(value, torch.Tensor):
                        self.test_set[key] = value.cpu() if value.device.type == 'cuda' else value
                    else:
                        self.test_set[key] = value
                print(f"✅ Test set loaded: {len(self.test_set['inputs'])} samples (on CPU)")
            except Exception as e:
                print(f"⚠️  Warning: Could not load NYUv2 test set: {e}")
                print("   Using labeled pool for evaluation instead.")
                self.test_set = None
        elif self.config.get('dataset') == 'cityscapes':
            cityscapes_root = self.config.get('cityscapes_root')
            cityscapes_test_split = self.config.get('cityscapes_test_split', 'val')

            if cityscapes_root is None:
                print("⚠️  Warning: Cityscapes root not specified. Using labeled pool for evaluation.")
                self.test_set = None
                return

            try:
                print(f"📊 Loading Cityscapes test set from {cityscapes_root}...")
                # Load test set on CPU
                _, test_set_loaded = load_cityscapes_dataset(
                    root=cityscapes_root,
                    split=cityscapes_test_split,
                    tasks=self.config['tasks'],
                    image_size=(self.config.get('image_height', 512), self.config.get('image_width', 512)),
                    device='cpu'  # Keep on CPU
                )
                # Ensure test_set is on CPU
                self.test_set = {}
                for key, value in test_set_loaded.items():
                    if isinstance(value, torch.Tensor):
                        self.test_set[key] = value.cpu() if value.device.type == 'cuda' else value
                    else:
                        self.test_set[key] = value
                print(f"✅ Test set loaded: {len(self.test_set['inputs'])} samples (on CPU)")
            except Exception as e:
                print(f"⚠️  Warning: Could not load Cityscapes test set: {e}")
                print("   Using labeled pool for evaluation instead.")
                self.test_set = None
        else:
            print("⚠️  Warning: Test set loading not implemented for this dataset.")
            self.test_set = None

    def _create_selectors(self) -> Dict[str, any]:
        """Create selectors for different methods."""
        selectors = {}

        # Random selector
        selectors['random'] = make_selector('random', self.config)

        # Uncertainty selector
        config_with_task_types = self.config.copy()
        config_with_task_types['task_types'] = {task: task for task in self.config['tasks']}
        selectors['uncertainty'] = make_selector(
            'uncertainty',
            config_with_task_types,
            model=self.model
        )

        # Diversity selector
        selectors['diversity'] = make_selector(
            'diversity',
            config_with_task_types,
            model=self.model
        )

        # CoreSet selector
        selectors['coreset'] = make_selector(
            'coreset',
            config_with_task_types,
            model=self.model
        )

        # BADGE / MT-BADGE (same selector: multi-task gradient concat + k-means++)
        selectors['badge'] = make_selector(
            'badge',
            config_with_task_types,
            model=self.model
        )
        selectors['mt-badge'] = make_selector(
            'mt-badge',
            config_with_task_types,
            model=self.model
        )

        # BatchBALD selector
        selectors['batchbald'] = make_selector(
            'batchbald',
            config_with_task_types,
            model=self.model
        )

        # Clustering selector
        selectors['clustering'] = make_selector(
            'clustering',
            config_with_task_types,
            model=self.model
        )

        # Hekimoglu et al. (WACV 2024) selector
        selectors['hekimoglu'] = make_selector(
            'hekimoglu',
            config_with_task_types,
            model=self.model
        )

        # PartAL (Durasov et al., 2022) selector
        selectors['partial'] = make_selector(
            'partial',
            config_with_task_types,
            model=self.model
        )

        # NeurIPS rebuttal baselines: filter-then-diversify methods cited by reviewers
        for rebuttal_method in ('cluster-margin', 'ppal', 'fass', 'd2pruning', 'ses',
                                'cluster-margin-ctmi'):
            selectors[rebuttal_method] = make_selector(
                rebuttal_method,
                config_with_task_types,
                model=self.model
            )

        # Our method variants: ours, ours-ctmi, ours-synergy
        ours_variants = {
            'ours': 'consistency',       # Default: confidence consistency
            'ours-ctmi': 'ctmi',          # Cross-Task Mutual Information
            'ours-synergy': 'synergy'     # Task Synergy Score
        }

        methods_list = self.config.get('methods', ['ours'])
        for variant_name, third_mode in ours_variants.items():
            if variant_name in methods_list:
                acquisition_function = MultiTaskActiveLearningAcquisition(
                    model=self.model,
                    alpha=self.config.get('alpha', 1.0),
                    beta=self.config.get('beta', 1.0),
                    gamma=self.config.get('gamma', 1.0),
                    delta=self.config.get('delta', 0.2),  # Diversity weight
                    hardness_mode=self.config.get('hardness_mode', 'entropy'),
                    conflict_mode=self.config.get('conflict_mode', 'cosine'),
                    consistency_mode=self.config.get('consistency_mode', 'kl'),
                    diversity_mode=self.config.get('diversity_mode', 'euclidean'),
                    tasks=self.config['tasks'],
                    criteria=self.config['criteria'],
                    auto_balance_weights=self.config.get('auto_balance_weights', True),
                    device=self.device,
                    third_component_mode=third_mode  # consistency, ctmi, or synergy
                )

                selectors[variant_name] = make_selector(
                    'ours',  # Use 'ours' selector logic
                    self.config,
                    acquisition_function=acquisition_function
                )

        # Two-Stage Orthogonal Selection (ODS) - our main contribution
        # Stage 1: Informativeness (Hardness + CTMI)
        # Stage 2: Diversity (K-Center Greedy)
        if 'ours-2stage' in methods_list:
            third_mode = self.config.get('third_component_mode', 'ctmi')
            acquisition_function = MultiTaskActiveLearningAcquisition(
                model=self.model,
                alpha=self.config.get('alpha', 1.0),
                beta=self.config.get('beta', 0.0),  # Conflict excluded in 2-stage
                gamma=self.config.get('gamma', 1.0),
                delta=self.config.get('delta', 0.0),  # Diversity handled in Stage 2
                hardness_mode=self.config.get('hardness_mode', 'entropy'),
                conflict_mode=self.config.get('conflict_mode', 'cosine'),
                consistency_mode=self.config.get('consistency_mode', 'kl'),
                diversity_mode=self.config.get('diversity_mode', 'euclidean'),
                tasks=self.config['tasks'],
                criteria=self.config['criteria'],
                auto_balance_weights=False,  # Manual control for 2-stage
                device=self.device,
                third_component_mode=third_mode,
                ctmi_use_mc_dropout=self.config.get('ctmi_use_mc_dropout', True),
                ctmi_num_samples=self.config.get('ctmi_num_samples', 10),
                ctmi_mode=self.config.get('ctmi_mode', 'covariance_negative'),  # Default to negative (1 - cov)
                score_normalization=self.config.get('score_normalization', 'minmax'),
                score_combination=self.config.get('score_combination', 'sum')
            )

            selectors['ours-2stage'] = make_selector(
                'ours-2stage',
                self.config,
                acquisition_function=acquisition_function
            )

        # Linear Scalarization baseline: score = α·H + γ·CTMI + δ·Diversity (single-stage top-k)
        if 'linear-scalarization' in methods_list:
            third_mode = self.config.get('third_component_mode', 'ctmi')
            acquisition_function = MultiTaskActiveLearningAcquisition(
                model=self.model,
                alpha=self.config.get('alpha', 1.0),
                beta=self.config.get('beta', 0.0),
                gamma=self.config.get('gamma', 1.0),
                delta=self.config.get('delta', 0.0),
                hardness_mode=self.config.get('hardness_mode', 'entropy'),
                conflict_mode=self.config.get('conflict_mode', 'cosine'),
                consistency_mode=self.config.get('consistency_mode', 'kl'),
                diversity_mode=self.config.get('diversity_mode', 'euclidean'),
                tasks=self.config['tasks'],
                criteria=self.config['criteria'],
                auto_balance_weights=False,
                device=self.device,
                third_component_mode=third_mode,
                ctmi_use_mc_dropout=self.config.get('ctmi_use_mc_dropout', True),
                ctmi_num_samples=self.config.get('ctmi_num_samples', 10),
                ctmi_mode=self.config.get('ctmi_mode', 'covariance_negative')
            )

            selectors['linear-scalarization'] = make_selector(
                'linear-scalarization',
                self.config,
                acquisition_function=acquisition_function
            )

        return selectors

    def _pcgrad_projection(self, task_grads: torch.Tensor) -> torch.Tensor:
        """
        PCGrad projection to reduce gradient conflicts.

        For 2 tasks: considers cls-det pair
        For 3 tasks: considers cls-det, cls-seg, det-seg all pairs

        Each task's gradient is projected onto the orthogonal complement
        of other tasks' gradients to resolve conflicts.

        Args:
            task_grads: Stacked gradients [num_tasks, grad_dim]

        Returns:
            Projected gradients [num_tasks, grad_dim]
        """
        num_tasks = len(task_grads)
        projected_grads = []

        # For each task, project its gradient onto orthogonal complement of other tasks
        for i in range(num_tasks):
            grad_i = task_grads[i].clone()

            # Project task i's gradient onto orthogonal complement of each other task
            for j in range(num_tasks):
                if i != j:
                    # Use original gradient j (not projected) for projection
                    # This ensures we project based on original conflicts, not sequential updates
                    dot_product = torch.dot(grad_i, task_grads[j])
                    if dot_product < 0:  # Only project if gradients conflict
                        projection = dot_product / (torch.norm(task_grads[j]) ** 2 + 1e-8)
                        grad_i = grad_i - projection * task_grads[j]

            projected_grads.append(grad_i)

        return torch.stack(projected_grads)

    def _train_model(self, model: nn.Module, labeled_indices: List[int] = None,
                     round_num: int = None, method_save_model_path: str = None,
                     logger: Optional[logging.Logger] = None) -> Tuple[nn.Module, Optional[Dict[str, Dict[str, float]]]]:
        """Train model on labeled data.

        Args:
            model: Model to train
            labeled_indices: List of indices in full_dataset_labels to use for training
                           If None, uses self.labeled_indices_set
            round_num: Round number for logging
            method_save_model_path: Method-specific path for saving models
            logger: Logger instance for logging training progress
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        # _print_memory_usage(self.device, f"[Round {round_num}] _train_model START:")
        model = model.to(self.device)

        # Get labeled indices
        if labeled_indices is None:
            labeled_indices = list(self.labeled_indices_set)

        if len(labeled_indices) == 0:
            raise ValueError(
                f"No labeled indices available for training. "
                f"labeled_indices_set size: {len(self.labeled_indices_set)}, "
                f"full_dataset_labels: {'exists' if self.full_dataset_labels is not None else 'None'}"
            )

        # Ensure indices are within valid range
        max_idx = len(self.full_dataset_labels['inputs']) - 1
        labeled_indices = [idx for idx in labeled_indices if 0 <= idx <= max_idx]

        if len(labeled_indices) == 0:
            raise ValueError(
                f"No valid labeled indices after filtering. "
                f"max_idx: {max_idx}, original indices: {list(self.labeled_indices_set)[:10]}..."
            )

        # Don't load labeled_data upfront - load batch-by-batch from full_dataset_labels
        # This prevents memory accumulation as labeled_indices_set grows
        n_samples = len(labeled_indices)
        # _print_memory_usage(self.device, f"[Round {round_num}] Starting training with {n_samples} labeled samples (batch-by-batch loading):")

        model.train()
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning Rate Scheduler: Configurable (StepLR for more aggressive decay)
        num_epochs = self.config.get('num_epochs', 10)
        scheduler_type = self.config.get('lr_scheduler', 'cosine').lower()

        if scheduler_type == 'step':
            step_size = self.config.get('lr_step_size', 30)
            gamma = self.config.get('lr_gamma', 0.5)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'plateau':
            factor = self.config.get('lr_factor', 0.5)
            patience = self.config.get('lr_patience', 10)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, verbose=True)
        else:  # cosine (default)
            eta_min = self.config.get('lr_eta_min', learning_rate * 0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)

        # Use fixed loss weights from config
        adaptive_loss_weights = self.config.get('loss_weights', {}).copy()

        loss_functions = {}
        logger.info(f"  🔧 Building loss functions for tasks: {self.config['tasks']}")
        for task in self.config['tasks']:
            if task == 'classification':
                loss_functions['classification'] = nn.CrossEntropyLoss()
            elif task == 'detection':
                def detection_loss(pred, target):
                    """
                    Detection loss with per-sample normalization for stable training.

                    Fixes loss fluctuation by:
                    1. Computing per-sample loss
                    2. Averaging over batch size (consistent normalization)
                    3. Handling zero valid boxes gracefully
                    """
                    num_classes = pred.shape[2] - 4
                    batch_size = pred.shape[0]

                    cls_pred = pred[:, :, :num_classes]
                    cls_target = target[:, :, :num_classes]
                    bbox_pred = pred[:, :, num_classes:]
                    bbox_target = target[:, :, num_classes:]

                    # Per-sample loss calculation for stable batch normalization
                    per_sample_losses = []

                    for i in range(batch_size):
                        # Find valid boxes for this sample
                        valid_mask_i = cls_target[i].sum(dim=-1) > 0  # [num_boxes]

                        if valid_mask_i.sum() == 0:
                            # No valid boxes: use small constant loss (prevents sudden drop to 0)
                            per_sample_losses.append(torch.tensor(0.01, device=pred.device, requires_grad=True))
                            continue

                        # Compute loss for valid boxes only
                        cls_pred_i_valid = cls_pred[i][valid_mask_i]  # [num_valid_i, num_classes]
                        cls_target_i_valid = cls_target[i][valid_mask_i]  # [num_valid_i, num_classes]
                        cls_target_i_flat = torch.argmax(cls_target_i_valid, dim=1)

                        cls_loss_i = F.cross_entropy(cls_pred_i_valid, cls_target_i_flat, reduction='mean')

                        bbox_pred_i_valid = bbox_pred[i][valid_mask_i]
                        bbox_target_i_valid = bbox_target[i][valid_mask_i]
                        bbox_loss_i = F.l1_loss(bbox_pred_i_valid, bbox_target_i_valid, reduction='mean')

                        per_sample_losses.append(cls_loss_i + bbox_loss_i)

                    # Average over batch size (consistent normalization across batches)
                    return torch.stack(per_sample_losses).mean()
                loss_functions['detection'] = detection_loss
            elif task == 'segmentation':
                # Custom segmentation loss with class weighting and ignore_index handling
                # Note: PyTorch's CrossEntropyLoss with weight + ignore_index can have issues
                # when ignore_index (255) is outside valid class range and targets are used to index weights
                num_classes = self.config.get('num_classes', 80)
                ignore_idx = 255

                def segmentation_loss(pred, target):
                    # pred: [B, C, H, W], target: [B, H, W]
                    # Create valid mask (exclude ignore_index)
                    valid_mask = (target != ignore_idx)

                    if not valid_mask.any():
                        # No valid pixels, return zero loss
                        return torch.tensor(0.0, device=pred.device, requires_grad=True)

                    # Flatten for easier indexing
                    B, C, H, W = pred.shape
                    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
                    target_flat = target.reshape(-1)  # [B*H*W]
                    valid_flat = valid_mask.reshape(-1)  # [B*H*W]

                    # Select only valid pixels
                    pred_valid = pred_flat[valid_flat]  # [N_valid, C]
                    target_valid = target_flat[valid_flat]  # [N_valid]

                    # Compute cross entropy loss on valid pixels only
                    # Apply class weighting: lower weight for background (class 0)
                    class_weights = torch.ones(num_classes, device=pred.device)
                    class_weights[0] = 0.1
                    loss = F.cross_entropy(pred_valid, target_valid, weight=class_weights)
                    return loss

                loss_functions['segmentation'] = segmentation_loss
            elif task == 'depth':
                # Depth estimation: use MSE loss (or L1 for robustness)
                # Scale the loss to match segmentation loss magnitude
                def depth_loss(pred, target):
                    # pred: [B, 1, H, W], target: [B, 1, H, W] or [B, H, W]
                    if target.dim() == 3:
                        target = target.unsqueeze(1)
                    # Resize pred to match target size if needed
                    if pred.shape[-2:] != target.shape[-2:]:
                        pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
                    # Use smooth L1 loss (Huber loss) for robustness
                    return F.smooth_l1_loss(pred, target)
                loss_functions['depth'] = depth_loss
            elif task == 'normal':
                # Surface normal prediction: use cosine similarity loss + L1 loss
                def normal_loss(pred, target, _logger=logger):
                    # pred: [B, 3, H, W], target: [B, 3, H, W]
                    # Debug: log shapes on first call
                    if not hasattr(normal_loss, '_first_call_done'):
                        normal_loss._first_call_done = True
                        _logger.info(f"  🔍 Normal loss - pred shape: {pred.shape}, target shape: {target.shape}")
                        _logger.info(
                            f"  🔍 Normal loss - pred range: [{pred.min():.4f}, {pred.max():.4f}], target range: [{target.min():.4f}, {target.max():.4f}]")
                    # Resize pred to match target size if needed
                    if pred.shape[-2:] != target.shape[-2:]:
                        pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
                    # Normalize predictions to unit vectors
                    pred_norm = F.normalize(pred, p=2, dim=1)
                    target_norm = F.normalize(target, p=2, dim=1)
                    # Cosine similarity loss (1 - cos_sim)
                    cos_sim = (pred_norm * target_norm).sum(dim=1)  # [B, H, W]
                    cos_loss = (1 - cos_sim).mean()
                    # L1 loss for additional supervision
                    l1_loss = F.l1_loss(pred_norm, target_norm)
                    total_loss = cos_loss + 0.5 * l1_loss
                    # Debug: log loss value on first call
                    if not hasattr(normal_loss, '_loss_logged'):
                        normal_loss._loss_logged = True
                        _logger.info(
                            f"  🔍 Normal loss computed: cos={cos_loss.item():.4f}, l1={l1_loss.item():.4f}, total={total_loss.item():.4f}")
                    return total_loss
                loss_functions['normal'] = normal_loss
            elif task == 'instance':
                # Instance segmentation: convert to binary foreground/background task
                # Instance IDs > 0 are foreground, 0 is background
                def instance_loss(pred, target):
                    # pred: [B, num_classes, H, W], target: [B, H, W] with instance IDs
                    # Convert instance IDs to binary mask (0=background, 1=foreground)
                    binary_target = (target > 0).long()
                    # Resize pred to match target if needed
                    if pred.shape[-2:] != target.shape[-2:]:
                        pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
                    # Use only first 2 channels for binary classification
                    if pred.shape[1] > 2:
                        pred = pred[:, :2, :, :]
                    return F.cross_entropy(pred, binary_target, ignore_index=255)
                loss_functions['instance'] = instance_loss

        logger.info(f"  ✅ Loss functions created: {list(loss_functions.keys())}")

        # Training loop with batch processing to avoid memory issues
        train_batch_size = self.config.get('train_batch_size', 32)  # Use train_batch_size from config
        epoch_pbar = tqdm(range(num_epochs), desc="Training", leave=True)

        # Track best loss for model saving and early stopping
        best_loss = float('inf')
        best_model_state = None
        best_metrics = None
        early_stopping_patience = self.config.get('early_stopping_patience', None)
        no_improvement_count = 0

        for epoch in epoch_pbar:
            indices = torch.randperm(n_samples, device='cpu')
            epoch_task_losses = {}
            epoch_batch_sizes = {}  # Track batch sizes for weighted averaging
            num_batches = 0

            # Clear previous epoch's GPU cache and gradients
            if epoch > 0 and self.device.type == 'cuda':
                optimizer.zero_grad()  # Clear any remaining gradients
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            for batch_start in range(0, n_samples, train_batch_size):
                batch_end = min(batch_start + train_batch_size, n_samples)
                actual_batch_size = batch_end - batch_start  # Actual batch size (may be smaller for last batch)
                # Get indices into labeled_indices list (not into labeled_data)
                batch_labeled_idx_positions = indices[batch_start:batch_end]
                # Map to actual full_dataset_labels indices
                batch_labeled_indices = [labeled_indices[i] for i in batch_labeled_idx_positions]

                # Load batch data directly from full_dataset_labels (batch-by-batch)
                batch_inputs = self.full_dataset_labels['inputs'][batch_labeled_indices].to(
                    self.device, non_blocking=False)
                batch_data = {'inputs': batch_inputs}
                for task in self.config['tasks']:
                    if task in self.full_dataset_labels:
                        batch_data[task] = self.full_dataset_labels[task][batch_labeled_indices].to(
                            self.device, non_blocking=False)

                optimizer.zero_grad()
                # Get image size from config for anchor-based detection
                image_size = (self.config.get('image_height', 480),
                              self.config.get('image_width', 480))
                outputs = model(batch_inputs, image_size=image_size)

                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                batch_task_losses = {}
                task_losses = {}

                # Compute losses for each task
                for task_name, loss_fn in loss_functions.items():
                    if task_name in outputs and task_name in batch_data:
                        # Get output and target
                        output = outputs[task_name]
                        target = batch_data[task_name]

                        # Compute loss
                        loss = loss_fn(output, target)
                        task_losses[task_name] = loss
                        batch_task_losses[task_name] = loss.item()
                    else:
                        # Debug: log when task is skipped
                        if epoch == 0 and num_batches == 0:
                            in_outputs = task_name in outputs
                            in_batch_data = task_name in batch_data
                            logger.info(
                                f"  ⚠️ Task '{task_name}' skipped: in_outputs={in_outputs}, in_batch_data={in_batch_data}")

                # Check if we have any losses
                if len(task_losses) == 0:
                    # No losses computed, skip this batch
                    del outputs, batch_inputs, batch_data
                    continue

                # Check if PCGrad is enabled
                use_pcgrad = self.config.get('use_pcgrad', False)

                if use_pcgrad and len(task_losses) > 1:
                    # PCGrad: Compute gradients per task and project conflicting gradients
                    # Get shared parameters (encoder parameters)
                    shared_params = []
                    for name, param in model.named_parameters():
                        if 'shared_encoder' in name and param.requires_grad:
                            shared_params.append(param)

                    if shared_params:
                        # Collect task gradients
                        task_grads = []
                        task_names_list = list(task_losses.keys())

                        # Compute gradient for each task
                        for task_name in task_names_list:
                            optimizer.zero_grad()
                            loss = task_losses[task_name]
                            weight = adaptive_loss_weights.get(
                                task_name, self.config.get('loss_weights', {}).get(task_name, 1.0))
                            weighted_loss = weight * loss

                            # Backward for this task only
                            weighted_loss.backward(retain_graph=True)

                            # Collect gradient from shared encoder
                            grad_list = []
                            for param in shared_params:
                                if param.grad is not None:
                                    grad_list.append(param.grad.view(-1))

                            if grad_list:
                                task_grad = torch.cat(grad_list)
                                task_grads.append(task_grad)
                            else:
                                # Fallback: use zero gradient
                                total_dim = sum(p.numel() for p in shared_params)
                                task_grads.append(torch.zeros(total_dim, device=self.device))

                        # Apply PCGrad projection
                        if len(task_grads) > 1:
                            task_grads_tensor = torch.stack(task_grads)  # [num_tasks, grad_dim]
                            projected_grads = self._pcgrad_projection(task_grads_tensor)

                            # Set projected gradients back to parameters
                            optimizer.zero_grad()
                            grad_idx = 0
                            for param in shared_params:
                                param_size = param.numel()
                                # Average projected gradients across tasks
                                avg_grad = projected_grads[:, grad_idx:grad_idx + param_size].mean(dim=0)
                                param.grad = avg_grad.view(param.shape)
                                grad_idx += param_size

                            # For task-specific parameters, accumulate gradients normally
                            # (PCGrad only applies to shared parameters)
                            optimizer.zero_grad()
                            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                            for task_name, loss in task_losses.items():
                                weight = adaptive_loss_weights.get(
                                    task_name, self.config.get('loss_weights', {}).get(task_name, 1.0))
                                weighted_loss = weight * loss
                                total_loss = total_loss + weighted_loss

                            # Backward for task-specific parameters only
                            # (shared parameters already have projected gradients)
                            total_loss.backward()

                            # Merge: keep projected gradients for shared params, use normal gradients for task-specific
                            # (gradients are already set correctly above)
                        else:
                            # Single task: use normal backward
                            optimizer.zero_grad()
                            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                            for task_name, loss in task_losses.items():
                                weight = adaptive_loss_weights.get(
                                    task_name, self.config.get('loss_weights', {}).get(task_name, 1.0))
                                weighted_loss = weight * loss
                                total_loss = total_loss + weighted_loss
                            total_loss.backward()
                    else:
                        # No shared parameters found, use normal backward
                        optimizer.zero_grad()
                        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        for task_name, loss in task_losses.items():
                            weight = adaptive_loss_weights.get(
                                task_name, self.config.get('loss_weights', {}).get(task_name, 1.0))
                            weighted_loss = weight * loss
                            total_loss = total_loss + weighted_loss
                        total_loss.backward()
                else:
                    # Normal training without PCGrad
                    # Compute weighted total loss
                    total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    for task_name, loss in task_losses.items():
                        weight = adaptive_loss_weights.get(
                            task_name, self.config.get('loss_weights', {}).get(task_name, 1.0))
                        weighted_loss = weight * loss
                        total_loss = total_loss + weighted_loss

                    # Backward and step
                    total_loss.backward()

                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()  # Clear gradients immediately after step

                # Clear task losses and outputs
                del task_losses
                for task_name in list(outputs.keys()):
                    if task_name in outputs:
                        del outputs[task_name]
                del outputs, total_loss

                for task_name, loss_val in batch_task_losses.items():
                    if task_name not in epoch_task_losses:
                        epoch_task_losses[task_name] = []
                        epoch_batch_sizes[task_name] = []
                    epoch_task_losses[task_name].append(loss_val)
                    epoch_batch_sizes[task_name].append(actual_batch_size)

                # Debug: log first batch losses for each epoch
                if num_batches == 0 and epoch == 0:
                    logger.info(f"  📊 First batch losses: {batch_task_losses}")

                num_batches += 1

                # Clear batch tensors to free memory immediately
                del batch_inputs, batch_data, batch_task_losses
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for GPU operations to complete

            # Compute metrics for progress bar (on training data) - sample subset to save memory
            model.eval()
            eval_sample_size = min(256, n_samples)  # Further reduce to avoid OOM
            eval_idx_positions = indices[:eval_sample_size]
            # Map to actual full_dataset_labels indices
            eval_labeled_indices = [labeled_indices[i] for i in eval_idx_positions]

            # Process eval in smaller batches to avoid OOM (especially for segmentation)
            eval_batch_size_for_metrics = min(64, eval_sample_size)  # Small batch for eval
            train_metrics = {}

            with torch.no_grad():
                # Process eval data in batches and accumulate metrics
                task_metrics_accumulators = {task: {} for task in self.config['tasks']}

                for eval_batch_start in range(0, eval_sample_size, eval_batch_size_for_metrics):
                    eval_batch_end = min(eval_batch_start + eval_batch_size_for_metrics, eval_sample_size)
                    eval_batch_indices = eval_labeled_indices[eval_batch_start:eval_batch_end]

                    # Load eval batch
                    eval_inputs_batch = self.full_dataset_labels['inputs'][eval_batch_indices].to(self.device)
                    train_outputs_batch = model(eval_inputs_batch)

                    eval_data_batch = {'inputs': eval_inputs_batch}
                    for task in self.config['tasks']:
                        if task in self.full_dataset_labels:
                            eval_data_batch[task] = self.full_dataset_labels[task][eval_batch_indices].to(self.device)

                    # Compute metrics for this batch
                    for task in self.config['tasks']:
                        if task == 'classification' and 'classification' in train_outputs_batch:
                            batch_metrics = PerTaskMetrics.classification_metrics(
                                train_outputs_batch['classification'],
                                eval_data_batch['classification']
                            )
                            if not task_metrics_accumulators[task]:
                                task_metrics_accumulators[task] = {k: [v] for k, v in batch_metrics.items()}
                            else:
                                for k, v in batch_metrics.items():
                                    if k in task_metrics_accumulators[task]:
                                        task_metrics_accumulators[task][k].append(v)
                        elif task == 'detection' and 'detection' in train_outputs_batch:
                            batch_metrics = PerTaskMetrics.detection_metrics(
                                train_outputs_batch['detection'],
                                eval_data_batch['detection']
                            )
                            if not task_metrics_accumulators[task]:
                                task_metrics_accumulators[task] = {k: [v] for k, v in batch_metrics.items()}
                            else:
                                for k, v in batch_metrics.items():
                                    if k in task_metrics_accumulators[task]:
                                        task_metrics_accumulators[task][k].append(v)
                        elif task == 'segmentation' and 'segmentation' in train_outputs_batch:
                            batch_metrics = PerTaskMetrics.segmentation_metrics(
                                train_outputs_batch['segmentation'],
                                eval_data_batch['segmentation']
                            )
                            if not task_metrics_accumulators[task]:
                                task_metrics_accumulators[task] = {k: [v] for k, v in batch_metrics.items()}
                            else:
                                for k, v in batch_metrics.items():
                                    if k in task_metrics_accumulators[task]:
                                        task_metrics_accumulators[task][k].append(v)
                        elif task == 'depth' and task in train_outputs_batch:
                            # Depth regression
                            batch_metrics = PerTaskMetrics.regression_metrics(
                                train_outputs_batch[task],
                                eval_data_batch[task]
                            )
                            if not task_metrics_accumulators[task]:
                                task_metrics_accumulators[task] = {k: [v] for k, v in batch_metrics.items()}
                            else:
                                for k, v in batch_metrics.items():
                                    if k in task_metrics_accumulators[task]:
                                        task_metrics_accumulators[task][k].append(v)
                                    else:
                                        task_metrics_accumulators[task][k] = [v]
                        elif task == 'normal' and task in train_outputs_batch:
                            # Surface normal prediction with angle error
                            batch_metrics = PerTaskMetrics.normal_metrics(
                                train_outputs_batch[task],
                                eval_data_batch[task]
                            )
                            if not task_metrics_accumulators[task]:
                                task_metrics_accumulators[task] = {k: [v] for k, v in batch_metrics.items()}
                            else:
                                for k, v in batch_metrics.items():
                                    if k in task_metrics_accumulators[task]:
                                        task_metrics_accumulators[task][k].append(v)
                                    else:
                                        task_metrics_accumulators[task][k] = [v]
                        elif task == 'instance' and 'instance' in train_outputs_batch:
                            # Instance segmentation: use instance_metrics with mAP
                            binary_target = (eval_data_batch['instance'] > 0).long()
                            batch_metrics = PerTaskMetrics.instance_metrics(
                                train_outputs_batch['instance'],
                                binary_target
                            )
                            if not task_metrics_accumulators[task]:
                                task_metrics_accumulators[task] = {k: [v] for k, v in batch_metrics.items()}
                            else:
                                for k, v in batch_metrics.items():
                                    if k in task_metrics_accumulators[task]:
                                        task_metrics_accumulators[task][k].append(v)
                                    else:
                                        task_metrics_accumulators[task][k] = [v]

                    # Clear batch immediately
                    del eval_inputs_batch, train_outputs_batch, eval_data_batch
                    if self.device.type == 'cuda':
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()

                # Average metrics across batches
                for task in self.config['tasks']:
                    if task_metrics_accumulators[task]:
                        metrics = {
                            k: sum(v) / len(v) if isinstance(v[0], (int, float)) else v[0]
                            for k, v in task_metrics_accumulators[task].items()
                        }
                        # Average loss across batches in this epoch
                        # Note: Each batch loss is already normalized by batch size
                        # (for detection: per-sample loss averaged over batch)
                        batch_losses = epoch_task_losses.get(task, [0.0])
                        # Weighted average by batch size to handle variable batch sizes
                        # This ensures consistent loss scale across batches of different sizes
                        batch_sizes = epoch_batch_sizes.get(task, [])
                        if batch_losses and batch_sizes and len(batch_losses) == len(batch_sizes):
                            total_samples = sum(batch_sizes)
                            weighted_sum = sum(loss * size for loss, size in zip(batch_losses, batch_sizes))
                            avg_loss = weighted_sum / total_samples if total_samples > 0 else 0.0
                        else:
                            # Fallback to simple average if batch sizes not tracked
                            avg_loss = sum(batch_losses) / max(len(batch_losses), 1)

                        if task == 'classification':
                            train_metrics[task] = {
                                'acc': metrics['accuracy'],
                                'loss': avg_loss
                            }
                        elif task == 'detection':
                            train_metrics[task] = {
                                'acc': metrics['accuracy'],
                                'box_l1': metrics['box_l1'],
                                'loss': avg_loss
                            }
                        elif task == 'segmentation':
                            train_metrics[task] = {
                                'dice': metrics['dice_mean'],
                                'iou': metrics['iou_mean'],
                                'loss': avg_loss
                            }
                        elif task == 'depth':
                            # Depth regression
                            train_metrics[task] = {
                                'mse': metrics.get('mse', 0.0),
                                'mae': metrics.get('mae', 0.0),
                                'rmse': metrics.get('rmse', 0.0),
                                'loss': avg_loss
                            }
                        elif task == 'normal':
                            # Surface normal with angle error
                            train_metrics[task] = {
                                'mse': metrics.get('mse', 0.0),
                                'mae': metrics.get('mae', 0.0),
                                'rmse': metrics.get('rmse', 0.0),
                                'mErr': metrics.get('mErr', 0.0),  # Mean angle error in degrees
                                'loss': avg_loss
                            }
                        elif task == 'instance':
                            # Instance segmentation with mAP
                            train_metrics[task] = {
                                'mAP': metrics.get('mAP', 0.0),
                                'dice': metrics.get('dice_mean', 0.0),
                                'iou': metrics.get('iou_mean', 0.0),
                                'loss': avg_loss
                            }
                    elif task == 'depth':
                        # Fallback for depth regression if no metrics were computed
                        batch_losses = epoch_task_losses.get(task, [0.0])
                        batch_sizes = epoch_batch_sizes.get(task, [])
                        if batch_losses and batch_sizes and len(batch_losses) == len(batch_sizes):
                            total_samples = sum(batch_sizes)
                            weighted_sum = sum(loss * size for loss, size in zip(batch_losses, batch_sizes))
                            avg_loss = weighted_sum / total_samples if total_samples > 0 else 0.0
                        else:
                            avg_loss = sum(batch_losses) / max(len(batch_losses), 1) if batch_losses else 0.0
                        train_metrics[task] = {
                            'mse': avg_loss if avg_loss > 0 else 0.0,
                            'mae': avg_loss * 0.8 if avg_loss > 0 else 0.0,
                            'rmse': np.sqrt(avg_loss) if avg_loss > 0 else 0.0,
                            'loss': avg_loss
                        }
                    elif task == 'normal':
                        # Fallback for normal prediction if no metrics were computed
                        batch_losses = epoch_task_losses.get(task, [0.0])
                        batch_sizes = epoch_batch_sizes.get(task, [])
                        if batch_losses and batch_sizes and len(batch_losses) == len(batch_sizes):
                            total_samples = sum(batch_sizes)
                            weighted_sum = sum(loss * size for loss, size in zip(batch_losses, batch_sizes))
                            avg_loss = weighted_sum / total_samples if total_samples > 0 else 0.0
                        else:
                            avg_loss = sum(batch_losses) / max(len(batch_losses), 1) if batch_losses else 0.0
                        train_metrics[task] = {
                            'mse': avg_loss if avg_loss > 0 else 0.0,
                            'mae': avg_loss * 0.8 if avg_loss > 0 else 0.0,
                            'rmse': np.sqrt(avg_loss) if avg_loss > 0 else 0.0,
                            'mErr': 90.0,  # Default angle error (worst case)
                            'loss': avg_loss
                        }
                    elif task == 'instance':
                        # Fallback for instance segmentation if no metrics were computed
                        batch_losses = epoch_task_losses.get(task, [0.0])
                        batch_sizes = epoch_batch_sizes.get(task, [])
                        if batch_losses and batch_sizes and len(batch_losses) == len(batch_sizes):
                            total_samples = sum(batch_sizes)
                            weighted_sum = sum(loss * size for loss, size in zip(batch_losses, batch_sizes))
                            avg_loss = weighted_sum / total_samples if total_samples > 0 else 0.0
                        else:
                            avg_loss = sum(batch_losses) / max(len(batch_losses), 1) if batch_losses else 0.0
                        train_metrics[task] = {
                            'mAP': 0.0,
                            'dice': 0.0,
                            'iou': 0.0,
                            'loss': avg_loss
                        }

                # Final cleanup
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

            model.train()  # Ensure model is in train mode

            # Update progress bar with losses and metrics
            pbar_info = []
            for task in self.config['tasks']:
                if task in train_metrics:
                    if task == 'classification':
                        pbar_info.append(
                            f"cls_loss={train_metrics[task]['loss']:.4f}, cls_acc={train_metrics[task]['acc']:.4f}")
                    elif task == 'detection':
                        pbar_info.append(
                            f"det_loss={train_metrics[task]['loss']:.4f}, det_acc={train_metrics[task]['acc']:.4f}, box_l1={train_metrics[task]['box_l1']:.4f}")
                    elif task == 'segmentation':
                        pbar_info.append(
                            f"seg_loss={train_metrics[task]['loss']:.4f}, dice={train_metrics[task]['dice']:.4f}, iou={train_metrics[task]['iou']:.4f}")
                    elif task == 'depth':
                        pbar_info.append(
                            f"depth_loss={train_metrics[task]['loss']:.4f}, rmse={train_metrics[task]['rmse']:.4f}, mae={train_metrics[task]['mae']:.4f}")
                    elif task == 'normal':
                        pbar_info.append(
                            f"normal_loss={train_metrics[task]['loss']:.4f}, mErr={train_metrics[task]['mErr']:.2f}°, rmse={train_metrics[task]['rmse']:.4f}")
                    elif task == 'instance':
                        pbar_info.append(
                            f"inst_loss={train_metrics[task]['loss']:.4f}, mAP={train_metrics[task]['mAP']:.4f}, inst_dice={train_metrics[task]['dice']:.4f}")

            epoch_pbar.set_postfix_str(", ".join(pbar_info))

            # Track best loss (weighted sum of all task losses) and best metrics
            loss_weights = self.config.get('loss_weights', {})
            total_loss = sum(
                loss_weights.get(task, 1.0) * train_metrics[task]['loss']
                for task in train_metrics
            )

            # Update learning rate scheduler (after total_loss is computed)
            if scheduler_type == 'plateau':
                scheduler.step(total_loss)
                current_lr = optimizer.param_groups[0]['lr']
            else:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0] if hasattr(
                    scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

            if total_loss < best_loss:
                best_loss = total_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = train_metrics.copy()  # Store best metrics
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping check
            if early_stopping_patience is not None and no_improvement_count >= early_stopping_patience:
                logger.info(
                    f"\n  ⏹️  Early stopping triggered after {epoch + 1} epochs (no improvement for {no_improvement_count} epochs)")
                break

        # Save best model if save_model_path is provided
        save_path = method_save_model_path if method_save_model_path is not None else self.save_model_path
        if save_path is not None and round_num is not None and best_model_state is not None:
            import os

            # Create path: save_model_path/round_N
            round_dir = os.path.join(save_path, f'round_{round_num}')
            os.makedirs(round_dir, exist_ok=True)
            model_path = os.path.join(round_dir, 'best_loss.pth')
            torch.save(best_model_state, model_path)
            logger.info(f"  💾 Saved best model (loss={best_loss:.4f}) to {model_path}")

        # Load best model state back to model
        if best_model_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            # Clear CPU copy to free memory
            del best_model_state

        # Clear GPU cache after training (on specific device)
        # Note: labeled_data was never loaded, so no need to delete it
        # _print_memory_usage(self.device, f"[Round {round_num}] Before cleanup in _train_model:")
        del epoch_task_losses, train_metrics, optimizer, loss_functions
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
                # Force memory release by resetting peak stats and clearing cache again
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # _print_memory_usage(self.device, f"[Round {round_num}] _train_model END:", reset_peak=True)
        # Return best metrics for printing at round end
        return model, best_metrics

    def _evaluate_model(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Evaluate model on test data.

        Similar to spatial active learning: process batch-by-batch and accumulate metrics.
        Uses train_batch_size to keep GPU memory usage consistent.
        """
        model.eval()
        per_task_metrics = {}

        # Use train_batch_size to keep GPU memory usage consistent with training
        eval_batch_size = self.config.get('train_batch_size', 128)
        n_samples = len(test_data['inputs'])

        # Initialize accumulators for each task (similar to spatial active learning)
        task_metrics_accumulators = {task: {} for task in self.config['tasks']}
        batch_count = 0

        with torch.no_grad():
            for i in range(0, n_samples, eval_batch_size):
                end_idx = min(i + eval_batch_size, n_samples)

                # Move batch to GPU
                batch_inputs = test_data['inputs'][i:end_idx].to(self.device)
                # Get image size from config for anchor-based detection
                image_size = (self.config.get('image_height', 480),
                              self.config.get('image_width', 480))
                batch_outputs = model(batch_inputs, image_size=image_size)

                # Process each task batch-by-batch (like spatial active learning)
                batch_output = None
                batch_target = None
                for task in self.config['tasks']:
                    # Debug: log missing tasks
                    if task not in test_data:
                        import logging
                        logger = logging.getLogger(__name__)
                        if batch_count == 0:  # Only log once per evaluation
                            logger.debug(
                                f"    Task '{task}' not in test_data. Available keys: {list(test_data.keys())}")
                        continue
                    if task not in batch_outputs:
                        import logging
                        logger = logging.getLogger(__name__)
                        if batch_count == 0:  # Only log once per evaluation
                            logger.debug(
                                f"    Task '{task}' not in batch_outputs. Available keys: {list(batch_outputs.keys())}")
                        continue

                    batch_output = batch_outputs[task]
                    batch_target = test_data[task][i:end_idx]
                    if isinstance(batch_target, torch.Tensor) and batch_target.device.type == 'cuda':
                        batch_target = batch_target.to(self.device)
                    else:
                        batch_target = batch_target.to(self.device)

                    # Compute metrics for this batch
                    if task == 'classification':
                        batch_metrics = PerTaskMetrics.classification_metrics(batch_output, batch_target)
                    elif task == 'segmentation':
                        batch_metrics = PerTaskMetrics.segmentation_metrics(batch_output, batch_target)
                    elif task == 'detection':
                        batch_metrics = PerTaskMetrics.detection_metrics(batch_output, batch_target)
                    elif task == 'depth':
                        # Depth regression
                        batch_metrics = PerTaskMetrics.regression_metrics(batch_output, batch_target)
                    elif task == 'normal':
                        # Surface normal prediction with angle error
                        batch_metrics = PerTaskMetrics.normal_metrics(batch_output, batch_target)
                    elif task == 'instance':
                        # Instance segmentation: use instance_metrics with mAP
                        # Convert target to binary (0=background, >0=foreground)
                        binary_target = (batch_target > 0).long()
                        batch_metrics = PerTaskMetrics.instance_metrics(batch_output, binary_target)
                    else:
                        continue

                    # Accumulate metrics (average will be computed at the end)
                    if not task_metrics_accumulators[task]:
                        # First batch: initialize
                        task_metrics_accumulators[task] = {k: [v] for k, v in batch_metrics.items()}
                    else:
                        # Subsequent batches: append
                        for k, v in batch_metrics.items():
                            if k in task_metrics_accumulators[task]:
                                task_metrics_accumulators[task][k].append(v)

                batch_count += 1
                # Clear batch data from GPU immediately
                # Only delete variables that were actually created
                del batch_outputs, batch_inputs
                if batch_output is not None:
                    del batch_output
                if batch_target is not None:
                    del batch_target
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for GPU operations to complete

        # Average accumulated metrics across all batches
        for task in self.config['tasks']:
            if task_metrics_accumulators[task]:
                # Average metrics across batches
                per_task_metrics[task] = {
                    k: sum(v) / len(v) if isinstance(v[0], (int, float)) else v[0]
                    for k, v in task_metrics_accumulators[task].items()
                }

        # Verify that all expected tasks have metrics
        missing_tasks = [task for task in self.config['tasks'] if task not in per_task_metrics]
        if missing_tasks:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"⚠️  Some tasks have no metrics computed: {missing_tasks}. "
                f"This may indicate a problem with model outputs or evaluation data."
            )

        # Final cleanup
        # _print_memory_usage(self.device, "Before cleanup in _evaluate_model:")
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # _print_memory_usage(self.device, "_evaluate_model END:", reset_peak=True)
        return per_task_metrics

    def _adjust_hyperparameters_dynamically(self, round_num: int, per_task_metrics: Dict[str, Dict[str, float]]):
        """Dynamically adjust hyperparameters based on round number and task performance.

        Args:
            round_num: Current round number
            per_task_metrics: Current round's per-task metrics
        """
        import logging
        logger = logging.getLogger(__name__)

        # Check if adaptive scheduling is enabled
        adaptive_scheduling = self.config.get('adaptive_hyperparameter_scheduling', False)
        task_aware = self.config.get('task_aware_weighting', False)

        if not adaptive_scheduling and not task_aware:
            # Both features disabled, skip
            return

        # Get the acquisition function from 'ours' selector
        ours_selector = self.selectors.get('ours')
        if not ours_selector or not hasattr(ours_selector, 'acquisition_function'):
            return

        acquisition_function = ours_selector.acquisition_function
        if not hasattr(acquisition_function, 'update_hyperparameters'):
            return

        num_rounds = self.config.get('num_rounds', 30)
        base_alpha = self.config.get('alpha', 1.0)
        base_beta = self.config.get('beta', 1.5)
        base_gamma = self.config.get('gamma', 1.2)
        base_delta = self.config.get('delta', 0.25)

        # Initialize with base values
        alpha = base_alpha
        beta = base_beta
        gamma = base_gamma
        delta = base_delta

        # 1. Adaptive Hyperparameter Scheduling (round-based adjustment)
        if adaptive_scheduling:
            # Get adaptive schedule from config (with defaults)
            schedule = self.config.get('adaptive_schedule', {})
            early_schedule = schedule.get(
                'early', {'alpha_mult': 1.0, 'beta_mult': 0.3, 'gamma_mult': 0.3, 'delta_mult': 1.5})
            mid_schedule = schedule.get('mid', {'alpha_mult': 1.1, 'beta_mult': 1.0,
                                        'gamma_mult': 1.0, 'delta_mult': 1.0})
            late_schedule = schedule.get(
                'late', {'alpha_mult': 0.8, 'beta_mult': 1.3, 'gamma_mult': 1.3, 'delta_mult': 1.0})

            # Round-based adjustment: Different strategies for early/mid/late rounds
            if round_num < num_rounds // 3:
                # Early rounds: Low conflict/consistency, high diversity (for stable learning)
                alpha = base_alpha * early_schedule.get('alpha_mult', 1.0)
                beta = base_beta * early_schedule.get('beta_mult', 0.3)
                gamma = base_gamma * early_schedule.get('gamma_mult', 0.3)
                delta = base_delta * early_schedule.get('delta_mult', 1.5)
                logger.info(f"  🔄 [Round {round_num}] Early stage: Diversity focus - "
                            f"α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")
            elif round_num < 2 * num_rounds // 3:
                # Mid rounds: Balanced approach
                alpha = base_alpha * mid_schedule.get('alpha_mult', 1.1)
                beta = base_beta * mid_schedule.get('beta_mult', 1.0)
                gamma = base_gamma * mid_schedule.get('gamma_mult', 1.0)
                delta = base_delta * mid_schedule.get('delta_mult', 1.0)
                logger.info(f"  🔄 [Round {round_num}] Mid stage: Balanced - "
                            f"α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")
            else:
                # Late rounds: Focus on conflict and consistency
                alpha = base_alpha * late_schedule.get('alpha_mult', 0.8)
                beta = base_beta * late_schedule.get('beta_mult', 1.3)
                gamma = base_gamma * late_schedule.get('gamma_mult', 1.3)
                delta = base_delta * late_schedule.get('delta_mult', 1.0)
                logger.info(f"  🔄 [Round {round_num}] Late stage: Conflict/Consistency focus - "
                            f"α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")

        # 2. Task-aware Weighting (dynamic adjustment based on task performance)
        if task_aware and round_num > 0 and round_num - 1 in self.previous_round_metrics:
            prev_metrics = self.previous_round_metrics[round_num - 1]

            # Identify underperforming tasks
            task_performance_changes = {}
            task_relative_performance = {}

            for task_name in self.config['tasks']:
                if task_name in per_task_metrics and task_name in prev_metrics:
                    # Get primary metric for each task
                    if task_name == 'classification':
                        current_metric = per_task_metrics[task_name].get('accuracy', 0.0)
                        prev_metric = prev_metrics[task_name].get('accuracy', 0.0)
                    elif task_name == 'detection':
                        current_metric = per_task_metrics[task_name].get('accuracy', 0.0)
                        prev_metric = prev_metrics[task_name].get('accuracy', 0.0)
                    elif task_name == 'segmentation':
                        current_metric = per_task_metrics[task_name].get('dice_mean', 0.0)
                        prev_metric = prev_metrics[task_name].get('dice_mean', 0.0)
                    else:
                        continue

                    # Calculate improvement rate
                    if prev_metric > 0:
                        improvement_rate = (current_metric - prev_metric) / prev_metric
                        task_performance_changes[task_name] = improvement_rate
                        task_relative_performance[task_name] = current_metric

            # Calculate average performance for normalization
            if task_relative_performance:
                avg_performance = np.mean(list(task_relative_performance.values()))

                # Adjust hyperparameters based on task performance
                # Underperforming tasks get higher weights for conflict/consistency

                # Get task-aware sensitivity from config (default: 1.05 for stability)
                sensitivity = self.config.get('task_aware_sensitivity', 1.05)

                # Classification underperforming: increase hardness (alpha)
                if 'classification' in task_performance_changes:
                    cls_change = task_performance_changes['classification']
                    cls_perf = task_relative_performance.get('classification', 0.0)

                    if cls_change < -0.05 or cls_perf < avg_performance * 0.9:
                        alpha = alpha * sensitivity
                        logger.info(f"  📊 Classification underperforming (change: {cls_change:.3f}, "
                                    f"perf: {cls_perf:.3f}), increasing α to {alpha:.3f}")

                # Detection underperforming: increase conflict weight (beta)
                if 'detection' in task_performance_changes:
                    det_change = task_performance_changes['detection']
                    det_perf = task_relative_performance.get('detection', 0.0)

                    if det_change < -0.05 or det_perf < avg_performance * 0.9:
                        beta = beta * sensitivity
                        logger.info(f"  📊 Detection underperforming (change: {det_change:.3f}, "
                                    f"perf: {det_perf:.3f}), increasing β to {beta:.3f}")

                # Segmentation underperforming: increase consistency weight (gamma)
                if 'segmentation' in task_performance_changes:
                    seg_change = task_performance_changes['segmentation']
                    seg_perf = task_relative_performance.get('segmentation', 0.0)

                    if seg_change < -0.05 or seg_perf < avg_performance * 0.9:
                        gamma = gamma * sensitivity
                        logger.info(f"  📊 Segmentation underperforming (change: {seg_change:.3f}, "
                                    f"perf: {seg_perf:.3f}), increasing γ to {gamma:.3f}")

                # If all tasks are improving well, increase diversity
                if all(rate > 0.02 for rate in task_performance_changes.values()):
                    delta = delta * sensitivity
                    logger.info(f"  📊 All tasks improving well, increasing δ to {delta:.3f}")

        # Update acquisition function hyperparameters
        acquisition_function.update_hyperparameters(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta
        )

        logger.info(f"  ✅ Updated hyperparameters: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")

        logger = logging.getLogger(__name__)
        logger.info(
            f"  🔧 Dynamic hyperparameters (Round {round_num}): α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")

    def run_single_round(
        self,
        method: str,
        round_num: int,
        batch_size: int,
        method_save_model_path: str = None,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[Dict[str, Dict[str, float]], float, List[int], Dict[str, Dict[str, float]]]:
        """Run a single AL round.

        Returns:
            Tuple of (per_task_metrics, mtl_score, selected_indices, best_train_metrics)
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.info(f"  🔄 Round {round_num}: {method}")
        # _print_memory_usage(self.device, f"[Round {round_num}] run_single_round START:")

        # Debug: Check labeled_indices_set before training
        if len(self.labeled_indices_set) == 0:
            raise ValueError(
                f"labeled_indices_set is empty at start of round {round_num}. "
                f"This should not happen. Check if _store_full_coco_labels was called correctly."
            )

        self.model, best_train_metrics = self._train_model(
            self.model, labeled_indices=None, round_num=round_num, method_save_model_path=method_save_model_path,
            logger=logger)
        # _print_memory_usage(self.device, f"[Round {round_num}] After _train_model:")

        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
                torch.cuda.empty_cache()

        # _print_memory_usage(self.device, f"[Round {round_num}] After cleanup after training:")
        if best_train_metrics is not None:
            logger.info(f"\n  📊 Best training metrics (lowest total loss):")
            for task in self.config['tasks']:
                if task in best_train_metrics:
                    task_metrics = best_train_metrics[task]
                    metric_strs = [f"{k}={v:.4f}" for k, v in task_metrics.items()]
                    logger.info(f"    {task}: {', '.join(metric_strs)}")

        # For evaluation, use test_set if available, otherwise load labeled data on-demand
        logger.info(f"  🔍 Starting evaluation (Round {round_num})...")
        try:
            if self.test_set is not None:
                eval_data = self.test_set
                logger.info(f"    Using test_set with {len(eval_data['inputs'])} samples")
            else:
                # Load labeled data from full_dataset_labels for evaluation
                labeled_indices = list(self.labeled_indices_set)
                eval_data = {'inputs': self.full_dataset_labels['inputs'][labeled_indices]}
                for task in self.config['tasks']:
                    if task in self.full_dataset_labels:
                        eval_data[task] = self.full_dataset_labels[task][labeled_indices]
                # Ensure on CPU
                for key in eval_data:
                    if isinstance(eval_data[key], torch.Tensor) and eval_data[key].device.type == 'cuda':
                        eval_data[key] = eval_data[key].cpu()
                logger.info(f"    Using labeled data with {len(eval_data['inputs'])} samples")

            per_task_metrics = self._evaluate_model(self.model, eval_data)
            logger.info(f"  ✅ Evaluation completed. Metrics: {list(per_task_metrics.keys())}")

            # Print detailed evaluation metrics (validation/test results)
            logger.info(f"  📊 Evaluation metrics (validation/test):")
            for task, metrics in per_task_metrics.items():
                metric_strs = []
                for k, v in metrics.items():
                    try:
                        if isinstance(v, (int, float)):
                            metric_strs.append(f"{k}={v:.4f}")
                        elif isinstance(v, np.ndarray):
                            # Handle numpy arrays
                            metric_strs.append(f"{k}={float(v.mean()):.4f}")
                        elif isinstance(v, torch.Tensor):
                            # Handle torch tensors
                            metric_strs.append(f"{k}={float(v.mean().item()):.4f}")
                        elif isinstance(v, list) and len(v) > 0:
                            # For list values, show mean
                            if all(isinstance(x, (int, float)) for x in v):
                                metric_strs.append(f"{k}={sum(v) / len(v):.4f}")
                            else:
                                metric_strs.append(f"{k}={v}")
                        else:
                            metric_strs.append(f"{k}={v}")
                    except Exception:
                        metric_strs.append(f"{k}={v}")
                logger.info(f"    {task}: {', '.join(metric_strs)}")
        except Exception as e:
            logger.error(f"  ❌ Evaluation failed in round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            raise
        # _print_memory_usage(self.device, f"[Round {round_num}] After _evaluate_model:")

        # Clear eval_data after evaluation
        del eval_data
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
                torch.cuda.empty_cache()

        # _print_memory_usage(self.device, f"[Round {round_num}] After cleanup after evaluation:")

        # Select new samples (skip for round 0 as initial pool is already set)
        selected_full_indices = []
        if round_num > 0:
            logger.info(f"  🔍 Starting sample selection (Round {round_num}, method={method})...")
            logger.info(f"    Unlabeled pool size: {len(self.unlabeled_pool)}, batch_size: {batch_size}")
            try:
                # _print_memory_usage(self.device, f"[Round {round_num}] Before selector.select:")
                if method in self.selectors:
                    selector = self.selectors[method]
                    # Use train_batch_size to keep GPU memory usage consistent
                    train_batch_size = self.config.get('train_batch_size', 128)
                    logger.info(f"    Calling selector.select with k={batch_size}, train_batch_size={train_batch_size}")
                    select_kwargs = dict(
                        task_types={task: task for task in self.config['tasks']},
                        train_batch_size=train_batch_size,
                    )
                    if (method == 'hekimoglu'
                            and self.full_dataset_labels is not None
                            and self.labeled_indices_set):
                        lab_idx = list(self.labeled_indices_set)
                        select_kwargs['labeled_pool'] = self.full_dataset_labels['inputs'][lab_idx]
                    if (method == 'ppal'
                            and self.full_dataset_labels is not None
                            and self.labeled_indices_set):
                        lab_idx = list(self.labeled_indices_set)
                        select_kwargs['labeled_pool'] = self.full_dataset_labels['inputs'][lab_idx]
                        select_kwargs['labeled_targets'] = {
                            task: self.full_dataset_labels[task][lab_idx]
                            for task in self.config['tasks']
                            if task in self.full_dataset_labels
                        }
                    selected_indices = selector.select(
                        self.unlabeled_pool,
                        k=batch_size,
                        **select_kwargs,
                    )
                    logger.info(f"  ✅ Selection completed. Selected {len(selected_indices)} samples")
                else:
                    logger.warning(f"    Method {method} not in selectors, using random selection")
                    # Fallback to random selection (on CPU since unlabeled_pool is on CPU)
                    n_samples = len(self.unlabeled_pool)
                    selected_indices = torch.randperm(n_samples, device='cpu', dtype=torch.long)[:batch_size]
                    logger.info(f"  ✅ Random selection completed. Selected {len(selected_indices)} samples")
            except Exception as e:
                logger.error(f"  ❌ Selection failed in round {round_num}: {e}")
                import traceback
                traceback.print_exc()
                raise

            # Clear memory after selection (on specific device)
            if self.device.type == 'cuda':
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

            # Ensure selected_indices are valid and on CPU (since unlabeled_pool is on CPU)
            if isinstance(selected_indices, torch.Tensor):
                selected_indices = selected_indices.cpu().long()
            else:
                selected_indices = torch.tensor(selected_indices, device='cpu', dtype=torch.long)

            # Clamp indices to valid range and ensure long dtype
            n_unlabeled = len(self.unlabeled_pool)
            selected_indices = torch.clamp(selected_indices, 0, n_unlabeled - 1).long()

            # Remove duplicates if any
            selected_indices = torch.unique(selected_indices).long()

            # Limit to batch_size if more than requested
            if len(selected_indices) > batch_size:
                selected_indices = selected_indices[:batch_size].long()

            # Get actual number of selected samples (after deduplication and truncation)
            actual_num_selected = len(selected_indices)
            # _print_memory_usage(self.device, f"[Round {round_num}] After selector.select (selected {actual_num_selected} samples):")

            # Map unlabeled pool indices to full dataset indices
            if self.config.get('dataset') == 'coco' and self.full_dataset_labels is not None:
                # Retrieve actual labels from full dataset
                # Map unlabeled pool indices to full dataset indices
                unlabeled_to_full = self._get_unlabeled_to_full_mapping()
                # Ensure indices are within bounds and have correct dtype
                selected_indices_clamped = torch.clamp(selected_indices, 0, len(unlabeled_to_full) - 1).long()
                full_indices = unlabeled_to_full[selected_indices_clamped].long()

                # Verify that full_indices has the correct length
                if len(full_indices) != actual_num_selected:
                    raise ValueError(
                        f"Index mapping mismatch: selected_indices={actual_num_selected}, "
                        f"full_indices={len(full_indices)}"
                    )

                # Ensure full_indices are within valid range of full_dataset_labels
                max_valid_idx = len(self.full_dataset_labels['inputs']) - 1
                full_indices_clamped = torch.clamp(full_indices, 0, max_valid_idx).long()

                # Check if any indices were clamped (this indicates a bug in _unlabeled_to_full)
                if not torch.equal(full_indices, full_indices_clamped):
                    clamped_count = (full_indices != full_indices_clamped).sum().item()
                    print(f"⚠️  Warning: {clamped_count} indices were clamped to valid range "
                          f"(max_valid_idx={max_valid_idx}, max_full_indices={full_indices.max().item()})")
                    full_indices = full_indices_clamped

                # Update labeled indices set (no need to store actual data)
                full_indices_list = full_indices.cpu().tolist()
                self.labeled_indices_set.update(full_indices_list)
                # Store full_indices for logging (full dataset indices)
                selected_full_indices = full_indices_list
            else:
                # Synthetic data - just update indices (data will be generated on-demand)
                # Map selected_indices to full dataset indices
                if hasattr(self, '_unlabeled_to_full'):
                    full_indices = self._unlabeled_to_full[selected_indices].long()
                else:
                    full_indices = selected_indices.long()
                full_indices_list = full_indices.cpu().tolist()
                self.labeled_indices_set.update(full_indices_list)
                # Store full_indices for logging
                selected_full_indices = full_indices_list

            # Remove selected samples from unlabeled pool (keep on CPU)
            # _print_memory_usage(self.device, f"[Round {round_num}] Before updating unlabeled_pool (size={len(self.unlabeled_pool)}):")
            if isinstance(selected_indices, torch.Tensor):
                selected_set = set(selected_indices.cpu().long().tolist())
            else:
                selected_set = set(selected_indices)
            remaining_indices = torch.tensor([i for i in range(len(self.unlabeled_pool))
                                             if i not in selected_set], device='cpu', dtype=torch.long)
            # Create new unlabeled_pool by indexing (old reference will be garbage collected)
            self.unlabeled_pool = self.unlabeled_pool[remaining_indices]  # Should stay on CPU
            # _print_memory_usage(self.device, f"[Round {round_num}] After updating unlabeled_pool (new_size={len(self.unlabeled_pool)}):")

            # Update unlabeled-to-full mapping if using COCO
            if self.config.get('dataset') == 'coco' and hasattr(self, '_unlabeled_to_full'):
                old_mapping = self._unlabeled_to_full
                # remaining_indices are indices into the old unlabeled pool
                # Map them to full dataset indices
                if len(remaining_indices) > 0:
                    remaining_indices_clamped = torch.clamp(remaining_indices, 0, len(old_mapping) - 1).long()
                    self._unlabeled_to_full = old_mapping[remaining_indices_clamped].long()
                    # Ensure updated indices are within valid range of full_dataset_labels
                    if self.full_dataset_labels is not None:
                        max_valid_idx = len(self.full_dataset_labels['inputs']) - 1
                        self._unlabeled_to_full = torch.clamp(self._unlabeled_to_full, 0, max_valid_idx).long()
                else:
                    self._unlabeled_to_full = torch.tensor([], dtype=torch.long, device='cpu')
                del old_mapping

            # Clear intermediate tensors and force garbage collection
            del selected_indices, remaining_indices, selected_set
            # old_unlabeled is already deleted above (line 930)
            if self.device.type == 'cuda':
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        else:
            # Round 0: return initial labeled indices
            selected_full_indices = list(self.labeled_indices_set)

        # Verify per_task_metrics is not empty and contains expected tasks
        logger.info(f"  🔍 Verifying evaluation metrics (Round {round_num})...")
        if not per_task_metrics:
            raise ValueError(
                f"per_task_metrics is empty after evaluation in round {round_num}. "
                f"This indicates a problem with model evaluation."
            )
        missing_tasks = [task for task in self.config['tasks'] if task not in per_task_metrics]
        if missing_tasks:
            logger.warning(
                f"⚠️  Round {round_num}: Missing metrics for tasks: {missing_tasks}. "
                f"Available tasks: {list(per_task_metrics.keys())}"
            )
        logger.info(f"  ✅ Metrics verified. Tasks: {list(per_task_metrics.keys())}")

        # Adjust hyperparameters dynamically if enabled
        logger.info(f"  🔍 Adjusting hyperparameters (Round {round_num})...")
        try:
            self._adjust_hyperparameters_dynamically(round_num, per_task_metrics)
            logger.info(f"  ✅ Hyperparameters adjusted")
        except Exception as e:
            logger.warning(f"  ⚠️  Hyperparameter adjustment failed: {e}")

        # Compute MTL score: simple sum of task performances (cls + det + seg)
        logger.info(f"  🔍 Computing MTL score (Round {round_num})...")
        try:
            mtl_score = MTLMetrics.simple_sum(per_task_metrics)
            logger.info(f"  ✅ MTL score computed: {mtl_score:.4f}")
        except Exception as e:
            logger.error(f"  ❌ MTL score computation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Compute MTL Active Score: improvement from previous round
        previous_round_metrics = self.previous_round_metrics.get(round_num - 1) if round_num > 0 else None
        mtl_active_score = MTLMetrics.active_improvement(per_task_metrics, previous_round_metrics)

        # Store metrics for next round's active score calculation
        self.previous_round_metrics[round_num] = per_task_metrics.copy()

        # Clear GPU cache and force garbage collection after each round to prevent memory accumulation (on specific device)
        # _print_memory_usage(self.device, f"[Round {round_num}] Before final cleanup:")
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations are complete before clearing
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # _print_memory_usage(self.device, f"[Round {round_num}] run_single_round END:", reset_peak=True)
        return per_task_metrics, mtl_score, selected_full_indices, best_train_metrics

    def _get_primary_metric_name(self, task_name: str) -> str:
        """Get primary metric name for a task."""
        if task_name == 'classification':
            return 'accuracy'
        elif task_name == 'detection':
            return 'accuracy'
        elif task_name == 'segmentation':
            return 'dice_mean'
        else:
            return 'accuracy'

    def _get_unlabeled_to_full_mapping(self) -> torch.Tensor:
        """Get mapping from unlabeled pool indices to full dataset indices."""
        if hasattr(self, '_unlabeled_to_full'):
            return self._unlabeled_to_full.long()
        else:
            # Fallback: assume identity mapping
            return torch.arange(len(self.unlabeled_pool), device=self.device, dtype=torch.long)

    def run_ablation(self):
        """Run complete ablation study."""
        print("🎯 Starting ablation study...")

        # Get methods to test
        methods = self.config.get('methods', ['random', 'uncertainty', 'diversity', 'ours'])
        num_rounds = self.config.get('num_rounds', 5)
        batch_size = self.config.get('batch_size', 20)

        # Save config to main save directory
        import json
        import os
        import shutil
        config_save_path = os.path.join(self.save_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"📋 Config saved to: {config_save_path}")

        for method in methods:
            print(f"\n📊 Testing method: {method}")
            method_file_logger = None

            # Create method-specific save directories
            method_save_dir = os.path.join(self.save_dir, method)
            os.makedirs(method_save_dir, exist_ok=True)

            # Save config to method-specific directory as well
            method_config_path = os.path.join(method_save_dir, 'config.json')
            shutil.copy(config_save_path, method_config_path)
            method_save_model_path = None
            if self.save_model_path is not None:
                method_save_model_path = os.path.join(self.save_model_path, method)
                os.makedirs(method_save_model_path, exist_ok=True)

            # Create method-specific logger for metrics
            method_logger = MetricsLogger(method_save_dir)

            # Setup logging to file for this method
            log_file_path = os.path.join(method_save_dir, f"{method}_run.log")
            method_file_logger = logging.getLogger(f"method_{method}")
            method_file_logger.setLevel(logging.INFO)
            # Remove existing handlers to avoid duplicates
            method_file_logger.handlers = []
            # File handler
            file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            method_file_logger.addHandler(file_handler)
            # Console handler (to also print to stdout)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            method_file_logger.addHandler(console_handler)
            method_file_logger.propagate = False

            try:
                # Reset seeds for each method to ensure reproducibility
                # This ensures all methods start with the same initial labeled pool
                seed = self.config.get('seed', 42)
                self._set_seeds(seed)

                # Reset for each method
                self.model = self._create_model()

                # Reset base metrics for each method
                self.base_metrics = None
                self.oracle_metrics = None

                # For COCO, data is already loaded in __init__, just reset indices
                method_file_logger.info(f"  🔍 Checking dataset: {self.config.get('dataset')}")
                if self.config.get('dataset') == 'coco':
                    # Data is already loaded in __init__, just reset labeled_indices_set
                    if self.full_dataset_labels is None:
                        raise RuntimeError(
                            f"full_dataset_labels not initialized. This should not happen.")

                    # Reset labeled_indices_set and _unlabeled_to_full (but keep full_dataset_labels)
                    self.labeled_indices_set = set()
                    self._unlabeled_to_full = None

                    # Reset labeled_indices_set using same seed to get same initial split
                    # This ensures all methods start with the same initial labeled pool
                    initial_labeled_size = self.config.get('initial_labeled_size', 100)
                    seed = self.config.get('seed', 42)
                    n_sampled = len(self.full_dataset_labels['inputs'])

                    # Reset all random seeds to ensure reproducibility
                    self._set_seeds(seed)
                    split_indices = torch.randperm(n_sampled, device='cpu', dtype=torch.long)
                    labeled_local_indices = split_indices[:initial_labeled_size].long()
                    unlabeled_local_indices = split_indices[initial_labeled_size:].long()

                    # Ensure all indices are within valid range [0, n_sampled-1]
                    max_valid_idx = n_sampled - 1
                    labeled_local_indices = torch.clamp(labeled_local_indices, 0, max_valid_idx).long()
                    unlabeled_local_indices = torch.clamp(unlabeled_local_indices, 0, max_valid_idx).long()

                    self.labeled_indices_set = set(labeled_local_indices.tolist())
                    self._unlabeled_to_full = unlabeled_local_indices.long()

                    method_file_logger.info(
                        f"  ✅ Reset labeled_indices_set: {len(self.labeled_indices_set)} samples from {n_sampled} total")

                    # Create unlabeled_pool from already-loaded data
                    self.unlabeled_pool = self._create_unlabeled_pool()

                    # Test set is already loaded in __init__, no need to reload
                else:
                    # For non-COCO datasets, create unlabeled_pool normally
                    method_file_logger.info(f"  ⚠️ Dataset is not 'coco', skipping _store_full_coco_labels")
                    self.unlabeled_pool = self._create_unlabeled_pool()

            except Exception as e:
                method_file_logger.error(f"  ❌ Failed to initialize for method {method}: {e}")
                import traceback
                traceback.print_exc()
                raise

            for round_num in range(num_rounds):
                # Run single round with method-specific save paths
                per_task_metrics, mtl_score, selected_indices, best_train_metrics = self.run_single_round(
                    method, round_num, batch_size, method_save_model_path=method_save_model_path,
                    logger=method_file_logger)

                # Get MTL Active Score (computed in run_single_round)
                previous_round_metrics = self.previous_round_metrics.get(round_num - 1) if round_num > 0 else None
                mtl_active_score = MTLMetrics.active_improvement(per_task_metrics, previous_round_metrics)

                # Log results to method-specific logger
                method_logger.log_round(
                    round_num=round_num,
                    method=method,
                    criteria=self.config['criteria'],
                    tasks=self.config['tasks'],
                    per_task_metrics=per_task_metrics,
                    mtl_score=mtl_score,
                    seed=self.config.get('seed', 42),
                    label_budget=(len(self.labeled_indices_set) /
                                  (len(self.labeled_indices_set) + len(self.unlabeled_pool))),
                    selected_indices=selected_indices,
                    total_selected=len(self.labeled_indices_set),
                    best_train_metrics=best_train_metrics,
                    mtl_active_score=mtl_active_score
                )

                method_file_logger.info(f"    MTL Score: {mtl_score:.4f}")

            # Save results for this method
            try:
                method_logger.save_summary()

                # Generate all plots
                if method_logger.metrics_history:
                    method_file_logger.info("  📊 Generating plots...")

                    # MTL Score plot
                    try:
                        method_logger.plot_metrics('mtl_score')
                        method_file_logger.info("    ✅ Generated mtl_score_plot.png")
                    except Exception as e:
                        method_file_logger.warning(f"    ⚠️  Failed to generate mtl_score plot: {e}")

                    # MTL Active Score plot (if available)
                    if method_logger.metrics_history and len(method_logger.metrics_history) > 0:
                        if 'mtl_active_score' in method_logger.metrics_history[0]:
                            try:
                                method_logger.plot_metrics('mtl_active_score')
                                method_file_logger.info("    ✅ Generated mtl_active_score_plot.png")
                            except Exception as e:
                                method_file_logger.warning(f"    ⚠️  Failed to generate mtl_active_score plot: {e}")

                    # Task Balance plot
                    try:
                        method_logger.plot_task_balance()
                        method_file_logger.info("    ✅ Generated task_balance_plot.png")
                    except Exception as e:
                        method_file_logger.warning(f"    ⚠️  Failed to generate task_balance plot: {e}")

                    # Convergence Rate plot
                    try:
                        method_logger.plot_convergence_rate()
                        method_file_logger.info("    ✅ Generated convergence_rate_plot.png")
                    except Exception as e:
                        method_file_logger.warning(f"    ⚠️  Failed to generate convergence_rate plot: {e}")

                    # Label Efficiency plot
                    try:
                        method_logger.plot_label_efficiency()
                        method_file_logger.info("    ✅ Generated label_efficiency_plot.png")
                    except Exception as e:
                        method_file_logger.warning(f"    ⚠️  Failed to generate label_efficiency plot: {e}")

                    # Task Performance plot
                    try:
                        method_logger.plot_performance()
                        method_file_logger.info("    ✅ Generated perf.png")
                    except Exception as e:
                        method_file_logger.warning(f"    ⚠️  Failed to generate performance plot: {e}")

                    # Save task performance JSON
                    try:
                        method_logger.save_task_performance()
                        method_file_logger.info("    ✅ Generated task_perf.json")
                    except Exception as e:
                        method_file_logger.warning(f"    ⚠️  Failed to save task_performance: {e}")
                else:
                    method_file_logger.warning("  ⚠️  No metrics history found, skipping plot generation")

                method_file_logger.info(f"  💾 Results for {method} saved to: {method_save_dir}")
            except Exception as e:
                method_file_logger.error(f"  ❌ Error saving results/plots for {method}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✅ Ablation study completed!")
        print(f"   All results saved to: {self.save_dir}")
        print(f"   Each method has its own subdirectory")


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_default_config() -> Dict:
    """Create default configuration."""
    return {
        'tasks': ['classification', 'segmentation'],
        'criteria': ['hardness', 'conflict', 'consistency'],
        'methods': ['random', 'uncertainty', 'diversity', 'ours'],
        'num_rounds': 5,
        'batch_size': 20,
        'unlabeled_pool_size': 1000,
        'initial_labeled_size': 100,
        'num_classes': 10,
        'input_channels': 3,
        'image_height': 64,
        'image_width': 64,
        'hidden_dim': 256,
        'num_encoder_layers': 4,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1.0,
        'hardness_mode': 'entropy',
        'conflict_mode': 'cosine',
        'consistency_mode': 'kl',
        'auto_balance_weights': True,
        'seed': 42
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Task Active Learning Runner')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--save_model_path', type=str, default=None,
                        help='Directory to save best loss models (e.g., ./models). Models saved as round_N/best_loss.pth')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0, cuda:1, cpu)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config['save_dir'] = args.save_dir
    # Use save_model_path from config if not provided via command line
    save_model_path = args.save_model_path or config.get('save_model_path')

    print(f"=" * 60)
    print(f"🚀 Multi-Task Active Learning Experiment")
    print(f"=" * 60)
    print(f"Config: {args.config}")
    print(f"Save dir: {args.save_dir}")
    if save_model_path:
        print(f"Save model path: {save_model_path}")
    print(f"Device: {args.device}")
    print(f"Methods: {config.get('methods', ['ours'])}")
    print(f"=" * 60)

    # Create runner
    runner = AblationRunner(config=config, device=args.device, save_dir=args.save_dir,
                            save_model_path=save_model_path)

    # Run experiment
    runner.run_ablation()

    print(f"\n✅ Experiment completed!")
    print(f"Results saved to: {args.save_dir}")

"""
Multi-Task Active Learning Framework

A novel active learning framework for multi-task learning that combines:
- Hardness (uncertainty) estimation
- Gradient conflict measurement  
- Task consistency enforcement

This framework addresses the limitations of existing approaches by considering
destructive gradient interference and ensuring optimization progress.
"""

from .acquisition import MultiTaskActiveLearningAcquisition
from .al_selectors import (LinearScalarizationSelector, OursSelector,
                            TwoStageSelector, make_selector)
from .baseline_selectors import (BADGESelector, BatchBALDSelector,
                                 ClusteringSelector, CoreSetSelector,
                                 DiversitySelector, LearningLossSelector,
                                 RandomSelector, UncertaintySelector)
from .metrics import MetricsLogger, MTLMetrics, PerTaskMetrics
from .models import MultiTaskModel, SharedEncoder, TaskHead, create_toy_model
from .runner import AblationRunner, create_default_config, load_config
from .utils import compute_entropy, compute_margin, normalize_scores

__version__ = "0.1.0"
__all__ = [
    "MultiTaskActiveLearningAcquisition",
    "SharedEncoder",
    "TaskHead",
    "MultiTaskModel",
    "create_toy_model",
    "normalize_scores",
    "compute_entropy",
    "compute_margin",
    "make_selector",
    # Baseline selectors
    "RandomSelector",
    "UncertaintySelector",
    "DiversitySelector",
    "CoreSetSelector",
    "BADGESelector",
    "BatchBALDSelector",
    "ClusteringSelector",
    "LearningLossSelector",
    # Our method
    "OursSelector",
    "TwoStageSelector",
    "LinearScalarizationSelector",
    # Metrics
    "PerTaskMetrics",
    "MTLMetrics",
    "MetricsLogger",
    # Runner
    "AblationRunner",
    "create_default_config",
    "load_config"
]

"""
Metrics for Multi-Task Active Learning

Implements per-task metrics and MTL summary metrics.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


class PerTaskMetrics:
    """Per-task metrics computation."""

    @staticmethod
    def classification_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Dictionary of metrics
        """
        pred_labels = torch.argmax(predictions, dim=-1)
        accuracy = (pred_labels == targets).float().mean().item()

        pred_probs = F.softmax(predictions, dim=-1).cpu().numpy()
        targets_np = targets.cpu().numpy()

        try:
            auroc = roc_auc_score(targets_np, pred_probs, multi_class='ovr', average='macro')
        except:
            auroc = 0.0

        try:
            auprc = average_precision_score(targets_np, pred_probs, average='macro')
        except:
            auprc = 0.0

        ece = PerTaskMetrics._compute_ece(predictions, targets)

        return {
            'accuracy': accuracy,
            'auroc': auroc,
            'auprc': auprc,
            'ece': ece
        }

    @staticmethod
    def segmentation_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                             ignore_index: int = 255) -> Dict[str, float]:
        """
        Compute segmentation metrics.

        Args:
            predictions: Model predictions [batch_size, num_classes, height, width]
            targets: Ground truth masks [batch_size, height, width]
            ignore_index: Label value to ignore in metrics computation (default: 255)

        Returns:
            Dictionary of metrics
        """
        device = predictions.device
        pred_labels = torch.argmax(predictions, dim=1)
        num_classes = predictions.shape[1]

        # Create mask for valid pixels (not ignore_index)
        valid_mask = (targets != ignore_index)

        # Clamp targets to valid range for one_hot encoding
        # Invalid pixels will be masked out anyway
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0  # Set to 0 (will be masked)

        pred_one_hot = F.one_hot(pred_labels, num_classes).permute(0, 3, 1, 2).float()
        target_one_hot = F.one_hot(targets_clamped.long(), num_classes).permute(0, 3, 1, 2).float()

        # Apply valid mask to both predictions and targets
        valid_mask_expanded = valid_mask.unsqueeze(1).float()  # [B, 1, H, W]
        pred_one_hot = pred_one_hot * valid_mask_expanded
        target_one_hot = target_one_hot * valid_mask_expanded

        intersection = (pred_one_hot * target_one_hot).sum(dim=(0, 2, 3))
        pred_sum = pred_one_hot.sum(dim=(0, 2, 3))
        target_sum = target_one_hot.sum(dim=(0, 2, 3))
        union = pred_sum + target_sum

        dice_per_class = torch.where(
            union > 0,
            2.0 * intersection / union,
            torch.where(intersection == 0, torch.ones_like(intersection), torch.zeros_like(intersection))
        )

        union_iou = pred_sum + target_sum - intersection
        iou_per_class = torch.where(
            union_iou > 0,
            intersection / union_iou,
            torch.where(intersection == 0, torch.ones_like(intersection), torch.zeros_like(intersection))
        )

        dice_scores = dice_per_class.cpu().numpy().tolist()
        iou_scores = iou_per_class.cpu().numpy().tolist()
        mean_dice = float(dice_per_class.mean().item())
        mean_iou = float(iou_per_class.mean().item())

        return {
            'dice_mean': mean_dice,
            'dice_per_class': dice_scores,
            'iou_mean': mean_iou,
            'iou_per_class': iou_scores
        }

    @staticmethod
    def detection_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute detection metrics.

        Args:
            predictions: Model predictions [batch_size, num_boxes, num_classes + 4]
            targets: Ground truth [batch_size, num_boxes, num_classes + 4]

        Returns:
            Dictionary of metrics
        """
        # Simplified detection metrics (full mAP computation is complex)
        # Here we compute basic accuracy and confidence metrics

        # Extract confidence scores
        # Format: [batch, num_boxes, num_classes + 4]
        # First num_classes are class scores, last 4 are bbox coordinates
        num_classes = predictions.shape[2] - 4
        conf_scores = predictions[:, :, :num_classes]  # [batch, num_boxes, num_classes]
        target_conf = targets[:, :, :num_classes]     # [batch, num_boxes, num_classes]

        # Find valid boxes (non-empty boxes)
        valid_mask = target_conf.sum(dim=-1) > 0  # [batch, num_boxes]

        if valid_mask.sum() == 0:
            accuracy = 0.0
        else:
            # Classification accuracy (only for valid boxes)
            pred_labels = torch.argmax(conf_scores, dim=-1)
            target_labels = torch.argmax(target_conf, dim=-1)
            accuracy = (pred_labels[valid_mask] == target_labels[valid_mask]).float().mean().item()

        conf_mean = conf_scores.mean().item()
        conf_std = conf_scores.std().item()

        pred_boxes = predictions[:, :, num_classes:]
        target_boxes = targets[:, :, num_classes:]
        box_l1 = F.l1_loss(pred_boxes, target_boxes).item()

        return {
            'accuracy': accuracy,
            'confidence_mean': conf_mean,
            'confidence_std': conf_std,
            'box_l1': box_l1
        }

    @staticmethod
    def _compute_ece(predictions: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
        """Compute Expected Calibration Error."""
        pred_probs = F.softmax(predictions, dim=-1)
        pred_conf, pred_labels = torch.max(pred_probs, dim=-1)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=predictions.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = torch.tensor(0.0, device=predictions.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (pred_conf > bin_lower) & (pred_conf <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                accuracy_in_bin = (pred_labels[in_bin] == targets[in_bin]).float().mean()
                avg_confidence_in_bin = pred_conf[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()

    @staticmethod
    def regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute regression metrics (for depth, normal, etc.).

        Args:
            predictions: Model predictions [batch_size, channels, H, W] or [batch_size, features]
            targets: Ground truth [batch_size, channels, H, W] or [batch_size, features]

        Returns:
            Dictionary of metrics (mse, mae, rmse)
        """
        # Flatten spatial dimensions if present
        if predictions.dim() == 4:  # [batch, channels, H, W]
            pred_flat = predictions.view(predictions.shape[0], -1)
            target_flat = targets.view(targets.shape[0], -1)
        elif predictions.dim() == 3:  # [batch, H, W] or [batch, channels, features]
            if predictions.shape[1] == 3:  # [batch, 3, H*W] for normal
                pred_flat = predictions.view(predictions.shape[0], -1)
                target_flat = targets.view(targets.shape[0], -1)
            else:
                pred_flat = predictions.view(predictions.shape[0], -1)
                target_flat = targets.view(targets.shape[0], -1)
        else:
            pred_flat = predictions
            target_flat = targets

        # Compute metrics
        mse = F.mse_loss(pred_flat, target_flat).item()
        mae = F.l1_loss(pred_flat, target_flat).item()
        rmse = np.sqrt(mse)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }

    @staticmethod
    def normal_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute surface normal prediction metrics including angle error.

        Args:
            predictions: Model predictions [batch_size, 3, H, W] - surface normal vectors
            targets: Ground truth [batch_size, 3, H, W] - surface normal vectors

        Returns:
            Dictionary of metrics (mse, mae, rmse, mErr - mean angle error in degrees)
        """
        # Ensure 4D tensor
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        # Normalize predictions to unit vectors
        pred_norm = F.normalize(predictions, p=2, dim=1)  # [B, 3, H, W]
        target_norm = F.normalize(targets, p=2, dim=1)    # [B, 3, H, W]

        # Compute cosine similarity (dot product of normalized vectors)
        # cos_sim shape: [B, H, W]
        cos_sim = (pred_norm * target_norm).sum(dim=1)

        # Clamp to valid range for arccos [-1, 1]
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # Compute angle error in radians, then convert to degrees
        angle_error_rad = torch.acos(cos_sim)
        angle_error_deg = angle_error_rad * (180.0 / np.pi)

        # Mean angle error
        mErr = angle_error_deg.mean().item()

        # Also compute standard regression metrics
        pred_flat = predictions.view(predictions.shape[0], -1)
        target_flat = targets.view(targets.shape[0], -1)
        mse = F.mse_loss(pred_flat, target_flat).item()
        mae = F.l1_loss(pred_flat, target_flat).item()
        rmse = np.sqrt(mse)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mErr': mErr  # Mean angle error in degrees
        }

    @staticmethod
    def instance_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute instance segmentation metrics (binary foreground/background).

        Args:
            predictions: Model predictions [batch_size, 2, H, W] - binary class logits
            targets: Ground truth [batch_size, H, W] - binary mask (0=background, 1=foreground)

        Returns:
            Dictionary of metrics (mAP, dice, iou, precision, recall, f1)
        """
        with torch.no_grad():
            # Get predicted class (argmax over channel dimension)
            if predictions.dim() == 4 and predictions.shape[1] >= 2:
                # Get softmax probabilities for foreground class
                probs = F.softmax(predictions, dim=1)[:, 1, :, :]  # [B, H, W] - foreground prob
                pred_labels = predictions.argmax(dim=1)  # [B, H, W]
            else:
                probs = torch.sigmoid(predictions.squeeze(1))
                pred_labels = (probs > 0.5).long()

            # Flatten for metric computation (use reshape for non-contiguous tensors)
            probs_flat = probs.reshape(-1).cpu().numpy()
            pred_flat = pred_labels.reshape(-1).cpu().numpy()
            target_flat = targets.reshape(-1).cpu().numpy()

            # Filter out ignore index (255)
            valid_mask = target_flat != 255
            probs_flat = probs_flat[valid_mask]
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]

            # Compute mAP (Average Precision)
            try:
                if len(np.unique(target_flat)) > 1:  # Need both classes for AP
                    mAP = average_precision_score(target_flat, probs_flat)
                else:
                    mAP = 1.0 if (target_flat == pred_flat).all() else 0.0
            except:
                mAP = 0.0

            # Compute Dice and IoU for foreground class
            tp = ((pred_flat == 1) & (target_flat == 1)).sum()
            fp = ((pred_flat == 1) & (target_flat == 0)).sum()
            fn = ((pred_flat == 0) & (target_flat == 1)).sum()
            tn = ((pred_flat == 0) & (target_flat == 0)).sum()

            # Precision, Recall, F1
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            # Dice coefficient
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

            # IoU (Intersection over Union) for foreground
            iou = tp / (tp + fp + fn + 1e-8)

            return {
                'mAP': float(mAP),
                'dice_mean': float(dice),
                'iou_mean': float(iou),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }

    @staticmethod
    def _get_primary_metric(metrics: Dict[str, float]) -> float:
        """Get primary metric for a task."""
        if 'accuracy' in metrics:
            return metrics['accuracy']
        elif 'dice_mean' in metrics:
            return metrics['dice_mean']
        elif 'iou_mean' in metrics:
            return metrics['iou_mean']
        elif 'mAP' in metrics:
            return metrics['mAP']
        elif 'auroc' in metrics:
            return metrics['auroc']
        elif 'rmse' in metrics:
            # For regression tasks, use negative RMSE (lower is better, so negate for higher-is-better)
            return -metrics['rmse']
        elif 'mae' in metrics:
            # For regression tasks, use negative MAE
            return -metrics['mae']
        else:
            if len(metrics) > 0:
                return list(metrics.values())[0]
            else:
                return 0.0


class MTLMetrics:
    """MTL summary metrics."""

    # Task difficulty weights: harder tasks get higher weights
    # Based on empirical observation: Segmentation is hardest (0.55-0.65),
    # Detection is medium (~0.95-1.0), Classification is easiest (~0.98-0.99)
    # These weights emphasize performance on harder tasks
    TASK_DIFFICULTY_WEIGHTS = {
        'classification': 0.5,   # Easiest task - lower weight
        'detection': 1.0,        # Medium difficulty - baseline weight
        'segmentation': 2.0      # Hardest task - highest weight
    }

    @staticmethod
    def simple_sum(
        per_task_metrics: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Compute simple sum MTL score: cls + det + seg performance.
        No normalization, just sum of task performances.

        Args:
            per_task_metrics: Current per-task metrics

        Returns:
            Simple sum MTL score
        """
        total_score = 0.0

        for task_name, metrics in per_task_metrics.items():
            primary_metric = PerTaskMetrics._get_primary_metric(metrics)
            total_score += primary_metric

        return total_score

    @staticmethod
    def weighted_sum(
        per_task_metrics: Dict[str, Dict[str, float]],
        task_weights: Dict[str, float] = None
    ) -> float:
        """
        Compute weighted sum MTL score with task difficulty weights.
        Harder tasks (segmentation) get higher weights.

        Args:
            per_task_metrics: Current per-task metrics
            task_weights: Optional custom weights. Default uses TASK_DIFFICULTY_WEIGHTS.

        Returns:
            Weighted sum MTL score (normalized by total weight)
        """
        if task_weights is None:
            task_weights = MTLMetrics.TASK_DIFFICULTY_WEIGHTS

        total_score = 0.0
        total_weight = 0.0

        for task_name, metrics in per_task_metrics.items():
            primary_metric = PerTaskMetrics._get_primary_metric(metrics)
            weight = task_weights.get(task_name, 1.0)
            total_score += weight * primary_metric
            total_weight += weight

        # Normalize by total weight to keep score in reasonable range
        return total_score / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def weighted_harmonic_mean(
        per_task_metrics: Dict[str, Dict[str, float]],
        task_weights: Dict[str, float] = None,
        epsilon: float = 1e-8
    ) -> float:
        """
        Compute weighted harmonic mean of per-task metrics.
        Harmonic mean penalizes poor performance on any task more than arithmetic mean.
        Weights emphasize harder tasks.

        Args:
            per_task_metrics: Current per-task metrics
            task_weights: Optional custom weights. Default uses TASK_DIFFICULTY_WEIGHTS.
            epsilon: Small constant to avoid division by zero

        Returns:
            Weighted harmonic mean score
        """
        if task_weights is None:
            task_weights = MTLMetrics.TASK_DIFFICULTY_WEIGHTS

        weighted_reciprocal_sum = 0.0
        total_weight = 0.0

        for task_name, metrics in per_task_metrics.items():
            primary_metric = PerTaskMetrics._get_primary_metric(metrics)
            weight = task_weights.get(task_name, 1.0)
            weighted_reciprocal_sum += weight / (primary_metric + epsilon)
            total_weight += weight

        if total_weight == 0 or weighted_reciprocal_sum == 0:
            return 0.0

        # Weighted harmonic mean = total_weight / sum(weight_i / score_i)
        return total_weight / weighted_reciprocal_sum

    @staticmethod
    def active_improvement(
        current_metrics: Dict[str, Dict[str, float]],
        previous_metrics: Optional[Dict[str, Dict[str, float]]]
    ) -> float:
        """
        Compute MTL Active Score: improvement from previous round.
        Sum of (current - previous) for each task.

        Args:
            current_metrics: Current round per-task metrics
            previous_metrics: Previous round per-task metrics (None for round 0)

        Returns:
            MTL Active Score (improvement from previous round)
        """
        if previous_metrics is None:
            # Round 0: no improvement, return 0
            return 0.0

        total_improvement = 0.0

        for task_name, current_task_metrics in current_metrics.items():
            if task_name in previous_metrics:
                current_metric = PerTaskMetrics._get_primary_metric(current_task_metrics)
                previous_metric = PerTaskMetrics._get_primary_metric(previous_metrics[task_name])
                improvement = current_metric - previous_metric
                total_improvement += improvement

        return total_improvement

    @staticmethod
    def normalized_average(
        per_task_metrics: Dict[str, Dict[str, float]],
        base_metrics: Dict[str, Dict[str, float]],
        oracle_metrics: Dict[str, Dict[str, float]],
        epsilon: float = 1e-8
    ) -> float:
        """
        Compute normalized average MTL score.

        Args:
            per_task_metrics: Current per-task metrics
            base_metrics: Baseline (round 0) metrics
            oracle_metrics: Oracle (fully labeled) metrics
            epsilon: Small constant to avoid division by zero

        Returns:
            Normalized average MTL score
        """
        scores = []

        for task_name in per_task_metrics:
            if task_name in base_metrics and task_name in oracle_metrics:
                # Get primary metric for each task
                current_metric = PerTaskMetrics._get_primary_metric(per_task_metrics[task_name])
                base_metric = PerTaskMetrics._get_primary_metric(base_metrics[task_name])
                oracle_metric = PerTaskMetrics._get_primary_metric(oracle_metrics[task_name])

                if oracle_metric - base_metric > epsilon:
                    normalized_score = (current_metric - base_metric) / (oracle_metric - base_metric + epsilon)
                    scores.append(normalized_score)

        return np.mean(scores) if scores else 0.0

    @staticmethod
    def harmonic_mean(
        per_task_metrics: Dict[str, Dict[str, float]],
        epsilon: float = 1e-8
    ) -> float:
        """
        Compute harmonic mean of per-task metrics.

        Args:
            per_task_metrics: Current per-task metrics
            epsilon: Small constant to avoid division by zero

        Returns:
            Harmonic mean score
        """
        scores = []

        for task_name, metrics in per_task_metrics.items():
            primary_metric = PerTaskMetrics._get_primary_metric(metrics)
            scores.append(primary_metric + epsilon)

        if not scores:
            return 0.0

        harmonic_mean = len(scores) / sum(1.0 / score for score in scores)
        return harmonic_mean

    @staticmethod
    def task_balance_score(
        per_task_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute task balance score: measures how balanced performance is across tasks.

        Args:
            per_task_metrics: Current per-task metrics

        Returns:
            Dictionary with balance metrics:
            - std_dev: Standard deviation of task scores (lower is better)
            - balance_score: Normalized balance score (higher is better, 0-1)
            - min_max_ratio: Ratio of min to max task score (higher is better, 0-1)
        """
        task_scores = []
        for task_name, metrics in per_task_metrics.items():
            primary_metric = PerTaskMetrics._get_primary_metric(metrics)
            task_scores.append(primary_metric)

        if not task_scores:
            return {
                'std_dev': 0.0,
                'balance_score': 1.0,
                'min_max_ratio': 1.0
            }

        std_dev = float(np.std(task_scores))
        mean_score = float(np.mean(task_scores))

        # Balance score: higher is better (1.0 = perfectly balanced)
        # Normalize by mean to make it scale-invariant
        balance_score = 1.0 / (1.0 + std_dev / (mean_score + 1e-8))

        # Min-max ratio: ratio of worst to best task
        min_score = float(np.min(task_scores))
        max_score = float(np.max(task_scores))
        min_max_ratio = min_score / (max_score + 1e-8)

        return {
            'std_dev': std_dev,
            'balance_score': balance_score,
            'min_max_ratio': min_max_ratio
        }

    @staticmethod
    def difficulty_weighted_balance(
        per_task_metrics: Dict[str, Dict[str, float]],
        task_weights: Dict[str, float] = None,
        difficulty_ceiling: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Compute difficulty-weighted balance score.

        Key insight: A method achieving 0.65 dice on segmentation (hard task)
        while getting 0.99 on classification (easy task) should have HIGHER
        balance score than a method with 0.99/0.99/0.55 that ignores the hard task.

        This metric:
        1. Normalizes each task by its "difficulty ceiling" (max achievable)
        2. Weights tasks by their difficulty
        3. Penalizes methods that neglect harder tasks

        Args:
            per_task_metrics: Current per-task metrics
            task_weights: Difficulty weights. Default uses TASK_DIFFICULTY_WEIGHTS.
            difficulty_ceiling: Expected max performance per task. Default based on empirics.

        Returns:
            Dictionary with difficulty-weighted balance metrics
        """
        if task_weights is None:
            task_weights = MTLMetrics.TASK_DIFFICULTY_WEIGHTS

        if difficulty_ceiling is None:
            # Empirical ceiling: typical max performance each task can achieve
            difficulty_ceiling = {
                'classification': 1.0,    # Classification can reach ~99%
                'detection': 1.0,         # Detection can reach ~100%
                'segmentation': 0.70      # Segmentation typically maxes at ~65-70%
            }

        normalized_scores = {}
        weighted_scores = []
        total_weight = 0.0

        for task_name, metrics in per_task_metrics.items():
            primary_metric = PerTaskMetrics._get_primary_metric(metrics)
            ceiling = difficulty_ceiling.get(task_name, 1.0)
            weight = task_weights.get(task_name, 1.0)

            # Normalize by difficulty ceiling: how close to max achievable?
            normalized = min(primary_metric / (ceiling + 1e-8), 1.0)
            normalized_scores[task_name] = normalized

            weighted_scores.append(weight * normalized)
            total_weight += weight

        if not weighted_scores:
            return {
                'difficulty_balance_score': 0.0,
                'weighted_mean': 0.0,
                'normalized_scores': {},
                'hardest_task_score': 0.0
            }

        # Weighted mean of normalized scores
        weighted_mean = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0

        # Standard deviation of normalized scores (not weighted, for fairness check)
        norm_values = list(normalized_scores.values())
        norm_std = float(np.std(norm_values))

        # Difficulty-aware balance: high weighted mean + low normalized std
        # Penalize high variance in normalized scores
        difficulty_balance = weighted_mean * (1.0 - 0.5 * norm_std)

        # Hardest task bonus: methods excelling at hard tasks get extra credit
        hardest_task = max(task_weights.keys(), key=lambda k: task_weights.get(k, 0))
        hardest_task_score = normalized_scores.get(hardest_task, 0.0)

        return {
            'difficulty_balance_score': difficulty_balance,
            'weighted_mean': weighted_mean,
            'normalized_scores': normalized_scores,
            'hardest_task_score': hardest_task_score,
            'normalized_std': norm_std
        }

    @staticmethod
    def mtl_composite_score(
        per_task_metrics: Dict[str, Dict[str, float]],
        task_weights: Dict[str, float] = None,
        balance_weight: float = 0.3
    ) -> float:
        """
        Compute composite MTL score combining performance and balance.

        Score = (1 - balance_weight) * weighted_harmonic_mean + balance_weight * difficulty_balance

        This score:
        1. Uses harmonic mean to penalize poor performance on any task
        2. Weights harder tasks more heavily
        3. Includes balance as a component to reward balanced improvement

        Args:
            per_task_metrics: Current per-task metrics
            task_weights: Difficulty weights. Default uses TASK_DIFFICULTY_WEIGHTS.
            balance_weight: Weight for balance component (0-1). Default 0.3.

        Returns:
            Composite MTL score
        """
        if task_weights is None:
            task_weights = MTLMetrics.TASK_DIFFICULTY_WEIGHTS

        # Performance component: weighted harmonic mean
        perf_score = MTLMetrics.weighted_harmonic_mean(per_task_metrics, task_weights)

        # Balance component: difficulty-weighted balance
        balance_result = MTLMetrics.difficulty_weighted_balance(per_task_metrics, task_weights)
        balance_score = balance_result['difficulty_balance_score']

        # Composite: combine performance and balance
        composite = (1 - balance_weight) * perf_score + balance_weight * balance_score

        return composite

    @staticmethod
    def convergence_rate(
        rounds_history: List[Dict[str, Dict[str, float]]],
        compute_score_func=None
    ) -> Dict[str, float]:
        """
        Compute convergence rate: how fast performance improves across rounds.

        Args:
            rounds_history: List of per_task_metrics for each round
            compute_score_func: Function to compute score from metrics (default: simple_sum)

        Returns:
            Dictionary with convergence metrics:
            - avg_improvement: Average improvement per round
            - slope: Linear regression slope (positive = improving)
            - early_improvement: Average improvement in first half
            - late_improvement: Average improvement in second half
        """
        if compute_score_func is None:
            compute_score_func = MTLMetrics.simple_sum

        if len(rounds_history) < 2:
            return {
                'avg_improvement': 0.0,
                'slope': 0.0,
                'early_improvement': 0.0,
                'late_improvement': 0.0
            }

        # Compute scores for each round
        scores = [compute_score_func(metrics) for metrics in rounds_history]

        # Average improvement per round
        improvements = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
        avg_improvement = float(np.mean(improvements))

        # Linear regression slope
        rounds = np.arange(len(scores))
        if len(scores) > 1:
            slope = float(np.polyfit(rounds, scores, 1)[0])
        else:
            slope = 0.0

        # Early vs late improvement
        mid_point = len(improvements) // 2
        early_improvements = improvements[:mid_point] if mid_point > 0 else improvements
        late_improvements = improvements[mid_point:] if mid_point < len(improvements) else []

        early_improvement = float(np.mean(early_improvements)) if early_improvements else 0.0
        late_improvement = float(np.mean(late_improvements)) if late_improvements else 0.0

        return {
            'avg_improvement': avg_improvement,
            'slope': slope,
            'early_improvement': early_improvement,
            'late_improvement': late_improvement
        }

    @staticmethod
    def label_efficiency(
        final_metrics: Dict[str, Dict[str, float]],
        total_labels_used: int,
        target_score: float = None,
        compute_score_func=None
    ) -> Dict[str, float]:
        """
        Compute label efficiency: performance per label used.

        Args:
            final_metrics: Final round per-task metrics
            total_labels_used: Total number of labels used
            target_score: Target MTL score to achieve (optional)
            compute_score_func: Function to compute score from metrics (default: simple_sum)

        Returns:
            Dictionary with efficiency metrics:
            - efficiency: Final score per label
            - labels_to_target: Labels needed to reach target (if target provided)
        """
        if compute_score_func is None:
            compute_score_func = MTLMetrics.simple_sum

        final_score = compute_score_func(final_metrics)

        # Efficiency: score per label
        efficiency = final_score / (total_labels_used + 1e-8)

        result = {'efficiency': efficiency}

        # If target provided, estimate labels needed (linear extrapolation)
        if target_score is not None and final_score > 0:
            # Simple linear extrapolation (can be improved with curve fitting)
            labels_to_target = (target_score / final_score) * total_labels_used if final_score > 0 else float('inf')
            result['labels_to_target'] = labels_to_target

        return result


class MetricsLogger:
    """Logger for tracking metrics across AL rounds."""

    def __init__(self, save_dir: str = './results'):
        self.save_dir = save_dir
        self.metrics_history = []
        self.current_method = None
        self.results_file = None

        # Create save directory
        import os
        os.makedirs(save_dir, exist_ok=True)

    def log_round(
        self,
        round_num: int,
        method: str,
        criteria: List[str],
        tasks: List[str],
        per_task_metrics: Dict[str, Dict[str, float]],
        mtl_score: float,
        seed: int,
        label_budget: float,
        selected_indices: List[int] = None,
        total_selected: int = None,
        best_train_metrics: Dict[str, Dict[str, float]] = None,
        mtl_active_score: float = None
    ):
        """Log metrics for a single AL round and append to results JSON."""
        import json
        import os

        log_entry = {
            'round': round_num,
            'method': method,
            'criteria': criteria,
            'tasks': tasks,
            'per_task_metrics': per_task_metrics,
            'mtl_score': mtl_score,
            'seed': seed,
            'label_budget': label_budget
        }

        # Add MTL Active Score if provided
        if mtl_active_score is not None:
            log_entry['mtl_active_score'] = mtl_active_score

        # Add selected_indices if provided
        if selected_indices is not None:
            log_entry['selected_indices'] = selected_indices
        if total_selected is not None:
            log_entry['total_selected'] = total_selected
        if best_train_metrics is not None:
            log_entry['best_train_metrics'] = best_train_metrics

        self.metrics_history.append(log_entry)

        # Save to single JSON file per method (append to array)
        results_file = f"{self.save_dir}/{method}_results.json"

        # Load existing results if file exists
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = []

        # Append new round entry
        results.append(log_entry)

        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def save_summary(self):
        """Save summary of all rounds with additional metrics."""
        import json
        import os

        import pandas as pd

        # Compute additional metrics
        additional_metrics = self._compute_additional_metrics()

        # Add additional metrics to metrics_history for JSON saving
        summary_data = {
            'rounds': self.metrics_history,
            'summary_metrics': additional_metrics
        }

        # Save JSON with summary metrics
        json_path = f"{self.save_dir}/all_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"  💾 Saved: {json_path}")

        # Also save summary metrics separately
        summary_json_path = f"{self.save_dir}/summary_metrics.json"
        with open(summary_json_path, 'w') as f:
            json.dump(additional_metrics, f, indent=2)
        print(f"  💾 Saved: {summary_json_path}")

        # Save CSV
        df_data = []
        for entry in self.metrics_history:
            row = {
                'round': entry['round'],
                'method': entry['method'],
                'criteria': str(entry['criteria']),
                'tasks': str(entry['tasks']),
                'mtl_score': entry['mtl_score'],
                'seed': entry['seed'],
                'label_budget': entry['label_budget']
            }

            # Add MTL Active Score if available
            if 'mtl_active_score' in entry:
                row['mtl_active_score'] = entry['mtl_active_score']

            # Add per-task metrics
            for task_name, metrics in entry['per_task_metrics'].items():
                for metric_name, value in metrics.items():
                    row[f"{task_name}_{metric_name}"] = value

            df_data.append(row)

        csv_path = f"{self.save_dir}/metrics_summary.csv"
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        print(f"  💾 Saved: {csv_path}")

        # List all result files for each method
        method_files = {}
        for entry in self.metrics_history:
            method = entry['method']
            if method not in method_files:
                method_files[method] = []
            method_files[method].append(entry['round'])

        for method, rounds in method_files.items():
            method_json_path = f"{self.save_dir}/{method}_results.json"
            if os.path.exists(method_json_path):
                file_size = os.path.getsize(method_json_path) / 1024  # KB
                print(f"  💾 Saved: {method_json_path} ({len(rounds)} rounds, {file_size:.1f} KB)")

    def _compute_additional_metrics(self) -> Dict:
        """Compute additional metrics (balance, convergence, efficiency) from history."""
        if not self.metrics_history:
            return {}

        # Extract per_task_metrics for each round
        rounds_history = [entry['per_task_metrics'] for entry in self.metrics_history]

        # Compute task balance for each round (original)
        balance_scores = []
        for metrics in rounds_history:
            balance = MTLMetrics.task_balance_score(metrics)
            balance_scores.append(balance['balance_score'])

        # Compute NEW difficulty-weighted balance for each round
        difficulty_balance_scores = []
        for metrics in rounds_history:
            diff_balance = MTLMetrics.difficulty_weighted_balance(metrics)
            difficulty_balance_scores.append(diff_balance['difficulty_balance_score'])

        # Compute NEW weighted scores for each round
        weighted_scores = []
        weighted_harmonic_scores = []
        composite_scores = []
        for metrics in rounds_history:
            weighted_scores.append(MTLMetrics.weighted_sum(metrics))
            weighted_harmonic_scores.append(MTLMetrics.weighted_harmonic_mean(metrics))
            composite_scores.append(MTLMetrics.mtl_composite_score(metrics))

        # Compute convergence rate
        convergence = MTLMetrics.convergence_rate(rounds_history)

        # Compute label efficiency (using final round)
        final_metrics = rounds_history[-1] if rounds_history else {}
        total_labels = self.metrics_history[-1].get('total_selected', 0) if self.metrics_history else 0
        efficiency = MTLMetrics.label_efficiency(final_metrics, total_labels)

        # Get original final MTL score (simple sum)
        final_mtl_score = self.metrics_history[-1]['mtl_score'] if self.metrics_history else 0.0

        # Compute average balance scores
        avg_balance = float(np.mean(balance_scores)) if balance_scores else 0.0
        avg_difficulty_balance = float(np.mean(difficulty_balance_scores)) if difficulty_balance_scores else 0.0

        return {
            'final_mtl_score': final_mtl_score,
            # NEW: Weighted MTL scores
            'final_weighted_mtl_score': weighted_scores[-1] if weighted_scores else 0.0,
            'final_weighted_harmonic_score': weighted_harmonic_scores[-1] if weighted_harmonic_scores else 0.0,
            'final_composite_score': composite_scores[-1] if composite_scores else 0.0,
            # Original balance
            'task_balance': {
                'average_balance_score': avg_balance,
                'final_balance_score': balance_scores[-1] if balance_scores else 0.0,
                'balance_scores_per_round': balance_scores
            },
            # NEW: Difficulty-weighted balance
            'difficulty_weighted_balance': {
                'average_score': avg_difficulty_balance,
                'final_score': difficulty_balance_scores[-1] if difficulty_balance_scores else 0.0,
                'scores_per_round': difficulty_balance_scores
            },
            # Score trajectories for analysis
            'score_trajectories': {
                'weighted_sum': weighted_scores,
                'weighted_harmonic': weighted_harmonic_scores,
                'composite': composite_scores
            },
            'convergence': convergence,
            'label_efficiency': efficiency,
            'total_rounds': len(self.metrics_history),
            'total_labels_used': total_labels
        }

    def plot_metrics(self, metric_name: str = 'mtl_score'):
        """Plot metrics across rounds."""
        import os

        rounds = [entry['round'] for entry in self.metrics_history]
        scores = [entry[metric_name] for entry in self.metrics_history]

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, scores, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('AL Round')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'{metric_name.replace("_", " ").title()} vs AL Rounds')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = f"{self.save_dir}/{metric_name}_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path) / 1024  # KB
            print(f"  💾 Saved: {plot_path} ({file_size:.1f} KB)")

    def save_task_performance(self):
        """Save task performance summary to task_perf.json."""
        import json
        import os

        task_perf = []
        for entry in self.metrics_history:
            round_num = entry['round']
            method = entry['method']
            best_train_metrics = entry.get('best_train_metrics', {})

            round_perf = {
                'round': round_num,
                'method': method
            }

            # Format task performance - dynamically handle all tasks
            for task_name in best_train_metrics:
                task_metrics = best_train_metrics[task_name]
                if task_name == 'classification':
                    round_perf[task_name] = {
                        'acc': task_metrics.get('acc', 0.0),
                        'loss': task_metrics.get('loss', 0.0)
                    }
                elif task_name == 'detection':
                    round_perf[task_name] = {
                        'acc': task_metrics.get('acc', 0.0),
                        'box_l1': task_metrics.get('box_l1', 0.0),
                        'loss': task_metrics.get('loss', 0.0)
                    }
                elif task_name == 'segmentation':
                    round_perf[task_name] = {
                        'dice': task_metrics.get('dice', 0.0),
                        'iou': task_metrics.get('iou', 0.0),
                        'loss': task_metrics.get('loss', 0.0)
                    }
                elif task_name in ['depth', 'normal']:
                    # Regression tasks
                    round_perf[task_name] = {
                        'mse': task_metrics.get('mse', 0.0),
                        'mae': task_metrics.get('mae', 0.0),
                        'rmse': task_metrics.get('rmse', 0.0),
                        'loss': task_metrics.get('loss', 0.0)
                    }

            task_perf.append(round_perf)

        json_path = f"{self.save_dir}/task_perf.json"
        with open(json_path, 'w') as f:
            json.dump(task_perf, f, indent=2)

        if os.path.exists(json_path):
            file_size = os.path.getsize(json_path) / 1024  # KB
            print(f"  💾 Saved: {json_path} ({file_size:.1f} KB)")

    def plot_performance(self):
        """Plot task performance across rounds."""
        import os

        # Group by method
        methods = {}
        for entry in self.metrics_history:
            method = entry['method']
            if method not in methods:
                methods[method] = []
            methods[method].append(entry)

        # Sort each method's entries by round
        for method in methods:
            methods[method].sort(key=lambda x: x['round'])

        # Get all tasks from metrics_history
        all_tasks = set()
        for entry in self.metrics_history:
            per_task_metrics = entry.get('per_task_metrics', {})
            all_tasks.update(per_task_metrics.keys())

        # Also check best_train_metrics
        for entry in self.metrics_history:
            best_train = entry.get('best_train_metrics', {})
            all_tasks.update(best_train.keys())

        all_tasks = sorted(list(all_tasks))

        if not all_tasks:
            return

        # Create figure with subplots (one for each task)
        num_tasks = len(all_tasks)
        fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 5))
        if num_tasks == 1:
            axes = [axes]

        for task_idx, task_name in enumerate(all_tasks):
            ax = axes[task_idx]

            for method, entries in methods.items():
                rounds = [e['round'] for e in entries]
                metric_values = []
                losses = []

                for entry in entries:
                    best_train = entry.get('best_train_metrics', {})
                    if task_name in best_train:
                        task_metrics = best_train[task_name]

                        # Determine metric based on task type
                        if task_name in ['classification', 'detection']:
                            metric_val = task_metrics.get('acc', 0.0)
                            metric_label = 'Accuracy'
                        elif task_name == 'segmentation':
                            metric_val = task_metrics.get('dice', task_metrics.get('iou', 0.0))
                            metric_label = 'Dice Score' if 'dice' in task_metrics else 'IoU'
                        elif task_name in ['depth', 'normal']:
                            # Regression tasks: use RMSE (lower is better, so we'll plot negative or inverse)
                            metric_val = task_metrics.get('rmse', 0.0)
                            metric_label = 'RMSE'
                        else:
                            # Default: try common metrics
                            metric_val = task_metrics.get('acc', task_metrics.get(
                                'dice', task_metrics.get('rmse', 0.0)))
                            metric_label = 'Metric'

                        loss_val = task_metrics.get('loss', 0.0)
                        metric_values.append(metric_val)
                        losses.append(loss_val)
                    else:
                        metric_values.append(0.0)
                        losses.append(0.0)

                if metric_values and losses:
                    ax_twin = ax.twinx()

                    # For regression tasks (depth, normal), RMSE is lower-is-better, so plot it directly
                    if task_name in ['depth', 'normal']:
                        line1 = ax.plot(rounds, metric_values, 'o-',
                                        label=f'{method} (RMSE)', linewidth=2, markersize=6, color='blue')
                        ax.set_ylabel('RMSE', fontsize=12, color='blue')
                    else:
                        line1 = ax.plot(rounds, metric_values, 'o-',
                                        label=f'{method} ({metric_label.lower()})', linewidth=2, markersize=6, color='blue')
                        ax.set_ylabel(metric_label, fontsize=12, color='blue')

                    line2 = ax_twin.plot(rounds, losses, 's--',
                                         label=f'{method} (loss)', linewidth=2, markersize=6, alpha=0.7, color='red')
                    ax.set_xlabel('AL Round', fontsize=12)
                    ax_twin.set_ylabel('Loss', fontsize=12, color='red')
                    ax.set_title(f'{task_name.capitalize()}: {metric_label} & Loss', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='y', labelcolor='blue')
                    ax_twin.tick_params(axis='y', labelcolor='red')

                    # Combine legends
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='best')

        plt.tight_layout()
        plot_path = f"{self.save_dir}/perf.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path) / 1024  # KB
            print(f"  💾 Saved: {plot_path} ({file_size:.1f} KB)")

    def plot_task_balance(self):
        """Plot task balance score across rounds."""
        import os

        if not self.metrics_history:
            return

        rounds = [entry['round'] for entry in self.metrics_history]
        balance_scores = []

        for entry in self.metrics_history:
            balance = MTLMetrics.task_balance_score(entry['per_task_metrics'])
            balance_scores.append(balance['balance_score'])

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, balance_scores, 'g-o', linewidth=2, markersize=6)
        plt.xlabel('AL Round')
        plt.ylabel('Task Balance Score')
        plt.title('Task Balance Score vs AL Rounds (Higher = More Balanced)')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.1])
        plt.tight_layout()

        plot_path = f"{self.save_dir}/task_balance_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path) / 1024  # KB
            print(f"  💾 Saved: {plot_path} ({file_size:.1f} KB)")

    def plot_convergence_rate(self):
        """Plot convergence metrics."""
        import os

        if len(self.metrics_history) < 2:
            return

        rounds = [entry['round'] for entry in self.metrics_history]
        rounds_history = [entry['per_task_metrics'] for entry in self.metrics_history]

        # Compute MTL scores for each round
        mtl_scores = [MTLMetrics.simple_sum(metrics) for metrics in rounds_history]

        # Compute improvements
        improvements = [mtl_scores[i] - mtl_scores[i - 1] for i in range(1, len(mtl_scores))]
        improvement_rounds = rounds[1:]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot 1: MTL Score over rounds with trend line
        ax1.plot(rounds, mtl_scores, 'b-o', linewidth=2, markersize=6, label='MTL Score')

        # Add linear trend line
        if len(rounds) > 1:
            z = np.polyfit(rounds, mtl_scores, 1)
            p = np.poly1d(z)
            ax1.plot(rounds, p(rounds), 'r--', linewidth=2, alpha=0.7, label=f'Trend (slope={z[0]:.4f})')

        ax1.set_xlabel('AL Round')
        ax1.set_ylabel('MTL Score')
        ax1.set_title('MTL Score Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Improvement per round
        ax2.bar(improvement_rounds, improvements, alpha=0.7, color='green')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('AL Round')
        ax2.set_ylabel('Improvement per Round')
        ax2.set_title('Convergence Rate (Improvement per Round)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f"{self.save_dir}/convergence_rate_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path) / 1024  # KB
            print(f"  💾 Saved: {plot_path} ({file_size:.1f} KB)")

    def plot_label_efficiency(self):
        """Plot label efficiency metrics."""
        import os

        if not self.metrics_history:
            return

        rounds = [entry['round'] for entry in self.metrics_history]
        total_labels_per_round = []
        mtl_scores = []
        efficiency_per_round = []

        cumulative_labels = 0
        for entry in self.metrics_history:
            # Estimate labels added this round (approximate)
            total_selected = entry.get('total_selected', 0)
            cumulative_labels = total_selected
            total_labels_per_round.append(cumulative_labels)

            mtl_score = entry['mtl_score']
            mtl_scores.append(mtl_score)

            # Efficiency = score / labels
            if cumulative_labels > 0:
                efficiency = mtl_score / cumulative_labels
            else:
                efficiency = 0.0
            efficiency_per_round.append(efficiency)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: MTL Score vs Total Labels
        ax1.plot(total_labels_per_round, mtl_scores, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Total Labels Used')
        ax1.set_ylabel('MTL Score')
        ax1.set_title('MTL Score vs Label Budget')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Efficiency (Score per Label)
        ax2.plot(rounds, efficiency_per_round, 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('AL Round')
        ax2.set_ylabel('Efficiency (Score per Label)')
        ax2.set_title('Label Efficiency Over Rounds')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f"{self.save_dir}/label_efficiency_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path) / 1024  # KB
            print(f"  💾 Saved: {plot_path} ({file_size:.1f} KB)")


# Example usage
if __name__ == "__main__":
    # Example metrics computation
    batch_size, num_classes = 32, 10

    # Classification metrics
    pred_cls = torch.randn(batch_size, num_classes)
    target_cls = torch.randint(0, num_classes, (batch_size,))

    cls_metrics = PerTaskMetrics.classification_metrics(pred_cls, target_cls)
    print("Classification metrics:", cls_metrics)

    # Segmentation metrics
    pred_seg = torch.randn(batch_size, num_classes, 32, 32)
    target_seg = torch.randint(0, num_classes, (batch_size, 32, 32))

    seg_metrics = PerTaskMetrics.segmentation_metrics(pred_seg, target_seg)
    print("Segmentation metrics:", seg_metrics)

    # MTL metrics
    per_task_metrics = {
        'classification': cls_metrics,
        'segmentation': seg_metrics
    }

    base_metrics = {
        'classification': {'accuracy': 0.5},
        'segmentation': {'dice_mean': 0.3}
    }

    oracle_metrics = {
        'classification': {'accuracy': 0.9},
        'segmentation': {'dice_mean': 0.8}
    }

    mtl_score = MTLMetrics.normalized_average(per_task_metrics, base_metrics, oracle_metrics)
    print("MTL score:", mtl_score)

    print("Metrics module working correctly")

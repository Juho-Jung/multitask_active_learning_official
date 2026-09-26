"""
Multi-Task Active Learning Acquisition Function

Implements the novel acquisition function that combines:
S(x) = [α'·Û(x) + β'·(1-Ĉg(x)) + γ'·(1-Ĉc(x))] · (1 + δ·D̂(x))

Where:
- Û(x): Normalized hardness (uncertainty)
- Ĉg(x): Gradient conflict measure (normalized to [0, 1], higher = more conflict)
- Ĉc(x): Task consistency measure (normalized to [0, 1], higher = more consistent)
- D̂(x): Diversity score (distance to other samples, normalized to [0, 1])
- α', β', γ': Normalized weights (α' = α/(α+β+γ), etc.) to ensure scale consistency
- δ: Diversity weight (default: 0.2)

Key improvements:
1. Normalized weights: Ensures each component has equal scale influence
2. Low conflict preference: (1-Ĉg(x)) prefers samples with low gradient conflict (helpful for all tasks)
3. Multiplicative diversity: Diversity amplifies base score rather than adding independently
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskActiveLearningAcquisition:
    """
    Multi-task active learning acquisition function that combines hardness,
    gradient conflict, and task consistency measures.

    Args:
        model: Multi-task model with shared encoder and task-specific heads
        alpha: Weight for hardness component (default: 1.0)
        beta: Weight for gradient conflict component (default: 1.0) 
        gamma: Weight for task consistency component (default: 1.0)
        hardness_mode: Method for computing hardness ('entropy' or 'margin')
        conflict_mode: Method for computing gradient conflict ('cosine' or 'pcgrad')
        consistency_mode: Method for computing task consistency ('kl' or 'mse')
        device: Device to run computations on
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 0.2,
        hardness_mode: str = 'entropy',
        conflict_mode: str = 'cosine',
        consistency_mode: str = 'kl',
        diversity_mode: str = 'euclidean',
        device: torch.device = None,
        # Ablation parameters
        tasks: List[str] = None,
        criteria: List[str] = None,
        auto_balance_weights: bool = True,
        # New: Third component mode (consistency, ctmi, or synergy)
        third_component_mode: str = 'consistency',
        # CTMI parameters: number of MC-dropout samples for parameter uncertainty
        ctmi_num_samples: int = 10,
        ctmi_use_mc_dropout: bool = True,
        # CTMI mode: 'covariance_positive' (high cov = high CTMI) or 'covariance_negative' (low cov = high CTMI)
        ctmi_mode: str = 'covariance_negative',
        # Rebuttal ablations (E5/E6): how Hardness and CTMI are normalized and
        # combined into the Stage-1 informativeness score.
        score_normalization: str = 'minmax',   # 'minmax' | 'zscore' | 'rank'
        score_combination: str = 'sum'         # 'sum' | 'product'
    ):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta  # Diversity weight (lambda)
        self.hardness_mode = hardness_mode
        self.conflict_mode = conflict_mode
        self.consistency_mode = consistency_mode
        self.diversity_mode = diversity_mode
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tasks = tasks or ['classification', 'segmentation']
        self.criteria = criteria or ['hardness', 'conflict', 'consistency']
        self.auto_balance_weights = auto_balance_weights

        # Third component mode: 'consistency' (default), 'ctmi', or 'synergy'
        self.third_component_mode = third_component_mode

        # CTMI parameters
        self.ctmi_num_samples = ctmi_num_samples
        self.ctmi_use_mc_dropout = ctmi_use_mc_dropout
        self.ctmi_mode = ctmi_mode

        # E5/E6 ablation knobs
        assert score_normalization in ['minmax', 'zscore', 'rank'], \
            f"Invalid score_normalization: {score_normalization}"
        assert score_combination in ['sum', 'product'], \
            f"Invalid score_combination: {score_combination}"
        self.score_normalization = score_normalization
        self.score_combination = score_combination

        self._validate_ablation_params()

        if self.auto_balance_weights:
            self._balance_weights()
        assert hardness_mode in ['entropy', 'margin'], f"Invalid hardness_mode: {hardness_mode}"
        assert conflict_mode in ['cosine', 'pcgrad'], f"Invalid conflict_mode: {conflict_mode}"
        assert consistency_mode in ['kl', 'mse'], f"Invalid consistency_mode: {consistency_mode}"
        assert diversity_mode in ['euclidean', 'cosine'], f"Invalid diversity_mode: {diversity_mode}"
        assert third_component_mode in ['consistency', 'ctmi', 'synergy'], \
            f"Invalid third_component_mode: {third_component_mode}"

        # Move model to device
        self.model.to(self.device)

    def update_hyperparameters(self, alpha: float = None, beta: float = None,
                               gamma: float = None, delta: float = None):
        """Update hyperparameters dynamically.

        Args:
            alpha: New weight for hardness component
            beta: New weight for gradient conflict component
            gamma: New weight for task consistency component
            delta: New weight for diversity component
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if delta is not None:
            self.delta = delta

    def _validate_ablation_params(self):
        """Validate ablation parameters."""
        # Validate tasks
        valid_tasks = ['classification', 'detection', 'segmentation', 'depth', 'normal', 'instance']
        for task in self.tasks:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task: {task}. Must be one of {valid_tasks}")

        # Validate criteria
        valid_criteria = ['hardness', 'conflict', 'consistency']
        for criterion in self.criteria:
            if criterion not in valid_criteria:
                raise ValueError(f"Invalid criterion: {criterion}. Must be one of {valid_criteria}")

        # Validate task-criteria combinations
        num_tasks = len(self.tasks)
        if num_tasks == 1:
            # Single task: only hardness makes sense
            invalid_criteria = []
            if 'conflict' in self.criteria:
                invalid_criteria.append('conflict')
            if 'consistency' in self.criteria:
                invalid_criteria.append('consistency')

            if invalid_criteria:
                raise ValueError(
                    f"Invalid criteria combination: Single task ({self.tasks}) cannot use "
                    f"{invalid_criteria}. "
                    f"Conflict and consistency require at least 2 tasks. "
                    f"Use only 'hardness' for single task."
                )
        elif num_tasks == 2:
            # Two tasks: all criteria are valid
            pass
        else:
            # Three or more tasks: all criteria are valid
            pass

    def _balance_weights(self):
        """Automatically balance weights based on active criteria."""
        if len(self.criteria) == 1:
            # Single criterion gets full weight
            if 'hardness' in self.criteria:
                self.alpha, self.beta, self.gamma = 1.0, 0.0, 0.0
            elif 'conflict' in self.criteria:
                self.alpha, self.beta, self.gamma = 0.0, 1.0, 0.0
            elif 'consistency' in self.criteria:
                self.alpha, self.beta, self.gamma = 0.0, 0.0, 1.0
        elif len(self.criteria) == 2:
            # Two criteria get equal weight
            if 'hardness' in self.criteria and 'conflict' in self.criteria:
                self.alpha, self.beta, self.gamma = 0.5, 0.5, 0.0
            elif 'hardness' in self.criteria and 'consistency' in self.criteria:
                self.alpha, self.beta, self.gamma = 0.5, 0.0, 0.5
            elif 'conflict' in self.criteria and 'consistency' in self.criteria:
                self.alpha, self.beta, self.gamma = 0.0, 0.5, 0.5
        else:
            # Three criteria get equal weight
            self.alpha, self.beta, self.gamma = 1.0 / 3, 1.0 / 3, 1.0 / 3

    def compute_hardness(
        self,
        outputs: Dict[str, torch.Tensor],
        task_types: Dict[str, str]
    ) -> torch.Tensor:
        """
        Compute normalized hardness (uncertainty) for each sample.

        Args:
            outputs: Dictionary of model outputs for each task
            task_types: Dictionary mapping task names to their types ('classification', 'detection', 'segmentation')

        Returns:
            Normalized hardness scores [batch_size]
        """
        hardness_scores = []

        # Filter outputs based on active tasks
        active_outputs = {k: v for k, v in outputs.items() if k in self.tasks}

        for task_name, output in active_outputs.items():
            # Skip if output is not a tensor (e.g., empty list or None)
            if not isinstance(output, torch.Tensor):
                continue

            task_type = task_types.get(task_name, 'classification')

            if task_type == 'classification':
                if self.hardness_mode == 'entropy':
                    # Compute entropy: -sum(p * log(p))
                    probs = F.softmax(output, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                    hardness_scores.append(entropy)
                elif self.hardness_mode == 'margin':
                    # Compute margin: difference between top-2 predictions
                    probs = F.softmax(output, dim=-1)
                    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                    hardness_scores.append(1.0 - margin)  # Higher margin = lower hardness

            elif task_type == 'detection':
                # For detection, use confidence uncertainty + box regression uncertainty
                if output.dim() == 3:  # [batch, num_boxes, num_classes + 4]
                    # First num_classes are class scores, last 4 are bbox coordinates
                    num_classes = output.shape[2] - 4
                    # Extract confidence scores [batch, num_boxes, num_classes]
                    conf_scores = output[:, :, :num_classes]
                    # Extract box coordinates [batch, num_boxes, 4]
                    bbox_pred = output[:, :, num_classes:]

                    if self.hardness_mode == 'entropy':
                        # Classification uncertainty
                        probs = F.softmax(conf_scores, dim=-1)  # [batch, num_boxes, num_classes]
                        cls_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [batch, num_boxes]

                        # Box regression uncertainty: variance across boxes
                        # Higher variance indicates more uncertainty in localization
                        bbox_variance = bbox_pred.var(dim=1, keepdim=False)  # [batch, 4] - variance per coordinate
                        bbox_uncertainty = bbox_variance.mean(dim=-1)  # [batch] - average variance across coordinates

                        # Object-wise hardness: Use Top-K boxes sum to emphasize high-entropy objects
                        # This focuses on samples with multiple uncertain objects, which are more informative
                        num_boxes = cls_entropy.shape[1]
                        topk_boxes = min(5, num_boxes)  # Top-5 boxes or all if fewer
                        cls_entropy_topk = torch.topk(cls_entropy, k=topk_boxes, dim=1)[0]  # [batch, topk]
                        # Sum of top-k box entropies (more aggressive, better for detection)
                        cls_entropy_object_wise = cls_entropy_topk.sum(dim=1)

                        # Combined uncertainty: object-wise classification + box regression
                        combined_uncertainty = cls_entropy_object_wise + 0.5 * bbox_uncertainty
                        hardness_scores.append(combined_uncertainty)
                    elif self.hardness_mode == 'margin':
                        # Classification margin
                        probs = F.softmax(conf_scores, dim=-1)  # [batch, num_boxes, num_classes]
                        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)  # [batch, num_boxes, num_classes]
                        margin = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]  # [batch, num_boxes]
                        margin_mean = (1.0 - margin).mean(dim=1)  # [batch] - Average over boxes

                        # Box regression uncertainty: variance across boxes
                        bbox_variance = bbox_pred.var(dim=1, keepdim=False)  # [batch, 4]
                        bbox_uncertainty = bbox_variance.mean(dim=-1)  # [batch]

                        # Combined uncertainty
                        combined_uncertainty = margin_mean + 0.5 * bbox_uncertainty
                        hardness_scores.append(combined_uncertainty)

            elif task_type == 'segmentation':
                # For segmentation, use pixel-wise entropy
                if self.hardness_mode == 'entropy':
                    probs = F.softmax(output, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [batch, H, W]

                    # Object-wise hardness: Use 90th percentile to focus on high-entropy regions (objects)
                    # This prevents small object entropy from being diluted by large background regions
                    # Using 90th percentile to select more informative samples with high uncertainty
                    entropy_flat = entropy.flatten(1)  # [batch, H*W]
                    # Use 90th percentile (top 10% pixels) to focus on high-entropy regions (objects)
                    hardness = torch.quantile(entropy_flat, q=0.90, dim=1)  # [batch]
                    hardness_scores.append(hardness)
                elif self.hardness_mode == 'margin':
                    probs = F.softmax(output, dim=1)
                    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                    hardness_scores.append((1.0 - margin).mean(dim=(1, 2)))

            elif task_type in ['depth', 'normal']:
                # For regression tasks (depth, normal), use variance as uncertainty measure
                # Higher variance = more uncertain = harder sample
                if output.dim() == 4:  # [batch, channels, H, W]
                    # Spatial variance across spatial dimensions
                    variance = output.var(dim=(2, 3))  # [batch, channels]
                    uncertainty = variance.mean(dim=1)  # [batch]
                elif output.dim() == 2:  # [batch, features]
                    # Feature variance
                    uncertainty = output.var(dim=1)  # [batch]
                else:
                    # Fallback: use output magnitude variance
                    uncertainty = output.view(output.shape[0], -1).var(dim=1)
                hardness_scores.append(uncertainty)

            elif task_type in ['instance', 'instance_segmentation']:
                # Instance segmentation: handle both segmentation and detection formats
                if output.dim() == 4:  # [batch, num_classes, H, W] - segmentation format
                    if self.hardness_mode == 'entropy':
                        probs = F.softmax(output, dim=1)
                        pixel_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                        mean_entropy = pixel_entropy.mean(dim=(1, 2))
                        hardness_scores.append(mean_entropy)
                    elif self.hardness_mode == 'margin':
                        probs = F.softmax(output, dim=1)
                        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                        margin = sorted_probs[:, 0, :, :] - sorted_probs[:, 1, :, :]
                        margin_mean = (1.0 - margin).mean(dim=(1, 2))
                        hardness_scores.append(margin_mean)
                elif output.dim() == 3:  # [batch, num_boxes, num_classes + 4] - detection format
                    num_classes = output.shape[2] - 4
                    conf_scores = output[:, :, :num_classes]
                    if self.hardness_mode == 'entropy':
                        probs = F.softmax(conf_scores, dim=-1)
                        cls_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                        cls_entropy_mean = cls_entropy.mean(dim=1)
                        hardness_scores.append(cls_entropy_mean)
                    elif self.hardness_mode == 'margin':
                        probs = F.softmax(conf_scores, dim=-1)
                        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                        margin = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
                        margin_mean = (1.0 - margin).mean(dim=1)
                        hardness_scores.append(margin_mean)

        if not hardness_scores:
            raise ValueError("No valid hardness scores computed")

        # Average across tasks and normalize
        combined_hardness = torch.stack(hardness_scores).mean(dim=0)
        return self._normalize_scores(combined_hardness)

    def compute_gradient_conflict(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: List[torch.nn.Parameter],
        batch_size: int = None
    ) -> torch.Tensor:
        """
        Compute PER-SAMPLE gradient alignment between tasks using PARALLEL computation.

        IMPROVED VERSION: Instead of batch-level conflict, we compute per-sample
        gradient alignment using per-sample loss gradients with GPU parallelization.

        Args:
            losses: Dictionary of task losses [batch_size] for each task
            shared_params: List of shared encoder parameters

        Returns:
            Normalized conflict scores [batch_size] where higher values indicate more conflict.
            In the acquisition function, we use (1 - conflict) to prefer aligned samples.

        Novelty:
            - Per-sample gradient computation instead of batch-level
            - GPU-parallelized using torch.autograd.grad with batched grad_outputs
            - Measures how well each sample aligns gradients across all tasks
            - Samples with high alignment help ALL tasks simultaneously
        """
        # Filter losses based on active tasks
        active_losses = {k: v for k, v in losses.items() if k in self.tasks}

        if len(active_losses) < 2:
            # No conflict if only one task
            if batch_size is None:
                first_loss = list(active_losses.values())[0]
                batch_size = len(first_loss) if first_loss.dim() > 0 else 1
            return torch.zeros(batch_size, device=self.device)

        # Infer batch size from losses
        first_loss = list(active_losses.values())[0]
        if first_loss.dim() > 0:
            batch_size = len(first_loss)
        else:
            batch_size = 1

        # Compute per-sample gradient embeddings for each task (PARALLELIZED)
        task_grad_embeddings = {}

        # Select a subset of shared parameters for efficiency (last few layers)
        selected_params = []
        param_count = 0
        for param in reversed(list(shared_params)):
            if param.requires_grad and param.numel() > 0:
                selected_params.append(param)
                param_count += 1
                if param_count >= 3:  # Use last 3 layers
                    break
        selected_params = list(reversed(selected_params))

        if not selected_params:
            return torch.zeros(batch_size, device=self.device)

        for task_name, loss in active_losses.items():
            if loss.dim() == 0:
                # Scalar loss - expand to batch
                loss = loss.unsqueeze(0).expand(batch_size)

            try:
                # PARALLEL per-sample gradient computation using torch.autograd.grad
                # Create identity matrix as grad_outputs for per-sample gradients
                # Each row selects one sample's gradient
                grad_outputs = torch.eye(batch_size, device=self.device)

                # Compute gradients for all samples in parallel
                # grad_outputs[i] = one-hot vector selecting sample i
                per_sample_grads = []

                for param in selected_params:
                    # Compute jacobian: d(loss) / d(param) for each sample
                    # Using batched gradient computation
                    param_grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=param,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                        is_grads_batched=True  # PyTorch 2.0+ parallel per-sample grads
                    )[0]

                    if param_grads is not None:
                        # param_grads shape: [batch_size, *param.shape]
                        # Flatten to [batch_size, param_numel]
                        per_sample_grads.append(param_grads.view(batch_size, -1))

                if per_sample_grads:
                    # Concatenate gradients from all selected params
                    # [batch_size, total_grad_dim]
                    task_grad_embeddings[task_name] = torch.cat(per_sample_grads, dim=1)
                else:
                    task_grad_embeddings[task_name] = torch.zeros(batch_size, 100, device=self.device)

            except (RuntimeError, TypeError) as e:
                # Fallback to sequential if parallel fails (older PyTorch)
                task_grad_embeddings[task_name] = self._compute_grads_sequential(
                    loss, selected_params, batch_size
                )

        self.model.zero_grad()

        if len(task_grad_embeddings) < 2:
            return torch.zeros(batch_size, device=self.device)

        # Ensure all tasks have same gradient dimension
        task_names = list(task_grad_embeddings.keys())
        min_dim = min(task_grad_embeddings[t].shape[1] for t in task_names)
        for t in task_names:
            task_grad_embeddings[t] = task_grad_embeddings[t][:, :min_dim]

        # Compute per-sample alignment across all task pairs (already parallel - tensor ops)
        alignment_scores = torch.zeros(batch_size, device=self.device)
        num_pairs = 0

        for i, task1 in enumerate(task_names):
            for task2 in task_names[i + 1:]:
                g1 = task_grad_embeddings[task1]  # [batch_size, grad_dim]
                g2 = task_grad_embeddings[task2]  # [batch_size, grad_dim]

                # Per-sample cosine similarity (already vectorized)
                cos_sim = F.cosine_similarity(g1, g2, dim=1)  # [batch_size]
                alignment_scores += cos_sim
                num_pairs += 1

        if num_pairs > 0:
            alignment_scores /= num_pairs

        # Convert alignment [-1, 1] to conflict [0, 1]
        # High alignment (1) → Low conflict (0)
        # Low alignment (-1) → High conflict (1)
        conflict_scores = (1.0 - alignment_scores) / 2.0

        return self._normalize_scores(conflict_scores)

    def _compute_grads_sequential(
        self,
        loss: torch.Tensor,
        selected_params: List[torch.nn.Parameter],
        batch_size: int
    ) -> torch.Tensor:
        """
        Fallback: Sequential per-sample gradient computation for older PyTorch versions.

        Args:
            loss: Per-sample loss [batch_size]
            selected_params: List of parameters to compute gradients for
            batch_size: Number of samples

        Returns:
            Per-sample gradients [batch_size, grad_dim]
        """
        grad_embeddings = []

        for i in range(batch_size):
            self.model.zero_grad()

            try:
                loss[i].backward(retain_graph=True)

                grad_list = []
                for param in selected_params:
                    if param.grad is not None and param.grad.numel() > 0:
                        grad_list.append(param.grad.view(-1).clone())

                if grad_list:
                    grad_embeddings.append(torch.cat(grad_list))
                else:
                    grad_embeddings.append(torch.zeros(100, device=self.device))

            except RuntimeError:
                grad_embeddings.append(torch.zeros(100, device=self.device))

            # Clear gradients
            for param in selected_params:
                if param.grad is not None:
                    param.grad.zero_()

        # Pad to same size and stack
        if grad_embeddings:
            max_len = max(g.shape[0] for g in grad_embeddings)
            padded = []
            for g in grad_embeddings:
                if g.shape[0] < max_len:
                    padded.append(F.pad(g, (0, max_len - g.shape[0])))
                else:
                    padded.append(g[:max_len])
            return torch.stack(padded)  # [batch_size, grad_dim]

        return torch.zeros(batch_size, 100, device=self.device)

    def compute_ctmi(
        self,
        outputs: Dict[str, torch.Tensor],
        task_types: Dict[str, str]
    ) -> torch.Tensor:
        """
        Compute Cross-Task Mutual Information (CTMI) acquisition score.

        IMPROVED VERSION: Information-theoretic CTMI based on covariance of task entropies.

        Key insight: Mutual Information I(Z_i; Z_j) between task outputs can be approximated by
        the covariance of their entropies across parameter samples (MC-dropout).

        I(Z_i; Z_j) ≈ Cov_θ[H(p_θ(y_i|x)), H(p_θ(y_j|x))]

        Selection Strategy:
        - LOW covariance = tasks are processed INDEPENDENTLY = model hasn't learned relationships
        - HIGH covariance = tasks share information = model has learned relationships

        We select samples with HIGH covariance (high MI between tasks), as these samples
        contain more inter-task information.

        Final score: We return normalized_covariance so that higher score = more informative

        Args:
            outputs: Dictionary of model outputs for each task
            task_types: Dictionary mapping task names to their types

        Returns:
            Normalized scores [batch_size] - higher means model needs to learn task relationships
        """
        active_outputs = {k: v for k, v in outputs.items() if k in self.tasks}

        if len(active_outputs) < 2:
            first_output = list(active_outputs.values())[0] if active_outputs else None
            batch_size = first_output.shape[0] if first_output is not None else 1
            return torch.zeros(batch_size, device=self.device)

        batch_size = list(active_outputs.values())[0].shape[0]

        # Check if we should use MC-dropout for parameter uncertainty
        if self.ctmi_use_mc_dropout and self.ctmi_num_samples > 1:
            return self._compute_ctmi_mc_dropout(active_outputs, task_types, batch_size)
        else:
            return self._compute_ctmi_single_pass(active_outputs, task_types, batch_size)

    def _compute_ctmi_mc_dropout(
        self,
        outputs: Dict[str, torch.Tensor],
        task_types: Dict[str, str],
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute CTMI using MC-dropout for parameter uncertainty estimation.

        This provides a more accurate MI approximation by sampling from the posterior:
        I(Z_i; Z_j) ≈ Cov_θ[H(p_θ(y_i|x)), H(p_θ(y_j|x))]
        """
        # Fallback to single pass if no cached inputs
        if not hasattr(self, '_cached_inputs') or self._cached_inputs is None:
            return self._compute_ctmi_single_pass(outputs, task_types, batch_size)

        return self._compute_ctmi_mc_dropout_with_inputs(
            self._cached_inputs, task_types, batch_size
        )

    def _compute_ctmi_mc_dropout_with_inputs(
        self,
        inputs: torch.Tensor,
        task_types: Dict[str, str],
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute CTMI using MC-dropout with actual inputs (batched to avoid OOM).

        Performs K forward passes with dropout enabled to estimate parameter uncertainty.
        Then computes covariance of task entropies across these K samples.

        Key insight: For each sample x, we get K entropy values per task.
        Cov(H_i, H_j) computed across K samples approximates MI(Z_i; Z_j).

        Args:
            inputs: Input tensor [batch_size, C, H, W]
            task_types: Dictionary mapping task names to their types
            batch_size: Number of samples

        Returns:
            CTMI scores [batch_size] - higher means high MI (more informative)
        """
        K = self.ctmi_num_samples  # Number of MC samples
        num_tasks = len(self.tasks)
        model_device = next(self.model.parameters()).device

        # Process in batches to avoid OOM
        mc_batch_size = 64  # Small batch for MC-dropout
        n_samples = inputs.shape[0]
        all_ctmi_scores = []

        # Enable dropout for MC sampling
        self.model.train()

        with torch.no_grad():
            for batch_start in range(0, n_samples, mc_batch_size):
                batch_end = min(batch_start + mc_batch_size, n_samples)
                batch_inputs = inputs[batch_start:batch_end].to(model_device)
                current_batch_size = batch_end - batch_start

                # Collect entropies for this batch: [K, num_tasks, current_batch_size]
                batch_entropies = []

                for k in range(K):
                    # Forward pass with dropout
                    outputs = self.model(batch_inputs)

                    # Compute per-task entropy for this pass
                    pass_entropies = []
                    for task_name in self.tasks:
                        if task_name in outputs:
                            task_type = task_types.get(task_name, 'classification')
                            entropy = self._compute_task_entropy(outputs[task_name], task_type)
                            if entropy is not None:
                                pass_entropies.append(entropy.cpu())  # Move to CPU immediately

                    if len(pass_entropies) == num_tasks:
                        batch_entropies.append(torch.stack(pass_entropies, dim=0))

                    # Clean up
                    del outputs

                # Clean up batch inputs
                del batch_inputs
                if model_device.type == 'cuda':
                    torch.cuda.empty_cache()

                if len(batch_entropies) < 2:
                    all_ctmi_scores.append(torch.zeros(current_batch_size))
                    continue

                # Stack all passes: [K, num_tasks, current_batch_size]
                entropy_samples = torch.stack(batch_entropies, dim=0)

                # Compute CTMI scores for this batch
                batch_ctmi = torch.zeros(current_batch_size)
                for i in range(current_batch_size):
                    sample_entropies = entropy_samples[:, :, i]  # [K, num_tasks]

                    if K >= 2:
                        try:
                            cov_matrix = torch.cov(sample_entropies.T)  # [num_tasks, num_tasks]
                            mi_sum = 0.0
                            num_pairs = 0
                            for t1 in range(num_tasks):
                                for t2 in range(t1 + 1, num_tasks):
                                    mi_sum += cov_matrix[t1, t2].abs().item()
                                    num_pairs += 1
                            batch_ctmi[i] = mi_sum / num_pairs if num_pairs > 0 else 0.0
                        except RuntimeError:
                            batch_ctmi[i] = 0.0

                all_ctmi_scores.append(batch_ctmi)

        self.model.eval()

        # Concatenate all batch results
        ctmi_scores = torch.cat(all_ctmi_scores, dim=0).to(self.device)

        # Normalize CTMI scores
        ctmi_normalized = self._normalize_scores(ctmi_scores)

        # Apply CTMI mode: positive (high cov = high score) or negative (low cov = high score)
        if self.ctmi_mode == 'covariance_positive':
            # High covariance = high CTMI (tasks share information)
            return ctmi_normalized
        else:  # covariance_negative (default)
            # Low covariance = high CTMI (tasks are independent, need learning)
            return 1.0 - ctmi_normalized

    def set_cached_inputs(self, inputs: torch.Tensor):
        """Cache inputs for MC-dropout CTMI computation."""
        self._cached_inputs = inputs

    def clear_cached_inputs(self):
        """Clear cached inputs to free memory."""
        self._cached_inputs = None

    def _compute_ctmi_single_pass(
        self,
        outputs: Dict[str, torch.Tensor],
        task_types: Dict[str, str],
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute CTMI using single forward pass outputs.

        Uses covariance-based MI approximation:
        - Compute entropy for each task
        - Compute pairwise covariance of entropies across samples
        - Low covariance = independent tasks = informative for learning relationships
        """
        # Extract per-task entropies
        task_entropies = []
        task_names = []

        for task_name, output in outputs.items():
            task_type = task_types.get(task_name, 'classification')
            entropy = self._compute_task_entropy(output, task_type)
            if entropy is not None:
                task_entropies.append(entropy)
                task_names.append(task_name)

        if len(task_entropies) < 2:
            return torch.zeros(batch_size, device=self.device)

        # Stack entropies: [num_tasks, batch_size]
        entropy_stack = torch.stack(task_entropies, dim=0).to(self.device)  # Ensure on correct device
        num_tasks = entropy_stack.shape[0]

        # Method 1: Pairwise covariance-based MI approximation
        # For each sample, compute the covariance of task entropies
        # This captures how much tasks "move together" in their uncertainty

        # Compute mean-centered entropies
        entropy_mean = entropy_stack.mean(dim=0, keepdim=True)  # [1, batch]
        entropy_centered = entropy_stack - entropy_mean  # [num_tasks, batch]

        # Compute pairwise covariance for each sample
        # Cov(H_i, H_j) = E[(H_i - μ_i)(H_j - μ_j)]
        # For a single sample, we approximate using the product of centered values
        pairwise_cov = torch.zeros(batch_size, device=self.device)
        num_pairs = 0

        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                # Product of centered entropies (approximates covariance contribution)
                cov_ij = entropy_centered[i] * entropy_centered[j]
                pairwise_cov += cov_ij.abs()  # Use absolute value for magnitude
                num_pairs += 1

        if num_pairs > 0:
            pairwise_cov /= num_pairs

        # Method 2: Entropy variance as additional signal
        # High variance = tasks have different uncertainty patterns
        entropy_variance = entropy_stack.var(dim=0)

        # Method 3: Confidence consistency
        task_confidences = []
        for task_name, output in outputs.items():
            task_type = task_types.get(task_name, 'classification')
            conf = self._compute_task_confidence(output, task_type)
            if conf is not None:
                task_confidences.append(conf)

        if len(task_confidences) >= 2:
            conf_stack = torch.stack(task_confidences, dim=0).to(self.device)  # Ensure on correct device
            conf_variance = conf_stack.var(dim=0)
        else:
            conf_variance = torch.zeros(batch_size, device=self.device)

        # Combine components:
        # - High covariance = tasks share information = HIGH acquisition score
        # - High variance = tasks disagree = HIGH acquisition score

        # Normalize covariance to [0, 1] range
        cov_normalized = self._normalize_scores(pairwise_cov)

        # Apply CTMI mode: positive (high cov = high score) or negative (low cov = high score)
        if self.ctmi_mode == 'covariance_positive':
            # High covariance = high CTMI (tasks share information)
            cov_score = cov_normalized
        else:  # covariance_negative (default)
            # Low covariance = high CTMI (tasks are independent, need learning)
            cov_score = 1.0 - cov_normalized

        # Normalize variance components
        entropy_var_normalized = self._normalize_scores(entropy_variance)
        conf_var_normalized = self._normalize_scores(conf_variance)

        # Combined score:
        # - Covariance-based: 0.5 (main MI approximation)
        # - Entropy variance: 0.3 (uncertainty disagreement)
        # - Confidence variance: 0.2 (confidence disagreement)
        ctmi_score = 0.5 * cov_score + 0.3 * entropy_var_normalized + 0.2 * conf_var_normalized

        return self._normalize_scores(ctmi_score)

    def _compute_task_entropy(
        self,
        output: torch.Tensor,
        task_type: str
    ) -> Optional[torch.Tensor]:
        """Compute entropy for a single task output."""
        if task_type == 'classification':
            probs = F.softmax(output, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            return entropy

        elif task_type == 'segmentation':
            probs = F.softmax(output, dim=1)
            pixel_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            entropy = pixel_entropy.mean(dim=(1, 2))
            return entropy

        elif task_type == 'detection':
            num_classes = output.shape[2] - 4
            cls_scores = output[:, :, :num_classes]
            probs = F.softmax(cls_scores, dim=-1)
            box_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            entropy = box_entropy.mean(dim=1)
            return entropy

        elif task_type in ['depth', 'normal']:
            # For regression tasks, use variance as uncertainty proxy
            if output.dim() == 4:  # [batch, channels, H, W]
                variance = output.var(dim=(2, 3))  # [batch, channels]
                uncertainty = variance.mean(dim=1)  # [batch]
            elif output.dim() == 2:  # [batch, features]
                uncertainty = output.var(dim=1)  # [batch]
            else:
                uncertainty = output.view(output.shape[0], -1).var(dim=1)
            return uncertainty

        elif task_type in ['instance', 'instance_segmentation']:
            # Instance segmentation: treat like segmentation (pixel-wise classification)
            if output.dim() == 4:  # [batch, num_classes, H, W] - segmentation format
                probs = F.softmax(output, dim=1)
                pixel_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                entropy = pixel_entropy.mean(dim=(1, 2))
                return entropy
            else:  # Detection-like format [batch, num_boxes, num_classes + 4]
                num_classes = output.shape[2] - 4
                cls_scores = output[:, :, :num_classes]
                probs = F.softmax(cls_scores, dim=-1)
                box_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                entropy = box_entropy.mean(dim=1)
                return entropy

        return None

    def _compute_task_confidence(
        self,
        output: torch.Tensor,
        task_type: str
    ) -> Optional[torch.Tensor]:
        """Compute confidence (max probability) for a single task output."""
        if task_type == 'classification':
            probs = F.softmax(output, dim=-1)
            confidence = probs.max(dim=-1)[0]
            return confidence

        elif task_type == 'segmentation':
            probs = F.softmax(output, dim=1)
            confidence = probs.max(dim=1)[0].mean(dim=(1, 2))
            return confidence

        elif task_type == 'detection':
            num_classes = output.shape[2] - 4
            cls_scores = output[:, :, :num_classes]
            probs = F.softmax(cls_scores, dim=-1)
            confidence = probs.max(dim=-1)[0].mean(dim=1)
            return confidence

        elif task_type in ['depth', 'normal']:
            # For regression tasks, use inverse variance as confidence proxy
            # Lower variance = higher confidence
            if output.dim() == 4:  # [batch, channels, H, W]
                variance = output.var(dim=(2, 3))  # [batch, channels]
                mean_variance = variance.mean(dim=1)  # [batch]
            elif output.dim() == 2:  # [batch, features]
                mean_variance = output.var(dim=1)  # [batch]
            else:
                mean_variance = output.view(output.shape[0], -1).var(dim=1)
            # Convert variance to confidence: 1 / (1 + variance)
            confidence = 1.0 / (1.0 + mean_variance)
            return confidence

        elif task_type in ['instance', 'instance_segmentation']:
            # Instance segmentation: treat like segmentation (pixel-wise)
            if output.dim() == 4:  # [batch, num_classes, H, W] - segmentation format
                probs = F.softmax(output, dim=1)
                confidence = probs.max(dim=1)[0].mean(dim=(1, 2))
                return confidence
            else:  # Detection-like format
                num_classes = output.shape[2] - 4
                cls_scores = output[:, :, :num_classes]
                probs = F.softmax(cls_scores, dim=-1)
                confidence = probs.max(dim=-1)[0].mean(dim=1)
                return confidence

        return None

    def compute_synergy(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: List[torch.nn.Parameter],
        batch_size: int = None
    ) -> torch.Tensor:
        """
        Compute Task Synergy Score.

        Synergy measures whether learning a sample in MTL setting provides more
        benefit than learning it in separate single-task settings.

        Mathematical foundation:
            Synergy(x) = ||g_MTL||² - Σₜ wₜ ||gₜ||²

        Where:
            - g_MTL = gradient from MTL loss (all tasks combined)
            - gₜ = gradient from single task t
            - wₜ = task weight

        High synergy means:
            - The combined gradient is stronger than sum of individual gradients
            - Tasks are providing complementary information
            - This sample promotes positive transfer

        Args:
            losses: Dictionary of per-sample task losses [batch_size]
            shared_params: List of shared encoder parameters
            batch_size: Number of samples

        Returns:
            Normalized synergy scores [batch_size] - higher means more synergy
        """
        active_losses = {k: v for k, v in losses.items() if k in self.tasks}

        if len(active_losses) < 2:
            if batch_size is None:
                first_loss = list(active_losses.values())[0]
                batch_size = len(first_loss) if first_loss.dim() > 0 else 1
            return torch.zeros(batch_size, device=self.device)

        # Infer batch size
        first_loss = list(active_losses.values())[0]
        if first_loss.dim() > 0:
            batch_size = len(first_loss)
        else:
            batch_size = 1

        # Select parameters for gradient computation (last few layers for efficiency)
        selected_params = []
        param_count = 0
        for param in reversed(list(shared_params)):
            if param.requires_grad and param.numel() > 0:
                selected_params.append(param)
                param_count += 1
                if param_count >= 3:
                    break
        selected_params = list(reversed(selected_params))

        if not selected_params:
            return torch.zeros(batch_size, device=self.device)

        # Compute gradient norms for each task
        task_grad_norms = {}
        for task_name, loss in active_losses.items():
            if loss.dim() == 0:
                loss = loss.unsqueeze(0).expand(batch_size)

            try:
                # Compute per-sample gradients
                grad_outputs = torch.eye(batch_size, device=self.device)
                per_sample_grad_norms = torch.zeros(batch_size, device=self.device)

                for param in selected_params:
                    param_grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=param,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                        is_grads_batched=True
                    )[0]

                    if param_grads is not None:
                        # [batch_size, *param.shape] -> [batch_size, -1] -> norm per sample
                        per_sample_grad_norms += param_grads.view(batch_size, -1).norm(dim=1) ** 2

                task_grad_norms[task_name] = per_sample_grad_norms

            except (RuntimeError, TypeError):
                # Fallback for older PyTorch
                task_grad_norms[task_name] = torch.zeros(batch_size, device=self.device)

        self.model.zero_grad()

        # Compute MTL gradient norm (sum of all task losses)
        mtl_loss = sum(active_losses.values())
        if mtl_loss.dim() == 0:
            mtl_loss = mtl_loss.unsqueeze(0).expand(batch_size)

        try:
            grad_outputs = torch.eye(batch_size, device=self.device)
            mtl_grad_norm = torch.zeros(batch_size, device=self.device)

            for param in selected_params:
                param_grads = torch.autograd.grad(
                    outputs=mtl_loss,
                    inputs=param,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                    is_grads_batched=True
                )[0]

                if param_grads is not None:
                    mtl_grad_norm += param_grads.view(batch_size, -1).norm(dim=1) ** 2

        except (RuntimeError, TypeError):
            mtl_grad_norm = torch.zeros(batch_size, device=self.device)

        self.model.zero_grad()

        # Synergy = ||g_MTL||² - Σ||gₜ||²
        # Positive synergy = MTL gradient is stronger than sum of single-task gradients
        single_task_sum = sum(task_grad_norms.values())
        synergy = mtl_grad_norm - single_task_sum

        # Shift to positive range and normalize
        # Synergy can be negative (redundant) or positive (complementary)
        synergy_shifted = synergy - synergy.min() + 1e-8

        return self._normalize_scores(synergy_shifted)

    def compute_task_consistency(
        self,
        outputs: Dict[str, torch.Tensor],
        task_types: Dict[str, str]
    ) -> torch.Tensor:
        """
        Compute CONFIDENCE CONSISTENCY across tasks.

        IMPROVED VERSION: Instead of comparing output distributions (which is
        meaningless across different task types), we measure whether the model's
        CONFIDENCE is consistent across all tasks.

        Intuition:
        - If the model is confident in ALL tasks → easy sample (low priority)
        - If the model is uncertain in ALL tasks → hard but consistent sample
        - If confident in some, uncertain in others → INCONSISTENT (high priority!)

        The inconsistency score captures samples where tasks disagree on difficulty,
        which are the most informative for multi-task learning.

        Args:
            outputs: Dictionary of model outputs for each task
            task_types: Dictionary mapping task names to their types

        Returns:
            Normalized inconsistency scores [batch_size] - higher means more inconsistent
        """
        # Filter outputs based on active tasks
        active_outputs = {k: v for k, v in outputs.items() if k in self.tasks}

        if not active_outputs:
            return torch.zeros(1, device=self.device)

        # Compute per-task confidence (max probability)
        task_confidences = []

        for task_name, output in active_outputs.items():
            task_type = task_types.get(task_name, 'classification')

            if task_type == 'classification':
                # Classification: max probability
                probs = F.softmax(output, dim=-1)
                confidence = probs.max(dim=-1)[0]  # [batch_size]

            elif task_type == 'segmentation':
                # Segmentation: average max probability over pixels
                probs = F.softmax(output, dim=1)  # [batch, classes, H, W]
                pixel_confidence = probs.max(dim=1)[0]  # [batch, H, W]
                confidence = pixel_confidence.mean(dim=(1, 2))  # [batch_size]

            elif task_type == 'detection':
                # Detection: average max probability over boxes
                num_classes = output.shape[2] - 4
                cls_scores = output[:, :, :num_classes]  # [batch, boxes, classes]
                probs = F.softmax(cls_scores, dim=-1)
                box_confidence = probs.max(dim=-1)[0]  # [batch, boxes]
                confidence = box_confidence.mean(dim=1)  # [batch_size]

            else:
                # Default: uniform confidence
                batch_size = output.shape[0] if output.dim() > 0 else 1
                confidence = torch.ones(batch_size, device=self.device) * 0.5

            task_confidences.append(confidence)

        if len(task_confidences) < 2:
            # Need at least 2 tasks for consistency measure
            return torch.zeros_like(task_confidences[0])

        # Stack confidences: [num_tasks, batch_size]
        conf_stack = torch.stack(task_confidences)

        # Compute INCONSISTENCY as variance of confidences across tasks
        # High variance = tasks disagree on difficulty = inconsistent = informative!
        inconsistency = conf_stack.std(dim=0)  # [batch_size]

        # Additional: Also consider samples where one task is very confident
        # but another is very uncertain (max-min gap)
        conf_range = conf_stack.max(dim=0)[0] - conf_stack.min(dim=0)[0]  # [batch_size]

        # Combine std and range (both indicate inconsistency)
        combined_inconsistency = inconsistency + 0.5 * conf_range

        return self._normalize_scores(combined_inconsistency)

    def _compute_pairwise_consistency(
        self,
        output1: torch.Tensor,
        output2: torch.Tensor,
        type1: str,
        type2: str
    ) -> torch.Tensor:
        """Compute consistency between two task outputs."""

        if self.consistency_mode == 'kl':
            if type1 == 'classification' and type2 == 'classification':
                # KL divergence between classification outputs
                probs1 = F.softmax(output1, dim=-1)
                probs2 = F.softmax(output2, dim=-1)
                kl_div = F.kl_div(
                    probs1.log(), probs2, reduction='none'
                ).sum(dim=-1)
                return kl_div

            elif type1 == 'segmentation' and type2 == 'segmentation':
                # KL divergence between segmentation outputs
                probs1 = F.softmax(output1, dim=1)
                probs2 = F.softmax(output2, dim=1)
                kl_div = F.kl_div(
                    probs1.log(), probs2, reduction='none'
                ).sum(dim=1).mean(dim=(1, 2))  # Average over spatial dimensions
                return kl_div

            else:
                # Cross-task consistency (e.g., classification vs segmentation vs detection)
                # Convert all to classification-like format

                # Process output1
                if type1 == 'segmentation':
                    # Global average pooling + classification head
                    if output1.numel() == 0:
                        # Empty tensor - return zeros
                        batch_size = output1.shape[0] if output1.shape[0] > 0 else 1
                        num_classes = output1.shape[1] if len(output1.shape) > 1 else 80
                        return torch.zeros(batch_size, device=output1.device)
                    seg_pooled = F.adaptive_avg_pool2d(output1, (1, 1))
                    # Remove spatial dimensions only (keep batch and channel dims)
                    if seg_pooled.dim() == 4:
                        seg_pooled = seg_pooled.squeeze(-1).squeeze(-1)  # [batch, channels]
                    elif seg_pooled.dim() == 3:
                        seg_pooled = seg_pooled.mean(dim=0)  # Average over channels if needed
                    # Ensure batch dimension exists
                    if seg_pooled.dim() == 1:
                        batch_size = output1.shape[0]
                        seg_pooled = seg_pooled.unsqueeze(0).expand(batch_size, -1)
                    # Handle empty tensor after pooling
                    if seg_pooled.numel() == 0:
                        batch_size = output1.shape[0] if output1.shape[0] > 0 else 1
                        num_classes = output1.shape[1] if len(output1.shape) > 1 else 80
                        return torch.zeros(batch_size, device=output1.device)
                    probs1 = F.softmax(seg_pooled, dim=-1)
                elif type1 == 'detection':
                    # Detection: [batch, num_boxes, num_classes + 4]
                    # Extract class scores and average over boxes
                    num_classes = output1.shape[2] - 4
                    cls_scores = output1[:, :, :num_classes]  # [batch, num_boxes, num_classes]
                    cls_scores = cls_scores.mean(dim=1)  # [batch, num_classes] - average over boxes
                    probs1 = F.softmax(cls_scores, dim=-1)
                else:  # classification
                    probs1 = F.softmax(output1, dim=-1)

                # Process output2
                if type2 == 'segmentation':
                    if output2.numel() == 0:
                        # Empty tensor - return zeros
                        batch_size = output2.shape[0] if output2.shape[0] > 0 else 1
                        num_classes = output2.shape[1] if len(output2.shape) > 1 else 80
                        return torch.zeros(batch_size, device=output2.device)
                    seg_pooled = F.adaptive_avg_pool2d(output2, (1, 1))
                    # Remove spatial dimensions only (keep batch and channel dims)
                    if seg_pooled.dim() == 4:
                        seg_pooled = seg_pooled.squeeze(-1).squeeze(-1)  # [batch, channels]
                    elif seg_pooled.dim() == 3:
                        seg_pooled = seg_pooled.mean(dim=0)  # Average over channels if needed
                    # Ensure batch dimension exists
                    if seg_pooled.dim() == 1:
                        batch_size = output2.shape[0]
                        seg_pooled = seg_pooled.unsqueeze(0).expand(batch_size, -1)
                    # Handle empty tensor after pooling
                    if seg_pooled.numel() == 0:
                        batch_size = output2.shape[0] if output2.shape[0] > 0 else 1
                        num_classes = output2.shape[1] if len(output2.shape) > 1 else 80
                        return torch.zeros(batch_size, device=output2.device)
                    probs2 = F.softmax(seg_pooled, dim=-1)
                elif type2 == 'detection':
                    # Detection: [batch, num_boxes, num_classes + 4]
                    num_classes = output2.shape[2] - 4
                    cls_scores = output2[:, :, :num_classes]  # [batch, num_boxes, num_classes]
                    cls_scores = cls_scores.mean(dim=1)  # [batch, num_classes] - average over boxes
                    probs2 = F.softmax(cls_scores, dim=-1)
                else:  # classification
                    probs2 = F.softmax(output2, dim=-1)

                kl_div = F.kl_div(probs1.log(), probs2, reduction='none').sum(dim=-1)
                return kl_div

        elif self.consistency_mode == 'mse':
            # Mean squared error between outputs
            if output1.shape != output2.shape:
                # Handle different output shapes
                # Convert all to classification-like format for comparison

                # Process output1
                if type1 == 'segmentation':
                    if output1.numel() == 0:
                        batch_size = output1.shape[0] if output1.shape[0] > 0 else 1
                        return torch.zeros(batch_size, device=output1.device)
                    seg_pooled = F.adaptive_avg_pool2d(output1, (1, 1))
                    # Remove spatial dimensions only (keep batch and channel dims)
                    if seg_pooled.dim() == 4:
                        seg_pooled = seg_pooled.squeeze(-1).squeeze(-1)  # [batch, channels]
                    elif seg_pooled.dim() == 3:
                        seg_pooled = seg_pooled.mean(dim=0)  # Average over channels if needed
                    # Ensure batch dimension exists
                    if seg_pooled.dim() == 1:
                        batch_size = output1.shape[0]
                        seg_pooled = seg_pooled.unsqueeze(0).expand(batch_size, -1)
                    if seg_pooled.numel() == 0:
                        batch_size = output1.shape[0] if output1.shape[0] > 0 else 1
                        return torch.zeros(batch_size, device=output1.device)
                    out1_flat = seg_pooled
                elif type1 == 'detection':
                    # Average detection outputs over boxes dimension
                    num_classes = output1.shape[2] - 4
                    cls_scores = output1[:, :, :num_classes].mean(dim=1)  # [batch, num_classes]
                    out1_flat = cls_scores
                else:  # classification
                    out1_flat = output1

                # Process output2
                if type2 == 'segmentation':
                    if output2.numel() == 0:
                        batch_size = output2.shape[0] if output2.shape[0] > 0 else 1
                        return torch.zeros(batch_size, device=output2.device)
                    seg_pooled = F.adaptive_avg_pool2d(output2, (1, 1))
                    # Remove spatial dimensions only (keep batch and channel dims)
                    if seg_pooled.dim() == 4:
                        seg_pooled = seg_pooled.squeeze(-1).squeeze(-1)  # [batch, channels]
                    elif seg_pooled.dim() == 3:
                        seg_pooled = seg_pooled.mean(dim=0)  # Average over channels if needed
                    # Ensure batch dimension exists
                    if seg_pooled.dim() == 1:
                        batch_size = output2.shape[0]
                        seg_pooled = seg_pooled.unsqueeze(0).expand(batch_size, -1)
                    if seg_pooled.numel() == 0:
                        batch_size = output2.shape[0] if output2.shape[0] > 0 else 1
                        return torch.zeros(batch_size, device=output2.device)
                    out2_flat = seg_pooled
                elif type2 == 'detection':
                    num_classes = output2.shape[2] - 4
                    cls_scores = output2[:, :, :num_classes].mean(dim=1)  # [batch, num_classes]
                    out2_flat = cls_scores
                else:  # classification
                    out2_flat = output2

                # Ensure same shape for comparison
                if out1_flat.shape != out2_flat.shape:
                    min_dim = min(out1_flat.shape[-1], out2_flat.shape[-1])
                    out1_flat = out1_flat[..., :min_dim]
                    out2_flat = out2_flat[..., :min_dim]

                mse = F.mse_loss(out1_flat, out2_flat, reduction='none').sum(dim=-1)
            else:
                mse = F.mse_loss(output1, output2, reduction='none').sum(dim=-1)
            return mse

        # Handle empty tensor case
        if output1.numel() == 0:
            batch_size = output1.shape[0] if len(output1.shape) > 0 and output1.shape[0] > 0 else 1
            return torch.zeros(batch_size, device=output1.device)
        return torch.zeros(len(output1), device=self.device)

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

    def _jensen_shannon_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jensen-Shannon divergence between two probability distributions.

        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        where M = 0.5 * (P + Q)

        Args:
            p: Probability distribution [batch_size, num_classes] or [batch_size, ...]
            q: Probability distribution [batch_size, num_classes] or [batch_size, ...]

        Returns:
            JS divergence [batch_size]
        """
        # Ensure probabilities sum to 1
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
        q = q / (q.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute mixture M = 0.5 * (P + Q)
        m = 0.5 * (p + q)

        # Compute KL divergences
        kl_pm = F.kl_div(m.log(), p, reduction='none').sum(dim=-1)
        kl_qm = F.kl_div(m.log(), q, reduction='none').sum(dim=-1)

        # JS divergence
        js = 0.5 * kl_pm + 0.5 * kl_qm

        return js

    def _compute_entropy_from_output(
        self,
        output: torch.Tensor,
        task_type: str
    ) -> torch.Tensor:
        """
        Compute entropy from model output for a single task.

        Args:
            output: Model output tensor
            task_type: Type of task ('classification', 'segmentation', 'detection')

        Returns:
            Entropy scores [batch_size]
        """
        if task_type == 'classification':
            probs = F.softmax(output, dim=-1)  # [batch, num_classes]
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [batch]
        elif task_type == 'segmentation':
            probs = F.softmax(output, dim=1)  # [batch, classes, H, W]
            pixel_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [batch, H, W]
            entropy = pixel_entropy.mean(dim=(1, 2))  # [batch]
        elif task_type == 'detection':
            num_classes = output.shape[2] - 4
            cls_scores = output[:, :, :num_classes]  # [batch, boxes, classes]
            probs = F.softmax(cls_scores, dim=-1)  # [batch, boxes, classes]
            box_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [batch, boxes]
            entropy = box_entropy.mean(dim=1)  # [batch]
        else:
            # Default: treat as classification
            probs = F.softmax(output, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        return entropy

    def _get_average_predictive_distribution(
        self,
        outputs_list: List[torch.Tensor],
        task_type: str
    ) -> torch.Tensor:
        """
        Compute average predictive distribution across MC samples.

        Args:
            outputs_list: List of K output tensors from MC samples
            task_type: Type of task

        Returns:
            Average probability distribution [batch_size, num_classes] or [batch_size, ...]
        """
        # Convert all outputs to probabilities
        probs_list = []
        for output in outputs_list:
            if task_type == 'classification':
                probs = F.softmax(output, dim=-1)  # [batch, num_classes]
            elif task_type == 'segmentation':
                probs = F.softmax(output, dim=1)  # [batch, classes, H, W]
                # Average over spatial dimensions for comparison
                probs = probs.mean(dim=(2, 3))  # [batch, classes]
            elif task_type == 'detection':
                num_classes = output.shape[2] - 4
                cls_scores = output[:, :, :num_classes]  # [batch, boxes, classes]
                probs = F.softmax(cls_scores, dim=-1)  # [batch, boxes, classes]
                probs = probs.mean(dim=1)  # [batch, classes] - average over boxes
            else:
                probs = F.softmax(output, dim=-1)

            probs_list.append(probs)

        # Average across MC samples
        avg_probs = torch.stack(probs_list).mean(dim=0)  # [K, batch, ...] -> [batch, ...]
        return avg_probs

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: Raw scores to normalize

        Returns:
            Normalized scores in [0, 1] range
        """
        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score < 1e-8:
            # All scores are the same
            return torch.zeros_like(scores)

        normalized = (scores - min_score) / (max_score - min_score)
        return torch.clamp(normalized, 0.0, 1.0)

    def compute_diversity(
        self,
        inputs: torch.Tensor,
        selected_indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute diversity score based on distance to other samples.

        Args:
            inputs: Input samples [batch_size, ...]
            selected_indices: Indices of already selected samples (optional)

        Returns:
            Normalized diversity scores [batch_size] - higher scores indicate more diverse samples
        """
        batch_size = len(inputs)

        if selected_indices is not None and len(selected_indices) > 0:
            # Compute distance to already selected samples
            selected_inputs = inputs[selected_indices]

            if self.diversity_mode == 'euclidean':
                # Flatten inputs for distance computation
                inputs_flat = inputs.view(batch_size, -1)
                selected_flat = selected_inputs.view(len(selected_indices), -1)
                distances = torch.cdist(inputs_flat, selected_flat, p=2)  # [batch_size, num_selected]
                min_distances = distances.min(dim=1)[0]  # [batch_size] - distance to closest selected sample
            elif self.diversity_mode == 'cosine':
                # Normalize for cosine distance
                inputs_flat = inputs.view(batch_size, -1)
                selected_flat = selected_inputs.view(len(selected_indices), -1)
                inputs_norm = F.normalize(inputs_flat, p=2, dim=1)
                selected_norm = F.normalize(selected_flat, p=2, dim=1)
                similarities = torch.mm(inputs_norm, selected_norm.t())  # [batch_size, num_selected]
                min_distances = 1.0 - similarities.max(dim=1)[0]  # [batch_size] - cosine distance
        else:
            # No selected samples: compute diversity within batch
            # Use pairwise distances within the batch
            if batch_size < 2:
                return torch.ones(batch_size, device=inputs.device)

            inputs_flat = inputs.view(batch_size, -1)

            if self.diversity_mode == 'euclidean':
                # Compute pairwise distances
                distances = torch.cdist(inputs_flat, inputs_flat, p=2)  # [batch_size, batch_size]
                # For each sample, compute average distance to other samples
                # Exclude self-distance (diagonal)
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=inputs.device)
                distances_masked = distances[mask].view(batch_size, batch_size - 1)
                min_distances = distances_masked.mean(dim=1)  # [batch_size] - average distance
            elif self.diversity_mode == 'cosine':
                inputs_norm = F.normalize(inputs_flat, p=2, dim=1)
                similarities = torch.mm(inputs_norm, inputs_norm.t())  # [batch_size, batch_size]
                # Exclude self-similarity (diagonal)
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=inputs.device)
                similarities_masked = similarities[mask].view(batch_size, batch_size - 1)
                min_distances = 1.0 - similarities_masked.mean(dim=1)  # [batch_size] - average cosine distance

        return self._normalize_scores(min_distances)

    def compute_acquisition_scores(
        self,
        outputs: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor],
        shared_params: List[torch.nn.Parameter],
        task_types: Dict[str, str],
        inputs: torch.Tensor = None,
        selected_indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the complete acquisition scores.

        Args:
            outputs: Dictionary of model outputs for each task
            losses: Dictionary of task losses
            shared_params: List of shared encoder parameters
            task_types: Dictionary mapping task names to their types
            inputs: Input samples for diversity computation (optional)
            selected_indices: Indices of already selected samples for diversity (optional)

        Returns:
            Acquisition scores [batch_size] - higher scores indicate more informative samples
        """
        # Compute individual components based on active criteria
        if not outputs:
            return torch.zeros(0, device=self.device)
        first_output = list(outputs.values())[0]
        if first_output.numel() == 0 or len(first_output) == 0:
            return torch.zeros(0, device=self.device)
        batch_size = len(first_output)

        # Compute hardness, conflict, consistency components
        # Step 1: Compute normalized weights to ensure scale consistency
        active_criteria_weights = []
        if 'hardness' in self.criteria:
            active_criteria_weights.append(self.alpha)
        if 'conflict' in self.criteria:
            active_criteria_weights.append(self.beta)
        if 'consistency' in self.criteria:
            active_criteria_weights.append(self.gamma)

        total_weight = sum(active_criteria_weights) if active_criteria_weights else 1.0

        # Normalize weights: α' = α / (α + β + γ), etc.
        alpha_norm = (self.alpha / total_weight) if 'hardness' in self.criteria else 0.0
        beta_norm = (self.beta / total_weight) if 'conflict' in self.criteria else 0.0
        gamma_norm = (self.gamma / total_weight) if 'consistency' in self.criteria else 0.0

        hardness_conflict_consistency_scores = torch.zeros(batch_size, device=self.device)

        if 'hardness' in self.criteria:
            hardness = self.compute_hardness(outputs, task_types)
            hardness_conflict_consistency_scores += alpha_norm * hardness

        if 'conflict' in self.criteria:
            gradient_conflict = self.compute_gradient_conflict(losses, shared_params, batch_size)
            # Prefer low conflict (samples with low conflict help all tasks)
            # gradient_conflict is normalized to [0, 1] range (higher = more conflict)
            # Use (1 - conflict) to prefer low conflict samples
            hardness_conflict_consistency_scores += beta_norm * (1.0 - gradient_conflict)

        if 'consistency' in self.criteria:
            # Third component depends on mode: consistency, ctmi, or synergy
            if self.third_component_mode == 'ctmi':
                # CTMI: Cross-Task Mutual Information
                # Higher score = higher CTMI = tasks disagree = informative for learning relationships
                third_component_score = self.compute_ctmi(outputs, task_types)
            elif self.third_component_mode == 'synergy':
                # Synergy: MTL gradient benefit vs single-task gradients
                # Higher score = more synergy = positive transfer opportunity
                third_component_score = self.compute_synergy(losses, shared_params, batch_size)
            else:
                # Default: Confidence Consistency
                # Higher inconsistency = tasks disagree on confidence = informative
                third_component_score = self.compute_task_consistency(outputs, task_types)

            hardness_conflict_consistency_scores += gamma_norm * third_component_score

        # Step 2: Combine with diversity using multiplicative approach
        # S(x) = base_score · (1 + δ·diversity)
        # This makes diversity amplify the base score rather than adding independently
        if inputs is not None and self.delta > 0:
            diversity = self.compute_diversity(inputs, selected_indices)
            # Diversity amplifies base score: higher diversity → higher final score
            acquisition_scores = hardness_conflict_consistency_scores * (1.0 + self.delta * diversity)
        else:
            acquisition_scores = hardness_conflict_consistency_scores

        return acquisition_scores

    def select_samples(
        self,
        outputs: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor],
        shared_params: List[torch.nn.Parameter],
        task_types: Dict[str, str],
        num_samples: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select the most informative samples based on acquisition scores.

        Args:
            outputs: Dictionary of model outputs for each task
            losses: Dictionary of task losses
            shared_params: List of shared encoder parameters
            task_types: Dictionary mapping task names to their types
            num_samples: Number of samples to select (if None, return all ranked)

        Returns:
            Tuple of (selected_indices, acquisition_scores)
        """
        acquisition_scores = self.compute_acquisition_scores(
            outputs, losses, shared_params, task_types
        )

        # Sort by acquisition score (descending)
        sorted_indices = torch.argsort(acquisition_scores, descending=True)

        if num_samples is not None:
            selected_indices = sorted_indices[:num_samples]
        else:
            selected_indices = sorted_indices

        return selected_indices, acquisition_scores

    def select_samples_two_stage(
        self,
        outputs: Dict[str, torch.Tensor],
        task_types: Dict[str, str],
        num_samples: int,
        pool_factor: int = 3,
        inputs: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [Key Improvement] 2-Stage Selection Strategy.

        Addresses previous issues:
        1. Diversity dilution: Separate into 2 stages instead of multiplicative approach
        2. Conflict paradox: Exclude conflict (PCGrad handles it)
        3. Computational efficiency: Expensive operations only on candidate pool

        Process:
            Stage 1 (Informativeness): Select top candidates using Hardness + CTMI
            Stage 2 (Representativeness): Ensure diversity using K-Center Greedy

        Args:
            outputs: Dictionary of model outputs for each task
            task_types: Dictionary mapping task names to their types
            num_samples: Number of samples to select
            pool_factor: Candidate pool multiplier (how many times final count to select in stage 1)
            inputs: Input features for diversity computation

        Returns:
            Tuple of (selected_indices, acquisition_scores)
        """
        # --- Stage 1: Informativeness Filtering ---
        # Hardness (information magnitude)
        hardness_score = self.compute_hardness(outputs, task_types).to(self.device)

        # CTMI (information structure - inter-task relationships)
        if self.third_component_mode == 'ctmi':
            ctmi_score = self.compute_ctmi(outputs, task_types).to(self.device)
        else:
            ctmi_score = torch.zeros_like(hardness_score, device=self.device)

        # Informativeness Score (Conflict excluded!)
        # Samples with high conflict can still be important if CTMI is high
        info_scores = self._combine_info_scores(hardness_score, ctmi_score)

        # Select top N * pool_factor candidates
        num_candidates = min(len(info_scores), num_samples * pool_factor)
        candidate_values, candidate_indices = torch.topk(info_scores, k=num_candidates)

        # --- Stage 2: Diversity Selection (Representativeness) ---
        if inputs is None or num_candidates <= num_samples:
            # If inputs unavailable or insufficient candidates, return by score order
            final_k = min(num_samples, len(candidate_indices))
            return candidate_indices[:final_k], candidate_values[:final_k]

        # Extract candidate features (Flatten)
        # Ensure indices are on the same device as inputs for indexing
        candidate_indices_for_inputs = candidate_indices.cpu() if inputs.device.type == 'cpu' else candidate_indices.to(inputs.device)
        candidate_features = inputs[candidate_indices_for_inputs].flatten(start_dim=1)

        # Final selection using K-Center Greedy
        final_selected_local_indices = self._k_center_greedy(
            candidate_features, num_samples
        )

        # Convert local indices to global indices
        final_selected_local_tensor = torch.tensor(
            final_selected_local_indices, device=candidate_indices.device, dtype=torch.long
        )
        final_selected_global_indices = candidate_indices[final_selected_local_tensor]
        final_scores = info_scores[final_selected_global_indices]

        return final_selected_global_indices, final_scores

    def _component_transform(self, scores: torch.Tensor) -> torch.Tensor:
        """E5 ablation: re-normalize a component score according to
        self.score_normalization. Input scores are already min-max normalized;
        z-score and rank are invariant to that monotone affine map, so applying
        them here is equivalent to applying them to the raw scores."""
        if self.score_normalization == 'minmax':
            return scores
        if self.score_normalization == 'zscore':
            std = scores.std()
            if std < 1e-8:
                return torch.zeros_like(scores)
            return (scores - scores.mean()) / std
        # 'rank': normalized ranks in [0, 1]
        n = len(scores)
        if n <= 1:
            return torch.zeros_like(scores)
        ranks = torch.argsort(torch.argsort(scores)).float()
        return ranks / (n - 1)

    def _combine_info_scores(self, hardness_score: torch.Tensor,
                             ctmi_score: torch.Tensor) -> torch.Tensor:
        """E5/E6 ablation-aware Stage-1 informativeness combination.
        Default (minmax + sum) reproduces the paper: alpha*H + gamma*CTMI."""
        h = self._component_transform(hardness_score)
        c = self._component_transform(ctmi_score)
        if self.score_combination == 'sum':
            return (self.alpha * h) + (self.gamma * c)
        # 'product': weights act as exponents; shift to positive to keep
        # the score monotone in each component.
        eps = 1e-3
        h_pos = h - h.min() + eps
        c_pos = c - c.min() + eps
        return h_pos.pow(self.alpha) * c_pos.pow(self.gamma)

    def _k_center_greedy(self, features: torch.Tensor, num_samples: int) -> List[int]:
        """
        K-Center Greedy Algorithm for Diversity Selection.
        Sequentially selects samples that are farthest apart.

        Args:
            features: [num_candidates, feature_dim] Features of candidate samples
            num_samples: Number of samples to select

        Returns:
            List of local indices of selected samples
        """
        num_candidates = features.shape[0]
        device = features.device

        if num_candidates <= num_samples:
            return list(range(num_candidates))

        selected_indices = []

        # 1. First sample: Select closest to feature center
        center = features.mean(dim=0, keepdim=True)
        dists_to_center = torch.cdist(features, center).squeeze()
        current_idx = torch.argmin(dists_to_center).item()
        selected_indices.append(current_idx)

        # Initialize distance table (minimum distance between all candidates and selected set)
        min_distances = torch.full((num_candidates,), float('inf'), device=device)

        # 2. Select remaining N-1 samples
        for _ in range(num_samples - 1):
            # Compute distances between last selected sample and all candidates
            last_selected_feature = features[selected_indices[-1]].unsqueeze(0)
            dists = torch.cdist(features, last_selected_feature).squeeze()

            # Update "minimum distance to selected set" for each candidate
            min_distances = torch.min(min_distances, dists)

            # Exclude already selected samples
            min_distances[selected_indices] = -float('inf')

            # Select sample with largest "minimum distance" (farthest from selected set)
            next_idx = torch.argmax(min_distances).item()
            selected_indices.append(next_idx)

        return selected_indices

"""
Selectors for Multi-Task Active Learning

Contains base selector class and our proposed method selector.
Baseline selectors are in baseline_selectors.py
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class BaseSelector(ABC):
    """Base class for all selectors."""

    @abstractmethod
    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """
        Select k samples from unlabeled pool.

        Args:
            unlabeled_pool: Unlabeled data
            k: Number of samples to select
            **kwargs: Additional arguments

        Returns:
            Indices of selected samples
        """
        pass


class OursSelector(BaseSelector):
    """Our proposed acquisition function selector."""

    def __init__(self, acquisition_function):
        self.acquisition_function = acquisition_function

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """Select samples using our acquisition function.

        Uses train_batch_size to keep GPU memory usage consistent with training.
        """
        model = self.acquisition_function.model
        model.train()

        # Use train_batch_size to keep GPU memory usage consistent with training
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        n_samples = len(unlabeled_pool)
        model_device = next(model.parameters()).device

        # Use CPU to store scores to avoid GPU memory accumulation
        # Process in batches and keep only top-k candidates
        all_acquisition_scores_cpu = []
        top_k_scores = None
        top_k_indices = None

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_pool = unlabeled_pool[i:end_idx].to(model_device)

            # Forward pass WITH gradients (needed for gradient conflict computation)
            # But we'll clear immediately after computing acquisition scores
            # Get image size from kwargs or use default
            image_size = kwargs.get('image_size', (480, 480))
            if isinstance(image_size, dict):
                image_size = (image_size.get('height', 480), image_size.get('width', 480))
            outputs = model(batch_pool, image_size=image_size)

            # Compute losses (needed for gradient conflict)
            # For unlabeled data, we use prediction-based losses (entropy/uncertainty)
            # since we don't have ground truth labels
            # IMPORTANT: We need per-sample losses for proper gradient conflict computation
            losses = {}
            task_types = kwargs.get('task_types', {})

            for task_name, output in outputs.items():
                task_type = task_types.get(task_name, 'classification')

                if task_type == 'classification':
                    # Use negative entropy as loss proxy (high entropy = high uncertainty = high loss)
                    # Keep per-sample: [batch_size]
                    with torch.enable_grad():  # Need gradients for loss computation
                        probs = F.softmax(output, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [batch_size]
                    # Negative entropy so higher uncertainty = higher loss
                    losses[task_name] = -entropy  # [batch_size] - per-sample loss
                    # Clear intermediate tensors immediately
                    del probs, entropy

                elif task_type == 'detection':
                    # For detection, use confidence entropy + box regression uncertainty
                    if output.dim() == 3:  # [batch, num_boxes, num_classes + 4]
                        num_classes = output.shape[2] - 4
                        conf_scores = output[:, :, :num_classes]  # [batch, num_boxes, num_classes]
                        bbox_pred = output[:, :, num_classes:]  # [batch, num_boxes, 4]
                        with torch.enable_grad():
                            # Classification entropy
                            probs = F.softmax(conf_scores, dim=-1)
                            cls_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [batch, num_boxes]

                            # Box regression uncertainty: variance across boxes
                            # Higher variance indicates more uncertainty in localization
                            bbox_variance = bbox_pred.var(dim=1, keepdim=False)  # [batch, 4] - variance per coordinate
                            # [batch] - average variance across coordinates
                            bbox_uncertainty = bbox_variance.mean(dim=-1)

                        # Combined uncertainty: classification + box regression
                        cls_entropy_mean = cls_entropy.mean(dim=1)  # [batch_size]
                        combined_uncertainty = cls_entropy_mean + 0.5 * bbox_uncertainty
                        losses[task_name] = -combined_uncertainty  # [batch_size] - per-sample loss
                        # Clear intermediate tensors immediately
                        del conf_scores, bbox_pred, probs, cls_entropy, bbox_variance, bbox_uncertainty, cls_entropy_mean, combined_uncertainty
                    else:
                        # Fallback: use output sum per sample
                        losses[task_name] = output.sum(dim=tuple(range(1, output.dim())))

                elif task_type == 'segmentation':
                    # For segmentation, use pixel-wise entropy
                    # CRITICAL: Segmentation outputs are huge [B, 80, 224, 224], need careful memory management
                    with torch.enable_grad():
                        probs = F.softmax(output, dim=1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [batch, H, W]
                    # Average over spatial dimensions: [batch_size]
                    entropy_per_sample = entropy.mean(dim=(1, 2))  # [batch_size]
                    losses[task_name] = -entropy_per_sample  # [batch_size] - per-sample loss
                    # CRITICAL: Delete large intermediate tensors immediately
                    del probs, entropy, entropy_per_sample
                    # Also delete the large output immediately after use
                    del output

                else:
                    # Fallback: use output sum per sample
                    losses[task_name] = output.sum(dim=tuple(range(1, output.dim())))

            # Compute acquisition scores (this computes gradients internally for conflict)
            shared_params = model.get_shared_parameters()

            # Cache inputs for MC-dropout CTMI computation
            if hasattr(self.acquisition_function, 'set_cached_inputs'):
                self.acquisition_function.set_cached_inputs(batch_pool)

            # Pass inputs for diversity computation
            batch_acquisition_scores = self.acquisition_function.compute_acquisition_scores(
                outputs, losses, shared_params, task_types,
                inputs=batch_pool,
                selected_indices=None  # Will be updated in future iterations if needed
            )

            # Clear cached inputs after computation
            if hasattr(self.acquisition_function, 'clear_cached_inputs'):
                self.acquisition_function.clear_cached_inputs()

            # Move scores to CPU immediately and detach to break gradient graph
            batch_scores_cpu = batch_acquisition_scores.detach().cpu()
            batch_indices_cpu = torch.arange(i, end_idx, dtype=torch.long)

            # CRITICAL: Clear all GPU tensors including outputs and losses immediately
            # This is critical to prevent memory accumulation, especially for large outputs
            # Delete outputs first (they may be very large, especially segmentation [B, 80, 224, 224])
            for task_name in list(outputs.keys()):
                if task_name in outputs:
                    del outputs[task_name]
            del outputs

            # Delete losses (they maintain gradient graph connections to outputs)
            for task_name in list(losses.keys()):
                if task_name in losses:
                    del losses[task_name]
            del losses

            # Delete acquisition scores
            del batch_acquisition_scores

            # Clear gradients after acquisition score computation
            model.zero_grad()

            # Force GPU memory cleanup after each batch
            if model_device.type == 'cuda':
                torch.cuda.empty_cache()

            # Merge with current top-k
            if top_k_scores is None:
                # First batch: initialize top-k
                top_k_scores = batch_scores_cpu
                top_k_indices = batch_indices_cpu
            else:
                # Merge and keep top-k
                merged_scores = torch.cat([top_k_scores, batch_scores_cpu], dim=0)
                merged_indices = torch.cat([top_k_indices, batch_indices_cpu], dim=0)

                # Keep top-k
                k_keep = min(k, len(merged_scores))
                top_k_values, top_k_positions = torch.topk(merged_scores, k=k_keep)
                top_k_scores = top_k_values
                top_k_indices = merged_indices[top_k_positions]
                # Clear merged tensors immediately
                del merged_scores, merged_indices, top_k_values, top_k_positions

            # Clear all GPU tensors immediately
            model.zero_grad()
            del batch_pool, batch_scores_cpu, batch_indices_cpu
            if model_device.type == 'cuda':
                with torch.cuda.device(model_device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        model.eval()  # Reset to eval mode

        # Return top-k indices (already on CPU)
        return top_k_indices[:k] if len(top_k_indices) > k else top_k_indices


class TwoStageSelector(BaseSelector):
    """
    2-Stage Selection Strategy.

    Key improvements:
    1. Resolve diversity dilution: Separate into 2 stages instead of multiplicative approach
    2. Resolve conflict paradox: Exclude conflict (PCGrad handles it)
    3. Computational efficiency: Expensive operations only on candidate pool

    Stage 1: Select candidates based on Informativeness (Hardness + CTMI)
    Stage 2: Final selection based on Diversity using K-Center Greedy
    """

    def __init__(self, acquisition_function, pool_factor: int = 3):
        self.acquisition_function = acquisition_function
        self.pool_factor = pool_factor

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """Select samples using 2-Stage strategy.

        IMPORTANT: This implementation is streaming to avoid OOM.
        It does NOT materialize all inputs/outputs for the entire unlabeled pool.

        Stage 1: compute informativeness per-sample (hardness + CTMI) and keep top-M candidates
        Stage 2: compute diversity on shared-encoder features for candidates only (K-center greedy)
        """
        model = self.acquisition_function.model
        model.eval()  # Use eval mode for consistent predictions

        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        n_samples = len(unlabeled_pool)
        model_device = next(model.parameters()).device
        task_types = kwargs.get('task_types', {})
        num_candidates = min(n_samples, int(k) * int(self.pool_factor))

        # --- Stage 1: Informativeness (streaming top-M) ---
        top_scores_cpu: Optional[torch.Tensor] = None
        top_indices_cpu: Optional[torch.Tensor] = None

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_inputs = unlabeled_pool[start_idx:end_idx].to(model_device)

                # Get image size from kwargs or use default
                image_size = kwargs.get('image_size', (480, 480))
                if isinstance(image_size, dict):
                    image_size = (image_size.get('height', 480), image_size.get('width', 480))
                outputs = model(batch_inputs, image_size=image_size)
                active_outputs = {t: outputs[t] for t in self.acquisition_function.tasks if t in outputs}

                hardness = self.acquisition_function.compute_hardness(active_outputs, task_types).to(model_device)

                if self.acquisition_function.third_component_mode == 'ctmi':
                    if self.acquisition_function.ctmi_use_mc_dropout and self.acquisition_function.ctmi_num_samples > 1:
                        # Compute CTMI from inputs for THIS batch only (avoid caching the full pool)
                        ctmi = self.acquisition_function._compute_ctmi_mc_dropout_with_inputs(
                            batch_inputs, task_types, batch_size=end_idx - start_idx
                        ).to(model_device)
                    else:
                        ctmi = self.acquisition_function._compute_ctmi_single_pass(
                            active_outputs, task_types, batch_size=end_idx - start_idx
                        ).to(model_device)
                else:
                    ctmi = torch.zeros_like(hardness, device=model_device)

                info_scores = (self.acquisition_function.alpha * hardness) + (self.acquisition_function.gamma * ctmi)

                batch_scores_cpu = info_scores.detach().cpu()
                batch_indices_cpu = torch.arange(start_idx, end_idx, device='cpu', dtype=torch.long)

                if top_scores_cpu is None:
                    top_scores_cpu = batch_scores_cpu
                    top_indices_cpu = batch_indices_cpu
                else:
                    merged_scores = torch.cat([top_scores_cpu, batch_scores_cpu], dim=0)
                    merged_indices = torch.cat([top_indices_cpu, batch_indices_cpu], dim=0)

                    k_keep = min(num_candidates, len(merged_scores))
                    top_vals, top_pos = torch.topk(merged_scores, k=k_keep)
                    top_scores_cpu = top_vals
                    top_indices_cpu = merged_indices[top_pos]

                    del merged_scores, merged_indices, top_vals, top_pos

                # Cleanup
                del outputs, active_outputs, hardness, ctmi, info_scores, batch_scores_cpu, batch_indices_cpu, batch_inputs
                if model_device.type == 'cuda':
                    with torch.cuda.device(model_device):
                        torch.cuda.empty_cache()

        if top_indices_cpu is None or len(top_indices_cpu) == 0:
            # Fallback: random if something went wrong
            return torch.randperm(n_samples, device='cpu', dtype=torch.long)[:k]

        # Candidate indices (CPU)
        candidate_indices = top_indices_cpu[:num_candidates].long()

        # --- Stage 2: Diversity on shared-encoder features (candidates only) ---
        if len(candidate_indices) <= k:
            return candidate_indices[:k]

        candidate_features_cpu: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(candidate_indices), batch_size):
                end = min(start + batch_size, len(candidate_indices))
                idx = candidate_indices[start:end]
                cand_inputs = unlabeled_pool[idx].to(model_device)

                if hasattr(model, 'shared_encoder'):
                    feats = model.shared_encoder(cand_inputs)
                else:
                    feats = cand_inputs

                if feats.dim() == 4:
                    feats = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)

                candidate_features_cpu.append(feats.detach().cpu())
                del cand_inputs, feats
        if model_device.type == 'cuda':
            with torch.cuda.device(model_device):
                torch.cuda.empty_cache()

        features = torch.cat(candidate_features_cpu, dim=0)

        # K-center greedy (reuse acquisition implementation)
        selected_local = self.acquisition_function._k_center_greedy(features, k)
        selected_local_tensor = torch.tensor(selected_local, device='cpu', dtype=torch.long)
        selected_indices = candidate_indices[selected_local_tensor]

        return selected_indices.cpu()


class LinearScalarizationSelector(BaseSelector):
    """
    Linear Scalarization baseline.

    Computes a single weighted sum: score = α·Hardness + γ·CTMI + δ·Diversity
    then selects top-k by score. This demonstrates that naively combining
    good metrics via linear combination suffers from diversity dilution
    compared to ODS's orthogonal 2-stage decomposition.
    """

    def __init__(self, acquisition_function, diversity_weight: float = 0.3):
        self.acquisition_function = acquisition_function
        self.diversity_weight = diversity_weight

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        model = self.acquisition_function.model
        model.eval()

        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        n_samples = len(unlabeled_pool)
        model_device = next(model.parameters()).device
        task_types = kwargs.get('task_types', {})

        all_scores = []
        all_features = []

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_inputs = unlabeled_pool[start_idx:end_idx].to(model_device)

                image_size = kwargs.get('image_size', (480, 480))
                if isinstance(image_size, dict):
                    image_size = (image_size.get('height', 480), image_size.get('width', 480))
                outputs = model(batch_inputs, image_size=image_size)
                active_outputs = {t: outputs[t] for t in self.acquisition_function.tasks if t in outputs}

                hardness = self.acquisition_function.compute_hardness(active_outputs, task_types).to(model_device)

                if self.acquisition_function.third_component_mode == 'ctmi':
                    if self.acquisition_function.ctmi_use_mc_dropout and self.acquisition_function.ctmi_num_samples > 1:
                        ctmi = self.acquisition_function._compute_ctmi_mc_dropout_with_inputs(
                            batch_inputs, task_types, batch_size=end_idx - start_idx
                        ).to(model_device)
                    else:
                        ctmi = self.acquisition_function._compute_ctmi_single_pass(
                            active_outputs, task_types, batch_size=end_idx - start_idx
                        ).to(model_device)
                else:
                    ctmi = torch.zeros_like(hardness, device=model_device)

                info_scores = (self.acquisition_function.alpha * hardness) + (self.acquisition_function.gamma * ctmi)
                all_scores.append(info_scores.detach().cpu())

                if hasattr(model, 'shared_encoder'):
                    feats = model.shared_encoder(batch_inputs)
                else:
                    feats = batch_inputs
                if feats.dim() == 4:
                    feats = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)
                all_features.append(feats.detach().cpu())

                del outputs, active_outputs, hardness, ctmi, info_scores, batch_inputs, feats
                if model_device.type == 'cuda':
                    with torch.cuda.device(model_device):
                        torch.cuda.empty_cache()

        info_scores_all = torch.cat(all_scores, dim=0)
        features_all = torch.cat(all_features, dim=0)

        # Compute batch-level diversity: average pairwise distance per sample
        # For memory efficiency, compute in chunks against random reference set
        ref_size = min(500, n_samples)
        ref_indices = torch.randperm(n_samples)[:ref_size]
        ref_features = features_all[ref_indices]

        diversity_scores = []
        chunk = 512
        for i in range(0, n_samples, chunk):
            end = min(i + chunk, n_samples)
            dists = torch.cdist(features_all[i:end], ref_features, p=2)
            diversity_scores.append(dists.mean(dim=1))
        diversity_all = torch.cat(diversity_scores, dim=0)

        # Normalize each component to [0, 1]
        def _norm(t):
            t_min, t_max = t.min(), t.max()
            if t_max - t_min < 1e-8:
                return torch.zeros_like(t)
            return (t - t_min) / (t_max - t_min)

        info_norm = _norm(info_scores_all)
        div_norm = _norm(diversity_all)

        # Linear combination: score = (1-δ)·info + δ·diversity
        combined = (1.0 - self.diversity_weight) * info_norm + self.diversity_weight * div_norm

        _, top_indices = torch.topk(combined, k=min(k, n_samples))
        return top_indices.cpu()


def make_selector(
    name: str,
    config: Dict,
    model=None,
    acquisition_function=None
) -> BaseSelector:
    """
    Factory function to create selectors.

    Args:
        name: Selector name ('random', 'uncertainty', 'diversity', 'coreset', 'badge', 'batchbald', 'clustering', 'learningloss', 'ours')
        config: Configuration dictionary
        model: Model for uncertainty/diversity selectors
        acquisition_function: Acquisition function for 'ours' selector

    Returns:
        Selector instance
    """
    # Import baseline selectors here to avoid circular import
    from .baseline_selectors import (BADGESelector, BatchBALDSelector,
                                     ClusteringSelector, CoreSetSelector,
                                     DiversitySelector, HekimogluSelector,
                                     LearningLossSelector, PartALSelector,
                                     RandomSelector, UncertaintySelector)

    if name == 'random':
        return RandomSelector(seed=config.get('seed'))

    elif name == 'uncertainty':
        if model is None:
            raise ValueError("Model required for uncertainty selector")
        return UncertaintySelector(
            model=model,
            task_types=config.get('task_types', {}),
            aggregation=config.get('aggregation', 'mean')
        )

    elif name == 'diversity':
        if model is None:
            raise ValueError("Model required for diversity selector")
        return DiversitySelector(
            model=model,
            metric=config.get('diversity_metric', 'euclidean'),
            cache_features=config.get('cache_features', True)
        )

    elif name == 'coreset':
        if model is None:
            raise ValueError("Model required for coreset selector")
        return CoreSetSelector(
            model=model,
            metric=config.get('diversity_metric', 'euclidean'),
            cache_features=config.get('cache_features', True)
        )

    elif name in ('badge', 'mt-badge'):
        if model is None:
            raise ValueError("Model required for badge selector")
        return BADGESelector(
            model=model,
            task_types=config.get('task_types', {}),
            cache_features=config.get('cache_features', True)
        )

    elif name == 'batchbald':
        if model is None:
            raise ValueError("Model required for batchbald selector")
        return BatchBALDSelector(
            model=model,
            task_types=config.get('task_types', {}),
            aggregation=config.get('aggregation', 'mean')
        )

    elif name == 'clustering':
        if model is None:
            raise ValueError("Model required for clustering selector")
        return ClusteringSelector(
            model=model,
            n_clusters=config.get('n_clusters', None),
            metric=config.get('diversity_metric', 'euclidean'),
            cache_features=config.get('cache_features', True)
        )

    elif name == 'hekimoglu':
        if model is None:
            raise ValueError("Model required for hekimoglu selector")
        return HekimogluSelector(
            model=model,
            task_types=config.get('task_types', {}),
            mc_samples=config.get('hekimoglu_mc_samples', 5),
            latent_dim=config.get('hekimoglu_latent_dim', config.get('hekimoglu_pca_dim', 512)),
            diversity_method=config.get('hekimoglu_diversity_method', 'autoencoder'),
            ae_epochs=config.get('hekimoglu_ae_epochs', 50),
            ae_lr=config.get('hekimoglu_ae_lr', 1e-3),
        )

    elif name == 'partial' or name == 'partal':
        if model is None:
            raise ValueError("Model required for PartAL selector")
        return PartALSelector(
            model=model,
            task_types=config.get('task_types', {}),
            mc_samples=config.get('partial_mc_samples', 10),
            aggregation=config.get('partial_aggregation', 'sum'),
        )

    elif name in ('cluster-margin', 'clustermargin'):
        if model is None:
            raise ValueError("Model required for cluster-margin selector")
        from .filter_diversify_selectors import ClusterMarginSelector
        return ClusterMarginSelector(
            model=model,
            task_types=config.get('task_types', {}),
            margin_factor=config.get('clustermargin_factor', 10.0),
            seed=config.get('seed'),
        )

    elif name in ('cluster-margin-ctmi', 'cmctmi'):
        if model is None:
            raise ValueError("Model required for cluster-margin-ctmi selector")
        from .filter_diversify_selectors import ClusterMarginCTMISelector
        return ClusterMarginCTMISelector(
            model=model,
            task_types=config.get('task_types', {}),
            margin_factor=config.get('clustermargin_factor', 10.0),
            gamma=config.get('gamma', 0.5),
            mc_samples=config.get('ctmi_num_samples', 15),
            ctmi_mode=config.get('ctmi_mode', 'covariance_negative'),
            seed=config.get('seed'),
        )

    elif name == 'ppal':
        if model is None:
            raise ValueError("Model required for ppal selector")
        from .filter_diversify_selectors import PPALSelector
        return PPALSelector(
            model=model,
            task_types=config.get('task_types', {}),
            delta=config.get('ppal_delta', 4.0),
            xi=config.get('ppal_xi', 0.6),
            alpha=config.get('ppal_alpha', 0.3),
            beta=config.get('ppal_beta', 0.2),
            kmeans_iters=config.get('ppal_kmeans_iters', 100),
            seed=config.get('seed'),
        )

    elif name == 'fass':
        if model is None:
            raise ValueError("Model required for fass selector")
        from .filter_diversify_selectors import FASSSelector
        return FASSSelector(
            model=model,
            task_types=config.get('task_types', {}),
            eta_factor=config.get('fass_eta_factor', 10.0),
        )

    elif name in ('d2pruning', 'd2-pruning'):
        if model is None:
            raise ValueError("Model required for d2pruning selector")
        from .filter_diversify_selectors import D2PruningSelector
        return D2PruningSelector(
            model=model,
            task_types=config.get('task_types', {}),
            k_nn=config.get('d2_k_nn', 15),
            gamma_f=config.get('d2_gamma_f', 1.0),
            gamma_r=config.get('d2_gamma_r', 1.0),
        )

    elif name == 'ses':
        if model is None:
            raise ValueError("Model required for ses selector")
        from .filter_diversify_selectors import SESSelector
        return SESSelector(
            model=model,
            task_types=config.get('task_types', {}),
            k_nn=config.get('ses_k_nn', 15),
            n_communities=config.get('ses_n_communities'),
            seed=config.get('seed'),
        )

    elif name == 'learningloss' or name == 'learning_loss':
        if model is None:
            raise ValueError("Model required for learning loss selector")
        return LearningLossSelector(
            model=model,
            task_types=config.get('task_types', {}),
            aggregation=config.get('aggregation', 'mean')
        )

    elif name == 'ours':
        if acquisition_function is None:
            raise ValueError("Acquisition function required for 'ours' selector")
        return OursSelector(acquisition_function)

    elif name == 'ours-2stage' or name == 'ours-twostage':
        if acquisition_function is None:
            raise ValueError("Acquisition function required for 'ours-2stage' selector")
        pool_factor = config.get('pool_factor', 3)
        return TwoStageSelector(acquisition_function, pool_factor=pool_factor)

    elif name == 'linear-scalarization':
        if acquisition_function is None:
            raise ValueError("Acquisition function required for 'linear-scalarization' selector")
        diversity_weight = config.get('linear_diversity_weight', 0.3)
        return LinearScalarizationSelector(acquisition_function, diversity_weight=diversity_weight)

    else:
        raise ValueError(f"Unknown selector: {name}")


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'seed': 42,
        'task_types': {'classification': 'classification', 'segmentation': 'segmentation'},
        'aggregation': 'mean',
        'diversity_metric': 'euclidean',
        'cache_features': True
    }

    # Create selectors
    random_selector = make_selector('random', config)
    print("Random selector created")

    # Note: Other selectors require model/acquisition_function
    print("Selector factory working correctly")

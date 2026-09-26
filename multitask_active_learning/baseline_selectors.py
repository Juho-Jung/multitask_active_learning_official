"""
Baseline Selectors for Multi-Task Active Learning

Implements random, uncertainty-based, and diversity-based selection strategies.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .al_selectors import BaseSelector


class RandomSelector(BaseSelector):
    """Random selection baseline."""

    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """Randomly select k samples."""
        n_samples = len(unlabeled_pool)
        if k >= n_samples:
            return torch.arange(n_samples, device=unlabeled_pool.device, dtype=torch.long)

        indices = torch.randperm(n_samples, device=unlabeled_pool.device, dtype=torch.long)[:k]
        return indices


class UncertaintySelector(BaseSelector):
    """Uncertainty-based selection."""

    def __init__(self, model, task_types: Dict[str, str], aggregation: str = 'mean'):
        self.model = model
        self.task_types = task_types
        self.aggregation = aggregation  # 'mean' or 'max'

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """Select samples based on uncertainty."""
        # Store original device (CPU) for return values
        original_device = unlabeled_pool.device
        model_device = next(self.model.parameters()).device

        # Use train_batch_size to keep GPU memory usage consistent
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        n_samples = len(unlabeled_pool)

        # Process in batches and accumulate uncertainties on CPU
        all_uncertainties = []  # List of lists: one list per task

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_pool = unlabeled_pool[i:end_idx].to(model_device)

                outputs = self.model(batch_pool)

                # Compute uncertainty for each task in this batch
                batch_uncertainties = []
                for task_name, output in outputs.items():
                    task_type = self.task_types.get(task_name, 'classification')

                    if task_type == 'classification':
                        probs = F.softmax(output, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                        batch_uncertainties.append(entropy.cpu())  # Move to CPU immediately
                    elif task_type == 'segmentation':
                        probs = F.softmax(output, dim=1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                        entropy = entropy.mean(dim=(1, 2))  # Average over spatial dimensions
                        batch_uncertainties.append(entropy.cpu())  # Move to CPU immediately
                    elif task_type == 'detection':
                        if output.dim() == 3:  # [batch, num_boxes, num_classes + 4]
                            num_classes = output.shape[2] - 4
                            conf_scores = output[:, :, :num_classes]
                            probs = F.softmax(conf_scores, dim=-1)
                            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [batch, num_boxes]
                            entropy = entropy.mean(dim=1)  # [batch] - Average over boxes
                            entropy = entropy.view(-1)
                            batch_uncertainties.append(entropy.cpu())  # Move to CPU immediately
                        else:
                            print(f"Warning: detection output has unexpected shape: {output.shape}")
                            batch_uncertainties.append(torch.zeros(output.shape[0], device='cpu'))

                # Accumulate uncertainties per task
                if not all_uncertainties:
                    # First batch: initialize lists
                    all_uncertainties = [[u] for u in batch_uncertainties]
                else:
                    # Append to existing lists
                    for task_idx, u in enumerate(batch_uncertainties):
                        all_uncertainties[task_idx].append(u)

                # Clean up GPU memory
                del batch_pool, outputs
                if model_device.type == 'cuda':
                    torch.cuda.empty_cache()

        if not all_uncertainties or not all_uncertainties[0]:
            # Fallback to random selection
            if k >= n_samples:
                return torch.arange(n_samples, device=original_device, dtype=torch.long)
            return torch.randperm(n_samples, device=original_device, dtype=torch.long)[:k]

        # Concatenate uncertainties for each task
        uncertainties = []
        for task_uncertainties in all_uncertainties:
            task_combined = torch.cat(task_uncertainties, dim=0)
            uncertainties.append(task_combined)

        # Aggregate uncertainties across tasks
        for i, u in enumerate(uncertainties):
            if u.dim() != 1:
                raise ValueError(f"Uncertainty {i} has wrong shape: {u.shape}, expected 1D tensor")

        if self.aggregation == 'mean':
            combined_uncertainty = torch.stack(uncertainties).mean(dim=0)
        elif self.aggregation == 'max':
            combined_uncertainty = torch.stack(uncertainties).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Select top-k most uncertain samples
        _, indices = torch.topk(combined_uncertainty, k=min(k, len(combined_uncertainty)))
        # Return indices on original device (CPU)
        return indices.long()


class DiversitySelector(BaseSelector):
    """Diversity-based selection using farthest-point sampling."""

    def __init__(self, model, metric: str = 'euclidean', cache_features: bool = True):
        self.model = model
        self.metric = metric  # 'euclidean' or 'cosine'
        self.cache_features = cache_features
        self._feature_cache = {}

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """Select diverse samples using farthest-point sampling."""
        # Get features from shared encoder (with batching)
        features = self._get_features(unlabeled_pool, **kwargs)

        n_samples = len(features)
        if k >= n_samples:
            return torch.arange(n_samples, device=unlabeled_pool.device)

        # Initialize with random first point
        selected_indices = [torch.randint(0, n_samples, (1,), device=unlabeled_pool.device).item()]

        # Farthest-point sampling
        for _ in range(k - 1):
            # Compute distances to already selected points
            selected_features = features[selected_indices]

            if self.metric == 'euclidean':
                distances = torch.cdist(features, selected_features, p=2)
            elif self.metric == 'cosine':
                # Normalize features for cosine distance
                features_norm = F.normalize(features, p=2, dim=1)
                selected_norm = F.normalize(selected_features, p=2, dim=1)
                distances = 1 - torch.mm(features_norm, selected_norm.t())
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            # Find point with maximum minimum distance
            min_distances = distances.min(dim=1)[0]
            next_idx = min_distances.argmax().item()
            selected_indices.append(next_idx)

        # Return tensor on the same device as unlabeled_pool
        return torch.tensor(selected_indices, device=unlabeled_pool.device, dtype=torch.long)

    def _get_features(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract features from shared encoder."""
        # Check cache first (using original CPU tensor's identifier)
        if self.cache_features:
            cache_key = hash(inputs.data_ptr())
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]

        model_device = next(self.model.parameters()).device
        original_device = inputs.device

        # Use train_batch_size to keep GPU memory usage consistent
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        n_samples = len(inputs)

        # Process in batches and accumulate features on CPU
        all_features = []

        # Extract features from shared encoder
        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_inputs = inputs[i:end_idx].to(model_device)

                if hasattr(self.model, 'shared_encoder'):
                    batch_features = self.model.shared_encoder(batch_inputs)
                else:
                    # Fallback: use first layer of model
                    batch_features = batch_inputs

                # Global average pooling if needed
                if batch_features.dim() == 4:  # [batch, channels, height, width]
                    batch_features = F.adaptive_avg_pool2d(batch_features, (1, 1)).squeeze()

                # Move to CPU immediately
                all_features.append(batch_features.cpu())

                # Clean up GPU memory
                del batch_inputs, batch_features
                if model_device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Concatenate all features
        features = torch.cat(all_features, dim=0)

        # Cache features
        if self.cache_features:
            self._feature_cache[cache_key] = features

        return features


class CoreSetSelector(BaseSelector):
    """CoreSet selection: selects samples that best represent the unlabeled pool.

    Implements the k-center greedy algorithm to select samples that maximize
    coverage of the unlabeled pool in feature space.
    """

    def __init__(self, model, metric: str = 'euclidean', cache_features: bool = True):
        self.model = model
        self.metric = metric  # 'euclidean' or 'cosine'
        self.cache_features = cache_features
        self._feature_cache = {}

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """Select samples using CoreSet (k-center greedy algorithm)."""
        # Get features from shared encoder (with batching)
        features = self._get_features(unlabeled_pool, **kwargs)

        n_samples = len(features)
        if k >= n_samples:
            return torch.arange(n_samples, device=unlabeled_pool.device, dtype=torch.long)

        # CoreSet: k-center greedy algorithm
        # Initialize with random first point
        selected_indices = [torch.randint(0, n_samples, (1,), device=unlabeled_pool.device).item()]

        # Greedily select points that maximize minimum distance to selected set
        for _ in range(k - 1):
            # Compute distances from all points to selected points
            selected_features = features[selected_indices]

            if self.metric == 'euclidean':
                distances = torch.cdist(features, selected_features, p=2)  # [n_samples, n_selected]
            elif self.metric == 'cosine':
                # Normalize features for cosine distance
                features_norm = F.normalize(features, p=2, dim=1)
                selected_norm = F.normalize(selected_features, p=2, dim=1)
                distances = 1 - torch.mm(features_norm, selected_norm.t())  # [n_samples, n_selected]
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            # For each unselected point, find minimum distance to selected set
            min_distances = distances.min(dim=1)[0]  # [n_samples]

            # Select the point with maximum minimum distance (farthest from selected set)
            # Exclude already selected indices
            mask = torch.ones(n_samples, dtype=torch.bool, device=unlabeled_pool.device)
            mask[selected_indices] = False
            min_distances[~mask] = -1  # Set selected points to -1 so they won't be chosen

            next_idx = min_distances.argmax().item()
            selected_indices.append(next_idx)

        # Return tensor on the same device as unlabeled_pool
        return torch.tensor(selected_indices, device=unlabeled_pool.device, dtype=torch.long)

    def _get_features(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract features from shared encoder."""
        # Check cache first (using original CPU tensor's identifier)
        if self.cache_features:
            cache_key = hash(inputs.data_ptr())
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]

        model_device = next(self.model.parameters()).device
        original_device = inputs.device

        # Use train_batch_size to keep GPU memory usage consistent
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        n_samples = len(inputs)

        # Process in batches and accumulate features on CPU
        all_features = []

        # Extract features from shared encoder
        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_inputs = inputs[i:end_idx].to(model_device)

                if hasattr(self.model, 'shared_encoder'):
                    batch_features = self.model.shared_encoder(batch_inputs)
                else:
                    # Fallback: use first layer of model
                    batch_features = batch_inputs

                # Global average pooling if needed
                if batch_features.dim() == 4:  # [batch, channels, height, width]
                    batch_features = F.adaptive_avg_pool2d(batch_features, (1, 1)).squeeze()

                # Move to CPU immediately
                all_features.append(batch_features.cpu())

                # Clean up GPU memory
                del batch_inputs, batch_features
                if model_device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Concatenate all features
        features = torch.cat(all_features, dim=0)

        # Cache features
        if self.cache_features:
            self._feature_cache[cache_key] = features

        return features


class BADGESelector(BaseSelector):
    """BADGE: Batch Active learning by Diverse Gradient Embeddings.

    Combines uncertainty (via gradient embeddings) with diversity (via k-means++).
    For each sample, computes gradient embedding w.r.t. loss, then uses k-means++
    to select diverse samples with high uncertainty.
    """

    def __init__(self, model, task_types: Dict[str, str], cache_features: bool = True):
        self.model = model
        self.task_types = task_types
        self.cache_features = cache_features
        self._gradient_cache = {}

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """Select samples using BADGE (gradient embeddings + k-means++)."""
        import time
        start_time = time.time()

        self.model.train()  # Need gradients

        n_samples = len(unlabeled_pool)
        if k >= n_samples:
            return torch.arange(n_samples, device=unlabeled_pool.device, dtype=torch.long)

        print(f"  🔄 BADGE: Computing gradient embeddings for {n_samples} samples...")
        # Compute gradient embeddings for all samples (with batching)
        gradient_embeddings = self._compute_gradient_embeddings(unlabeled_pool, **kwargs)
        embedding_time = time.time() - start_time
        print(f"  ✅ Gradient embeddings computed in {embedding_time:.1f}s")

        print(f"  🔄 BADGE: Running k-means++ selection for {k} samples...")
        kmeans_start = time.time()
        # Use k-means++ initialization for diverse selection
        selected_indices = self._kmeans_plus_plus(gradient_embeddings, k)
        kmeans_time = time.time() - kmeans_start
        print(f"  ✅ k-means++ completed in {kmeans_time:.1f}s")

        total_time = time.time() - start_time
        print(f"  ✅ BADGE selection completed in {total_time:.1f}s total")

        self.model.eval()  # Reset to eval mode
        return torch.tensor(selected_indices, device=unlabeled_pool.device, dtype=torch.long)

    def _compute_gradient_embeddings(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute gradient embeddings for BADGE.

        For each sample, computes gradient of loss w.r.t. last layer parameters,
        then concatenates and normalizes to get embedding.
        """
        from tqdm import tqdm

        model_device = next(self.model.parameters()).device
        original_device = inputs.device

        # Use larger batch size for better efficiency
        # Gradient computation needs memory, but we can use larger batches
        base_batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        batch_size = max(16, base_batch_size // 2)  # Use half of train_batch_size for better speed

        self.model.train()  # Need gradients
        all_embeddings = []
        n_samples = len(inputs)

        # Process in batches with progress bar
        with tqdm(total=n_samples, desc="  Computing BADGE embeddings", unit="samples") as pbar:
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = inputs[i:end_idx].to(model_device)
                batch_embeddings_list = []

                # Process each sample individually to get per-sample gradients
                for j in range(len(batch)):
                    sample = batch[j:j + 1]

                    # Get model outputs
                    outputs = self.model(sample)

                    # Compute loss for each task and get gradients
                    sample_grads = []
                    for task_name, output in outputs.items():
                        task_type = self.task_types.get(task_name, 'classification')

                        # Compute pseudo-loss (entropy for uncertainty)
                        if task_type == 'classification':
                            probs = F.softmax(output, dim=-1)
                            loss = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                        elif task_type == 'segmentation':
                            probs = F.softmax(output, dim=1)
                            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                            loss = entropy.mean(dim=(1, 2)).mean()
                        elif task_type == 'detection':
                            num_classes = output.shape[2] - 4
                            conf_scores = output[:, :, :num_classes]
                            probs = F.softmax(conf_scores, dim=-1)
                            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                            loss = entropy.mean(dim=1).mean()
                        else:
                            loss = output.mean()

                        # Compute gradient w.r.t. last layer
                        self.model.zero_grad()
                        loss.backward(retain_graph=(task_name != list(outputs.keys())[-1]))

                        # Extract gradient from last layer (task head)
                        grad_embedding = None
                        for name, param in self.model.named_parameters():
                            if task_name in name and ('head' in name or 'classifier' in name) and param.grad is not None:
                                # Use last layer gradient as embedding
                                grad_embedding = param.grad.flatten().cpu()  # Move to CPU immediately
                                break

                        if grad_embedding is not None:
                            sample_grads.append(grad_embedding)

                        # Clear gradients immediately
                        self.model.zero_grad()

                    # Concatenate gradients from all tasks for this sample
                    if sample_grads:
                        combined_embedding = torch.cat(sample_grads)
                        batch_embeddings_list.append(combined_embedding)
                    else:
                        # Fallback: use zero embedding (on CPU)
                        batch_embeddings_list.append(torch.zeros(128, device='cpu'))

                    # Clean up outputs for this sample
                    del outputs

                # Stack batch embeddings (already on CPU)
                if batch_embeddings_list:
                    batch_embeddings = torch.stack(batch_embeddings_list)
                    all_embeddings.append(batch_embeddings)

                # Clean up batch and update progress
                del batch
                pbar.update(end_idx - i)

                # Only clear cache periodically (every 10 batches) to avoid overhead
                if (i // batch_size) % 10 == 0 and model_device.type == 'cuda':
                    torch.cuda.empty_cache()

        if not all_embeddings:
            # Fallback: use random selection
            return torch.randn(n_samples, 128, device=original_device)

        # Stack and normalize (all on CPU)
        gradient_embeddings = torch.cat(all_embeddings, dim=0)
        # Normalize each embedding
        gradient_embeddings = F.normalize(gradient_embeddings, p=2, dim=1)

        return gradient_embeddings

    def _kmeans_plus_plus(self, embeddings: torch.Tensor, k: int) -> List[int]:
        """K-means++ initialization for diverse selection."""
        from tqdm import tqdm

        n_samples = len(embeddings)
        selected_indices = []

        # First center: random
        first_idx = torch.randint(0, n_samples, (1,)).item()
        selected_indices.append(first_idx)

        # Subsequent centers: probability proportional to distance^2
        # Use progress bar for k-means++ selection
        with tqdm(total=k - 1, desc="  k-means++ selection", unit="centers", leave=False) as pbar:
            for iteration in range(k - 1):
                # Compute distances to all selected centers
                selected_centers = embeddings[selected_indices]  # [n_selected, dim]
                distances = torch.cdist(embeddings, selected_centers, p=2)  # [n_samples, n_selected]
                min_distances = distances.min(dim=1)[0]  # [n_samples] - distance to nearest center
                min_distances_sq = min_distances ** 2

                # Set already selected to 0 probability
                min_distances_sq[selected_indices] = 0

                # Sample with probability proportional to distance^2
                probs = min_distances_sq / (min_distances_sq.sum() + 1e-8)
                next_idx = torch.multinomial(probs, 1).item()
                selected_indices.append(next_idx)

                pbar.update(1)

        return selected_indices


class BatchBALDSelector(BaseSelector):
    """BatchBALD: Batch Bayesian Active Learning by Disagreement.

    Simplified version: combines uncertainty (entropy) with diversity (mutual information).
    For multi-task, aggregates uncertainty across tasks and uses diversity sampling.
    """

    def __init__(self, model, task_types: Dict[str, str], aggregation: str = 'mean'):
        self.model = model
        self.task_types = task_types
        self.aggregation = aggregation  # 'mean' or 'max'

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """Select samples using BatchBALD (uncertainty + diversity)."""
        self.model.eval()

        n_samples = len(unlabeled_pool)
        if k >= n_samples:
            return torch.arange(n_samples, device=unlabeled_pool.device, dtype=torch.long)

        # Compute uncertainty scores (with batching)
        uncertainties = self._compute_uncertainties(unlabeled_pool, **kwargs)

        # Greedy selection: at each step, select sample with highest uncertainty
        # that is also diverse from already selected samples
        selected_indices = []
        remaining_indices = list(range(n_samples))

        # Get features for diversity computation (with batching)
        features = self._get_features(unlabeled_pool, **kwargs)

        for _ in range(k):
            if not remaining_indices:
                break

            # Compute scores: uncertainty + diversity bonus
            scores = uncertainties.clone()

            if selected_indices:
                # Add diversity bonus: distance to selected samples
                selected_features = features[selected_indices]
                distances = torch.cdist(features[remaining_indices], selected_features, p=2)
                min_distances = distances.min(dim=1)[0]
                # Normalize diversity bonus
                if min_distances.max() > min_distances.min():
                    diversity_bonus = (min_distances - min_distances.min()) / \
                        (min_distances.max() - min_distances.min() + 1e-8)
                else:
                    diversity_bonus = torch.zeros_like(min_distances)

                # Combine: 0.7 * uncertainty + 0.3 * diversity
                scores[remaining_indices] = 0.7 * uncertainties[remaining_indices] + 0.3 * diversity_bonus
            else:
                # First selection: use uncertainty only
                pass

            # Select highest scoring remaining sample
            remaining_scores = scores[remaining_indices]
            best_idx_in_remaining = remaining_scores.argmax().item()
            best_idx = remaining_indices[best_idx_in_remaining]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return torch.tensor(selected_indices, device=unlabeled_pool.device, dtype=torch.long)

    def _compute_uncertainties(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute uncertainty scores (entropy) for each sample."""
        model_device = next(self.model.parameters()).device
        original_device = inputs.device

        # Use train_batch_size to keep GPU memory usage consistent
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        n_samples = len(inputs)

        # Process in batches and accumulate uncertainties on CPU
        all_uncertainties = []  # List of lists: one list per task

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_inputs = inputs[i:end_idx].to(model_device)

                outputs = self.model(batch_inputs)

                # Compute uncertainty for each task in this batch
                batch_uncertainties = []
                for task_name, output in outputs.items():
                    task_type = self.task_types.get(task_name, 'classification')

                    if task_type == 'classification':
                        probs = F.softmax(output, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                        batch_uncertainties.append(entropy.cpu())  # Move to CPU immediately
                    elif task_type == 'segmentation':
                        probs = F.softmax(output, dim=1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                        entropy = entropy.mean(dim=(1, 2))  # Average over spatial dimensions
                        batch_uncertainties.append(entropy.cpu())  # Move to CPU immediately
                    elif task_type == 'detection':
                        num_classes = output.shape[2] - 4
                        conf_scores = output[:, :, :num_classes]
                        probs = F.softmax(conf_scores, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                        entropy = entropy.mean(dim=1)  # Average over boxes
                        batch_uncertainties.append(entropy.cpu())  # Move to CPU immediately

                # Accumulate uncertainties per task
                if not all_uncertainties:
                    # First batch: initialize lists
                    all_uncertainties = [[u] for u in batch_uncertainties]
                else:
                    # Append to existing lists
                    for task_idx, u in enumerate(batch_uncertainties):
                        all_uncertainties[task_idx].append(u)

                # Clean up GPU memory
                del batch_inputs, outputs
                if model_device.type == 'cuda':
                    torch.cuda.empty_cache()

        if not all_uncertainties or not all_uncertainties[0]:
            return torch.zeros(n_samples, device=original_device)

        # Concatenate uncertainties for each task
        uncertainties = []
        for task_uncertainties in all_uncertainties:
            task_combined = torch.cat(task_uncertainties, dim=0)
            uncertainties.append(task_combined)

        # Aggregate across tasks
        uncertainties_stack = torch.stack(uncertainties)
        if self.aggregation == 'mean':
            combined = uncertainties_stack.mean(dim=0)
        elif self.aggregation == 'max':
            combined = uncertainties_stack.max(dim=0)[0]
        else:
            combined = uncertainties_stack.mean(dim=0)

        return combined

    def _get_features(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract features from shared encoder."""
        model_device = next(self.model.parameters()).device
        original_device = inputs.device

        # Use train_batch_size to keep GPU memory usage consistent
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        n_samples = len(inputs)

        # Process in batches and accumulate features on CPU
        all_features = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_inputs = inputs[i:end_idx].to(model_device)

                if hasattr(self.model, 'shared_encoder'):
                    batch_features = self.model.shared_encoder(batch_inputs)
                else:
                    batch_features = batch_inputs

                # Global average pooling if needed
                if batch_features.dim() == 4:
                    batch_features = F.adaptive_avg_pool2d(batch_features, (1, 1)).squeeze()

                # Move to CPU immediately
                all_features.append(batch_features.cpu())

                # Clean up GPU memory
                del batch_inputs, batch_features
                if model_device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Concatenate all features
        features = torch.cat(all_features, dim=0)
        return features


class ClusteringSelector(BaseSelector):
    """Clustering-based selection: uses feature space clustering for diversity.

    Clusters samples in feature space and selects from different clusters
    to ensure diversity.
    """

    def __init__(self, model, n_clusters: int = None, metric: str = 'euclidean', cache_features: bool = True):
        self.model = model
        self.n_clusters = n_clusters  # If None, will be set to k
        self.metric = metric
        self.cache_features = cache_features
        self._feature_cache = {}

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """Select samples using clustering-based diversity."""
        features = self._get_features(unlabeled_pool, **kwargs)

        n_samples = len(features)
        if k >= n_samples:
            return torch.arange(n_samples, device=unlabeled_pool.device, dtype=torch.long)

        # Determine number of clusters
        n_clusters = self.n_clusters if self.n_clusters is not None else min(k, n_samples // 2)
        n_clusters = min(n_clusters, n_samples)

        # Simple k-means clustering
        cluster_centers, cluster_assignments = self._kmeans(features, n_clusters)

        # Select samples from each cluster (proportional to cluster size, but at least 1 per cluster)
        selected_indices = []
        samples_per_cluster = max(1, k // n_clusters)
        remaining_samples = k

        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_assignments == cluster_id)
            cluster_indices = torch.where(cluster_mask)[0].tolist()

            if not cluster_indices:
                continue

            # Select from this cluster
            n_select = min(samples_per_cluster, len(cluster_indices), remaining_samples)
            if n_select > 0:
                selected_from_cluster = torch.randperm(len(cluster_indices))[:n_select].tolist()
                selected_indices.extend([cluster_indices[i] for i in selected_from_cluster])
                remaining_samples -= n_select

            if remaining_samples <= 0:
                break

        # If we need more samples, randomly select from remaining
        if len(selected_indices) < k:
            all_indices = set(range(n_samples))
            remaining = list(all_indices - set(selected_indices))
            n_needed = k - len(selected_indices)
            if remaining:
                additional = torch.randperm(len(remaining))[:n_needed].tolist()
                selected_indices.extend([remaining[i] for i in additional])

        return torch.tensor(selected_indices[:k], device=unlabeled_pool.device, dtype=torch.long)

    def _kmeans(self, features: torch.Tensor, n_clusters: int, max_iters: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple k-means clustering."""
        n_samples, n_features = features.shape

        # Initialize centers randomly
        center_indices = torch.randperm(n_samples)[:n_clusters]
        centers = features[center_indices].clone()

        for _ in range(max_iters):
            # Assign points to nearest center
            distances = torch.cdist(features, centers, p=2)
            assignments = distances.argmin(dim=1)

            # Update centers
            new_centers = torch.zeros_like(centers)
            for i in range(n_clusters):
                cluster_mask = (assignments == i)
                if cluster_mask.sum() > 0:
                    new_centers[i] = features[cluster_mask].mean(dim=0)
                else:
                    new_centers[i] = centers[i]  # Keep old center if cluster is empty

            # Check convergence
            if torch.allclose(centers, new_centers, atol=1e-4):
                break

            centers = new_centers

        # Final assignment
        distances = torch.cdist(features, centers, p=2)
        assignments = distances.argmin(dim=1)

        return centers, assignments

    def _get_features(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract features from shared encoder."""
        # Check cache first (using original CPU tensor's identifier)
        if self.cache_features:
            cache_key = hash(inputs.data_ptr())
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]

        model_device = next(self.model.parameters()).device
        original_device = inputs.device

        # Use train_batch_size to keep GPU memory usage consistent
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        n_samples = len(inputs)

        # Process in batches and accumulate features on CPU
        all_features = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_inputs = inputs[i:end_idx].to(model_device)

                if hasattr(self.model, 'shared_encoder'):
                    batch_features = self.model.shared_encoder(batch_inputs)
                else:
                    batch_features = batch_inputs

                if batch_features.dim() == 4:
                    batch_features = F.adaptive_avg_pool2d(batch_features, (1, 1)).squeeze()

                # Move to CPU immediately
                all_features.append(batch_features.cpu())

                # Clean up GPU memory
                del batch_inputs, batch_features
                if model_device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Concatenate all features
        features = torch.cat(all_features, dim=0)

        if self.cache_features:
            self._feature_cache[cache_key] = features

        return features


class LearningLossSelector(BaseSelector):
    """Learning Loss: Uses loss prediction to select informative samples.

    Reference: "Learning Loss for Active Learning" (CVPR 2019)
    Paper: https://arxiv.org/abs/1905.03677

    For multi-task settings, we predict loss for each task and aggregate.
    If the model has a loss_prediction_module, we use it. Otherwise, we use
    entropy-based pseudo-loss as a proxy.
    """

    def __init__(self, model, task_types: Dict[str, str], aggregation: str = 'mean'):
        self.model = model
        self.task_types = task_types
        self.aggregation = aggregation  # 'mean' or 'max'

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        """Select samples using Learning Loss strategy."""
        original_device = unlabeled_pool.device
        model_device = next(self.model.parameters()).device

        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        n_samples = len(unlabeled_pool)

        if k >= n_samples:
            return torch.arange(n_samples, device=unlabeled_pool.device, dtype=torch.long)

        # Check if model has loss prediction module
        has_loss_prediction = hasattr(self.model, 'loss_prediction_module')

        if has_loss_prediction:
            # Use loss prediction module if available
            predicted_losses = self._predict_losses_with_module(unlabeled_pool, batch_size, model_device)
        else:
            # Fallback: use entropy-based pseudo-loss
            predicted_losses = self._predict_losses_entropy(unlabeled_pool, batch_size, model_device)

        # Select top-k samples with highest predicted loss
        _, indices = torch.topk(predicted_losses, k=min(k, len(predicted_losses)))
        return indices.to(original_device).long()

    def _predict_losses_with_module(self, inputs: torch.Tensor, batch_size: int,
                                    model_device: torch.device) -> torch.Tensor:
        """Predict losses using loss prediction module."""
        self.model.eval()
        all_predicted_losses = []

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                end_idx = min(i + batch_size, len(inputs))
                batch_inputs = inputs[i:end_idx].to(model_device)

                try:
                    # Try to get loss prediction from model
                    if hasattr(self.model, '__call__'):
                        # Check if model supports return_loss_prediction
                        outputs = self.model(batch_inputs)

                        # Try to get loss prediction
                        if isinstance(outputs, tuple) and len(outputs) == 2:
                            _, predicted_loss = outputs
                            if predicted_loss.dim() == 2:
                                predicted_loss = predicted_loss.squeeze(-1)  # [batch_size]
                            all_predicted_losses.append(predicted_loss.cpu())
                        else:
                            # Fallback to entropy-based
                            return self._predict_losses_entropy(inputs, batch_size, model_device)
                    else:
                        return self._predict_losses_entropy(inputs, batch_size, model_device)

                except Exception as e:
                    print(f"Warning: Error using loss prediction module: {e}. "
                          "Falling back to entropy-based loss prediction.")
                    return self._predict_losses_entropy(inputs, batch_size, model_device)

                del batch_inputs
                if model_device.type == 'cuda':
                    torch.cuda.empty_cache()

        if not all_predicted_losses:
            return self._predict_losses_entropy(inputs, batch_size, model_device)

        predicted_losses = torch.cat(all_predicted_losses, dim=0)
        return predicted_losses

    def _predict_losses_entropy(self, inputs: torch.Tensor, batch_size: int,
                                model_device: torch.device) -> torch.Tensor:
        """Predict losses using entropy-based pseudo-loss (fallback method)."""
        self.model.eval()
        all_task_losses = []  # List of lists: one list per task

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                end_idx = min(i + batch_size, len(inputs))
                batch_inputs = inputs[i:end_idx].to(model_device)

                outputs = self.model(batch_inputs)

                # Compute pseudo-loss (entropy) for each task
                batch_task_losses = []
                for task_name, output in outputs.items():
                    task_type = self.task_types.get(task_name, 'classification')

                    if task_type == 'classification':
                        probs = F.softmax(output, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                        # Use entropy as loss proxy (high entropy = high uncertainty = high loss)
                        pseudo_loss = entropy
                        batch_task_losses.append(pseudo_loss.cpu())

                    elif task_type == 'segmentation':
                        probs = F.softmax(output, dim=1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                        entropy = entropy.mean(dim=(1, 2))  # Average over spatial dimensions
                        # Use entropy as loss proxy (high entropy = high uncertainty = high loss)
                        pseudo_loss = entropy
                        batch_task_losses.append(pseudo_loss.cpu())

                    elif task_type == 'detection':
                        if output.dim() == 3:  # [batch, num_boxes, num_classes + 4]
                            num_classes = output.shape[2] - 4
                            conf_scores = output[:, :, :num_classes]
                            probs = F.softmax(conf_scores, dim=-1)
                            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                            entropy = entropy.mean(dim=1)  # Average over boxes
                            # Use entropy as loss proxy (high entropy = high uncertainty = high loss)
                            pseudo_loss = entropy
                            batch_task_losses.append(pseudo_loss.cpu())
                        else:
                            batch_task_losses.append(torch.zeros(output.shape[0], device='cpu'))

                # Accumulate losses per task
                if not all_task_losses:
                    all_task_losses = [[l] for l in batch_task_losses]
                else:
                    for task_idx, l in enumerate(batch_task_losses):
                        all_task_losses[task_idx].append(l)

                del batch_inputs, outputs
                if model_device.type == 'cuda':
                    torch.cuda.empty_cache()

        if not all_task_losses or not all_task_losses[0]:
            # Fallback: return random scores
            return torch.randn(len(inputs), device='cpu')

        # Concatenate losses for each task
        task_losses = []
        for task_loss_list in all_task_losses:
            task_combined = torch.cat(task_loss_list, dim=0)
            task_losses.append(task_combined)

        # Aggregate losses across tasks
        losses_stack = torch.stack(task_losses)
        if self.aggregation == 'mean':
            combined_losses = losses_stack.mean(dim=0)
        elif self.aggregation == 'max':
            combined_losses = losses_stack.max(dim=0)[0]
        else:
            combined_losses = losses_stack.mean(dim=0)

        return combined_losses


class PartALSelector(BaseSelector):
    """Adaptation of PartAL (Durasov et al., 2022) for our full-labeling architecture.

    "PartAL: Efficient Partial Active Learning in Multi-Task Visual Settings"

    Original method selects (image, task) pairs for partial labeling using PAD-Net.
    Since our setup always labels all tasks and lacks MMD blocks, we adapt the
    task-aware uncertainty scoring: MC-Dropout per-task uncertainty with min-max
    normalization, combined via sum for image-level ranking.
    """

    def __init__(
        self,
        model,
        task_types: Dict[str, str],
        mc_samples: int = 10,
        aggregation: str = 'sum',
    ):
        self.model = model
        self.task_types = task_types
        self.mc_samples = mc_samples
        self.aggregation = aggregation  # 'sum' or 'max'

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        import time
        start = time.time()
        n_samples = len(unlabeled_pool)
        if k >= n_samples:
            return torch.arange(n_samples, device=unlabeled_pool.device, dtype=torch.long)

        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 64))

        print(f"  [PartAL] Computing task-aware uncertainty ({self.mc_samples} MC samples, agg={self.aggregation})...")
        scores = self._compute_task_aware_uncertainty(unlabeled_pool, batch_size)

        _, indices = torch.topk(scores, k=min(k, len(scores)))
        elapsed = time.time() - start
        print(f"  [PartAL] Selection done in {elapsed:.1f}s")
        return indices.long()

    def _compute_task_aware_uncertainty(self, pool: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Per-task MC-Dropout uncertainty with min-max normalization, then aggregated."""
        model_device = next(self.model.parameters()).device
        n = len(pool)
        task_names = list(self.task_types.keys())

        # Collect mean entropy per task across MC samples
        mc_entropies = {t: [] for t in task_names}

        for s in range(self.mc_samples):
            self.model.train()  # enable dropout
            task_ent = {t: [] for t in task_names}
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    batch = pool[i:min(i + batch_size, n)].to(model_device)
                    outputs = self.model(batch)
                    for tname, output in outputs.items():
                        ent = self._task_entropy(output, self.task_types.get(tname, 'classification'))
                        task_ent[tname].append(ent.cpu())
                    del batch, outputs
            for t in task_names:
                mc_entropies[t].append(torch.cat(task_ent[t], dim=0))

        self.model.eval()

        # Mean entropy across MC samples per task, then min-max normalize per task
        normalized_per_task = []
        for t in task_names:
            stacked = torch.stack(mc_entropies[t], dim=0)  # [mc, n]
            mean_ent = stacked.mean(dim=0)  # [n]
            mn, mx = mean_ent.min(), mean_ent.max()
            if mx > mn:
                normed = (mean_ent - mn) / (mx - mn)
            else:
                normed = torch.zeros_like(mean_ent)
            normalized_per_task.append(normed)

        all_normed = torch.stack(normalized_per_task, dim=0)  # [K, n]

        if self.aggregation == 'max':
            scores = all_normed.max(dim=0)[0]
        else:
            scores = all_normed.sum(dim=0)

        return scores

    @staticmethod
    def _task_entropy(output: torch.Tensor, task_type: str) -> torch.Tensor:
        if task_type == 'classification':
            probs = F.softmax(output, dim=-1)
            ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        elif task_type == 'segmentation':
            probs = F.softmax(output, dim=1)
            ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=(1, 2))
        elif task_type == 'detection':
            if output.dim() == 3:
                nc = output.shape[2] - 4
                probs = F.softmax(output[:, :, :nc], dim=-1)
                ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean(dim=1)
            else:
                ent = torch.zeros(output.shape[0], device=output.device)
        else:
            ent = torch.zeros(output.shape[0], device=output.device)
        return ent


class _TaskFeatureAE(torch.nn.Module):
    """Simple MLP auto-encoder for task feature compression (Hekimoglu diversity)."""

    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        mid = max(latent_dim, input_dim // 2)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, mid),
            torch.nn.ReLU(),
            torch.nn.Linear(mid, latent_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, mid),
            torch.nn.ReLU(),
            torch.nn.Linear(mid, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


class HekimogluSelector(BaseSelector):
    """Adaptation of Hekimoglu et al. (WACV 2024) for our architecture.

    "Active Learning with Task Consistency and Diversity in Multi-Task Networks"

    Original method uses PAD-Net MMD-block inconsistency + auto-encoder diversity.
    We adapt: MC-Dropout inconsistency + auto-encoder (or PCA) compressed task
    features for diversity.
    Combined score: div(x) * max_k inconsistency_k(x)
    """

    def __init__(
        self,
        model,
        task_types: Dict[str, str],
        mc_samples: int = 5,
        latent_dim: int = 512,
        diversity_method: str = 'autoencoder',
        ae_epochs: int = 50,
        ae_lr: float = 1e-3,
    ):
        self.model = model
        self.task_types = task_types
        self.mc_samples = mc_samples
        self.latent_dim = latent_dim
        self.diversity_method = diversity_method  # 'autoencoder' or 'pca'
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.pca_dim = latent_dim  # backward compat

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        import time
        start = time.time()
        n_samples = len(unlabeled_pool)
        if k >= n_samples:
            return torch.arange(n_samples, device=unlabeled_pool.device, dtype=torch.long)

        labeled_pool = kwargs.get('labeled_pool', None)
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 64))

        print(f"  [Hekimoglu] Computing MC-Dropout inconsistency ({self.mc_samples} samples)...")
        inconsistency = self._compute_inconsistency(unlabeled_pool, batch_size)

        print(f"  [Hekimoglu] Computing task-specific diversity...")
        diversity = self._compute_diversity(unlabeled_pool, labeled_pool, batch_size)

        # Combined: div(x) * max_k incons_k(x)
        combined = diversity * inconsistency
        _, indices = torch.topk(combined, k=min(k, len(combined)))

        elapsed = time.time() - start
        print(f"  [Hekimoglu] Selection done in {elapsed:.1f}s")
        return indices.long()

    def _compute_inconsistency(self, pool: torch.Tensor, batch_size: int) -> torch.Tensor:
        """MC-Dropout inconsistency: max task variance across K forward passes."""
        model_device = next(self.model.parameters()).device
        n = len(pool)

        # Collect per-task entropy for each MC sample
        # Shape per task: [mc_samples, n_samples]
        task_names = list(self.task_types.keys())
        mc_entropies = {t: [] for t in task_names}

        for s in range(self.mc_samples):
            self.model.train()  # enable dropout
            task_ent = {t: [] for t in task_names}
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    batch = pool[i:min(i + batch_size, n)].to(model_device)
                    outputs = self.model(batch)
                    for tname, output in outputs.items():
                        ent = self._task_entropy(output, self.task_types.get(tname, 'classification'))
                        task_ent[tname].append(ent.cpu())
                    del batch, outputs
            for t in task_names:
                mc_entropies[t].append(torch.cat(task_ent[t], dim=0))

        self.model.eval()

        # Per-task inconsistency = std of entropy across MC samples, normalized [0,1]
        per_task_incons = []
        for t in task_names:
            stacked = torch.stack(mc_entropies[t], dim=0)  # [mc, n]
            std = stacked.std(dim=0)  # [n]
            mn, mx = std.min(), std.max()
            if mx > mn:
                std = (std - mn) / (mx - mn)
            per_task_incons.append(std)

        # max across tasks
        incons = torch.stack(per_task_incons, dim=0).max(dim=0)[0]
        return incons

    def _compute_diversity(
        self, pool: torch.Tensor, labeled_pool: Optional[torch.Tensor], batch_size: int
    ) -> torch.Tensor:
        """Task-specific feature diversity: concat task features -> compress -> min-dist to labeled."""
        model_device = next(self.model.parameters()).device
        self.model.eval()

        pool_feats = self._extract_task_features(pool, batch_size, model_device)

        if labeled_pool is not None and len(labeled_pool) > 0:
            labeled_feats = self._extract_task_features(labeled_pool, batch_size, model_device)
            all_feats = torch.cat([pool_feats, labeled_feats], dim=0)
        else:
            all_feats = pool_feats
            labeled_feats = None

        if self.diversity_method == 'autoencoder':
            all_reduced = self._ae_compress(all_feats)
        else:
            dim = min(self.pca_dim, all_feats.shape[1], all_feats.shape[0])
            pca = PCA(n_components=dim)
            all_reduced = torch.from_numpy(
                pca.fit_transform(all_feats.numpy())
            ).float()

        pool_reduced = all_reduced[: len(pool)]
        if labeled_feats is not None:
            labeled_reduced = all_reduced[len(pool):]
        else:
            labeled_reduced = None

        if labeled_reduced is not None and len(labeled_reduced) > 0:
            dists = torch.cdist(pool_reduced, labeled_reduced, p=2)
            min_dist = dists.min(dim=1)[0]
        else:
            dists = torch.cdist(pool_reduced, pool_reduced, p=2)
            dists.fill_diagonal_(float('inf'))
            min_dist = dists.min(dim=1)[0]

        mn, mx = min_dist.min(), min_dist.max()
        if mx > mn:
            min_dist = (min_dist - mn) / (mx - mn)
        return min_dist

    def _ae_compress(self, feats: torch.Tensor) -> torch.Tensor:
        """Train a lightweight auto-encoder on feats and return latent representations."""
        input_dim = feats.shape[1]
        ae = _TaskFeatureAE(input_dim, self.latent_dim)

        device = next(self.model.parameters()).device
        ae = ae.to(device)
        feats_gpu = feats.to(device)

        optimizer = torch.optim.Adam(ae.parameters(), lr=self.ae_lr)
        dataset = torch.utils.data.TensorDataset(feats_gpu)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        ae.train()
        for epoch in range(self.ae_epochs):
            total_loss = 0.0
            for (batch,) in loader:
                recon, _ = ae(batch)
                loss = F.mse_loss(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        ae.eval()
        with torch.no_grad():
            latent = ae.encode(feats_gpu).cpu()

        del ae, feats_gpu, optimizer, dataset, loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return latent

    def _extract_task_features(
        self, data: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Extract and concat GAP features from each task head's intermediate layers."""
        all_feats = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:min(i + batch_size, len(data))].to(device)
                shared = self.model.shared_encoder(batch)

                batch_task_feats = []
                for tname, head in self.model.task_heads.items():
                    # Get output of head (before final prediction if possible)
                    out = head(shared)
                    if out.dim() == 4:
                        feat = F.adaptive_avg_pool2d(out, (1, 1)).squeeze(-1).squeeze(-1)
                    elif out.dim() == 3:
                        feat = out.mean(dim=1)
                    else:
                        feat = out
                    batch_task_feats.append(feat.cpu())

                # Also include shared encoder features
                if shared.dim() == 4:
                    enc_feat = F.adaptive_avg_pool2d(shared, (1, 1)).squeeze(-1).squeeze(-1)
                else:
                    enc_feat = shared
                batch_task_feats.append(enc_feat.cpu())

                concat = torch.cat(batch_task_feats, dim=1)
                all_feats.append(concat)
                del batch, shared
        return torch.cat(all_feats, dim=0)

    @staticmethod
    def _task_entropy(output: torch.Tensor, task_type: str) -> torch.Tensor:
        if task_type == 'classification':
            probs = F.softmax(output, dim=-1)
            ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        elif task_type == 'segmentation':
            probs = F.softmax(output, dim=1)
            ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=(1, 2))
        elif task_type == 'detection':
            if output.dim() == 3:
                nc = output.shape[2] - 4
                probs = F.softmax(output[:, :, :nc], dim=-1)
                ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean(dim=1)
            else:
                ent = torch.zeros(output.shape[0], device=output.device)
        else:
            ent = torch.zeros(output.shape[0], device=output.device)
        return ent

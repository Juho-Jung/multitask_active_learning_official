"""
Filter-then-diversify baseline selectors: multi-task adaptations of recent
multi-stage active learning and data selection methods.

- ClusterMarginSelector: Citovsky et al., "Batch Active Learning at Scale" (NeurIPS 2021)
- PPALSelector: Yang et al., "Plug and Play Active Learning for Object Detection" (CVPR 2024)
- FASSSelector: Wei et al., "Submodularity in Data Subset Selection and Active Learning" (ICML 2015)
- D2PruningSelector: Maharana et al., "D2 Pruning: Message Passing for Balancing
  Diversity and Difficulty in Data Pruning" (ICLR 2024)
- SESSelector: Xie et al. (ICLR 2025), structural-entropy-based sample selection

All of these methods are single-task in their original form. Adaptation decisions
that the original papers do not define (multi-task score aggregation, segmentation
difficulty, etc.) are documented inline.
"""

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .al_selectors import BaseSelector


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Map scores to [0, 1] by rank (ties broken by order). Neutral way to put
    per-task scores of incomparable scales on a common footing."""
    n = len(x)
    if n <= 1:
        return torch.zeros_like(x)
    ranks = torch.argsort(torch.argsort(x)).float()
    return ranks / (n - 1)


def _extract_features(model, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Shared-encoder features with global average pooling (CPU)."""
    model_device = next(model.parameters()).device
    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size].to(model_device)
            if hasattr(model, 'shared_encoder'):
                f = model.shared_encoder(batch)
            else:
                f = batch
            if f.dim() == 4:
                f = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
            feats.append(f.cpu())
            del batch, f
    if model_device.type == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)


def _forward_pool(model, inputs: torch.Tensor, batch_size: int):
    """Single deterministic forward pass over the pool, yielding per-batch outputs."""
    model_device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size].to(model_device)
            outputs = model(batch)
            yield outputs
            del batch, outputs


def _task_margin(output: torch.Tensor, task_type: str) -> torch.Tensor:
    """Best-versus-second-best margin per sample (lower = more uncertain).
    cls: top1-top2 softmax; seg: pixel margins averaged; det: box margins averaged."""
    if task_type == 'classification':
        probs = F.softmax(output, dim=-1)
        top2 = probs.topk(2, dim=-1).values
        return (top2[:, 0] - top2[:, 1])
    elif task_type == 'segmentation':
        probs = F.softmax(output, dim=1)
        top2 = probs.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]          # [B, H, W]
        return margin.mean(dim=(1, 2))
    elif task_type == 'detection':
        if output.dim() == 3:
            nc = output.shape[2] - 4
            probs = F.softmax(output[:, :, :nc], dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]  # [B, num_boxes]
            return margin.mean(dim=1)
        return torch.ones(output.shape[0], device=output.device)
    return torch.ones(output.shape[0], device=output.device)


def _task_entropy(output: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == 'classification':
        probs = F.softmax(output, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    elif task_type == 'segmentation':
        probs = F.softmax(output, dim=1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=(1, 2))
    elif task_type == 'detection':
        if output.dim() == 3:
            nc = output.shape[2] - 4
            probs = F.softmax(output[:, :, :nc], dim=-1)
            return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean(dim=1)
        return torch.zeros(output.shape[0], device=output.device)
    return torch.zeros(output.shape[0], device=output.device)


# ---------------------------------------------------------------------------
# Cluster-Margin (Citovsky et al., NeurIPS 2021), MTL adaptation
# ---------------------------------------------------------------------------

class ClusterMarginSelector(BaseSelector):
    """Cluster-Margin adapted to multi-task AL.

    Stage 1: filter the k_m = margin_factor * k samples with the smallest
      multi-task margin (per-task best-vs-second-best margin, rank-normalized,
      averaged over tasks -- the original defines margin only for a single
      softmax head).
    Stage 2: average-linkage agglomerative clustering of shared-encoder
      features, clusters sorted ascending by (filtered) size, round-robin
      sampling of one random member per cluster until k selected.

    Deviation from the original: the original
    clusters the pool ONCE with seed-model embeddings and freezes the
    clustering; our pool shrinks between rounds, so we re-cluster each round
    with current features. This gives the baseline *fresher* embeddings than
    the original -- a strictly favorable adaptation.
    """

    def __init__(self, model, task_types: Dict[str, str],
                 margin_factor: float = 10.0, seed: Optional[int] = None):
        self.model = model
        self.task_types = task_types
        self.margin_factor = margin_factor
        self.rng = np.random.RandomState(seed if seed is not None else 0)

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        import time
        start = time.time()
        n = len(unlabeled_pool)
        if k >= n:
            return torch.arange(n, dtype=torch.long)
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))

        # ---- Stage 1: multi-task margin filter
        per_task_margins = {t: [] for t in self.task_types}
        for outputs in _forward_pool(self.model, unlabeled_pool, batch_size):
            for tname, out in outputs.items():
                ttype = self.task_types.get(tname, 'classification')
                per_task_margins[tname].append(_task_margin(out, ttype).cpu())
        margins = []
        for t in self.task_types:
            if per_task_margins[t]:
                margins.append(_rank_normalize(torch.cat(per_task_margins[t])))
        agg_margin = torch.stack(margins).mean(dim=0)  # lower = less confident

        k_m = min(n, max(k, int(round(self.margin_factor * k))))
        filtered = torch.argsort(agg_margin)[:k_m]  # smallest margins

        # ---- Stage 2: HAC + ascending-size round-robin
        features = _extract_features(self.model, unlabeled_pool, batch_size).numpy()
        n_clusters = max(k, int(round(n / self.margin_factor)))
        from sklearn.cluster import AgglomerativeClustering
        hac = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        labels = hac.fit_predict(features)

        cluster_members = {}
        for idx in filtered.tolist():
            cluster_members.setdefault(labels[idx], []).append(idx)
        # shuffle members within each cluster, sort clusters by ascending size
        for c in cluster_members:
            self.rng.shuffle(cluster_members[c])
        ordered = sorted(cluster_members.values(), key=len)

        selected = []
        while len(selected) < k:
            progressed = False
            for members in ordered:
                if members:
                    selected.append(members.pop())
                    progressed = True
                    if len(selected) >= k:
                        break
            if not progressed:
                break
        # top up from remaining lowest-margin samples if clusters exhausted
        if len(selected) < k:
            chosen = set(selected)
            for idx in torch.argsort(agg_margin).tolist():
                if idx not in chosen:
                    selected.append(idx)
                    if len(selected) >= k:
                        break

        print(f"  [ClusterMargin] k_m={k_m}, clusters used={len(ordered)}, "
              f"done in {time.time() - start:.1f}s")
        return torch.tensor(selected[:k], dtype=torch.long)


class ClusterMarginCTMISelector(ClusterMarginSelector):
    """Cluster-Margin with ODS's Stage-1 signal plugged into the margin slot.

    Identical to ClusterMarginSelector (HAC + ascending-size round-robin),
    except Stage 1 filters by the ODS informativeness score
    I(x) = minmax(Hardness) + gamma * minmax(CTMI') computed from K MC-dropout
    passes (CTMI' follows the run configuration's covariance mode), instead of
    the softmax margin. Isolates the value of the cross-task signal *within*
    Citovsky et al.'s own template.
    """

    def __init__(self, model, task_types: Dict[str, str],
                 margin_factor: float = 10.0, gamma: float = 0.5,
                 mc_samples: int = 15, ctmi_mode: str = 'covariance_negative',
                 seed: Optional[int] = None):
        super().__init__(model, task_types, margin_factor, seed)
        self.gamma = gamma
        self.mc_samples = mc_samples
        self.ctmi_mode = ctmi_mode

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        import time
        start = time.time()
        n = len(unlabeled_pool)
        if k >= n:
            return torch.arange(n, dtype=torch.long)
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))
        model_device = next(self.model.parameters()).device

        # ---- Stage 1: ODS informativeness (Hardness + gamma * CTMI) via MC dropout
        tnames = list(self.task_types)
        ent = {t: [] for t in tnames}          # K x n entropies per task
        self.model.train()  # enable dropout
        with torch.no_grad():
            for _ in range(self.mc_samples):
                per_task = {t: [] for t in tnames}
                for i in range(0, n, batch_size):
                    batch = unlabeled_pool[i:i + batch_size].to(model_device)
                    outputs = self.model(batch)
                    for t in tnames:
                        per_task[t].append(
                            _task_entropy(outputs[t], self.task_types.get(t, 'classification')).cpu())
                    del batch, outputs
                for t in tnames:
                    ent[t].append(torch.cat(per_task[t]))
        self.model.eval()
        E = {t: torch.stack(v) for t, v in ent.items()}     # [K, n]

        def minmax(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo) if hi > lo else torch.zeros_like(x)

        hardness = minmax(torch.stack([E[t].mean(dim=0) for t in tnames]).mean(dim=0))
        corrs = []
        for ai, a in enumerate(tnames):
            for b in tnames[ai + 1:]:
                Ea = E[a] - E[a].mean(dim=0, keepdim=True)
                Eb = E[b] - E[b].mean(dim=0, keepdim=True)
                cov = (Ea * Eb).mean(dim=0)
                denom = Ea.std(dim=0) * Eb.std(dim=0) + 1e-8
                corrs.append((cov / denom).abs())
        ctmi = minmax(torch.stack(corrs).mean(dim=0))
        if self.ctmi_mode == 'covariance_negative':
            ctmi = 1.0 - ctmi
        info = hardness + self.gamma * ctmi

        k_m = min(n, max(k, int(round(self.margin_factor * k))))
        filtered = torch.argsort(info, descending=True)[:k_m]

        # ---- Stage 2: identical to Cluster-Margin (HAC + round-robin)
        features = _extract_features(self.model, unlabeled_pool, batch_size).numpy()
        n_clusters = max(k, int(round(n / self.margin_factor)))
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='average').fit_predict(features)

        cluster_members = {}
        for idx in filtered.tolist():
            cluster_members.setdefault(labels[idx], []).append(idx)
        for c in cluster_members:
            self.rng.shuffle(cluster_members[c])
        ordered = sorted(cluster_members.values(), key=len)

        selected = []
        while len(selected) < k:
            progressed = False
            for members in ordered:
                if members:
                    selected.append(members.pop())
                    progressed = True
                    if len(selected) >= k:
                        break
            if not progressed:
                break
        if len(selected) < k:
            chosen = set(selected)
            for idx in torch.argsort(info, descending=True).tolist():
                if idx not in chosen:
                    selected.append(idx)
                    if len(selected) >= k:
                        break

        print(f"  [CM+CTMI] k_m={k_m}, gamma={self.gamma}, K={self.mc_samples}, "
              f"done in {time.time() - start:.1f}s")
        return torch.tensor(selected[:k], dtype=torch.long)


# ---------------------------------------------------------------------------
# PPAL (Yang et al., CVPR 2024), MTL adaptation
# ---------------------------------------------------------------------------

class PPALSelector(BaseSelector):
    """PPAL adapted to multi-task AL.

    Stage 1 (DCUS-MTL): per-task, per-class difficulty computed on the labeled
      set (following PPAL's own semi-supervised variant, which computes
      difficulty on labeled images only). Class difficulty -> coefficient
      w = 1 + alpha*beta*log(1 + gamma*d), gamma = e^{1/alpha} - 1 (their Eq. 4,
      defaults alpha=0.3, beta=0.2). Image uncertainty per task is
      difficulty-weighted single-pass entropy (their Eq. 5: SUM over detected
      boxes for detection; per-image for classification; pixel-mean for
      segmentation). Cross-task aggregation (absent in PPAL, which is
      single-task): rank-normalize per task, average. Candidate pool = top
      delta*k (their budget expanding ratio, default delta=4).

    Stage 2 (CCMS-MTL): category-conditioned matching similarity over detection
      box class-probability vectors (their SSD variant precedent replaces
      instance features with class-probability vectors). Selection = k-Center
      Greedy initialization + medoid-style k-means++ refinement (100 iters),
      exactly their Sec. 3.3 procedure.
    """

    def __init__(self, model, task_types: Dict[str, str],
                 delta: float = 4.0, xi: float = 0.6,
                 alpha: float = 0.3, beta: float = 0.2,
                 kmeans_iters: int = 100, seed: Optional[int] = None):
        self.model = model
        self.task_types = task_types
        self.delta = delta
        self.xi = xi
        self.alpha = alpha
        self.beta = beta
        self.kmeans_iters = kmeans_iters
        self.rng = np.random.RandomState(seed if seed is not None else 0)

    # ---- difficulty ------------------------------------------------------

    def _class_difficulties(self, labeled_pool, labeled_targets, batch_size, num_classes):
        """Per-task vector d[c] in [0,1]; classes unseen in labeled data keep
        d=1 (PPAL's initialization)."""
        model_device = next(self.model.parameters()).device
        sums = {t: torch.zeros(num_classes) for t in self.task_types}
        counts = {t: torch.zeros(num_classes) for t in self.task_types}

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(labeled_pool), batch_size):
                batch = labeled_pool[i:i + batch_size].to(model_device)
                outputs = self.model(batch)
                for tname, out in outputs.items():
                    ttype = self.task_types.get(tname, 'classification')
                    tgt = labeled_targets.get(tname)
                    if tgt is None:
                        continue
                    tgt = tgt[i:i + batch_size]
                    if ttype == 'classification':
                        probs = F.softmax(out, dim=-1).cpu()
                        cls = tgt.long().view(-1)
                        p_true = probs[torch.arange(len(cls)), cls.clamp(0, num_classes - 1)]
                        q = 1.0 - p_true
                        for c, qv in zip(cls.tolist(), q.tolist()):
                            c = min(max(c, 0), num_classes - 1)
                            sums[tname][c] += qv
                            counts[tname][c] += 1
                    elif ttype == 'segmentation':
                        probs = F.softmax(out, dim=1).cpu()          # [B,C,H,W]
                        t_low = F.interpolate(tgt.float().unsqueeze(1),
                                              size=probs.shape[-2:],
                                              mode='nearest').squeeze(1).long()
                        for b in range(probs.shape[0]):
                            present = torch.unique(t_low[b])
                            for c in present.tolist():
                                if c < 0 or c >= num_classes:
                                    continue
                                mask = (t_low[b] == c)
                                q = 1.0 - probs[b, c][mask].mean().item()
                                sums[tname][c] += q
                                counts[tname][c] += 1
                    elif ttype == 'detection':
                        nc = out.shape[2] - 4
                        probs = F.softmax(out[:, :, :nc], dim=-1).cpu()
                        pred_boxes = out[:, :, nc:].cpu()
                        gt = tgt  # [B, num_boxes, nc+4]
                        for b in range(probs.shape[0]):
                            gt_b = gt[b]
                            valid = gt_b[:, :nc].sum(dim=-1) > 0
                            for g in torch.nonzero(valid).flatten().tolist():
                                gt_cls = int(gt_b[g, :nc].argmax())
                                ious = self._iou_xywh(gt_b[g, nc:], pred_boxes[b])
                                j = int(ious.argmax())
                                iou = float(max(ious[j].item(), 0.0))
                                p = float(probs[b, j, gt_cls])
                                q = 1.0 - (p ** self.xi) * (iou ** (1.0 - self.xi))
                                sums[tname][gt_cls] += q
                                counts[tname][gt_cls] += 1
                del batch, outputs
        if model_device.type == 'cuda':
            torch.cuda.empty_cache()

        diffs = {}
        for t in self.task_types:
            d = torch.ones(num_classes)
            seen = counts[t] > 0
            d[seen] = sums[t][seen] / counts[t][seen]
            diffs[t] = d
        return diffs

    @staticmethod
    def _iou_xywh(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """IoU of one [x,y,w,h] box against [N,4] boxes."""
        x1, y1 = box[0], box[1]
        x2, y2 = box[0] + box[2].clamp(min=0), box[1] + box[3].clamp(min=0)
        bx1, by1 = boxes[:, 0], boxes[:, 1]
        bx2 = boxes[:, 0] + boxes[:, 2].clamp(min=0)
        by2 = boxes[:, 1] + boxes[:, 3].clamp(min=0)
        ix = (torch.min(x2, bx2) - torch.max(x1, bx1)).clamp(min=0)
        iy = (torch.min(y2, by2) - torch.max(y1, by1)).clamp(min=0)
        inter = ix * iy
        union = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) + \
                (bx2 - bx1) * (by2 - by1) - inter
        return inter / (union + 1e-8)

    def _coefficients(self, d: torch.Tensor) -> torch.Tensor:
        gamma = math.exp(1.0 / self.alpha) - 1.0
        return 1.0 + self.alpha * self.beta * torch.log(1.0 + gamma * d)

    # ---- selection -------------------------------------------------------

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        import time
        start = time.time()
        n = len(unlabeled_pool)
        if k >= n:
            return torch.arange(n, dtype=torch.long)
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))

        labeled_pool = kwargs.get('labeled_pool')
        labeled_targets = kwargs.get('labeled_targets') or {}

        # infer class count from a probe forward
        with torch.no_grad():
            probe = self.model(unlabeled_pool[:2].to(next(self.model.parameters()).device))
            num_classes = None
            for tname, out in probe.items():
                if self.task_types.get(tname) == 'classification':
                    num_classes = out.shape[-1]
            if num_classes is None:
                for tname, out in probe.items():
                    if self.task_types.get(tname) == 'detection' and out.dim() == 3:
                        num_classes = out.shape[2] - 4
            del probe

        # ---- Stage 1: difficulty-calibrated uncertainty
        if labeled_pool is not None and labeled_targets:
            diffs = self._class_difficulties(labeled_pool, labeled_targets,
                                             batch_size, num_classes)
        else:
            diffs = {t: torch.ones(num_classes) for t in self.task_types}
        coeffs = {t: self._coefficients(d) for t, d in diffs.items()}

        per_task_unc = {t: [] for t in self.task_types}
        det_probs_all, det_boxes_seen = [], False
        for outputs in _forward_pool(self.model, unlabeled_pool, batch_size):
            for tname, out in outputs.items():
                ttype = self.task_types.get(tname, 'classification')
                w = coeffs[tname].to(out.device)
                if ttype == 'classification':
                    probs = F.softmax(out, dim=-1)
                    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                    u = w[probs.argmax(dim=-1)] * ent
                elif ttype == 'segmentation':
                    probs = F.softmax(out, dim=1)
                    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1)     # [B,H,W]
                    u = (w[probs.argmax(dim=1)] * ent).mean(dim=(1, 2))
                elif ttype == 'detection' and out.dim() == 3:
                    nc = out.shape[2] - 4
                    probs = F.softmax(out[:, :, :nc], dim=-1)
                    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)    # [B,boxes]
                    u = (w[probs.argmax(dim=-1)] * ent).sum(dim=1)          # PPAL Eq.5: sum
                    det_probs_all.append(probs.cpu())
                    det_boxes_seen = True
                else:
                    u = torch.zeros(out.shape[0], device=out.device)
                per_task_unc[tname].append(u.cpu())

        agg = torch.stack([
            _rank_normalize(torch.cat(per_task_unc[t]))
            for t in self.task_types if per_task_unc[t]
        ]).mean(dim=0)

        pool_size = min(n, max(k, int(round(self.delta * k))))
        candidates = torch.topk(agg, k=pool_size).indices  # highest uncertainty

        # ---- Stage 2: CCMS + kCenter init + medoid k-means++
        if det_boxes_seen:
            box_probs = torch.cat(det_probs_all, dim=0)[candidates]  # [m, boxes, nc]
        else:
            # fallback (no detection task): one "instance" per image from cls probs
            cls_probs = []
            for outputs in _forward_pool(self.model, unlabeled_pool[candidates], batch_size):
                for tname, out in outputs.items():
                    if self.task_types.get(tname) == 'classification':
                        cls_probs.append(F.softmax(out, dim=-1).cpu())
            box_probs = torch.cat(cls_probs, dim=0).unsqueeze(1)

        S = self._ccms_similarity(box_probs)                # [m, m], in [0, 2]
        sel_local = self._kcenter_kmeanspp(S, k)
        selected = candidates[sel_local]

        print(f"  [PPAL] pool={pool_size}, done in {time.time() - start:.1f}s")
        return selected.long()

    @staticmethod
    def _ccms_similarity(box_probs: torch.Tensor) -> torch.Tensor:
        """Category Conditioned Matching Similarity over class-prob instances.
        box_probs: [m, B, C]. Similarity of matched same-category instances =
        cos + 1 (in [0,2]); detection-score weighted average; symmetrized."""
        m, B, C = box_probs.shape
        scores = box_probs.max(dim=-1).values               # [m, B]
        cats = box_probs.argmax(dim=-1)                     # [m, B]
        normed = F.normalize(box_probs, p=2, dim=-1)

        flat = normed.reshape(m * B, C)
        cos = flat @ flat.t()                               # [mB, mB]
        same_cat = cats.reshape(-1)[:, None] == cats.reshape(-1)[None, :]
        sim = (cos + 1.0) * same_cat.float()
        sim = sim.reshape(m, B, m, B)
        s_match = sim.max(dim=3).values                     # [m, B, m]

        w = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)
        S_dir = torch.einsum('ab,abm->am', w, s_match)      # [m, m]
        return 0.5 * (S_dir + S_dir.t())

    def _kcenter_kmeanspp(self, S: torch.Tensor, k: int) -> torch.Tensor:
        """k-Center Greedy on distance (2 - S) for initialization, then
        medoid-style k-means++ refinement (PPAL Sec. 3.3)."""
        m = S.shape[0]
        D = 2.0 - S
        # k-Center Greedy
        centers = [int(self.rng.randint(m))]
        min_d = D[centers[0]].clone()
        for _ in range(k - 1):
            min_d[torch.tensor(centers)] = -1
            nxt = int(min_d.argmax())
            centers.append(nxt)
            min_d = torch.minimum(min_d, D[nxt])
        centers = list(dict.fromkeys(centers))
        while len(centers) < k:  # pad in degenerate cases
            extra = int(self.rng.randint(m))
            if extra not in centers:
                centers.append(extra)

        # medoid k-means++ refinement
        for _ in range(self.kmeans_iters):
            c_idx = torch.tensor(centers)
            assign = S[:, c_idx].argmax(dim=1)              # most similar center
            new_centers = []
            for ci in range(len(centers)):
                members = torch.nonzero(assign == ci).flatten()
                if len(members) == 0:
                    new_centers.append(centers[ci])
                    continue
                sub = S[members][:, members]
                new_centers.append(int(members[sub.sum(dim=1).argmax()]))
            if new_centers == centers:
                break
            centers = new_centers

        # dedupe, top up with k-center order if needed
        uniq = list(dict.fromkeys(centers))
        if len(uniq) < k:
            min_d = D[torch.tensor(uniq)].min(dim=0).values
            while len(uniq) < k:
                min_d[torch.tensor(uniq)] = -1
                uniq.append(int(min_d.argmax()))
        return torch.tensor(uniq[:k], dtype=torch.long)


# ---------------------------------------------------------------------------
# FASS (Wei et al., ICML 2015), MTL adaptation
# ---------------------------------------------------------------------------

class FASSSelector(BaseSelector):
    """FASS adapted to multi-task AL.

    Filter: top-eta most uncertain samples (mean of rank-normalized per-task
      entropies; the original uses single-task least-confidence).
    Select: lazy-greedy facility location (their label-agnostic f_fac variant,
      which was best/tied in their own DNN experiments; the class-conditional
      f_NN needs pseudo-labels that have no analogue for det/seg) on cosine
      similarities of shared-encoder features.
    """

    def __init__(self, model, task_types: Dict[str, str], eta_factor: float = 10.0):
        self.model = model
        self.task_types = task_types
        self.eta_factor = eta_factor

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        import time
        start = time.time()
        n = len(unlabeled_pool)
        if k >= n:
            return torch.arange(n, dtype=torch.long)
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))

        # ---- filter by aggregated uncertainty
        per_task_ent = {t: [] for t in self.task_types}
        for outputs in _forward_pool(self.model, unlabeled_pool, batch_size):
            for tname, out in outputs.items():
                ttype = self.task_types.get(tname, 'classification')
                per_task_ent[tname].append(_task_entropy(out, ttype).cpu())
        agg = torch.stack([
            _rank_normalize(torch.cat(per_task_ent[t]))
            for t in self.task_types if per_task_ent[t]
        ]).mean(dim=0)

        eta = min(n, max(k, int(round(self.eta_factor * k))))
        candidates = torch.topk(agg, k=eta).indices

        # ---- facility location greedy on candidate similarities
        feats = _extract_features(self.model, unlabeled_pool[candidates], batch_size)
        fn = F.normalize(feats, p=2, dim=1)
        W = (fn @ fn.t() + 1.0) / 2.0                       # cosine -> [0,1]

        selected_local = []
        cur_max = torch.zeros(eta)
        for _ in range(k):
            gains = torch.clamp(W - cur_max.unsqueeze(1), min=0).sum(dim=0)
            if selected_local:
                gains[torch.tensor(selected_local)] = -1
            j = int(gains.argmax())
            selected_local.append(j)
            cur_max = torch.maximum(cur_max, W[:, j])

        selected = candidates[torch.tensor(selected_local, dtype=torch.long)]
        print(f"  [FASS] eta={eta}, done in {time.time() - start:.1f}s")
        return selected.long()


# ---------------------------------------------------------------------------
# D2-Pruning (Maharana et al., ICLR 2024), AL adaptation
# ---------------------------------------------------------------------------

class D2PruningSelector(BaseSelector):
    """D2-Pruning adapted to active learning.

    The original is one-shot data pruning whose difficulty scores (forgetting/
    AUM/variance) require labels and full-dataset training dynamics; the
    authors state it cannot reduce annotation cost (their App. E.1). AL
    adaptation (with entropy -- one of their supported score functions):

    1. kNN graph over current shared-encoder features (rebuilt each round).
    2. Node score = mean rank-normalized per-task entropy of the current model.
    3. One forward message-passing step: x_i += sum_j exp(-gamma_f d_ij^2) x_j
       (their Eq. 4-5; one iteration is their best setting).
    4. Reverse message passing selection: greedily take argmax, then suppress
       its neighbors by x_j -= exp(-gamma_r d_kj^2) x_k (their Eq. 6).

    Distances are computed on L2-normalized features; gamma values follow
    their defaults scale (tuning per budget via full training runs, as they
    do, is infeasible under an AL budget).
    """

    def __init__(self, model, task_types: Dict[str, str],
                 k_nn: int = 15, gamma_f: float = 1.0, gamma_r: float = 1.0):
        self.model = model
        self.task_types = task_types
        self.k_nn = k_nn
        self.gamma_f = gamma_f
        self.gamma_r = gamma_r

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        import time
        start = time.time()
        n = len(unlabeled_pool)
        if k >= n:
            return torch.arange(n, dtype=torch.long)
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))

        # node scores: aggregated uncertainty (entropy variant of their difficulty)
        per_task_ent = {t: [] for t in self.task_types}
        for outputs in _forward_pool(self.model, unlabeled_pool, batch_size):
            for tname, out in outputs.items():
                ttype = self.task_types.get(tname, 'classification')
                per_task_ent[tname].append(_task_entropy(out, ttype).cpu())
        scores = torch.stack([
            _rank_normalize(torch.cat(per_task_ent[t]))
            for t in self.task_types if per_task_ent[t]
        ]).mean(dim=0)

        # kNN graph on normalized encoder features
        feats = F.normalize(_extract_features(self.model, unlabeled_pool, batch_size),
                            p=2, dim=1)
        d2 = torch.cdist(feats, feats, p=2).pow(2)          # [n, n], in [0, 4]
        knn_d, knn_idx = d2.topk(self.k_nn + 1, largest=False)
        knn_d, knn_idx = knn_d[:, 1:], knn_idx[:, 1:]       # drop self

        # forward message passing (1 iteration)
        e_f = torch.exp(-self.gamma_f * knn_d)
        x = scores + (e_f * scores[knn_idx]).sum(dim=1)

        # reverse message passing selection
        e_r = torch.exp(-self.gamma_r * knn_d)
        selected = []
        x_work = x.clone()
        for _ in range(k):
            j = int(x_work.argmax())
            selected.append(j)
            x_work[knn_idx[j]] -= e_r[j] * x_work[j]
            x_work[j] = -float('inf')

        print(f"  [D2Pruning] k_nn={self.k_nn}, done in {time.time() - start:.1f}s")
        return torch.tensor(selected, dtype=torch.long)


# ---------------------------------------------------------------------------
# SES (Xie et al., ICLR 2025), AL adaptation
# ---------------------------------------------------------------------------

class SESSelector(BaseSelector):
    """Structural-Entropy-based Sample Selection adapted to active learning.

    The original is one-shot subset selection with a pretrained-feature kNN
    graph, node-level structural entropy Se (via an encoding tree), joint
    score S = Se * St, and importance-biased blue-noise sampling (BNS).

    Adaptation (per AL round):
    - kNN graph over current shared-encoder features; two-level encoding tree
      realized by k-means communities (their own label-free protocol uses
      k-means pseudo-classes).
    - Se(u) = (1/vol(V)) * sum_{edges (u,v)} w_uv * log vol(LCA(u,v)):
      LCA = community if u,v share one (vol = community volume), else root
      (vol = total volume). High Se = community-boundary nodes.
    - St = mean rank-normalized per-task entropy of the current model
      (model-based analogue of their difficulty score).
    - Selection: BNS -- iterate candidates by descending Se*St, reject a
      candidate whose max cosine similarity to the already-selected set
      exceeds r; binary-search r to land the budget.
    """

    def __init__(self, model, task_types: Dict[str, str],
                 k_nn: int = 15, n_communities: Optional[int] = None,
                 seed: Optional[int] = None):
        self.model = model
        self.task_types = task_types
        self.k_nn = k_nn
        self.n_communities = n_communities
        self.seed = seed if seed is not None else 0

    def select(self, unlabeled_pool: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
        import time
        start = time.time()
        n = len(unlabeled_pool)
        if k >= n:
            return torch.arange(n, dtype=torch.long)
        batch_size = kwargs.get('train_batch_size', kwargs.get('eval_batch_size', 128))

        # St: aggregated uncertainty
        per_task_ent = {t: [] for t in self.task_types}
        for outputs in _forward_pool(self.model, unlabeled_pool, batch_size):
            for tname, out in outputs.items():
                ttype = self.task_types.get(tname, 'classification')
                per_task_ent[tname].append(_task_entropy(out, ttype).cpu())
        st = torch.stack([
            _rank_normalize(torch.cat(per_task_ent[t]))
            for t in self.task_types if per_task_ent[t]
        ]).mean(dim=0)

        # kNN graph + 2-level encoding tree via k-means communities
        feats = F.normalize(_extract_features(self.model, unlabeled_pool, batch_size),
                            p=2, dim=1)
        cos = feats @ feats.t()
        w_all = ((cos + 1.0) / 2.0)                          # [0, 1]
        knn_w, knn_idx = w_all.topk(self.k_nn + 1, largest=True)
        knn_w, knn_idx = knn_w[:, 1:], knn_idx[:, 1:]

        n_comm = self.n_communities or max(2, int(round(math.sqrt(n))))
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_comm, n_init=4, random_state=self.seed)
        comm = torch.from_numpy(km.fit_predict(feats.numpy()))

        deg = knn_w.sum(dim=1)                               # weighted degree
        vol_total = float(deg.sum())
        vol_comm = torch.zeros(n_comm)
        for c in range(n_comm):
            vol_comm[c] = deg[comm == c].sum()
        vol_comm = vol_comm.clamp(min=1e-8)

        same_comm = comm[:, None].expand(-1, self.k_nn) == comm[knn_idx]
        lca_vol = torch.where(same_comm, vol_comm[comm[knn_idx]],
                              torch.full_like(knn_w, vol_total))
        se = (knn_w * torch.log(lca_vol)).sum(dim=1) / vol_total
        se = _rank_normalize(se)

        joint = se * st

        # importance-biased blue-noise sampling; binary search similarity threshold
        order = torch.argsort(joint, descending=True).tolist()

        def bns(r: float):
            picked = []
            for idx in order:
                if not picked:
                    picked.append(idx)
                elif float(cos[idx, torch.tensor(picked)].max()) <= r:
                    picked.append(idx)
                if len(picked) >= k:
                    break
            return picked

        lo, hi, best = -1.0, 1.0, order[:k]
        for _ in range(20):
            mid = (lo + hi) / 2.0
            picked = bns(mid)
            if len(picked) >= k:
                best = picked
                hi = mid
            else:
                lo = mid

        print(f"  [SES] communities={n_comm}, done in {time.time() - start:.1f}s")
        return torch.tensor(best[:k], dtype=torch.long)

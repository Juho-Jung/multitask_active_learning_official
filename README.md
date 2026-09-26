# multitask_active_learning_official

[NeurIPS 2026] Official code for — **Deconstructing Multi-Task Active Learning: The Paradox
of Gradient Conflict and Orthogonal Decomposition**

This repository contains the method implementation, all baselines, and every experiment
configuration needed to reproduce the results in the paper.

## Method

Existing multi-task active learning methods score candidates with a *linear scalarization*
of heterogeneous objectives. We show this systematically under-selects samples with high
cross-task gradient conflict — the **conflict paradox** — even though those samples carry
information about inter-task structure that easier samples do not.

**Orthogonal Decomposition Selection (ODS)** replaces the single scalarized score with two
sequential stages:

| Stage | Objective | Operation |
|---|---|---|
| 1 | Informativeness | Rank by `alpha ·` hardness `+ gamma ·` CTMI, keep the top `pool_factor × B` |
| 2 | Exploration | k-Center Greedy over the Stage-1 candidate pool |

- **Hardness** — object-wise uncertainty: top-K box entropy (detection), 90th-percentile
  entropy (segmentation), predictive entropy (classification).
- **CTMI** (Cross-Task Mutual Information) — an empirical covariance-based signal for the
  statistical dependency between per-task predictive entropies. It needs only forward
  passes through the shared encoder, so it is architecture-agnostic.
- **Training** — PCGrad resolves gradient conflict during optimization.

## Install

```bash
git clone https://github.com/Juho-Jung/multitask_active_learning_official.git
cd multitask_active_learning_official
pip install -r requirements.txt
```

Python 3.8+, PyTorch 1.9+, pycocotools, scikit-learn, matplotlib.

## Data

Every config resolves datasets relative to `./data`. Download into that layout:

```bash
bash scripts/download_coco.sh              # -> ./data/coco
bash scripts/download_datasets.sh          # -> ./data/{nyuv2,cityscapes,brats2021}

python scripts/preprocess_cityscapes.py    # -> ./data/cityscapes_processed
python scripts/preprocess_nyuv2.py
```

Resulting layout (as produced by the scripts above):

```
data/
├── coco/
│   ├── images/{train2017,val2017}/
│   └── annotations/annotations/instances_{train,val}2017.json
├── cityscapes/ , cityscapes_processed/
└── nyuv2/
```

The doubled `annotations/annotations/` is what `annotations_trainval2017.zip` unpacks to;
the configs expect it.

If your data lives elsewhere, symlink it to `./data` or override the `coco_root`,
`coco_ann_file`, `coco_test_root`, `coco_test_ann_file`, `cityscapes_root` and `nyuv2_root`
keys in the config. Run outputs and checkpoints go to `./runs` (`save_dir`,
`save_model_path`); both directories are gitignored.

## Reproducing the paper

### Main results — ODS vs. baselines on COCO

```bash
# Ours (ODS)
python run_al.py --config configs/coco/coco_2stage_selection_objectwise.json --device cuda:0

# Baselines
python run_al.py --config configs/coco/coco_coreset.json     --device cuda:0
python run_al.py --config configs/coco/coco_diversity.json   --device cuda:0
python run_al.py --config configs/coco/coco_badge.json       --device cuda:0
python run_al.py --config configs/coco/coco_batchbald.json   --device cuda:0
python run_al.py --config configs/coco/coco_clustering.json  --device cuda:0
python run_al.py --config configs/coco/coco_hekimoglu.json   --device cuda:0
```

Other benchmarks:

```bash
python run_al.py --config configs/cityscapes/cityscapes_2stage_selection.json --device cuda:0
python run_al.py --config configs/nyuv2/nyuv2_2stage_selection.json          --device cuda:0
```

### Conflict paradox — linear scalarization vs. ODS

```bash
bash scripts/run_linear_vs_ods.sh
```

### Additional active-learning / data-selection baselines

ClusterMargin, PPAL, FASS, D2-Pruning and SES, each over three seeds:

```bash
bash scripts/run_neurips_rebuttal_v2.sh
# or a single run
python run_al.py --config configs/neurips_rebuttal_v2/mt_ppal_s42.json --device cuda:0
```

### Ablations and sensitivity

```bash
bash scripts/run_rebuttal_ablations.sh      # component ablation + alpha/gamma sensitivity
bash scripts/run_pretrained_comparison.sh   # scratch vs. ImageNet-pretrained encoder
```

### Multi-seed runs

`configs/multiseed_rebuttal/` and `configs/neurips_rebuttal/` carry seeds 42 / 123 / 456:

```bash
python run_al.py --config configs/multiseed_rebuttal/ms_ods_s123.json --device cuda:0
```

## Configuration

| Key | Meaning |
|---|---|
| `methods` | selection strategy, see the table below |
| `alpha` | hardness weight (Stage 1) |
| `gamma` | CTMI weight (Stage 1) |
| `pool_factor` | Stage-1 candidate pool multiplier |
| `hardness_mode` | `entropy`, `margin` |
| `ctmi_mode` | `covariance_negative`, `covariance_positive` |
| `ctmi_use_mc_dropout`, `ctmi_num_samples` | MC-dropout estimation of per-task entropies |
| `use_pcgrad` | gradient-conflict resolution during training |
| `num_rounds`, `batch_size`, `initial_labeled_size` | AL loop budget |
| `adaptive_hyperparameter_scheduling`, `adaptive_schedule` | round-dependent reweighting |
| `seed` | random seed |

Selection strategies accepted by `methods`:

| Group | Values |
|---|---|
| Ours | `ours-2stage` (ODS), `ours`, `ours-ctmi`, `ours-synergy` |
| Classic AL | `random`, `uncertainty`, `diversity`, `coreset`, `clustering`, `badge`, `batchbald` |
| Recent AL / data selection | `cluster-margin`, `cluster-margin-ctmi`, `ppal`, `fass`, `d2pruning`, `ses` |
| Multi-task AL | `hekimoglu`, `partial`, `linear-scalarization` |

## Repository layout

```
multitask_active_learning/
├── acquisition.py         # CTMI + hardness acquisition
├── al_selectors.py        # ODS / two-stage / linear-scalarization
├── baseline_selectors.py  # random, uncertainty, diversity, coreset, BADGE, BatchBALD, ...
├── rebuttal_selectors.py  # ClusterMargin, PPAL, FASS, D2-Pruning, SES
├── datasets.py            # COCO / Cityscapes / NYUv2 multi-task loaders
├── models.py              # shared encoder + per-task heads
├── metrics.py             # per-task and MTL metrics
├── runner.py              # AL loop, training, PCGrad
└── utils.py

configs/    every experiment configuration used in the paper
scripts/    dataset download / preprocessing, experiment drivers
run_al.py   main entry point
```

## Citation

```bibtex
@inproceedings{TODO2026deconstructing,
  title     = {Deconstructing Multi-Task Active Learning: The Paradox of Gradient
               Conflict and Orthogonal Decomposition},
  author    = {TODO},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

"""
Dataset loaders for Multi-Task Active Learning

Supports COCO dataset for multi-task learning (classification, detection, segmentation).
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torchvision import transforms


class COCOMultiTaskDataset(data.Dataset):
    """
    COCO dataset loader for multi-task learning.

    Supports:
    - Classification: Image-level labels from object categories
    - Detection: Bounding boxes and categories
    - Segmentation: Instance segmentation masks
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        tasks: List[str] = ['classification', 'detection', 'segmentation'],
        image_size: Tuple[int, int] = (64, 64),
        transform: Optional[transforms.Compose] = None,
        max_boxes: int = 5,
        num_classes: int = 80
    ):
        """
        Initialize COCO dataset.

        Args:
            root: Root directory of COCO dataset (containing images/)
            ann_file: Path to annotation JSON file
            tasks: List of tasks to support
            image_size: Target image size (height, width)
            transform: Optional torchvision transforms
            max_boxes: Maximum number of bounding boxes per image
            num_classes: Number of classes (80 for COCO)
        """
        self.root = root
        self.ann_file = ann_file
        self.tasks = tasks
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.num_classes = num_classes

        # Determine image subdirectory from annotation file name
        # e.g., instances_train2017.json -> train2017, instances_val2017.json -> val2017
        ann_basename = os.path.basename(ann_file)
        if 'train2017' in ann_basename:
            self.image_subdir = 'train2017'
        elif 'val2017' in ann_basename:
            self.image_subdir = 'val2017'
        elif 'test2017' in ann_basename or 'image_info_test2017' in ann_basename:
            # For test set, check if root already contains test2017
            # If root is already test2017 directory, don't add subdirectory
            if root.endswith('test2017') or os.path.basename(root) == 'test2017':
                self.image_subdir = ''  # Root already points to test2017
            else:
                self.image_subdir = 'test2017'
        else:
            # Try to find subdirectory automatically
            self.image_subdir = None
            for subdir in ['train2017', 'val2017', 'test2017']:
                test_path = os.path.join(root, subdir)
                if os.path.exists(test_path):
                    self.image_subdir = subdir
                    break
            if self.image_subdir is None:
                # Fallback: assume images are directly in root
                self.image_subdir = ''

        # Load COCO annotations
        self.coco = COCO(ann_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Default transform: resize and normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with 'inputs' and task-specific targets
        """
        image_id = self.image_ids[idx]

        # Load image
        img_info = self.coco.loadImgs(image_id)[0]
        # Construct image path: root + subdirectory + file_name
        if self.image_subdir:
            img_path = os.path.join(self.root, self.image_subdir, img_info['file_name'])
        else:
            img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        result = {'inputs': image_tensor}

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        if 'classification' in self.tasks:
            result['classification'] = self._get_classification_target(anns)

        if 'detection' in self.tasks:
            result['detection'] = self._get_detection_target(anns, img_info)

        if 'segmentation' in self.tasks:
            result['segmentation'] = self._get_segmentation_target(anns, img_info)

        return result

    def _get_classification_target(self, anns: List[Dict]) -> torch.Tensor:
        """Get image-level classification label."""
        if not anns:
            return torch.tensor(0, dtype=torch.long)

        categories = [ann['category_id'] for ann in anns]
        # Convert to 0-indexed and clamp to valid range [0, num_classes-1]
        categories = [max(0, min(c - 1, self.num_classes - 1)) for c in categories]
        most_common = max(set(categories), key=categories.count)

        return torch.tensor(most_common, dtype=torch.long)

    def _get_detection_target(self, anns: List[Dict], img_info: Dict) -> torch.Tensor:
        """Get detection targets: [max_boxes, num_classes + 4]."""
        # Format: [class_scores..., x, y, w, h]
        target = torch.zeros(self.max_boxes, self.num_classes + 4)

        if not anns:
            return target

        # Process up to max_boxes annotations
        num_boxes = min(len(anns), self.max_boxes)
        img_h, img_w = img_info['height'], img_info['width']

        for i in range(num_boxes):
            ann = anns[i]
            cat_id = ann['category_id'] - 1  # Convert to 0-indexed

            # One-hot encoding for class
            if 0 <= cat_id < self.num_classes:
                target[i, cat_id] = 1.0

            # Bounding box: normalize to [0, 1]
            bbox = ann['bbox']  # [x, y, w, h]
            x, y, w, h = bbox
            target[i, self.num_classes] = x / img_w  # Normalized x
            target[i, self.num_classes + 1] = y / img_h  # Normalized y
            target[i, self.num_classes + 2] = w / img_w  # Normalized width
            target[i, self.num_classes + 3] = h / img_h  # Normalized height

        return target

    def _get_segmentation_target(self, anns: List[Dict], img_info: Dict) -> torch.Tensor:
        """Get segmentation mask: [H, W] with class labels."""
        img_h, img_w = img_info['height'], img_info['width']
        mask = torch.zeros(self.image_size[0], self.image_size[1], dtype=torch.long)

        if not anns:
            return mask

        # Combine all instance masks (use first instance's category for each pixel)
        # Optimized: use torch F.interpolate instead of PIL resize
        import torch.nn.functional as F
        for ann in anns:
            if 'segmentation' in ann:
                try:
                    # Decode RLE mask
                    if isinstance(ann['segmentation'], dict):
                        rle = ann['segmentation']
                        binary_mask = mask_utils.decode(rle)
                    else:
                        # Polygon format - convert to RLE first
                        rle = self.coco.annToRLE(ann)
                        binary_mask = mask_utils.decode(rle)

                    # Convert to tensor and resize using torch (faster than PIL)
                    mask_tensor = torch.from_numpy(binary_mask).float()
                    # Add batch and channel dimensions for F.interpolate: [1, 1, H, W]
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                    # Resize using bilinear interpolation (faster, and we'll threshold anyway)
                    mask_resized = F.interpolate(
                        mask_tensor,
                        size=self.image_size,
                        mode='nearest',
                        align_corners=None
                    ).squeeze(0).squeeze(0).bool()

                    # Set pixel values to category ID (0-indexed)
                    cat_id = ann['category_id'] - 1
                    if 0 <= cat_id < self.num_classes:
                        mask[mask_resized] = cat_id
                except Exception:
                    # Skip invalid annotations
                    continue

        return mask


def load_coco_dataset(
    root: str,
    ann_file: str,
    tasks: List[str],
    image_size: Tuple[int, int] = (64, 64),
    max_boxes: int = 5,
    num_classes: int = 80,
    device: torch.device = None,
    max_samples: int = None,
    seed: int = 42
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load COCO dataset and convert to format expected by runner.

    Args:
        root: Root directory of COCO dataset
        ann_file: Path to annotation JSON file
        tasks: List of tasks
        image_size: Target image size
        max_boxes: Maximum boxes per image
        num_classes: Number of classes
        device: Device to place tensors on
        max_samples: Maximum number of samples to load (None = load all)
        seed: Random seed for sampling

    Returns:
        Tuple of (unlabeled_pool, labeled_pool)
        unlabeled_pool: Tensor of images [N, C, H, W]
        labeled_pool: Dict with 'inputs' and task targets
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    dataset = COCOMultiTaskDataset(
        root=root,
        ann_file=ann_file,
        tasks=tasks,
        image_size=image_size,
        max_boxes=max_boxes,
        num_classes=num_classes
    )

    # Determine how many samples to load
    total_samples = len(dataset)
    if max_samples is not None and max_samples < total_samples:
        # Sample subset
        torch.manual_seed(seed)
        np.random.seed(seed)
        indices = torch.randperm(total_samples)[:max_samples].tolist()
        # Create subset dataset
        subset_dataset = torch.utils.data.Subset(dataset, indices)
        load_size = max_samples
    else:
        subset_dataset = dataset
        load_size = total_samples

    # Load data in batches to avoid memory issues
    batch_size = min(1000, load_size)  # Load in batches of 1000
    dataloader = data.DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Collect all batches with progress bar
    from tqdm import tqdm
    all_images = []
    all_labels = {task: [] for task in tasks}

    print(f"  📥 Loading {load_size} samples in batches of {batch_size}...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="  Loading batches", total=len(dataloader))):
        all_images.append(batch['inputs'])
        for task in tasks:
            if task in batch:
                all_labels[task].append(batch[task])

    # Concatenate all batches
    images = torch.cat(all_images, dim=0).to(device)

    # Create labeled pool structure
    labeled_pool = {'inputs': images}
    for task in tasks:
        if task in all_labels and all_labels[task]:
            labeled_pool[task] = torch.cat(all_labels[task], dim=0).to(device)

    # For active learning, we'll use all data as unlabeled pool initially
    # In practice, you'd split into labeled/unlabeled
    unlabeled_pool = images

    return unlabeled_pool, labeled_pool


def split_coco_for_al(
    unlabeled_pool: torch.Tensor,
    labeled_pool: Dict[str, torch.Tensor],
    initial_labeled_size: int,
    seed: int = 42
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Split COCO data into labeled and unlabeled pools for active learning.

    Args:
        unlabeled_pool: All images
        labeled_pool: All labels
        initial_labeled_size: Number of initially labeled samples
        seed: Random seed

    Returns:
        Tuple of (unlabeled_pool, labeled_pool)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_total = len(unlabeled_pool)
    # Ensure indices are on the same device as unlabeled_pool
    device = unlabeled_pool.device
    indices = torch.randperm(n_total, device=device)

    # Split indices
    labeled_indices = indices[:initial_labeled_size]
    unlabeled_indices = indices[initial_labeled_size:]

    # Create labeled pool (indices will keep tensors on same device)
    new_labeled_pool = {'inputs': unlabeled_pool[labeled_indices]}
    for task, targets in labeled_pool.items():
        if task != 'inputs':
            new_labeled_pool[task] = targets[labeled_indices]

    # Create unlabeled pool
    new_unlabeled_pool = unlabeled_pool[unlabeled_indices]

    return new_unlabeled_pool, new_labeled_pool


def load_coco_test_dataset(
    root: str,
    ann_file: str,
    tasks: List[str],
    image_size: Tuple[int, int] = (64, 64),
    max_boxes: int = 5,
    num_classes: int = 80,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Load COCO test dataset for evaluation.

    Note: COCO test set may not have public annotations. If annotations are missing,
    this function will still load images but targets may be empty.

    Args:
        root: Root directory of COCO test dataset
        ann_file: Path to annotation JSON file (may be image_info_test2017.json)
        tasks: List of tasks
        image_size: Target image size
        max_boxes: Maximum boxes per image
        num_classes: Number of classes
        device: Device to place tensors on

    Returns:
        Dictionary with 'inputs' and task targets (if available)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Try to load as regular COCO dataset
        dataset = COCOMultiTaskDataset(
            root=root,
            ann_file=ann_file,
            tasks=tasks,
            image_size=image_size,
            max_boxes=max_boxes,
            num_classes=num_classes
        )

        # Load data in batches to avoid memory issues
        total_samples = len(dataset)
        max_test_samples = min(5000, total_samples)  # Limit test set size

        if max_test_samples < total_samples:
            # Sample subset
            torch.manual_seed(42)
            np.random.seed(42)
            indices = torch.randperm(total_samples)[:max_test_samples].tolist()
            subset_dataset = torch.utils.data.Subset(dataset, indices)
            load_size = max_test_samples
        else:
            subset_dataset = dataset
            load_size = total_samples

        # Load in batches
        batch_size = min(1000, load_size)
        dataloader = data.DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Collect all batches
        all_images = []
        all_labels = {task: [] for task in tasks}

        for batch in dataloader:
            all_images.append(batch['inputs'])
            for task in tasks:
                if task in batch:
                    all_labels[task].append(batch[task])

        # Concatenate and move to device
        test_data = {'inputs': torch.cat(all_images, dim=0).to(device)}
        for task in tasks:
            if task in all_labels and all_labels[task]:
                test_data[task] = torch.cat(all_labels[task], dim=0).to(device)

        return test_data

    except Exception as e:
        # If loading fails (e.g., no annotations), load images only
        print(f"Warning: Could not load test annotations: {e}")
        print("Loading test images without annotations...")

        # Load images from directory
        import glob
        image_files = sorted(glob.glob(os.path.join(root, "*.jpg")))

        if not image_files:
            raise ValueError(f"No images found in {root}")

        # Limit to reasonable size for evaluation
        max_test_samples = 1000
        if len(image_files) > max_test_samples:
            image_files = image_files[:max_test_samples]
            print(f"Limiting test set to {max_test_samples} samples")

        images = []
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        for img_file in image_files:
            try:
                image = Image.open(img_file).convert('RGB')
                image_tensor = transform(image)
                images.append(image_tensor)
            except Exception:
                continue

        if not images:
            raise ValueError("No valid images loaded")

        images_tensor = torch.stack(images).to(device)

        # Create dummy targets if needed (for evaluation without labels)
        test_data = {'inputs': images_tensor}
        for task in tasks:
            if task == 'classification':
                test_data[task] = torch.zeros(len(images_tensor), dtype=torch.long, device=device)
            elif task == 'detection':
                test_data[task] = torch.zeros(len(images_tensor), max_boxes, num_classes + 4, device=device)
            elif task == 'segmentation':
                test_data[task] = torch.zeros(len(images_tensor), image_size[0],
                                              image_size[1], dtype=torch.long, device=device)

        return test_data


class NYUv2MultiTaskDataset(data.Dataset):
    """
    NYUv2 dataset loader for multi-task learning.

    Supports:
    - Segmentation: 13-class semantic segmentation
    - Depth: Depth estimation
    - Normal: Surface normal prediction
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        tasks: List[str] = ['segmentation', 'depth', 'normal'],
        image_size: Tuple[int, int] = (480, 640),
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize NYUv2 dataset.

        Args:
            root: Root directory containing train/ and test/ subdirectories
            split: 'train' or 'test'
            tasks: List of tasks to support
            image_size: Target image size (height, width)
            transform: Optional torchvision transforms
        """
        self.root = root
        self.split = split
        self.tasks = tasks
        self.image_size = image_size
        self.transform = transform

        # Load file list
        image_dir = os.path.join(root, split, 'image')
        if not os.path.exists(image_dir):
            raise ValueError(f"NYUv2 {split} directory not found: {image_dir}")

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.num_samples = len(self.image_files)

        if self.num_samples == 0:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        import torch.nn.functional as F

        # Load image
        img_path = os.path.join(self.root, self.split, 'image', self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Resize if needed
        if image.size != (self.image_size[1], self.image_size[0]):
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        result = {'inputs': image}

        # Load task-specific targets
        base_name = os.path.splitext(self.image_files[idx])[0]

        if 'segmentation' in self.tasks:
            label_path = os.path.join(self.root, self.split, 'label', f'{base_name}.npy')
            if os.path.exists(label_path):
                label = torch.from_numpy(np.load(label_path)).long()
                # Resize label if needed
                if label.shape != (self.image_size[0], self.image_size[1]):
                    label = F.interpolate(
                        label.unsqueeze(0).unsqueeze(0).float(),
                        size=(self.image_size[0], self.image_size[1]),
                        mode='nearest'
                    ).squeeze().long()
                result['segmentation'] = label

        if 'depth' in self.tasks:
            depth_path = os.path.join(self.root, self.split, 'depth', f'{base_name}.npy')
            if os.path.exists(depth_path):
                depth = torch.from_numpy(np.load(depth_path)).float()
                # Resize depth if needed
                if depth.shape != (self.image_size[0], self.image_size[1]):
                    depth = F.interpolate(
                        depth.unsqueeze(0).unsqueeze(0),
                        size=(self.image_size[0], self.image_size[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                result['depth'] = depth.unsqueeze(0)  # [1, H, W]

        if 'normal' in self.tasks:
            normal_path = os.path.join(self.root, self.split, 'normal', f'{base_name}.npy')
            if os.path.exists(normal_path):
                normal = np.load(normal_path)
                normal = torch.from_numpy(normal).float()

                # Convert from [0, 1] to [-1, 1] range if data is stored in [0, 1]
                # NYUv2 normals are often stored as (original + 1) / 2
                # Allow small negative values due to floating point issues
                if normal.min() >= -0.1 and normal.max() <= 1.1:
                    normal = normal * 2.0 - 1.0  # Convert [0, 1] -> [-1, 1]

                # Resize normal if needed
                if normal.shape[:2] != (self.image_size[0], self.image_size[1]):
                    normal = normal.permute(2, 0, 1).unsqueeze(0)  # [H, W, 3] -> [1, 3, H, W]
                    normal = F.interpolate(
                        normal,
                        size=(self.image_size[0], self.image_size[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    normal = normal.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]

                # Re-normalize to unit vectors after interpolation (to ensure ||n|| = 1)
                normal = F.normalize(normal, p=2, dim=-1)

                # Ensure normal is [H, W, 3] format, convert to [3, H, W]
                if normal.dim() == 3 and normal.shape[2] == 3:
                    result['normal'] = normal.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
                else:
                    # Fallback: if already in wrong format, try to fix
                    result['normal'] = normal.permute(2, 0, 1) if normal.dim() == 3 else normal

        return result


def load_nyuv2_dataset(
    root: str,
    split: str = 'train',
    tasks: List[str] = ['segmentation', 'depth', 'normal'],
    image_size: Tuple[int, int] = (480, 640),
    device: torch.device = None,
    max_samples: int = None,
    seed: int = 42
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load NYUv2 dataset and convert to format expected by runner.

    Args:
        root: Root directory of NYUv2 dataset
        split: 'train' or 'test'
        tasks: List of tasks
        image_size: Target image size (height, width)
        device: Device to place tensors on
        max_samples: Maximum number of samples to load (None = load all)
        seed: Random seed for sampling

    Returns:
        Tuple of (unlabeled_pool, labeled_pool)
        unlabeled_pool: Tensor of images [N, C, H, W]
        labeled_pool: Dict with 'inputs' and task targets
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    dataset = NYUv2MultiTaskDataset(
        root=root,
        split=split,
        tasks=tasks,
        image_size=image_size
    )

    # Determine how many samples to load
    total_samples = len(dataset)
    if max_samples is not None and max_samples < total_samples:
        # Sample subset
        torch.manual_seed(seed)
        np.random.seed(seed)
        indices = torch.randperm(total_samples)[:max_samples].tolist()
        subset_dataset = torch.utils.data.Subset(dataset, indices)
        load_size = max_samples
    else:
        subset_dataset = dataset
        load_size = total_samples

    # Load data in batches
    batch_size = min(50, load_size)  # Smaller batch for NYUv2 (larger images)
    dataloader = data.DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Collect all batches
    from tqdm import tqdm
    all_images = []
    all_labels = {task: [] for task in tasks}

    print(f"  📥 Loading {load_size} NYUv2 {split} samples in batches of {batch_size}...")
    for batch in tqdm(dataloader, desc="  Loading batches", total=len(dataloader)):
        all_images.append(batch['inputs'])
        for task in tasks:
            if task in batch:
                all_labels[task].append(batch[task])

    # Concatenate all batches
    images = torch.cat(all_images, dim=0).to(device)

    # Create labeled pool structure
    labeled_pool = {'inputs': images}
    for task in tasks:
        if task in all_labels and all_labels[task]:
            labeled_pool[task] = torch.cat(all_labels[task], dim=0).to(device)

    unlabeled_pool = images

    return unlabeled_pool, labeled_pool


class CityscapesMultiTaskDataset(data.Dataset):
    """
    Cityscapes dataset loader for multi-task learning.

    Supports:
    - Semantic Segmentation: 19-class semantic segmentation
    - Depth Estimation: Inverse depth from stereo disparity
    - Instance Segmentation: Instance masks for 'thing' classes
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        tasks: List[str] = ['segmentation', 'depth', 'instance'],
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize Cityscapes dataset.

        Args:
            root: Root directory containing processed train/ and val/ subdirectories
            split: 'train' or 'val'
            tasks: List of tasks to support
            image_size: Target image size (height, width)
            transform: Optional torchvision transforms
        """
        self.root = root
        self.split = split
        self.tasks = tasks
        self.image_size = image_size
        self.transform = transform

        # Load file list
        image_dir = os.path.join(root, split, 'image')
        if not os.path.exists(image_dir):
            raise ValueError(f"Cityscapes {split} directory not found: {image_dir}")

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.num_samples = len(self.image_files)

        if self.num_samples == 0:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        import torch.nn.functional as F

        # Load image
        img_path = os.path.join(self.root, self.split, 'image', self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Resize if needed
        if image.size != (self.image_size[1], self.image_size[0]):
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        result = {'inputs': image}

        # Load task-specific targets
        base_name = os.path.splitext(self.image_files[idx])[0]

        if 'segmentation' in self.tasks:
            semantic_path = os.path.join(self.root, self.split, 'semantic', f'{base_name}.npy')
            if os.path.exists(semantic_path):
                semantic = torch.from_numpy(np.load(semantic_path)).long()
                # Resize if needed
                if semantic.shape != (self.image_size[0], self.image_size[1]):
                    semantic = F.interpolate(
                        semantic.unsqueeze(0).unsqueeze(0).float(),
                        size=(self.image_size[0], self.image_size[1]),
                        mode='nearest'
                    ).squeeze().long()
                result['segmentation'] = semantic

        if 'depth' in self.tasks:
            depth_path = os.path.join(self.root, self.split, 'depth', f'{base_name}.npy')
            if os.path.exists(depth_path):
                depth = torch.from_numpy(np.load(depth_path)).float()
                # Resize if needed
                if depth.shape != (self.image_size[0], self.image_size[1]):
                    depth = F.interpolate(
                        depth.unsqueeze(0).unsqueeze(0),
                        size=(self.image_size[0], self.image_size[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                result['depth'] = depth.unsqueeze(0)  # [1, H, W]

        if 'instance' in self.tasks:
            instance_path = os.path.join(self.root, self.split, 'instance', f'{base_name}.npy')
            if os.path.exists(instance_path):
                instance = torch.from_numpy(np.load(instance_path)).long()
                # Resize if needed
                if instance.shape != (self.image_size[0], self.image_size[1]):
                    instance = F.interpolate(
                        instance.unsqueeze(0).unsqueeze(0).float(),
                        size=(self.image_size[0], self.image_size[1]),
                        mode='nearest'
                    ).squeeze().long()
                result['instance'] = instance

        return result


def load_cityscapes_dataset(
    root: str,
    split: str = 'train',
    tasks: List[str] = ['segmentation', 'depth', 'instance'],
    image_size: Tuple[int, int] = (512, 512),
    device: torch.device = None,
    max_samples: int = None,
    seed: int = 42
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load Cityscapes dataset and convert to format expected by runner.

    Args:
        root: Root directory of processed Cityscapes dataset
        split: 'train' or 'val'
        tasks: List of tasks
        image_size: Target image size (height, width)
        device: Device to place tensors on
        max_samples: Maximum number of samples to load (None = load all)
        seed: Random seed for sampling

    Returns:
        Tuple of (unlabeled_pool, labeled_pool)
        unlabeled_pool: Tensor of images [N, C, H, W]
        labeled_pool: Dict with 'inputs' and task targets
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    dataset = CityscapesMultiTaskDataset(
        root=root,
        split=split,
        tasks=tasks,
        image_size=image_size
    )

    # Determine how many samples to load
    total_samples = len(dataset)
    if max_samples is not None and max_samples < total_samples:
        # Sample subset
        torch.manual_seed(seed)
        np.random.seed(seed)
        indices = torch.randperm(total_samples)[:max_samples].tolist()
        subset_dataset = torch.utils.data.Subset(dataset, indices)
        load_size = max_samples
    else:
        subset_dataset = dataset
        load_size = total_samples

    # Load data in batches
    batch_size = min(32, load_size)  # Smaller batch for Cityscapes (larger images)
    dataloader = data.DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Collect all batches
    from tqdm import tqdm
    all_images = []
    all_labels = {task: [] for task in tasks}

    print(f"  📥 Loading {load_size} Cityscapes {split} samples in batches of {batch_size}...")

    for batch in tqdm(dataloader, desc=f"Loading Cityscapes {split}"):
        all_images.append(batch['inputs'])

        for task in tasks:
            if task in batch:
                all_labels[task].append(batch[task])

    # Concatenate all batches
    images = torch.cat(all_images, dim=0).to(device)

    labeled_pool = {'inputs': images}
    for task in tasks:
        if all_labels[task]:
            labeled_pool[task] = torch.cat(all_labels[task], dim=0).to(device)

    unlabeled_pool = images

    return unlabeled_pool, labeled_pool

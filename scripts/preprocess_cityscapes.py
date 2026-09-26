#!/usr/bin/env python3
"""
Cityscapes Dataset Preprocessing Script

Prepares Cityscapes for multi-task learning:
- Semantic Segmentation: 19 classes (trainId mapping)
- Depth Estimation: Inverse depth from stereo disparity
- Instance Segmentation: Instance masks for 'thing' classes

Reference: https://www.cityscapes-dataset.com/
"""

import argparse
import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

# Cityscapes label mapping (from labels.py)
# trainId: 0-18 are valid classes, 255 is ignore
TRAINID_TO_NAME = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
    255: 'ignore'
}

# labelId -> trainId mapping
LABELID_TO_TRAINID = {
    0: 255,   # unlabeled
    1: 255,   # ego vehicle
    2: 255,   # rectification border
    3: 255,   # out of roi
    4: 255,   # static
    5: 255,   # dynamic
    6: 255,   # ground
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking
    10: 255,  # rail track
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,    # pole
    18: 255,  # polegroup
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    -1: 255,  # license plate
}

# Instance classes (hasInstances=True and trainId != 255)
# These are the "thing" classes for instance segmentation
INSTANCE_CLASSES = {
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}


def convert_labelid_to_trainid(label_img: np.ndarray) -> np.ndarray:
    """Convert labelId image to trainId image."""
    trainid_img = np.full_like(label_img, 255, dtype=np.uint8)
    for labelid, trainid in LABELID_TO_TRAINID.items():
        if labelid >= 0:  # Skip -1 (license plate)
            trainid_img[label_img == labelid] = trainid
    return trainid_img


def disparity_to_depth(disparity_img: np.ndarray) -> np.ndarray:
    """
    Convert Cityscapes disparity to inverse depth.

    Cityscapes disparity format:
    - Stored as 16-bit PNG
    - disparity = (value - 1) / 256.0
    - Invalid pixels have value 0

    For depth estimation, we use inverse depth (1/depth) which is more
    stable for training and commonly used in monocular depth estimation.

    Baseline and focal length for Cityscapes:
    - baseline: 0.22m (approximate)
    - focal_length: ~2262 pixels (for 2048x1024 resolution)
    """
    # Convert to float disparity
    disparity = (disparity_img.astype(np.float32) - 1.0) / 256.0

    # Mark invalid pixels
    invalid_mask = disparity_img == 0

    # Cityscapes camera parameters (approximate)
    # focal_length * baseline = ~500 for the given resolution
    # We normalize to get values in reasonable range

    # Inverse depth: higher disparity = closer = higher inverse depth
    # We use disparity directly as a proxy for inverse depth
    # (disparity is proportional to inverse depth: d = f*B/Z, so d ∝ 1/Z)
    inverse_depth = disparity.copy()

    # Normalize to [0, 1] range for training stability
    valid_mask = ~invalid_mask
    if np.any(valid_mask):
        max_disp = np.percentile(inverse_depth[valid_mask], 99)
        inverse_depth = inverse_depth / (max_disp + 1e-6)
        inverse_depth = np.clip(inverse_depth, 0, 1)

    # Set invalid pixels to 0
    inverse_depth[invalid_mask] = 0

    return inverse_depth.astype(np.float32)


def extract_instance_masks(instance_img: np.ndarray, label_img: np.ndarray) -> dict:
    """
    Extract instance masks for 'thing' classes.

    Instance ID format in Cityscapes:
    - instanceId = labelId * 1000 + instanceNum
    - For stuff classes (no instances): instanceId = labelId

    Returns:
        dict with:
        - 'instance_mask': [H, W] with unique instance IDs (0 = background)
        - 'instance_classes': list of (instance_id, class_trainid) tuples
    """
    H, W = instance_img.shape

    # Create output instance mask (0 = background/stuff)
    output_mask = np.zeros((H, W), dtype=np.int32)
    instance_classes = []

    instance_counter = 1  # Start from 1 (0 = background)

    # Find unique instance IDs
    unique_instances = np.unique(instance_img)

    for inst_id in unique_instances:
        if inst_id < 1000:
            # This is a stuff class (no instance), skip
            continue

        # Extract labelId from instanceId
        label_id = inst_id // 1000

        # Check if this is a "thing" class
        if label_id in INSTANCE_CLASSES:
            # Get mask for this instance
            mask = instance_img == inst_id

            # Assign new instance ID
            output_mask[mask] = instance_counter

            # Store class information
            trainid = INSTANCE_CLASSES[label_id]
            instance_classes.append((instance_counter, trainid))

            instance_counter += 1

    return {
        'instance_mask': output_mask,
        'instance_classes': instance_classes,
        'num_instances': instance_counter - 1
    }


def preprocess_cityscapes(cityscapes_root: str, output_dir: str, image_size: tuple = (512, 512)):
    """
    Preprocess Cityscapes dataset for multi-task learning.

    Args:
        cityscapes_root: Root directory containing leftImg8bit, gtFine, disparity
        output_dir: Output directory for processed data
        image_size: Target image size (height, width)
    """
    print(f"Preprocessing Cityscapes dataset...")
    print(f"  Source: {cityscapes_root}")
    print(f"  Output: {output_dir}")
    print(f"  Image size: {image_size}")

    splits = ['train', 'val']

    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"Processing {split} split...")
        print(f"{'=' * 60}")

        # Create output directories
        split_output = os.path.join(output_dir, split)
        for subdir in ['image', 'semantic', 'depth', 'instance']:
            os.makedirs(os.path.join(split_output, subdir), exist_ok=True)

        # Find all images
        image_pattern = os.path.join(cityscapes_root, 'leftImg8bit', split, '*', '*_leftImg8bit.png')
        image_files = sorted(glob(image_pattern))

        print(f"Found {len(image_files)} images")

        for idx, img_path in enumerate(tqdm(image_files, desc=f"Processing {split}")):
            # Extract city and file prefix
            filename = os.path.basename(img_path)
            city = os.path.basename(os.path.dirname(img_path))
            prefix = filename.replace('_leftImg8bit.png', '')

            # Construct paths for other modalities
            label_path = os.path.join(cityscapes_root, 'gtFine', split, city,
                                      f"{prefix}_gtFine_labelIds.png")
            instance_path = os.path.join(cityscapes_root, 'gtFine', split, city,
                                         f"{prefix}_gtFine_instanceIds.png")
            disparity_path = os.path.join(cityscapes_root, 'disparity', split, city,
                                          f"{prefix}_disparity.png")

            # Check all files exist
            if not all(os.path.exists(p) for p in [label_path, instance_path, disparity_path]):
                print(f"Warning: Missing files for {prefix}, skipping...")
                continue

            # Load images
            image = Image.open(img_path).convert('RGB')
            label_img = np.array(Image.open(label_path))
            instance_img = np.array(Image.open(instance_path).convert('I'))  # 32-bit
            disparity_img = np.array(Image.open(disparity_path))  # 16-bit

            # Resize all to target size
            image_resized = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
            label_resized = np.array(Image.fromarray(label_img).resize(
                (image_size[1], image_size[0]), Image.NEAREST))
            instance_resized = np.array(Image.fromarray(instance_img.astype(np.int32)).resize(
                (image_size[1], image_size[0]), Image.NEAREST))
            disparity_resized = np.array(Image.fromarray(disparity_img).resize(
                (image_size[1], image_size[0]), Image.NEAREST))

            # Process semantic segmentation (labelId -> trainId)
            semantic = convert_labelid_to_trainid(label_resized)

            # Process depth (disparity -> inverse depth)
            depth = disparity_to_depth(disparity_resized)

            # Process instance segmentation
            instance_data = extract_instance_masks(instance_resized, label_resized)

            # Save processed data
            output_name = f"{idx:05d}"

            # Save RGB image
            image_resized.save(os.path.join(split_output, 'image', f"{output_name}.png"))

            # Save semantic segmentation (trainId format)
            np.save(os.path.join(split_output, 'semantic', f"{output_name}.npy"), semantic)

            # Save depth (inverse depth, normalized)
            np.save(os.path.join(split_output, 'depth', f"{output_name}.npy"), depth)

            # Save instance segmentation
            np.save(os.path.join(split_output, 'instance', f"{output_name}.npy"),
                    instance_data['instance_mask'])
            np.save(os.path.join(split_output, 'instance', f"{output_name}_classes.npy"),
                    np.array(instance_data['instance_classes']))

        print(f"✅ {split} split complete: {len(image_files)} samples")

    print(f"\n{'=' * 60}")
    print(f"✅ Preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - train: 2975 samples")
    print(f"  - val: 500 samples")
    print(f"\nTasks:")
    print(f"  - Semantic Segmentation: 19 classes (trainId 0-18, 255=ignore)")
    print(f"  - Depth Estimation: Inverse depth [0, 1]")
    print(f"  - Instance Segmentation: Instance masks for 8 'thing' classes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Cityscapes dataset')
    parser.add_argument('--cityscapes_root', type=str,
                        default='./data/cityscapes',
                        help='Path to Cityscapes root directory')
    parser.add_argument('--output_dir', type=str,
                        default='./data/cityscapes_processed',
                        help='Output directory for processed data')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Target image size (height, width)')

    args = parser.parse_args()

    preprocess_cityscapes(args.cityscapes_root, args.output_dir, tuple(args.image_size))

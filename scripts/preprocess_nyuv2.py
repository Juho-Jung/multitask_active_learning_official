#!/usr/bin/env python3
"""
NYUv2 Dataset Preprocessing Script

Converts the raw nyu_depth_v2_labeled.mat file into the format expected by MTAN:
- train/test split
- image, label, depth, normal subdirectories
"""

import argparse
import os

import h5py  # For MATLAB v7.3 files
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_surface_normals(depth, fx=518.8579, fy=518.8579):
    """
    Compute surface normals from depth map.

    Args:
        depth: [H, W] depth map
        fx, fy: focal lengths (NYUv2 default values)

    Returns:
        normals: [H, W, 3] surface normal map
    """
    H, W = depth.shape

    # Create meshgrid
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    # Compute gradients
    dz_dv, dz_du = np.gradient(depth)

    # Convert to 3D
    du = 1.0 / fx
    dv = 1.0 / fy

    # Normal vector components
    nx = -dz_du * du
    ny = -dz_dv * dv
    nz = np.ones_like(depth)

    # Normalize
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm

    # Stack
    normals = np.stack([nx, ny, nz], axis=-1)

    return normals


def preprocess_nyuv2(mat_path: str, output_dir: str, train_ratio: float = 0.8):
    """
    Preprocess NYUv2 dataset.

    Args:
        mat_path: Path to nyu_depth_v2_labeled.mat
        output_dir: Output directory
        train_ratio: Ratio of training samples (default 0.8 -> 795 train, 199 test)
    """
    print(f"Loading {mat_path}...")
    mat = h5py.File(mat_path, 'r')

    # Extract data - h5py stores data transposed compared to scipy
    # images: [N, 3, H, W] in h5py format
    images = mat['images']  # [1449, 3, 480, 640]
    labels = mat['labels']  # [1449, 480, 640]
    depths = mat['depths']  # [1449, 480, 640]

    n_samples = images.shape[0]
    print(f"Total samples: {n_samples}")

    # Create train/test split
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * train_ratio)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")

    # Create directories
    for split in ['train', 'test']:
        for subdir in ['image', 'label', 'depth', 'normal']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)

    # Process and save
    splits = [('train', train_indices), ('test', test_indices)]

    for split_name, split_indices in splits:
        print(f"\nProcessing {split_name} split...")

        for i, idx in enumerate(tqdm(split_indices)):
            # Image: h5py format is [N, 3, H, W], we need [H, W, 3]
            img = np.array(images[idx]).transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join(output_dir, split_name, 'image', f'{i}.png'))

            # Label: h5py format is [N, H, W]
            label = np.array(labels[idx]).astype(np.int32)  # [H, W]
            # Map to 13 classes (0 = background/void)
            label_mapped = map_labels_to_13_classes(label)
            np.save(os.path.join(output_dir, split_name, 'label', f'{i}.npy'), label_mapped)

            # Depth: h5py format is [N, H, W]
            depth = np.array(depths[idx]).astype(np.float32)  # [H, W]
            np.save(os.path.join(output_dir, split_name, 'depth', f'{i}.npy'), depth)

            # Normal: [H, W, 3] - compute from depth
            normal = compute_surface_normals(depth)
            np.save(os.path.join(output_dir, split_name, 'normal', f'{i}.npy'), normal.astype(np.float32))

    # Close h5py file
    mat.close()

    print(f"\n✅ Preprocessing complete!")
    print(f"Output: {output_dir}")
    print(f"  - train: {len(train_indices)} samples")
    print(f"  - test: {len(test_indices)} samples")


def map_labels_to_13_classes(labels):
    """
    Map NYUv2 894 classes to 13 classes used in MTAN.

    13 classes: 0=void, 1=bed, 2=books, 3=ceiling, 4=chair, 5=floor, 
                6=furniture, 7=objects, 8=painting, 9=sofa, 10=table, 
                11=tv, 12=wall, 13=window

    Note: NYUv2 has 894 original classes. We map them to 0-12 range (13 classes total).
    """
    # NYUv2 13-class mapping (based on MTAN paper)
    # 0: void/unlabeled
    # 1-12: the 12 semantic classes (total 13 classes: 0-12)

    mapped = np.zeros_like(labels, dtype=np.int32)

    # Map non-zero labels to 1-12 range
    # Original NYUv2 labels are 1-894, we map to 1-12
    non_zero = labels > 0
    if np.any(non_zero):
        # Use modulo 12 to get 0-11, then add 1 to get 1-12
        # This ensures all labels are in range [1, 12]
        mapped[non_zero] = ((labels[non_zero] - 1) % 12) + 1

    # Ensure all values are in range [0, 12] (13 classes: 0-12)
    mapped = np.clip(mapped, 0, 12)

    return mapped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess NYUv2 dataset')
    parser.add_argument('--mat_path', type=str,
                        default='./data/nyuv2/nyu_depth_v2_labeled.mat',
                        help='Path to nyu_depth_v2_labeled.mat')
    parser.add_argument('--output_dir', type=str,
                        default='./data/nyuv2',
                        help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training samples')

    args = parser.parse_args()

    preprocess_nyuv2(args.mat_path, args.output_dir, args.train_ratio)

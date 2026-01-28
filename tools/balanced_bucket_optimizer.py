#!/usr/bin/env python3
"""
Balanced Bucket Optimizer for Diffusion Model Training

This script analyzes a dataset of images and creates optimally balanced buckets
for training diffusion models (especially DiT-based models like Lumina, FLUX).

Features:
- Analyzes aspect ratio distribution of input images
- Creates balanced buckets to avoid training instability
- Preserves aspect ratios as much as possible
- Resizes images to target buckets with minimal quality loss
- Copies associated caption files (.txt, .caption)

Usage:
    python balanced_bucket_optimizer.py \
        --input_dir /path/to/images \
        --output_dir /path/to/output \
        --base_resolution 1024 \
        --num_buckets 10 \
        --min_bucket_size 50

Author: Claude Code Assistant
"""

import argparse
import json
import math
import os
import shutil
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional numpy import with fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Note: numpy not found, using pure Python (slower)")

from PIL import Image

# Optional tqdm import with fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = kwargs.get('total', None)
        if desc:
            print(f"{desc}...")
        return iterable


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ImageInfo:
    """Information about a single image"""
    path: Path
    width: int
    height: int
    aspect_ratio: float = field(init=False)
    total_pixels: int = field(init=False)

    def __post_init__(self):
        self.aspect_ratio = self.width / self.height
        self.total_pixels = self.width * self.height


@dataclass
class Bucket:
    """A bucket with target resolution"""
    width: int
    height: int
    aspect_ratio: float = field(init=False)
    total_pixels: int = field(init=False)
    images: List[ImageInfo] = field(default_factory=list)

    def __post_init__(self):
        self.aspect_ratio = self.width / self.height
        self.total_pixels = self.width * self.height

    @property
    def size(self) -> int:
        return len(self.images)

    def __repr__(self):
        return f"Bucket({self.width}x{self.height}, AR={self.aspect_ratio:.3f}, images={self.size})"


@dataclass
class OptimizationResult:
    """Result of bucket optimization"""
    buckets: List[Bucket]
    total_images: int
    avg_aspect_distortion: float
    max_bucket_size: int
    min_bucket_size: int
    imbalance_ratio: float


# ============================================================================
# Phase 1: Dataset Analysis
# ============================================================================

def get_image_extensions() -> set:
    """Supported image extensions"""
    return {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


def scan_image_folder(input_dir: Path, max_workers: int = 8) -> List[ImageInfo]:
    """
    Scan folder and extract image information

    Args:
        input_dir: Input directory containing images
        max_workers: Number of parallel workers

    Returns:
        List of ImageInfo objects
    """
    extensions = get_image_extensions()
    image_paths = []

    print(f"Scanning directory: {input_dir}")
    for ext in extensions:
        image_paths.extend(input_dir.rglob(f"*{ext}"))
        image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))

    image_paths = list(set(image_paths))  # Remove duplicates
    print(f"Found {len(image_paths)} images")

    def process_image(path: Path) -> Optional[ImageInfo]:
        try:
            with Image.open(path) as img:
                width, height = img.size
                if width > 0 and height > 0:
                    return ImageInfo(path=path, width=width, height=height)
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
        return None

    images = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, p): p for p in image_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading images"):
            result = future.result()
            if result:
                images.append(result)

    print(f"Successfully loaded {len(images)} images")
    return images


def _mean(values: List[float]) -> float:
    """Calculate mean without numpy"""
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    """Calculate standard deviation without numpy"""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _median(values: List[float]) -> float:
    """Calculate median without numpy"""
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def _quantile(values: List[float], q: float) -> float:
    """Calculate quantile without numpy"""
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = q * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def analyze_aspect_ratios(images: List[ImageInfo]) -> Dict:
    """
    Analyze aspect ratio distribution

    Args:
        images: List of ImageInfo objects

    Returns:
        Analysis dictionary with statistics
    """
    aspect_ratios = [img.aspect_ratio for img in images]

    # Common aspect ratio categories
    categories = {
        "ultra_wide": (2.0, float('inf')),      # > 2:1
        "wide": (1.5, 2.0),                      # 3:2 to 2:1
        "landscape": (1.1, 1.5),                 # ~4:3
        "square": (0.9, 1.1),                    # ~1:1
        "portrait": (0.67, 0.9),                 # ~3:4
        "tall": (0.5, 0.67),                     # 1:2 to 2:3
        "ultra_tall": (0.0, 0.5),                # < 1:2
    }

    distribution = defaultdict(list)
    for img in images:
        for cat_name, (low, high) in categories.items():
            if low <= img.aspect_ratio < high:
                distribution[cat_name].append(img)
                break

    # Use numpy if available, else fallback to pure Python
    if HAS_NUMPY:
        ar_mean = float(np.mean(aspect_ratios))
        ar_std = float(np.std(aspect_ratios))
        ar_median = float(np.median(aspect_ratios))
    else:
        ar_mean = _mean(aspect_ratios)
        ar_std = _std(aspect_ratios)
        ar_median = _median(aspect_ratios)

    analysis = {
        "total_images": len(images),
        "aspect_ratios": {
            "min": min(aspect_ratios),
            "max": max(aspect_ratios),
            "mean": ar_mean,
            "std": ar_std,
            "median": ar_median,
        },
        "distribution": {k: len(v) for k, v in distribution.items()},
        "distribution_percent": {k: len(v) / len(images) * 100 for k, v in distribution.items()},
    }

    return analysis


def print_analysis(analysis: Dict):
    """Print analysis results"""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    print(f"Total images: {analysis['total_images']}")
    print(f"\nAspect Ratio Statistics:")
    ar = analysis['aspect_ratios']
    print(f"  Min: {ar['min']:.3f}")
    print(f"  Max: {ar['max']:.3f}")
    print(f"  Mean: {ar['mean']:.3f}")
    print(f"  Median: {ar['median']:.3f}")
    print(f"  Std: {ar['std']:.3f}")

    print(f"\nAspect Ratio Distribution:")
    for cat, count in sorted(analysis['distribution'].items(), key=lambda x: -x[1]):
        pct = analysis['distribution_percent'][cat]
        bar = "█" * int(pct / 2)
        print(f"  {cat:12s}: {count:6d} ({pct:5.1f}%) {bar}")
    print("=" * 60 + "\n")


# ============================================================================
# Phase 2: Optimal Bucket Generation
# ============================================================================

def generate_candidate_buckets(
    base_resolution: int,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
    step: int = 64
) -> List[Tuple[int, int]]:
    """
    Generate candidate bucket resolutions

    Args:
        base_resolution: Target base resolution (e.g., 1024)
        min_ratio: Minimum aspect ratio
        max_ratio: Maximum aspect ratio
        step: Resolution step size

    Returns:
        List of (width, height) tuples
    """
    target_pixels = base_resolution * base_resolution
    candidates = []

    # Generate buckets with various aspect ratios
    # Keep total pixels approximately constant
    for width in range(step, base_resolution * 2 + step, step):
        # Calculate height to maintain similar pixel count
        height = int(round(target_pixels / width / step) * step)

        if height < step:
            continue

        ar = width / height
        if min_ratio <= ar <= max_ratio:
            candidates.append((width, height))

    # Remove duplicates and sort
    candidates = list(set(candidates))
    candidates.sort(key=lambda x: x[0] / x[1])  # Sort by aspect ratio

    return candidates


def calculate_assignment_cost(
    image: ImageInfo,
    bucket: Tuple[int, int],
    base_resolution: int
) -> float:
    """
    Calculate cost of assigning image to bucket

    Lower cost = better match

    Args:
        image: ImageInfo object
        bucket: (width, height) tuple
        base_resolution: Base resolution for normalization

    Returns:
        Cost value (lower is better)
    """
    bucket_ar = bucket[0] / bucket[1]

    # Aspect ratio distortion cost (most important)
    ar_diff = abs(image.aspect_ratio - bucket_ar)
    ar_cost = ar_diff ** 2  # Quadratic penalty

    # Resolution change cost
    # Penalize both upscaling (quality loss) and excessive downscaling
    scale_factor = math.sqrt(bucket[0] * bucket[1] / image.total_pixels)
    if scale_factor > 1.0:
        # Upscaling penalty (worse)
        scale_cost = (scale_factor - 1.0) ** 2 * 2.0
    else:
        # Downscaling penalty (less severe)
        scale_cost = (1.0 - scale_factor) ** 2 * 0.5

    # Combined cost (weighted)
    total_cost = ar_cost * 10.0 + scale_cost * 1.0

    return total_cost


# ============================================================================
# SD-Scripts Bucket Simulation (for data-driven auto mode)
# ============================================================================

def make_bucket_resolutions_sdscripts(
    max_reso: Tuple[int, int],
    min_size: int = 256,
    max_size: int = 1024,
    divisible: int = 64
) -> List[Tuple[int, int]]:
    """
    Generate bucket resolutions exactly like sd-scripts does.
    This is a copy of library/model_util.py:make_bucket_resolutions()

    Args:
        max_reso: (width, height) base resolution
        min_size: Minimum bucket dimension
        max_size: Maximum bucket dimension
        divisible: Step size (bucket_reso_steps)

    Returns:
        List of (width, height) tuples for all possible buckets
    """
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    # Add square bucket
    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    # Generate all other buckets
    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))
        width += divisible

    resos = list(resos)
    resos.sort()
    return resos


def simulate_bucket_assignment(
    images: List[ImageInfo],
    base_resolution: int,
    min_bucket_reso: int,
    max_bucket_reso: int,
    bucket_reso_steps: int,
    no_upscale: bool = False
) -> Dict[Tuple[int, int], List[ImageInfo]]:
    """
    Simulate bucket assignment exactly like sd-scripts BucketManager does.

    Args:
        images: List of ImageInfo objects
        base_resolution: Base resolution (e.g., 1024)
        min_bucket_reso: Minimum bucket resolution
        max_bucket_reso: Maximum bucket resolution
        bucket_reso_steps: Resolution step size
        no_upscale: If True, don't upscale images

    Returns:
        Dict mapping bucket resolution to list of images
    """
    max_reso = (base_resolution, base_resolution)
    max_area = base_resolution * base_resolution

    # Generate all possible bucket resolutions
    predefined_resos = make_bucket_resolutions_sdscripts(
        max_reso, min_bucket_reso, max_bucket_reso, bucket_reso_steps
    )
    predefined_resos_set = set(predefined_resos)

    if HAS_NUMPY:
        predefined_aspect_ratios = np.array([w / h for w, h in predefined_resos])
    else:
        predefined_aspect_ratios = [w / h for w, h in predefined_resos]

    # Assign each image to a bucket
    bucket_assignments = defaultdict(list)

    for img in images:
        aspect_ratio = img.aspect_ratio

        if not no_upscale:
            # Standard mode: find best matching bucket by aspect ratio
            reso = (img.width, img.height)
            if reso not in predefined_resos_set:
                # Find bucket with closest aspect ratio
                if HAS_NUMPY:
                    ar_errors = predefined_aspect_ratios - aspect_ratio
                    predefined_bucket_id = int(np.abs(ar_errors).argmin())
                else:
                    ar_errors = [abs(ar - aspect_ratio) for ar in predefined_aspect_ratios]
                    predefined_bucket_id = ar_errors.index(min(ar_errors))
                reso = predefined_resos[predefined_bucket_id]
        else:
            # No upscale mode: more complex logic
            if img.total_pixels > max_area:
                # Image too large, need to resize
                resized_width = math.sqrt(max_area * aspect_ratio)
                resized_height = max_area / resized_width

                def round_to_steps(x):
                    x = int(x + 0.5)
                    return x - x % bucket_reso_steps

                b_width_rounded = round_to_steps(resized_width)
                b_height_in_wr = round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr if b_height_in_wr > 0 else 0

                b_height_rounded = round_to_steps(resized_height)
                b_width_in_hr = round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded if b_height_rounded > 0 else 0

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    reso = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    reso = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
            else:
                # Round to steps
                bucket_width = img.width - img.width % bucket_reso_steps
                bucket_height = img.height - img.height % bucket_reso_steps
                reso = (bucket_width, bucket_height)

        bucket_assignments[reso].append(img)

    return dict(bucket_assignments)


def analyze_bucket_distribution(
    bucket_assignments: Dict[Tuple[int, int], List[ImageInfo]],
    num_repeats: int = 1
) -> Dict:
    """
    Analyze bucket distribution to find imbalance and issues.

    Args:
        bucket_assignments: Dict from simulate_bucket_assignment()
        num_repeats: Number of repeats per image (for training)

    Returns:
        Analysis dict with statistics and recommendations
    """
    if not bucket_assignments:
        return {"error": "No buckets found"}

    # Calculate sizes (with repeats)
    bucket_sizes = {
        reso: len(images) * num_repeats
        for reso, images in bucket_assignments.items()
    }

    sizes = list(bucket_sizes.values())
    total_images = sum(len(imgs) for imgs in bucket_assignments.values())
    total_with_repeats = sum(sizes)

    # Basic statistics
    min_size = min(sizes)
    max_size = max(sizes)
    avg_size = total_with_repeats / len(sizes)

    if HAS_NUMPY:
        std_size = float(np.std(sizes))
        median_size = float(np.median(sizes))
    else:
        std_size = _std(sizes)
        median_size = _median(sizes)

    # Imbalance ratio
    imbalance_ratio = max_size / min_size if min_size > 0 else float('inf')

    # Coefficient of variation (CV) - normalized measure of dispersion
    cv = std_size / avg_size if avg_size > 0 else 0

    # Find problematic buckets
    small_buckets = [(reso, count) for reso, count in bucket_sizes.items() if count < avg_size * 0.25]
    large_buckets = [(reso, count) for reso, count in bucket_sizes.items() if count > avg_size * 2.0]

    # Gini coefficient (measure of inequality)
    sorted_sizes = sorted(sizes)
    n = len(sorted_sizes)
    cumsum = 0
    for i, size in enumerate(sorted_sizes):
        cumsum += (2 * (i + 1) - n - 1) * size
    gini = cumsum / (n * sum(sorted_sizes)) if sum(sorted_sizes) > 0 else 0

    # Recommendations
    recommendations = []

    if imbalance_ratio > 10:
        recommendations.append(f"CRITICAL: Imbalance ratio {imbalance_ratio:.1f}x is very high. Training will be unstable.")
    elif imbalance_ratio > 5:
        recommendations.append(f"WARNING: Imbalance ratio {imbalance_ratio:.1f}x is high. Consider balancing buckets.")

    if len(small_buckets) > 0:
        recommendations.append(f"Found {len(small_buckets)} small buckets (<25% of avg). Consider merging them.")

    if len(large_buckets) > 0:
        recommendations.append(f"Found {len(large_buckets)} large buckets (>2x avg). Images are concentrated in few buckets.")

    if gini > 0.4:
        recommendations.append(f"High Gini coefficient ({gini:.2f}) indicates uneven distribution.")

    if len(bucket_assignments) > total_images / 50:
        recommendations.append(f"Too many buckets ({len(bucket_assignments)}) for {total_images} images. Consider increasing bucket_reso_steps.")

    return {
        "num_buckets": len(bucket_assignments),
        "total_images": total_images,
        "total_with_repeats": total_with_repeats,
        "bucket_sizes": bucket_sizes,
        "statistics": {
            "min": min_size,
            "max": max_size,
            "avg": avg_size,
            "std": std_size,
            "median": median_size,
            "imbalance_ratio": imbalance_ratio,
            "cv": cv,
            "gini": gini,
        },
        "problematic": {
            "small_buckets": small_buckets,
            "large_buckets": large_buckets,
        },
        "recommendations": recommendations,
    }


def auto_optimize_from_simulation(
    images: List[ImageInfo],
    base_resolution: int,
    target_max_imbalance: float = 4.0,
    target_min_bucket_size: int = 50
) -> Dict:
    """
    Automatically find optimal bucket parameters by simulating different configurations.

    This is DATA-DRIVEN: it actually simulates bucket creation and picks the best parameters.

    Args:
        images: List of ImageInfo objects
        base_resolution: Base resolution
        target_max_imbalance: Target maximum imbalance ratio
        target_min_bucket_size: Target minimum bucket size

    Returns:
        Dict with optimal parameters and analysis
    """
    total_images = len(images)

    # Try different bucket_reso_steps values
    step_candidates = [32, 64, 128, 256]

    # Try different min/max bucket resolutions
    min_reso_candidates = [256, 512, 768]
    max_reso_candidates = [1024, 1536, 2048, base_resolution * 2]

    best_config = None
    best_score = float('inf')
    all_results = []

    for steps in step_candidates:
        for min_reso in min_reso_candidates:
            for max_reso in max_reso_candidates:
                if min_reso >= max_reso:
                    continue
                if max_reso < base_resolution:
                    continue

                # Simulate bucket assignment
                try:
                    assignments = simulate_bucket_assignment(
                        images, base_resolution, min_reso, max_reso, steps
                    )
                except Exception as e:
                    continue

                if not assignments:
                    continue

                # Analyze distribution
                analysis = analyze_bucket_distribution(assignments)
                stats = analysis['statistics']

                # Score this configuration
                # Lower score = better
                score = 0

                # Penalize high imbalance
                if stats['imbalance_ratio'] > target_max_imbalance:
                    score += (stats['imbalance_ratio'] - target_max_imbalance) * 10

                # Penalize too few images in smallest bucket
                if stats['min'] < target_min_bucket_size:
                    score += (target_min_bucket_size - stats['min']) * 0.5

                # Penalize high CV (coefficient of variation)
                score += stats['cv'] * 5

                # Penalize too many or too few buckets
                ideal_buckets = max(3, min(10, total_images // 500))
                bucket_diff = abs(analysis['num_buckets'] - ideal_buckets)
                score += bucket_diff * 0.5

                # Penalize high Gini
                score += stats['gini'] * 3

                result = {
                    'bucket_reso_steps': steps,
                    'min_bucket_reso': min_reso,
                    'max_bucket_reso': max_reso,
                    'num_buckets': analysis['num_buckets'],
                    'imbalance_ratio': stats['imbalance_ratio'],
                    'min_bucket_size': stats['min'],
                    'cv': stats['cv'],
                    'gini': stats['gini'],
                    'score': score,
                }
                all_results.append(result)

                if score < best_score:
                    best_score = score
                    best_config = result
                    best_config['assignments'] = assignments
                    best_config['analysis'] = analysis

    if best_config is None:
        # Fallback to defaults
        return {
            'bucket_reso_steps': 64,
            'min_bucket_reso': 512,
            'max_bucket_reso': base_resolution * 2,
            'success': False,
            'message': 'Could not find optimal configuration'
        }

    # Sort all results by score for reporting
    all_results.sort(key=lambda x: x['score'])

    return {
        'optimal': best_config,
        'top_5': all_results[:5],
        'success': True,
        'message': f"Found optimal config with score {best_score:.2f}"
    }


def find_optimal_buckets(
    images: List[ImageInfo],
    base_resolution: int,
    num_buckets: int,
    min_bucket_size: int,
    max_imbalance: float = 5.0
) -> List[Bucket]:
    """
    Find optimal bucket configuration using clustering approach

    Args:
        images: List of ImageInfo objects
        base_resolution: Target base resolution
        num_buckets: Target number of buckets
        min_bucket_size: Minimum images per bucket
        max_imbalance: Maximum ratio between largest/smallest bucket

    Returns:
        List of Bucket objects with assigned images
    """
    print(f"\nFinding optimal buckets (target: {num_buckets} buckets)...")

    # Sort images by aspect ratio
    sorted_images = sorted(images, key=lambda x: x.aspect_ratio)

    # Strategy 1: Equal-count bucketing
    # Divide images into roughly equal groups by aspect ratio
    images_per_bucket = max(min_bucket_size, len(images) // num_buckets)

    # Find natural break points in aspect ratio distribution
    aspect_ratios = [img.aspect_ratio for img in sorted_images]

    # Use quantile-based bucketing for balanced distribution
    if HAS_NUMPY:
        quantiles = np.linspace(0, 1, num_buckets + 1)
        ar_breakpoints = np.quantile(aspect_ratios, quantiles)
    else:
        quantiles = [i / num_buckets for i in range(num_buckets + 1)]
        ar_breakpoints = [_quantile(aspect_ratios, q) for q in quantiles]

    # Generate candidate bucket resolutions
    candidates = generate_candidate_buckets(
        base_resolution,
        min_ratio=min(aspect_ratios) * 0.9,
        max_ratio=max(aspect_ratios) * 1.1,
        step=64
    )

    print(f"Generated {len(candidates)} candidate bucket resolutions")

    # Create buckets based on aspect ratio ranges
    buckets = []

    for i in range(num_buckets):
        ar_low = ar_breakpoints[i]
        ar_high = ar_breakpoints[i + 1]
        ar_mid = (ar_low + ar_high) / 2

        # Find best matching candidate bucket
        best_candidate = min(
            candidates,
            key=lambda c: abs(c[0] / c[1] - ar_mid)
        )

        bucket = Bucket(width=best_candidate[0], height=best_candidate[1])
        buckets.append(bucket)

    # Merge similar buckets (within 5% aspect ratio)
    merged_buckets = []
    for bucket in buckets:
        merged = False
        for existing in merged_buckets:
            ar_diff = abs(bucket.aspect_ratio - existing.aspect_ratio) / existing.aspect_ratio
            if ar_diff < 0.05:  # Within 5%
                merged = True
                break
        if not merged:
            merged_buckets.append(bucket)

    buckets = merged_buckets
    print(f"After merging similar: {len(buckets)} buckets")

    # Assign images to buckets
    for img in tqdm(sorted_images, desc="Assigning images"):
        # Find bucket with lowest cost
        best_bucket = min(
            buckets,
            key=lambda b: calculate_assignment_cost(img, (b.width, b.height), base_resolution)
        )
        best_bucket.images.append(img)

    # Check for empty buckets and redistribute
    non_empty_buckets = [b for b in buckets if b.size > 0]

    # Rebalance if needed
    if len(non_empty_buckets) > 1:
        sizes = [b.size for b in non_empty_buckets]
        max_size = max(sizes)
        min_size = min(sizes)

        if min_size > 0 and max_size / min_size > max_imbalance:
            print(f"\nRebalancing buckets (imbalance: {max_size / min_size:.2f}x)...")
            non_empty_buckets = rebalance_buckets(
                non_empty_buckets,
                min_bucket_size,
                max_imbalance
            )

    return non_empty_buckets


def rebalance_buckets(
    buckets: List[Bucket],
    min_bucket_size: int,
    max_imbalance: float
) -> List[Bucket]:
    """
    Rebalance buckets to reduce size imbalance

    Strategy:
    1. Merge very small buckets into nearest neighbors
    2. Split very large buckets if needed

    Args:
        buckets: List of Bucket objects
        min_bucket_size: Minimum images per bucket
        max_imbalance: Maximum ratio between largest/smallest bucket

    Returns:
        Rebalanced list of Bucket objects
    """
    # Sort by aspect ratio
    buckets = sorted(buckets, key=lambda b: b.aspect_ratio)

    # Merge small buckets
    merged = []
    i = 0
    while i < len(buckets):
        current = buckets[i]

        if current.size < min_bucket_size and i < len(buckets) - 1:
            # Merge with next bucket
            next_bucket = buckets[i + 1]
            # Create merged bucket with weighted average aspect ratio
            total = current.size + next_bucket.size
            if total > 0:
                avg_ar = (current.aspect_ratio * current.size +
                         next_bucket.aspect_ratio * next_bucket.size) / total

                # Find closest standard bucket size
                target_pixels = current.total_pixels
                new_height = int(math.sqrt(target_pixels / avg_ar))
                new_height = (new_height // 64) * 64
                new_width = int(new_height * avg_ar)
                new_width = (new_width // 64) * 64

                new_bucket = Bucket(width=max(64, new_width), height=max(64, new_height))
                new_bucket.images = current.images + next_bucket.images
                merged.append(new_bucket)
                i += 2
            else:
                merged.append(current)
                i += 1
        else:
            merged.append(current)
            i += 1

    return merged


# ============================================================================
# Phase 3: Image Resize and Copy
# ============================================================================

def calculate_resize_dimensions(
    image: ImageInfo,
    bucket: Bucket,
    preserve_ratio: bool = True,
    no_crop: bool = False
) -> Tuple[int, int, Tuple[int, int, int, int]]:
    """
    Calculate resize dimensions and crop box

    Args:
        image: ImageInfo object
        bucket: Target Bucket
        preserve_ratio: Whether to preserve aspect ratio (with letterbox/crop)
        no_crop: If True, only resize without any cropping (keep full image content)
                 Output dimensions will match bucket area but preserve original aspect ratio

    Returns:
        (new_width, new_height, crop_box) where crop_box is (left, top, right, bottom)
    """
    if not preserve_ratio:
        # Direct resize (may distort)
        return bucket.width, bucket.height, (0, 0, image.width, image.height)

    # NO CROP mode: resize to bucket area while preserving exact aspect ratio
    # This means the output may not match exact bucket dimensions
    if no_crop:
        # Calculate target pixel count from bucket
        target_pixels = bucket.total_pixels
        img_ar = image.aspect_ratio

        # Calculate new dimensions preserving aspect ratio with ~same pixel count
        # new_width * new_height = target_pixels
        # new_width / new_height = img_ar
        # => new_height = sqrt(target_pixels / img_ar)
        # => new_width = new_height * img_ar
        new_height = int(math.sqrt(target_pixels / img_ar))
        new_width = int(new_height * img_ar)

        # Round to nearest multiple of 64 for compatibility
        new_width = max(64, (new_width // 64) * 64)
        new_height = max(64, (new_height // 64) * 64)

        # No crop needed - use full original image
        return new_width, new_height, (0, 0, image.width, image.height)

    # Standard mode: Calculate scaling to fit bucket while preserving ratio
    img_ar = image.aspect_ratio
    bucket_ar = bucket.aspect_ratio

    if img_ar > bucket_ar:
        # Image is wider - fit to width, crop top/bottom
        scale = bucket.width / image.width
        scaled_height = int(image.height * scale)

        if scaled_height >= bucket.height:
            # Crop height
            crop_height = int(bucket.height / scale)
            crop_top = (image.height - crop_height) // 2
            crop_box = (0, crop_top, image.width, crop_top + crop_height)
            return bucket.width, bucket.height, crop_box
        else:
            # Letterbox (add padding) - but we'll just resize
            return bucket.width, scaled_height, (0, 0, image.width, image.height)
    else:
        # Image is taller - fit to height, crop left/right
        scale = bucket.height / image.height
        scaled_width = int(image.width * scale)

        if scaled_width >= bucket.width:
            # Crop width
            crop_width = int(bucket.width / scale)
            crop_left = (image.width - crop_width) // 2
            crop_box = (crop_left, 0, crop_left + crop_width, image.height)
            return bucket.width, bucket.height, crop_box
        else:
            # Letterbox - but we'll just resize
            return scaled_width, bucket.height, (0, 0, image.width, image.height)


def process_single_image(
    image: ImageInfo,
    bucket: Bucket,
    input_dir: Path,
    output_dir: Path,
    quality: int = 95,
    flat_output: bool = True,
    keep_extension: bool = False,
    no_crop: bool = False
) -> Tuple[bool, str]:
    """
    Process a single image: resize and copy with captions

    Args:
        image: ImageInfo object
        bucket: Target Bucket
        input_dir: Input directory root
        output_dir: Output directory root
        quality: JPEG quality (1-100)
        flat_output: If True, save all images directly to output_dir (keep original filename)
                    If False, create bucket subdirectories
        keep_extension: If True, keep original extension; If False, convert to jpg
        no_crop: If True, only resize without cropping (preserve full aspect ratio)

    Returns:
        (success, message)
    """
    try:
        if flat_output:
            # Flat structure: output_dir/filename.ext
            if keep_extension:
                output_path = output_dir / image.path.name
            else:
                output_path = output_dir / (image.path.stem + '.jpg')
        else:
            # Bucket subdirectory structure: output_dir/WxH/filename.ext
            rel_path = image.path.relative_to(input_dir)
            bucket_name = f"{bucket.width}x{bucket.height}"
            output_subdir = output_dir / bucket_name
            if keep_extension:
                output_path = output_subdir / rel_path
            else:
                output_path = (output_subdir / rel_path).with_suffix('.jpg')

        # Create directories
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate resize dimensions
        new_width, new_height, crop_box = calculate_resize_dimensions(image, bucket, no_crop=no_crop)

        # Open and process image
        with Image.open(image.path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Crop if needed
            if crop_box != (0, 0, image.width, image.height):
                img = img.crop(crop_box)

            # Resize
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save based on extension
            save_path = output_path
            if keep_extension:
                ext = image.path.suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    img.save(save_path, 'JPEG', quality=quality)
                elif ext == '.png':
                    img.save(save_path, 'PNG')
                elif ext == '.webp':
                    img.save(save_path, 'WEBP', quality=quality)
                else:
                    img.save(save_path, 'JPEG', quality=quality)
            else:
                img.save(save_path, 'JPEG', quality=quality)

        # Copy caption files
        caption_extensions = ['.txt', '.caption', '.tags']
        for ext in caption_extensions:
            caption_path = image.path.with_suffix(ext)
            if caption_path.exists():
                output_caption = output_path.with_suffix(ext)
                shutil.copy2(caption_path, output_caption)

        return True, f"Processed: {image.path.name}"

    except Exception as e:
        return False, f"Error processing {image.path}: {e}"


def process_images(
    buckets: List[Bucket],
    input_dir: Path,
    output_dir: Path,
    max_workers: int = 8,
    quality: int = 95,
    flat_output: bool = True,
    keep_extension: bool = False,
    no_crop: bool = False
) -> Dict:
    """
    Process all images: resize and copy

    Args:
        buckets: List of Bucket objects with assigned images
        input_dir: Input directory root
        output_dir: Output directory root
        max_workers: Number of parallel workers
        quality: JPEG quality
        flat_output: If True, save all images directly to output_dir
        keep_extension: If True, keep original file extension
        no_crop: If True, only resize without cropping (preserve full aspect ratio)

    Returns:
        Processing statistics
    """
    print(f"\nProcessing images to {output_dir}...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all tasks
    tasks = []
    for bucket in buckets:
        for image in bucket.images:
            tasks.append((image, bucket))

    stats = {
        "total": len(tasks),
        "success": 0,
        "failed": 0,
        "errors": []
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image,
                img, bucket, input_dir, output_dir, quality, flat_output, keep_extension, no_crop
            ): (img, bucket)
            for img, bucket in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Resizing"):
            success, message = future.result()
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(message)

    return stats


# ============================================================================
# Phase 4: Report Generation
# ============================================================================

def generate_report(
    buckets: List[Bucket],
    analysis: Dict,
    report_dir: Path,
    processing_stats: Dict
) -> str:
    """
    Generate detailed report

    Args:
        buckets: List of Bucket objects
        analysis: Dataset analysis results
        report_dir: Directory to save report files (usually current working directory)
        processing_stats: Processing statistics

    Returns:
        Report string
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("BALANCED BUCKET OPTIMIZER REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Dataset summary
    report_lines.append("DATASET SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total images: {analysis['total_images']}")
    ar = analysis['aspect_ratios']
    report_lines.append(f"Aspect ratio range: {ar['min']:.3f} - {ar['max']:.3f}")
    report_lines.append(f"Aspect ratio mean: {ar['mean']:.3f}")
    report_lines.append("")

    # Bucket summary
    report_lines.append("BUCKET CONFIGURATION")
    report_lines.append("-" * 40)
    report_lines.append(f"Number of buckets: {len(buckets)}")

    sizes = [b.size for b in buckets]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    report_lines.append(f"Images per bucket: min={min(sizes)}, max={max(sizes)}, avg={avg_size:.1f}")

    if min(sizes) > 0:
        imbalance = max(sizes) / min(sizes)
        report_lines.append(f"Imbalance ratio: {imbalance:.2f}x")
    report_lines.append("")

    # Bucket details
    report_lines.append("BUCKET DETAILS")
    report_lines.append("-" * 40)
    report_lines.append(f"{'Resolution':>12} {'AR':>8} {'Count':>8} {'Percent':>8} {'Distribution'}")
    report_lines.append("-" * 70)

    total = sum(b.size for b in buckets)
    for bucket in sorted(buckets, key=lambda b: -b.size):
        pct = bucket.size / total * 100
        bar = "█" * int(pct / 2)
        report_lines.append(
            f"{bucket.width}x{bucket.height:>4} {bucket.aspect_ratio:>8.3f} "
            f"{bucket.size:>8} {pct:>7.1f}% {bar}"
        )
    report_lines.append("")

    # Processing stats
    report_lines.append("PROCESSING STATISTICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Successfully processed: {processing_stats['success']}")
    report_lines.append(f"Failed: {processing_stats['failed']}")

    if processing_stats['errors']:
        report_lines.append(f"\nErrors (first 10):")
        for error in processing_stats['errors'][:10]:
            report_lines.append(f"  - {error}")

    report_lines.append("")
    report_lines.append("=" * 70)

    report = "\n".join(report_lines)

    # Save report to report_dir (current working directory)
    report_path = report_dir / "bucket_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # Save bucket config as JSON
    config = {
        "buckets": [
            {
                "resolution": f"{b.width}x{b.height}",
                "width": b.width,
                "height": b.height,
                "aspect_ratio": b.aspect_ratio,
                "image_count": b.size
            }
            for b in buckets
        ],
        "total_images": total,
        "num_buckets": len(buckets)
    }

    config_path = report_dir / "bucket_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    print(f"Config saved to: {config_path}")

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Balanced Bucket Optimizer for Diffusion Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. SIMULATE: Xem bucket distribution nhu sd-scripts se tao
  python balanced_bucket_optimizer.py -i ./data -o ./output -r 1024 --simulate
  python balanced_bucket_optimizer.py -i ./data -o ./output -r 1024 --simulate --bucket_reso_steps 128

  # 2. AUTO MODE (KHUYEN NGHI): Tu dong tim config toi uu qua simulation
  python balanced_bucket_optimizer.py -i ./data -o ./output -r 1024 --auto

  # 3. MANUAL: Chi dinh tham so thu cong
  python balanced_bucket_optimizer.py -i ./data -o ./output -r 1024 -n 8 --min_bucket_size 100

  # 4. Ket hop cac options
  python balanced_bucket_optimizer.py -i ./data -o ./output -r 1024 --auto --keep_extension
  python balanced_bucket_optimizer.py -i ./data -o ./output -r 1024 --bucket_folders

  # 5. Phan tich truoc
  python balanced_bucket_optimizer.py -i ./data -o ./output -r 1024 --analyze_only

Workflow khuyen nghi:
  1. Chay --simulate de xem bucket distribution hien tai
  2. Neu imbalance ratio > 5x, chay --auto de tim config toi uu
  3. Dung config toi uu de xu ly anh

Auto mode (--auto):
  - Script se simulate nhieu config khac nhau:
    + bucket_reso_steps: 32, 64, 128, 256
    + min_bucket_reso: 256, 512, 768
    + max_bucket_reso: 1024, 1536, 2048, base*2
  - Chon config co:
    + Imbalance ratio thap
    + Gini coefficient thap (phan phoi deu)
    + Du anh moi bucket (min >= 50)

Output:
  - output_dir/: Anh da resize + caption files
  - ./bucket_report.txt: Report chi tiet
  - ./bucket_config.json: Config bucket
        """
    )

    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Folder chua anh goc. Script se scan tat ca anh trong folder nay (bao gom subfolder)"
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Folder output chua anh da xu ly + file caption (.txt). Chi chua anh, khong chua report"
    )

    parser.add_argument(
        "--base_resolution", "-r",
        type=int,
        default=1024,
        help="Resolution co so (default: 1024). Bucket se duoc tao xung quanh resolution nay. "
             "Vi du: 1024 -> buckets co the la 1024x1024, 1152x896, 896x1152, ..."
    )

    parser.add_argument(
        "--num_buckets", "-n",
        type=int,
        default=None,
        help="So luong bucket muc tieu. Neu khong set, script se tu dong tinh dua tren dataset. "
             "Script se chia anh thanh N bucket dua tren aspect ratio"
    )

    parser.add_argument(
        "--min_bucket_size",
        type=int,
        default=None,
        help="So anh toi thieu moi bucket. Neu khong set, script se tu dong tinh. "
             "Bucket nho hon se duoc merge voi bucket gan nhat"
    )

    parser.add_argument(
        "--max_imbalance",
        type=float,
        default=None,
        help="Ti le chenh lech toi da giua bucket lon nhat va nho nhat. Neu khong set, mac dinh 4.0. "
             "Vi du: 4.0 nghia la bucket lon nhat khong duoc lon hon 4 lan bucket nho nhat"
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="[DATA-DRIVEN] Tu dong tim tham so toi uu bang cach simulate bucket creation nhu sd-scripts. "
             "Script se thu nhieu config va chon config tot nhat dua tren imbalance ratio, Gini coefficient, v.v."
    )

    # SD-Scripts simulation parameters
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="[Simulate] Buoc resolution (giong bucket_reso_steps cua sd-scripts). Default: 64. "
             "Tang len (128, 256) de giam so bucket, giam xuong (32) de tang so bucket"
    )

    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=256,
        help="[Simulate] Resolution toi thieu cho bucket (giong min_bucket_reso cua sd-scripts). Default: 256"
    )

    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=None,
        help="[Simulate] Resolution toi da cho bucket. Default: base_resolution * 2"
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="[Debug] Chi hien thi bucket distribution nhu sd-scripts, KHONG xu ly anh. "
             "Giup ban xem truoc cach sd-scripts se tao bucket voi dataset cua ban"
    )

    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="[Simulate] So lan lap anh (num_repeats) de tinh bucket size. Default: 1"
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Chat luong JPEG output 1-100 (default: 95). Chi ap dung khi khong dung --keep_extension"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="So luong worker xu ly song song. Mac dinh: auto-detect (so CPU cores). "
             "Set gia tri cu the neu muon gioi han, vd: --workers 4"
    )

    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="[KHUYEN NGHI] Chi resize anh, KHONG crop. Giu nguyen toan bo aspect ratio goc. "
             "Kich thuoc output se co cung so pixel voi bucket nhung giu nguyen AR. "
             "Vi du: Anh 1920x1080 (AR=1.78) se resize thanh ~1368x768 thay vi crop thanh 1024x1024"
    )

    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Chi phan tich dataset, KHONG resize anh. Dung de xem truoc bucket distribution"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Xem truoc ket qua ma khong thuc su xu ly. Hien thi bucket assignments"
    )

    parser.add_argument(
        "--bucket_folders",
        action="store_true",
        help="Tao subfolder cho moi bucket (vd: output/1024x1024/, output/1152x896/). "
             "Mac dinh: tat ca anh nam truc tiep trong output_dir"
    )

    parser.add_argument(
        "--keep_extension",
        action="store_true",
        help="Giu nguyen extension goc (png, webp, ...) thay vi convert sang JPEG"
    )

    args = parser.parse_args()

    # Auto-detect workers if not specified
    if args.workers is None:
        import multiprocessing
        args.workers = multiprocessing.cpu_count()
        print(f"Auto-detected CPU cores: {args.workers}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Phase 1: Scan and analyze
    print("\n" + "=" * 60)
    print("PHASE 1: DATASET ANALYSIS")
    print("=" * 60)

    images = scan_image_folder(input_dir, max_workers=args.workers)

    if not images:
        print("Error: No images found in input directory")
        sys.exit(1)

    analysis = analyze_aspect_ratios(images)
    print_analysis(analysis)

    if args.analyze_only:
        print("Analysis complete. Use without --analyze_only to process images.")
        sys.exit(0)

    # Set max_bucket_reso default
    max_bucket_reso = args.max_bucket_reso if args.max_bucket_reso else args.base_resolution * 2

    # ========================================================================
    # SIMULATE MODE: Show bucket distribution like sd-scripts
    # ========================================================================
    if args.simulate:
        print("\n" + "=" * 60)
        print("SD-SCRIPTS BUCKET SIMULATION")
        print("=" * 60)

        print(f"\nSimulation parameters:")
        print(f"  - base_resolution: {args.base_resolution}")
        print(f"  - bucket_reso_steps: {args.bucket_reso_steps}")
        print(f"  - min_bucket_reso: {args.min_bucket_reso}")
        print(f"  - max_bucket_reso: {max_bucket_reso}")
        print(f"  - num_repeats: {args.num_repeats}")

        # Simulate bucket creation
        bucket_assignments = simulate_bucket_assignment(
            images,
            args.base_resolution,
            args.min_bucket_reso,
            max_bucket_reso,
            args.bucket_reso_steps
        )

        # Analyze distribution
        dist_analysis = analyze_bucket_distribution(bucket_assignments, args.num_repeats)

        # Print in sd-scripts style
        print(f"\nmake buckets")
        print(f"number of images (including repeats) / 各bucketの画像枚数（繰り返し回数を含む）")

        sorted_buckets = sorted(bucket_assignments.items(), key=lambda x: (x[0][0], x[0][1]))
        for i, (reso, imgs) in enumerate(sorted_buckets):
            count = len(imgs) * args.num_repeats
            print(f"bucket {i}: resolution {reso}, count: {count}")

        # Print analysis
        stats = dist_analysis['statistics']
        print(f"\n--- Distribution Analysis ---")
        print(f"Total buckets: {dist_analysis['num_buckets']}")
        print(f"Total images: {dist_analysis['total_images']}")
        print(f"Bucket sizes: min={stats['min']}, max={stats['max']}, avg={stats['avg']:.1f}")
        print(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}x")
        print(f"Coefficient of variation: {stats['cv']:.3f}")
        print(f"Gini coefficient: {stats['gini']:.3f}")

        if dist_analysis['recommendations']:
            print(f"\n--- Recommendations ---")
            for rec in dist_analysis['recommendations']:
                print(f"  * {rec}")

        sys.exit(0)

    # ========================================================================
    # AUTO MODE: Data-driven optimization via simulation
    # ========================================================================
    if args.auto:
        print("\n" + "=" * 60)
        print("AUTO MODE: DATA-DRIVEN BUCKET OPTIMIZATION")
        print("=" * 60)

        print("\nSearching for optimal bucket configuration...")
        print("(Testing different bucket_reso_steps, min/max_bucket_reso combinations)")

        optimization_result = auto_optimize_from_simulation(
            images,
            args.base_resolution,
            target_max_imbalance=4.0,
            target_min_bucket_size=50
        )

        if optimization_result['success']:
            optimal = optimization_result['optimal']
            print(f"\n[OPTIMAL CONFIG FOUND] Score: {optimal['score']:.2f}")
            print(f"  - bucket_reso_steps: {optimal['bucket_reso_steps']}")
            print(f"  - min_bucket_reso: {optimal['min_bucket_reso']}")
            print(f"  - max_bucket_reso: {optimal['max_bucket_reso']}")
            print(f"  - Resulting buckets: {optimal['num_buckets']}")
            print(f"  - Imbalance ratio: {optimal['imbalance_ratio']:.2f}x")
            print(f"  - Min bucket size: {optimal['min_bucket_size']}")
            print(f"  - Gini coefficient: {optimal['gini']:.3f}")

            # Show top 5 alternatives
            print(f"\nTop 5 configurations:")
            for i, cfg in enumerate(optimization_result['top_5'][:5]):
                print(f"  {i+1}. steps={cfg['bucket_reso_steps']}, min={cfg['min_bucket_reso']}, "
                      f"max={cfg['max_bucket_reso']} -> {cfg['num_buckets']} buckets, "
                      f"imbalance={cfg['imbalance_ratio']:.1f}x, score={cfg['score']:.2f}")

            # Use the optimal assignments for processing
            bucket_assignments = optimal['assignments']

            # Convert to Bucket objects
            buckets = []
            for reso, imgs in bucket_assignments.items():
                bucket = Bucket(width=reso[0], height=reso[1])
                bucket.images = imgs
                buckets.append(bucket)

            # Print bucket summary
            print(f"\nUsing {len(buckets)} optimized buckets:")
            for bucket in sorted(buckets, key=lambda b: -b.size):
                print(f"  {bucket.width}x{bucket.height} (AR={bucket.aspect_ratio:.3f}): {bucket.size} images")

        else:
            print(f"\n{optimization_result['message']}")
            print("Falling back to default parameters...")
            # Fall through to manual mode with defaults
            args.auto = False

    # ========================================================================
    # MANUAL MODE: Use specified parameters
    # ========================================================================
    if not args.auto:
        print("\n" + "=" * 60)
        print("PHASE 2: OPTIMAL BUCKET GENERATION")
        print("=" * 60)

        # Use manual parameters with defaults
        num_buckets = args.num_buckets if args.num_buckets is not None else 10
        min_bucket_size = args.min_bucket_size if args.min_bucket_size is not None else 50
        max_imbalance = args.max_imbalance if args.max_imbalance is not None else 5.0

        print(f"Using parameters:")
        print(f"  - num_buckets: {num_buckets}")
        print(f"  - min_bucket_size: {min_bucket_size}")
        print(f"  - max_imbalance: {max_imbalance}")

        buckets = find_optimal_buckets(
            images,
            base_resolution=args.base_resolution,
            num_buckets=num_buckets,
            min_bucket_size=min_bucket_size,
            max_imbalance=max_imbalance
        )

        # Print bucket summary
        print(f"\nGenerated {len(buckets)} buckets:")
        for bucket in sorted(buckets, key=lambda b: -b.size):
            print(f"  {bucket.width}x{bucket.height} (AR={bucket.aspect_ratio:.3f}): {bucket.size} images")

    if args.dry_run:
        print("\nDry run complete. Use without --dry_run to process images.")
        sys.exit(0)

    # Phase 3: Process images
    print("\n" + "=" * 60)
    print("PHASE 3: IMAGE PROCESSING")
    print("=" * 60)

    processing_stats = process_images(
        buckets,
        input_dir,
        output_dir,
        max_workers=args.workers,
        quality=args.quality,
        flat_output=not args.bucket_folders,
        keep_extension=args.keep_extension,
        no_crop=args.no_crop
    )

    # Phase 4: Generate report
    print("\n" + "=" * 60)
    print("PHASE 4: REPORT GENERATION")
    print("=" * 60)

    # Report files saved to current working directory
    report = generate_report(buckets, analysis, Path.cwd(), processing_stats)
    print(report)

    print("\nDone!")


if __name__ == "__main__":
    main()

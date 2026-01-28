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


def auto_calculate_params(
    images: List[ImageInfo],
    analysis: Dict
) -> Tuple[int, int, float]:
    """
    Automatically calculate optimal num_buckets, min_bucket_size, max_imbalance
    based on dataset characteristics.

    Strategy:
    - num_buckets: Based on aspect ratio diversity (std) and total images
    - min_bucket_size: Based on total images (ensure enough samples per bucket)
    - max_imbalance: Fixed at reasonable value

    Args:
        images: List of ImageInfo objects
        analysis: Dataset analysis results

    Returns:
        (num_buckets, min_bucket_size, max_imbalance)
    """
    total_images = len(images)
    ar_std = analysis['aspect_ratios']['std']
    ar_range = analysis['aspect_ratios']['max'] - analysis['aspect_ratios']['min']

    # Calculate num_buckets based on aspect ratio diversity
    # More diverse AR -> more buckets needed
    # But cap based on total images
    if ar_std < 0.2:
        # Very uniform aspect ratios -> few buckets
        base_buckets = 3
    elif ar_std < 0.4:
        # Moderate diversity
        base_buckets = 5
    elif ar_std < 0.6:
        # High diversity
        base_buckets = 7
    else:
        # Very high diversity
        base_buckets = 10

    # Adjust based on total images (need enough images per bucket)
    # Target: at least 100-500 images per bucket for good training
    max_buckets_by_count = max(3, total_images // 500)
    num_buckets = min(base_buckets, max_buckets_by_count)

    # Ensure at least 3 buckets if we have enough images
    if total_images >= 1000:
        num_buckets = max(3, num_buckets)

    # Calculate min_bucket_size
    # Target: each bucket should have enough images for training stability
    if total_images < 1000:
        min_bucket_size = max(20, total_images // (num_buckets * 2))
    elif total_images < 10000:
        min_bucket_size = 50
    elif total_images < 50000:
        min_bucket_size = 100
    else:
        min_bucket_size = 200

    # max_imbalance: keep reasonable
    max_imbalance = 4.0

    return num_buckets, min_bucket_size, max_imbalance


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
    preserve_ratio: bool = True
) -> Tuple[int, int, Tuple[int, int, int, int]]:
    """
    Calculate resize dimensions and crop box

    Args:
        image: ImageInfo object
        bucket: Target Bucket
        preserve_ratio: Whether to preserve aspect ratio (with letterbox/crop)

    Returns:
        (new_width, new_height, crop_box) where crop_box is (left, top, right, bottom)
    """
    if not preserve_ratio:
        # Direct resize (may distort)
        return bucket.width, bucket.height, (0, 0, image.width, image.height)

    # Calculate scaling to fit bucket while preserving ratio
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
    keep_extension: bool = False
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
        new_width, new_height, crop_box = calculate_resize_dimensions(image, bucket)

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
    keep_extension: bool = False
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
                img, bucket, input_dir, output_dir, quality, flat_output, keep_extension
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
  # Tu dong (khuyen nghi): Script tu tinh num_buckets, min_bucket_size
  python balanced_bucket_optimizer.py -i ./data -o ./output_process -r 1024

  # Hoac dung flag --auto de ro rang hon
  python balanced_bucket_optimizer.py -i ./data -o ./output_process -r 1024 --auto

  # Chi dinh tham so thu cong
  python balanced_bucket_optimizer.py -i ./data -o ./output_process -r 1024 -n 8 --min_bucket_size 100

  # Giu extension goc (png, webp, ...)
  python balanced_bucket_optimizer.py -i ./data -o ./output_process -r 1024 --keep_extension

  # Tao subfolder theo bucket
  python balanced_bucket_optimizer.py -i ./data -o ./output_process -r 1024 --bucket_folders

  # Phan tich truoc (xem bucket distribution)
  python balanced_bucket_optimizer.py -i ./data -o ./output_process -r 1024 --analyze_only

Output:
  - output_dir/: Anh da resize + file caption (.txt)
  - ./bucket_report.txt: Report chi tiet
  - ./bucket_config.json: Config bucket

Auto mode logic:
  - num_buckets: Dua tren aspect ratio diversity (std)
    + std < 0.2 -> 3 buckets (uniform AR)
    + std < 0.4 -> 5 buckets
    + std < 0.6 -> 7 buckets
    + std >= 0.6 -> 10 buckets (diverse AR)
  - min_bucket_size: Dua tren tong so anh
    + < 1000 images -> 20
    + < 10000 images -> 50
    + < 50000 images -> 100
    + >= 50000 images -> 200
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
        help="Tu dong tinh tat ca tham so (num_buckets, min_bucket_size, max_imbalance) "
             "dua tren phan tich dataset. Khuyen nghi dung cho lan dau"
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
        default=8,
        help="So luong worker xu ly song song (default: 8). Tang len neu CPU nhieu core"
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

    # Phase 2: Find optimal buckets
    print("\n" + "=" * 60)
    print("PHASE 2: OPTIMAL BUCKET GENERATION")
    print("=" * 60)

    # Determine parameters (auto or manual)
    if args.auto or (args.num_buckets is None and args.min_bucket_size is None and args.max_imbalance is None):
        # Auto mode: calculate all parameters automatically
        auto_num_buckets, auto_min_bucket_size, auto_max_imbalance = auto_calculate_params(images, analysis)

        num_buckets = args.num_buckets if args.num_buckets is not None else auto_num_buckets
        min_bucket_size = args.min_bucket_size if args.min_bucket_size is not None else auto_min_bucket_size
        max_imbalance = args.max_imbalance if args.max_imbalance is not None else auto_max_imbalance

        print(f"[AUTO] Calculated parameters based on dataset analysis:")
        print(f"  - num_buckets: {num_buckets}")
        print(f"  - min_bucket_size: {min_bucket_size}")
        print(f"  - max_imbalance: {max_imbalance}")
    else:
        # Manual mode with defaults
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
        keep_extension=args.keep_extension
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

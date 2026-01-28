# Balanced Bucket Optimizer

Tool to analyze and optimize bucket distribution for Diffusion Model training (Lumina, FLUX, SD3, etc.).

## Problem

When training diffusion models with diverse aspect ratio images, sd-scripts creates buckets based on `bucket_reso_steps`. With default settings (`bucket_reso_steps=64`, `min_bucket_reso=256`, `max_bucket_reso=4096`), you may end up with:

- **Too many buckets** (100+ buckets for large datasets)
- **Imbalanced distribution** (some buckets have 1000 images, others have 5)
- **Training instability** due to uneven gradient updates

Example of imbalanced bucket distribution:
```
bucket 0: resolution (768, 1280), count: 4     <- Too few
bucket 1: resolution (832, 1216), count: 12
bucket 2: resolution (896, 1152), count: 12
bucket 3: resolution (960, 1088), count: 28
bucket 4: resolution (1024, 1024), count: 4    <- Too few
bucket 5: resolution (1088, 960), count: 8
bucket 6: resolution (1152, 896), count: 4     <- Too few
bucket 7: resolution (1344, 768), count: 96    <- Dominant bucket
```

## Solution

This tool helps you:
1. **Simulate** bucket creation exactly like sd-scripts
2. **Analyze** bucket distribution (imbalance ratio, Gini coefficient)
3. **Auto-optimize** parameters to find balanced configuration
4. **Resize images** to optimal bucket resolutions

## Installation

No additional dependencies required. Uses standard Python libraries.

Optional for better performance:
```bash
pip install numpy tqdm
```

## Usage

### 1. Simulate Mode - Preview Bucket Distribution

See how sd-scripts will create buckets with your current config:

```bash
python tools/balanced_bucket_optimizer.py \
    -i ./dataset \
    -o ./output \
    -r 1024 \
    --simulate
```

Output:
```
make buckets
bucket 0: resolution (704, 1408), count: 2
bucket 1: resolution (832, 1216), count: 5
bucket 2: resolution (1024, 1024), count: 5
...
--- Distribution Analysis ---
Total buckets: 8
Imbalance ratio: 5.00x    <- High = bad
Gini coefficient: 0.304   <- High = uneven
```

### 2. Auto Mode - Find Optimal Configuration (Recommended)

Let the tool find the best parameters automatically:

```bash
python tools/balanced_bucket_optimizer.py \
    -i ./dataset \
    -o ./output \
    -r 1024 \
    --auto
```

Output:
```
[OPTIMAL CONFIG FOUND] Score: 23.80
  - bucket_reso_steps: 128   <- Use this in training
  - min_bucket_reso: 768     <- Use this in training
  - max_bucket_reso: 1024    <- Use this in training
  - Resulting buckets: 3
  - Imbalance ratio: 1.60x   <- Much better!
  - Gini coefficient: 0.095  <- Very balanced
```

### 3. Manual Mode - Specify Parameters

Use quantile-based bucketing with manual parameters:

```bash
python tools/balanced_bucket_optimizer.py \
    -i ./dataset \
    -o ./output \
    -r 1024 \
    -n 8 \
    --min_bucket_size 100
```

## Workflow Recommendation

### Step 1: Analyze Current Distribution
```bash
python tools/balanced_bucket_optimizer.py -i ./dataset -o ./output -r 1024 --simulate
```

Check:
- Imbalance ratio > 5x? -> Needs optimization
- Gini coefficient > 0.3? -> Needs optimization
- Too many buckets? -> Increase `bucket_reso_steps`

### Step 2: Find Optimal Config
```bash
python tools/balanced_bucket_optimizer.py -i ./dataset -o ./output -r 1024 --auto --dry_run
```

Note the optimal parameters.

### Step 3: Use Optimal Config in Training

Use the suggested parameters in your training command:
```bash
accelerate launch lumina_train.py \
    --bucket_reso_steps 128 \     # From auto mode
    --min_bucket_reso 768 \       # From auto mode
    --max_bucket_reso 1024 \      # From auto mode
    ...
```

Or resize images to optimal buckets:
```bash
python tools/balanced_bucket_optimizer.py -i ./dataset -o ./output_optimized -r 1024 --auto
```

## Command Line Options

### Required
| Option | Description |
|--------|-------------|
| `-i, --input_dir` | Input folder containing images |
| `-o, --output_dir` | Output folder for processed images |

### Resolution Settings
| Option | Default | Description |
|--------|---------|-------------|
| `-r, --base_resolution` | 1024 | Base resolution (bucket area = base^2) |
| `--bucket_reso_steps` | 64 | Resolution step size (like sd-scripts) |
| `--min_bucket_reso` | 256 | Minimum bucket dimension |
| `--max_bucket_reso` | base*2 | Maximum bucket dimension |

### Mode Selection
| Option | Description |
|--------|-------------|
| `--simulate` | Only show bucket distribution (no processing) |
| `--auto` | Auto-optimize parameters via simulation |
| `--analyze_only` | Only show aspect ratio analysis |
| `--dry_run` | Show what would be done without processing |

### Manual Mode Options
| Option | Default | Description |
|--------|---------|-------------|
| `-n, --num_buckets` | 10 | Target number of buckets |
| `--min_bucket_size` | 50 | Minimum images per bucket |
| `--max_imbalance` | 5.0 | Maximum imbalance ratio allowed |

### Output Options
| Option | Default | Description |
|--------|---------|-------------|
| `--bucket_folders` | False | Create subfolder per bucket |
| `--keep_extension` | False | Keep original extension (vs convert to jpg) |
| `--quality` | 95 | JPEG quality (1-100) |
| `--workers` | auto | Parallel workers (auto-detect CPU cores if not set) |
| `--num_repeats` | 1 | Repeats for bucket size calculation |

### Resize Mode Options
| Option | Description |
|--------|-------------|
| (default) | Crop from center to fit exact bucket dimensions |
| `--resize` | Stretch/distort to exact bucket dimensions (guarantees bucket balance) |
| `--no_crop` | Simple resize preserving AR (creates different output dimensions) |

## Resize Modes Explained

The tool offers 3 different resize modes. Choose based on your priority:

### Comparison Table

| Mode | Content | Aspect Ratio | Bucket Size | Use Case |
|------|---------|--------------|-------------|----------|
| **Default (crop)** | Partial (cropped) | Preserved | Exact match | When AR match is important |
| **--resize** | Full (stretched) | Distorted | Exact match | When bucket balance is critical |
| **--no_crop** | Full (preserved) | Preserved | Different | When content preservation is critical |

### 1. Default Mode (Crop from Center)
```
Image: 1920x1080 (AR=1.78) → Bucket: 1024x1024 (AR=1.0)
Result: Crop center 1080x1080, then resize to 1024x1024
        ✓ Exact bucket dimensions
        ✓ Aspect ratio preserved (no distortion)
        ✗ Content lost (sides cropped)
```

### 2. --resize Mode (Stretch to Fit)
```
Image: 1920x1080 (AR=1.78) → Bucket: 1024x1024 (AR=1.0)
Result: Stretch directly to 1024x1024
        ✓ Exact bucket dimensions
        ✓ Full content preserved
        ✗ Aspect ratio distorted (image stretched)
```

### 3. --no_crop Mode (Preserve AR)
```
Image: 1920x1080 (AR=1.78) → Bucket max: 1024
Result: Resize to 1024x576 (keeping AR=1.78)
        ✓ Aspect ratio preserved
        ✓ Full content preserved
        ✗ Different output dimensions (creates new buckets)
```

## Usage Examples

```bash
# Default: crop from center (recommended for most cases)
python tools/balanced_bucket_optimizer.py -i ./data -o ./output -r 1024 --auto

# Stretch to exact bucket (when bucket balance is critical)
python tools/balanced_bucket_optimizer.py -i ./data -o ./output -r 1024 --auto --resize

# Preserve AR (for analysis or when content is critical)
python tools/balanced_bucket_optimizer.py -i ./data -o ./output -r 1024 --auto --no_crop
```

## Recommendation

| Scenario | Recommended Mode |
|----------|------------------|
| General training | Default (crop) |
| Strict bucket balance needed | `--resize` |
| Analysis/preview only | `--no_crop` |
| Portrait/landscape mix with similar AR | Default (crop) |
| Highly diverse AR dataset | Consider `--resize` |

## Understanding Metrics

### Imbalance Ratio
```
imbalance = max_bucket_size / min_bucket_size
```
- **< 3x**: Good - buckets are balanced
- **3-5x**: Acceptable - minor imbalance
- **5-10x**: Warning - may affect training
- **> 10x**: Critical - training will be unstable

### Gini Coefficient
Measures inequality in distribution (like wealth inequality).
- **0.0**: Perfect equality (all buckets same size)
- **0.2-0.3**: Low inequality
- **0.3-0.4**: Moderate inequality
- **> 0.4**: High inequality

### Coefficient of Variation (CV)
```
CV = std_deviation / mean
```
- **< 0.3**: Low variation
- **0.3-0.5**: Moderate variation
- **> 0.5**: High variation

## Auto Mode Mechanism

The auto mode is **data-driven**: it simulates bucket creation exactly like sd-scripts does, then scores each configuration to find the best one.

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Generate configuration grid                            │
├─────────────────────────────────────────────────────────────────┤
│  bucket_reso_steps: [32, 64, 128, 256]                          │
│  min_bucket_reso:   [256, 512, 768]                             │
│  max_bucket_reso:   [1024, 1536, 2048, base*2]                  │
│  → ~30 valid configurations to test                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Simulate bucket creation for each config               │
├─────────────────────────────────────────────────────────────────┤
│  For each image:                                                │
│  1. Calculate aspect ratio = width / height                     │
│  2. Find bucket with closest aspect ratio                       │
│  3. Assign image to that bucket                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Calculate metrics                                      │
├─────────────────────────────────────────────────────────────────┤
│  - Imbalance ratio = max_bucket_size / min_bucket_size          │
│  - Gini coefficient (inequality measure)                        │
│  - CV = standard deviation / mean                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Score configuration (lower = better)                   │
├─────────────────────────────────────────────────────────────────┤
│  score = 0                                                      │
│  if imbalance > 4: score += (imbalance - 4) × 10               │
│  if min_size < 50: score += (50 - min_size) × 0.5              │
│  score += cv × 5                                                │
│  score += gini × 3                                              │
│  score += |buckets - ideal| × 0.5                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Select config with LOWEST score                        │
└─────────────────────────────────────────────────────────────────┘
```

### Scoring Formula

```python
score = 0

# Penalize high imbalance (most important)
if imbalance > 4.0:
    score += (imbalance - 4.0) * 10

# Penalize small buckets (< 50 images)
if min_bucket_size < 50:
    score += (50 - min_bucket_size) * 0.5

# Penalize high CV (coefficient of variation)
score += cv * 5

# Penalize wrong bucket count
ideal = total_images // 500
score += abs(num_buckets - ideal) * 0.5

# Penalize high Gini (inequality)
score += gini * 3
```

### Example Optimization

| Config | Steps | Min | Max | Buckets | Imbalance | Score |
|--------|-------|-----|------|---------|-----------|-------|
| A | 64 | 256 | 2048 | 47 | 15.2x | 156.7 |
| B | 128 | 512 | 1536 | 12 | 3.8x | 8.4 |
| **C** | **128** | **768** | **1536** | **6** | **2.1x** | **3.2** ✓ |
| D | 256 | 768 | 1024 | 3 | 1.8x | 5.1 |

Config C is selected with the lowest score.

## Examples

### Large Dataset (100k+ images)

```bash
# Step 1: Analyze
python tools/balanced_bucket_optimizer.py \
    -i ./large_dataset \
    -o ./output \
    -r 1024 \
    --simulate \
    --num_repeats 4

# Step 2: Optimize
python tools/balanced_bucket_optimizer.py \
    -i ./large_dataset \
    -o ./output \
    -r 1024 \
    --auto
```

### Small Dataset with Diverse Aspect Ratios

```bash
python tools/balanced_bucket_optimizer.py \
    -i ./diverse_dataset \
    -o ./output \
    -r 1024 \
    --auto \
    --keep_extension
```

### Using with Training

After running auto mode, update your training config:

**Before (problematic):**
```yaml
bucket_reso_steps: 64
min_bucket_reso: 256
max_bucket_reso: 4096
```

**After (optimized):**
```yaml
bucket_reso_steps: 128    # From auto mode suggestion
min_bucket_reso: 768      # From auto mode suggestion
max_bucket_reso: 1536     # From auto mode suggestion
```

## Output Files

- `output_dir/`: Processed images + caption files
- `./bucket_report.txt`: Detailed analysis report
- `./bucket_config.json`: Bucket configuration (can be used for training)

## Troubleshooting

### "Too many buckets" warning
Increase `bucket_reso_steps`:
```bash
--bucket_reso_steps 128  # or 256
```

### "Imbalance ratio too high" warning
Options:
1. Use `--auto` mode to find better parameters
2. Increase `bucket_reso_steps` to reduce bucket count
3. Reduce `max_bucket_reso` to limit resolution range

### "Not enough images per bucket"
Options:
1. Reduce `--num_buckets` in manual mode
2. Use `--auto` mode which considers minimum bucket size
3. Add more images to dataset

## Technical Details

This tool replicates sd-scripts bucket creation logic from:
- `library/model_util.py:make_bucket_resolutions()`
- `library/train_util.py:BucketManager.select_bucket()`

The simulation is accurate for `no_upscale=False` (default training mode).

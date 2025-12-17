# Checkpointing Guide

## Overview

The evaluation system now supports automatic checkpointing to save progress during long runs. This prevents data loss if a run is interrupted and allows you to resume from where you left off.

## Features

1. **Periodic Checkpoints**: Save progress every N samples (default: 10)
2. **Automatic Resume**: Resume from latest checkpoint automatically
3. **Progress Tracking**: Checkpoints include sample count and remaining work
4. **Safe Recovery**: Checkpoints are separate from final results

## Usage

### Basic Checkpointing

Save checkpoint every 10 samples (default):
```bash
python -m cobra_eval --method scratchpad_compare --num_samples 100 --checkpoint_interval 10
```

Save checkpoint every 5 samples:
```bash
python -m cobra_eval --method scratchpad_compare --num_samples 100 --checkpoint_interval 5
```

Disable checkpointing:
```bash
python -m cobra_eval --method scratchpad_compare --num_samples 100 --checkpoint_interval 0
```

### Resuming from Checkpoint

**Automatic Resume** (recommended):
```bash
python -m cobra_eval --method scratchpad_compare --num_samples 100 --resume_from_checkpoint
```

This will:
1. Look for the latest checkpoint in the output directory
2. Load all processed samples
3. Skip those samples and continue from where it left off
4. Complete the remaining samples

**Manual Resume**:
```bash
python -m cobra_eval --method scratchpad_compare --num_samples 100 --load_results results/run_XXX/method/checkpoint_method_50_20251130_120000.json
```

## Checkpoint Files

### Naming Convention

Checkpoints are saved as:
```
checkpoint_{method}_{sample_count}_{timestamp}.json
```

Example:
```
checkpoint_scratchpad_compare_10_20251130_120000.json
checkpoint_scratchpad_compare_20_20251130_120500.json
checkpoint_scratchpad_compare_30_20251130_121000.json
```

### File Structure

```json
{
  "meta": {
    "timestamp": "2025-11-30T12:00:00",
    "config": { ... },
    "method": "scratchpad_compare",
    "samples_processed": 10,
    "total_samples": 100,
    "is_checkpoint": true
  },
  "checkpoint_info": {
    "processed_count": 10,
    "remaining_count": 90
  },
  "results": [
    {
      "image_id": 0,
      "reference_captions": [...],
      "generated_caption": "...",
      "reasoning_trace": "...",
      "metadata": {...},
      "metrics": {...}
    },
    ...
  ]
}
```

## Workflow Example

### Long Run with Checkpointing

```bash
# Start a long run (1000 samples, checkpoint every 50)
python -m cobra_eval \
  --method scratchpad_compare \
  --num_samples 1000 \
  --checkpoint_interval 50

# If interrupted, resume automatically:
python -m cobra_eval \
  --method scratchpad_compare \
  --num_samples 1000 \
  --checkpoint_interval 50 \
  --resume_from_checkpoint
```

### Progress Monitoring

Checkpoints are saved periodically, so you can monitor progress:

```bash
# List checkpoints
ls -lh results/run_*/method/checkpoint_*.json

# Check latest checkpoint
cat results/run_*/method/checkpoint_*_latest.json | jq '.checkpoint_info'
```

## Best Practices

### 1. Choose Appropriate Interval

- **Small datasets (<100)**: Checkpoint every 10-20 samples
- **Medium datasets (100-1000)**: Checkpoint every 50-100 samples
- **Large datasets (>1000)**: Checkpoint every 100-200 samples

**Trade-off**: More frequent checkpoints = more I/O overhead but better recovery

### 2. Use Automatic Resume

Always use `--resume_from_checkpoint` for long runs:
```bash
python -m cobra_eval --method scratchpad_compare --num_samples 1000 --resume_from_checkpoint
```

### 3. Monitor Disk Space

Checkpoints can accumulate. Clean up old checkpoints periodically:
```bash
# Keep only latest checkpoint
find results/ -name "checkpoint_*.json" -not -name "*_latest.json" -delete
```

### 4. Verify Checkpoints

Before resuming, verify the checkpoint is valid:
```python
import json
with open("checkpoint_file.json") as f:
    data = json.load(f)
    print(f"Processed: {data['checkpoint_info']['processed_count']}")
    print(f"Remaining: {data['checkpoint_info']['remaining_count']}")
    print(f"Results: {len(data['results'])}")
```

## Troubleshooting

### Checkpoint Not Found

If `--resume_from_checkpoint` doesn't find a checkpoint:
1. Check the output directory exists
2. Verify checkpoint files are present: `ls results/run_*/method/checkpoint_*.json`
3. Use `--load_results` with explicit path

### Checkpoint Outdated

If checkpoint is from a different configuration:
1. The system will detect mismatches
2. Use `--load_results` with explicit checkpoint path
3. Or start fresh without resume

### Partial Results

If a run completes but you want to add more samples:
1. Load the final results file (not checkpoint)
2. Use `--load_results` with the results file
3. Set `--num_samples` to total desired count

## Integration with Other Features

### Checkpointing + Scratchpad Compare

Checkpointing works seamlessly with `scratchpad_compare`:
```bash
python -m cobra_eval \
  --method scratchpad_compare \
  --num_samples 500 \
  --checkpoint_interval 50 \
  --resume_from_checkpoint
```

All pass comparisons are saved in each checkpoint.

### Checkpointing + Multiple Methods

When running multiple methods (`--method all`), each method gets its own checkpoints:
- `checkpoint_baseline_*.json`
- `checkpoint_scratchpad_*.json`
- `checkpoint_llava_cot_*.json`

Resume works per-method automatically.

## Performance Impact

- **Checkpoint Save Time**: ~0.1-0.5 seconds per checkpoint (depends on data size)
- **Memory**: Minimal (checkpoints are written to disk, not kept in memory)
- **Disk Space**: ~1-5 MB per checkpoint (depends on number of samples and method)

For 1000 samples with checkpoint every 50:
- ~20 checkpoints
- ~20-100 MB total disk space
- ~2-10 seconds total checkpoint overhead

## Example Output

```
Starting generation at 10:00:00...
[10:00:15] Processed 1/100 samples...
[10:00:30] Processed 2/100 samples...
...
[10:05:00] Processed 10/100 samples...
  → Checkpoint saved: checkpoint_scratchpad_compare_10_20251130_100500.json (10/100 samples)
[10:05:15] Processed 11/100 samples...
...
[10:10:00] Processed 20/100 samples...
  → Checkpoint saved: checkpoint_scratchpad_compare_20_20251130_101000.json (20/100 samples)
```

If interrupted and resumed:
```
Found checkpoint: checkpoint_scratchpad_compare_20_20251130_101000.json
Loaded checkpoint: 20 samples processed, 80 remaining
[10:15:00] Processed 21/100 samples (Cached)...
[10:15:15] Processed 22/100 samples (Cached)...
...
```


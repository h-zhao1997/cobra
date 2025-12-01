#!/bin/bash
# Script to run QLoRA fine-tuning with 100 training examples
# Output will be saved in a unique folder indicating 100 samples were used

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "env" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
fi

# Create unique output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./qlora_outputs_10000samples_${TIMESTAMP}"

echo "=================================================================================="
echo "Running QLoRA fine-tuning with 10000 training examples"
echo "Output directory: ${OUTPUT_DIR}"
echo "=================================================================================="

# Run the training script with memory-saving settings
python train_qlora.py \
    --model_id cobra+3b \
    --dataset_max_samples 10000 \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1

echo ""
echo "Training complete! Output saved to: ${OUTPUT_DIR}"


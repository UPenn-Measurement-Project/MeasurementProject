#!/bin/bash
# Run all segmentation experiments sequentially.
# Must be executed from SEGMENTATIONS/seg_experiments/
#   cd SEGMENTATIONS/seg_experiments && bash run_experiments.sh

set -e

mkdir -p model_saves results

run_exp() {
    local name=$1
    shift
    echo ""
    echo "=========================================="
    echo "TRAINING: $name"
    echo "=========================================="
    python3 train.py --name "$name" --esl 5 "$@"

    echo ""
    echo "=========================================="
    echo "TESTING: $name"
    echo "=========================================="
    python3 test.py --name "$name" --ds test

    echo ""
    echo "Done: $name"
    echo "=========================================="
}

# 1. Baseline — BCE loss, no augmentation, no scheduler (mirrors current mod_seg setup)
run_exp baseline

# 2. Dice loss only
run_exp loss_dice --loss dice

# 3. BCE + Dice (0.5/0.5 combo) — directly optimizes the evaluation metric
run_exp loss_bce_dice --loss bce_dice

# 4. Focal loss (gamma=2) — down-weights easy background pixels
run_exp loss_focal --loss focal

# 5. Geometric augmentation — random rotation ±12°, scale 0.85-1.15, small translation
run_exp aug_geom --aug

# 6. LR scheduler — ReduceLROnPlateau halves LR when val loss plateaus
run_exp scheduler --scheduler

# 7. Combined — best guesses together: BCE+Dice, augmentation, and scheduler
run_exp combined --loss bce_dice --aug --scheduler

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo ""
echo "Results:"
cat results/results.csv

# cnn_on_points: Code Guide

This document walks through how the cnn_on_points model works end-to-end, from raw DXA images to predicted femur measurements. It's meant to help new contributors understand the codebase.

## Overview

The goal: given a DXA hip X-ray image, predict 10 anatomical measurements of one femur. The approach is to train a CNN that outputs 11 numbers representing keypoint parameters in a relative coordinate system, then convert those to the 10 measurements using geometry.

## File Structure

```
cnn_on_points/
    data_utils/
        data_utils.py     # Data loading, preprocessing, augmentation
    model/
        model.py          # CNN architectures + coordinate/measurement math
    model_saves/          # Saved .pth model weights (gitignored)
    test_results/         # Test output images + CSVs (gitignored)
    train.py              # Training script
    test.py               # Testing script
    test_lateral_cortical.py  # Diagnostic for lateral cortical thickness
```

## The 10 Measurements

Each femur side (left and right) has these 10 measurements:

| Index | Measurement | What it is |
|-------|-------------|------------|
| 0 | Medial Cortical Thickness | Inner cortex thickness at the shaft |
| 1 | Lateral Cortical Thickness | Outer cortex thickness at the shaft |
| 2 | Shaft Width | Total width of the femoral shaft |
| 3 | Femoral Head Diameter | Diameter of the ball at the top of the femur |
| 4 | Horizontal Offset | Horizontal distance from shaft center to femoral head center |
| 5 | Vertical Offset | Vertical distance from shaft center to femoral head center |
| 6 | Femoral Neck Width | Width of the narrow neck connecting head to shaft |
| 7 | Hip Axis Length | Total length from lateral shaft to femoral head edge |
| 8 | Femoral Neck Axis Length | Length along the neck axis |
| 9 | Femoral Neck-Shaft Angle | Angle between the neck axis and the shaft (degrees) |

Measurements 0-8 are in mm. Measurement 9 is in degrees.

## The Keypoint System

Rather than predicting the 10 measurements directly, the model predicts 11 parameters that define keypoint positions in a **relative coordinate system** centered on the femur. These keypoints are labeled with letters:

```
    A ---- D          (femoral head top edge)
   / \
  /   \
 F     G              (neck axis endpoints)
 |   |
 H ---+--- H'         (neck width endpoints)
      |
      B                (neck-shaft intersection, on the neck axis line)
      |
 W -- X -- Y -- Z     (shaft: outer left, inner left, inner right, outer right)
      |
      C                (shaft center)
```

The model outputs 11 numbers. Here's what each one means:

| Output Index | Meaning |
|---|---|
| 0, 1 | D point: (x, y) of femoral head edge. Defines the neck-shaft axis slope |
| 2 | F point: x-coordinate (y is derived from the slope through D) |
| 3 | G point: x-coordinate (y is derived from the slope through D) |
| 4, 5 | H point: (x, y) of one side of the neck |
| 6, 7 | W point: (x, y) of the lateral shaft edge. y is shared by W, X, Y, Z |
| 8 | X point: x-coordinate (medial cortical inner edge) |
| 9 | Y point: x-coordinate (lateral cortical inner edge) |
| 10 | Z point: x-coordinate (medial shaft edge) |

Key constraints baked into this parameterization:
- F and G lie on the same line through the origin as D (the neck axis)
- W, X, Y, Z all share the same y-coordinate (they're on a horizontal line across the shaft)
- B is derived as the projection of H onto the line through D
- H' is the reflection of H across B

## Code Walkthrough

### 1. Data Loading (`data_utils/data_utils.py`)

**`DataProcessor.__init__`**: Sets everything up.
- Reads the measurement CSV, drops metadata columns, drops rows with NaN values
- Finds all `.dcm` DICOM images in the image directory
- Keeps only images that have corresponding measurement data
- Splits into train/val/test using a seeded random permutation

**`DataProcessor.create_ds`**: Creates a PyTorch DataLoader.
- For each image, creates **2 DataPoints** (original + horizontally flipped)
- When augmentation is enabled, creates additional DataPoints for each combination of rotation, scale, and crop values
- Without augmentation: 2 DataPoints per image (left femur + right femur)
- With augmentation: can be 2 * rotations * scales * crops per image

**`DataPoint.__init__`**: Preprocesses a single image.
1. Reads the DICOM file with pydicom
2. Normalizes pixel values to [0, 1]
3. Optionally flips horizontally (to get the other femur)
4. Applies augmentation: crop from top-left, scale down, pad back to target size, rotate
5. Selects the correct 10 measurements (left or right depending on flip)
6. Scales measurements by `aug_scale / aug_crop` to match the augmented image

The flip logic: the DXA image shows both femurs. Flipping the image horizontally turns the left femur into a right-femur-like orientation. When flipped, we use the right side measurements (indices 10-19); when not flipped, we use the left side measurements (indices 0-9).

Each DataPoint stores a tuple: `(image, measurements, aug_scale)`.

### 2. Model Architecture (`model/model.py`)

**`SimpleCNNModel`**: A straightforward CNN.
```
Input (1, 192, 240)
    -> Conv2d(1, 32) + ReLU + MaxPool    -> (32, 96, 120)
    -> Conv2d(32, 64) + ReLU + MaxPool   -> (64, 48, 60)
    -> Conv2d(64, 128) + ReLU + MaxPool  -> (128, 24, 30)
    -> Conv2d(128, 256) + ReLU + MaxPool -> (256, 12, 15)
    -> Flatten                           -> (46080)
    -> Linear(46080, 4096) + ReLU        -> (4096)
    -> Linear(4096, 11)                  -> (11)
```

The 11 outputs are the keypoint parameters described above.

**`AlexNet`**: A larger AlexNet-style variant with 5 conv layers and larger kernels.

**`BNRelu`**: Helper class for optional batch normalization. Can be "none" (just ReLU), "before" (BN then ReLU), or "after" (ReLU then BN). Note: batch norm has been found to cause issues with this model -- use "none" for best results.

### 3. Coordinate Geometry (`model/model.py`)

Three key functions convert between model outputs, keypoint coordinates, and measurements. All coordinates are in a **relative** system centered on the femur (origin = point A, the center of the femoral head).

**`model_to_coord(model_out)`**: Converts the 11 model outputs to 12 keypoint (x, y) coordinates.
- Uses the slope defined by D to place F and G on the neck axis
- Derives B by projecting H onto the neck axis line
- Derives H' by reflecting H across B
- Returns coordinates + the length AB (distance from head center to neck-shaft intersection)

**`measurements_to_coord(measurements, ab)`**: Converts ground truth measurements (in mm) to the same keypoint coordinate system.
- First scales mm to pixels: `measurements * pix_per_mm * img_scale_factor`
- Uses the neck-shaft angle to compute rotation, then places all points geometrically
- Needs `ab` from the model (since AB distance isn't a ground truth measurement)

**`coord_to_measurements(coord)`**: The inverse -- converts keypoint coordinates back to the 10 measurements in mm.
- Medial cortical = X_x - W_x
- Lateral cortical = Z_x - Y_x
- Shaft width = Z_x - W_x
- Head diameter = 2 * ||D||
- Horizontal offset = midpoint of W and Z x-coords
- Vertical offset = -W_y
- Neck width = 2 * distance from H to the neck axis
- Hip axis length = ||F - G||
- Neck axis length = ||D - G||
- Neck-shaft angle = derived from slope of D

### 4. Training (`train.py`)

**Settings**:
- `pix_per_mm = 2400 / 408`: The original DXA images are 2400 pixels wide representing 408mm
- `img_scale_factor = 0.1`: Images are downscaled to 10% (240x192 pixels)
- Default learning rate: 0.001 (Adam optimizer)
- Default early stopping: 5 epochs with no improvement

**Training loop** (each epoch):
1. Forward pass: image -> model -> 11 params -> `model_to_coord` -> predicted keypoints
2. Ground truth: measurements -> `measurements_to_coord(yvals, ab)` -> real keypoints
   - Note: `ab` comes from the **model's** prediction, not ground truth (AB isn't in the data)
3. Loss: mean over batch of (sum of Euclidean distances between each predicted and real keypoint, divided by aug_scale)
4. Validation: same forward pass + computes per-measurement percent error using `coord_to_measurements`
5. Early stopping: saves best model by validation loss

**Loss function**: `mean(sum(||pred_i - real_i||) / aug_scale)` -- this is the sum of Euclidean distances across all 12 keypoints, normalized by augmentation scale. This means all keypoints are weighted equally regardless of how important or precise they need to be.

### 5. Testing (`test.py`)

Loads a saved model and runs inference on the specified dataset split. Outputs:
- Per-measurement percent error
- Mean and std of absolute errors
- Mean and std of true measurements
- Per-sample scatter plots (predicted vs real keypoints)
- Optional CSV with per-sample predicted, true, and error values

### 6. Lateral Cortical Diagnostic (`test_lateral_cortical.py`)

A specialized test that focuses on lateral cortical thickness (the hardest measurement). Outputs a CSV with per-sample:
- Predicted Y and Z keypoint x-positions (pixels)
- Predicted lateral cortical thickness (pixels and mm)
- Real lateral cortical thickness (pixels and mm)
- Error in both units + percent error

This is useful for understanding why lateral cortical error is so high: the measurement is only ~3 pixels at 0.1x scale, so sub-pixel keypoint errors cause large measurement errors.

## Known Issues

- **Lateral cortical thickness**: ~98% error. At 0.1x scale, average lateral cortical is only 3.3 pixels. A 1-pixel error on either Y or Z is a ~30% measurement error.
- **Batch normalization**: Causes large errors. Likely because per-image normalization + zero-padded augmented images create inconsistent batch statistics. Use `--bn none`.
- **Loss function doesn't weight measurements**: The keypoint loss treats all points equally. Cortical thickness depends on two points 3px apart; hip axis length depends on two points 50px apart. The optimizer barely feels cortical errors.

## Quick Start

```bash
cd cnn_on_points

# Train (from the cnn_on_points directory)
python3 train.py --mdata 20260407.csv --idata box_images --model basic --esl 5

# Test
python3 test.py --mdata 20260407.csv --idata box_images --ds test --model basic --makecsv

# Lateral cortical diagnostic
python3 test_lateral_cortical.py --mdata 20260407.csv --idata box_images --ds test --model basic
```

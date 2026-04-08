# MeasurementProject

Automated femur measurement prediction from hip DXA images. Multiple deep learning approaches to predict 10 anatomical measurements per femur side (medial/lateral cortical thickness, shaft width, femoral head diameter, horizontal/vertical offset, neck width, hip axis length, neck axis length, neck-shaft angle).

## Active Models

| Directory | Approach | Description |
|---|---|---|
| `cnn_on_points/` | Direct regression | CNN regresses 11 keypoint params, converted to measurements via geometry |
| `heatmaps/` | Heatmap keypoints | Encoder produces 13 heatmaps, soft-argmax to coordinates |
| `seg_heatmaps/` | Seg backbone + heatmaps | Frozen pretrained segmentation encoder + heatmap decoder |
| `dual_heatmaps/` | Dual encoder + heatmaps | Frozen seg encoder + trainable raw encoder with fused skip connections |

## Other Models

| Directory | Approach |
|---|---|
| `autoencoder/` | U-Net autoencoder for image reconstruction (used as backbone) |
| `cnn_backbone/` | Pretrained autoencoder encoder + linear head |
| `keypoint_detection/` | Self-supervised keypoint detection via image reconstruction |
| `pretrained_cnn/` | DenseNet-121 (torchxrayvision) backbone + linear head |
| `null_model/` | Baseline: predicts training set mean |
| `SEGMENTATIONS/` | U-Net segmentation models (original 512x256, modified 192x240) |
| `NONACTIVE_MODELS/` | Archived/deprecated approaches |

## Data
- `data/measurements/` -- Dated CSVs of human-annotated measurements (~3094 images)
- `data/box_images/` -- DXA DICOM images (gitignored)
- `data/segmentation/` -- Segmentation masks (gitignored)

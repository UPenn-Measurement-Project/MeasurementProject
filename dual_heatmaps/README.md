# dual_heatmaps

Dual-encoder heatmap model. Combines a frozen pretrained segmentation encoder (bone boundaries) with a trainable raw encoder (image texture/intensity) via skip-level feature fusion.

## Architecture
- **Seg Encoder**: Frozen, pretrained from `SEGMENTATIONS/mod_seg` -- provides bone boundary features
- **Raw Encoder**: Trainable, learns intensity/texture features (especially for cortical thickness)
- **DualDecoder**: Concatenates skip connections from both encoders at every level, outputs 13 heatmaps
- Decoder input channels are doubled to handle fused skips (512->384->192->96)
- Input: 192x240 grayscale (0.1x scale)

## Usage
```bash
# train (run from project root)
python3 -m dual_heatmaps.train --mdata 20260407.csv --esl 5

# test
python3 -m dual_heatmaps.test --mdata 20260407.csv --ds test --makecsv
```

# seg_heatmaps

Heatmap model using a pretrained segmentation U-Net encoder as backbone. The seg encoder is frozen; only the decoder trains.

## Architecture
- **Encoder**: SegUNet encoder from `SEGMENTATIONS/mod_seg`, pretrained on femur segmentation, frozen
- **Decoder**: U-Net decoder with skip connections, outputs 13 keypoint heatmaps
- Soft-argmax to extract coordinates, then geometric conversion to 10 measurements
- Input: 192x240 grayscale (0.1x scale)

## Usage
```bash
# train (run from project root)
python3 -m seg_heatmaps.train --mdata 20260407.csv --esl 5

# test
python3 -m seg_heatmaps.test --mdata 20260407.csv --ds test --makecsv
```

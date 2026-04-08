# heatmaps

Heatmap-based keypoint detection model for femur measurements. Predicts 13 keypoint heatmaps, converts to coordinates via soft-argmax, then to 10 measurements via geometry.

## Architecture
- 4-layer encoder (32->64->128->256) with stride-2 downsampling
- 1x1 conv to produce 13 heatmaps at reduced resolution
- Soft-argmax to extract (x, y) coordinates per keypoint
- Input: 192x240 grayscale (0.1x scale)

## Usage
```bash
# train (run from project root)
python3 -m heatmaps.train --mdata 20260407.csv --idata box_images --model hm --esl 5

# test
python3 -m heatmaps.test --mdata 20260407.csv --idata box_images --ds test --model hm --makecsv
```

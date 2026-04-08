# keypoint_detection

Self-supervised keypoint detection using image reconstruction. A location encoder produces keypoint heatmaps from a target image, an image encoder encodes a source image, and a decoder reconstructs the source from the target's keypoints + source's encoding.

## Architecture
- **ImageEncoder**: Encodes source image to feature map
- **LocationEncoder**: Produces K keypoint heatmaps from target image, converted to Gaussian maps via soft-argmax
- **Decoder**: Reconstructs source image from concatenated source features + target Gaussian keypoints
- Learns keypoints without explicit coordinate supervision

## Usage
```bash
cd keypoint_detection
python3 train.py --mdata <csv> --idata <img_dir> --esl 5
python3 test.py --mdata <csv> --idata <img_dir> --ds test
```

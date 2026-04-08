# autoencoder

U-Net autoencoder for DXA image reconstruction. Encoder-decoder with optional skip connections (residual mode). Used as a backbone for other models (e.g., cnn_backbone).

## Architecture
- **Encoder**: 4-layer (32->64->128->256) with stride-2 downsampling, BatchNorm, ReLU
- **Decoder**: Mirrors encoder with bilinear upsampling, optional skip connections from encoder
- Sigmoid output for image reconstruction
- Input: 192x240 grayscale

## Usage
```bash
cd autoencoder
python3 train.py --mdata <csv> --idata <img_dir> --esl 5
python3 test.py --mdata <csv> --idata <img_dir> --ds test
```

# cnn_backbone

CNN measurement model that uses a pretrained autoencoder encoder as its backbone. The autoencoder encoder is frozen and a linear head is trained on top to regress 11 keypoint parameters.

## Architecture
- **Backbone**: Frozen encoder from `autoencoder/` model
- **Head**: Linear layers on top of encoded features -> 11 keypoint parameters
- Same coordinate-to-measurement geometry as cnn_on_points

## Usage
```bash
python3 -m cnn_backbone.train --mdata <csv> --idata <img_dir> --model basic --esl 5
python3 -m cnn_backbone.test --mdata <csv> --idata <img_dir> --ds test --model basic --makecsv
```

# pretrained_cnn

Measurement model using a pretrained DenseNet-121 backbone from torchxrayvision (trained on chest X-rays). Fine-tunes the backbone with additional conv + linear layers to regress 11 keypoint parameters.

## Architecture
- **Backbone**: First 7 feature blocks of `xrv.models.DenseNet(weights="densenet121-res224-all")` -> 512 channels at 28x28
- **Head**: Conv 512->256->256, MaxPool, Flatten, Linear -> 256 -> 11
- Input: 224x224 (resized to match DenseNet)

## Usage
```bash
cd pretrained_cnn
python3 train.py --mdata <csv> --idata <img_dir> --esl 5
python3 test.py --mdata <csv> --idata <img_dir> --ds test
```

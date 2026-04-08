# cnn_on_points

CNN that directly regresses 11 keypoint parameters from DXA images. Keypoints are converted to 10 femur measurements via geometric formulas.

## Architecture
- **SimpleCNNModel**: 4-layer CNN (32->64->128->256 channels) with MaxPool, flatten, linear (-> 4096 -> 11 outputs)
- **AlexNet**: AlexNet-style variant with larger kernels
- Optional batch norm (before/after ReLU, or none)
- Input: 192x240 grayscale (0.1x scale)

## Usage
```bash
# train
python3 train.py --mdata 20260407.csv --idata box_images --model basic --esl 5

# test
python3 test.py --mdata 20260407.csv --idata box_images --ds test --model basic --makecsv

# lateral cortical diagnostic
python3 test_lateral_cortical.py --mdata 20260407.csv --idata box_images --ds test --model basic
```

# DenseUNet-pytorch
A PyTorch implementation of [U-Net](https://arxiv.org/abs/1505.04597) using a [DenseNet-121](https://arxiv.org/abs/1608.06993) backbone for the encoding and deconding path.

The DenseNet blocks are based on the implementation available in [torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py).

The input is restricted to RGB images and has shape ![formula](https://render.githubusercontent.com/render/math?math=(N,3,H,W)).
The output has shape ![formula](https://render.githubusercontent.com/render/math?math=(N,C_{\text{out}},H,W)), where ![formula](https://render.githubusercontent.com/render/math?math=C_{\text{out}}) is the number of output classes.

Optionally a pretrained model can be used to initalize the encoder.

## Requirements
- pytorch
- torchvision

## Usage
 
```python
from dense_unet import DenseUNet

pretrained_encoder_uri = 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
#
# for a local file use
#
# from pathlib import Path
# pretrained_encoder_uri = Path('/path/to/local/model.pth').resolve().as_uri()
#

num_output_classes = 3
model = DenseUNet(num_output_classes, pretrained_encoder_uri)

```

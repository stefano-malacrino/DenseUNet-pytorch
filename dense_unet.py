from torchvision.models import DenseNet
from torchvision.models.densenet import _Transition, _load_state_dict
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

class _DenseUNetEncoder(DenseNet):
    def __init__(self, skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, downsample):
        super(_DenseUNetEncoder, self).__init__(growth_rate, block_config, num_init_features, bn_size, drop_rate)
        
        self.skip_connections = skip_connections

        # remove last norm, classifier
        features = OrderedDict(list(self.features.named_children())[:-1])
        delattr(self, 'classifier')
        if not downsample:
            features['conv0'].stride = 1
            del features['pool0']
        self.features = nn.Sequential(features)
        
        for module in self.features.modules():
            if isinstance(module, nn.AvgPool2d):
                module.register_forward_hook(lambda _, input, output : self.skip_connections.append(input[0]))

    def forward(self, x):
        return self.features(x)
        
class _DenseUNetDecoder(DenseNet):
    def __init__(self, skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, upsample):
        super(_DenseUNetDecoder, self).__init__(growth_rate, block_config, num_init_features, bn_size, drop_rate)
        
        self.skip_connections = skip_connections
        self.upsample = upsample
        
        # remove conv0, norm0, relu0, pool0, last denseblock, last norm, classifier
        features = list(self.features.named_children())[4:-2]
        delattr(self, 'classifier')

        num_features = num_init_features
        num_features_list = []
        for i, num_layers in enumerate(block_config):
            num_input_features = num_features + num_layers * growth_rate
            num_output_features = num_features // 2
            num_features_list.append((num_input_features, num_output_features))
            num_features = num_input_features // 2
        
        for i in range(len(features)):
            name, module = features[i]
            if isinstance(module, _Transition):
                num_input_features, num_output_features = num_features_list.pop(1)
                features[i] = (name, _TransitionUp(num_input_features, num_output_features, skip_connections))

        features.reverse()
        
        self.features = nn.Sequential(OrderedDict(features))
        
        num_input_features, _ = num_features_list.pop(0)
        
        if upsample:
            self.features.add_module('upsample0', nn.Upsample(scale_factor=4, mode='bilinear'))
        self.features.add_module('norm0', nn.BatchNorm2d(num_input_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('conv0', nn.Conv2d(num_input_features, num_init_features, kernel_size=1, stride=1, bias=False))
        self.features.add_module('norm1', nn.BatchNorm2d(num_init_features))

    def forward(self, x):
        return self.features(x)
          
        
class _Concatenate(nn.Module):
    def __init__(self, skip_connections):
        super(_Concatenate, self).__init__()
        self.skip_connections = skip_connections
        
    def forward(self, x):
        return torch.cat([x, self.skip_connections.pop()], 1)

          
class _TransitionUp(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, skip_connections):
        super(_TransitionUp, self).__init__()
        
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, num_output_features * 2,
                                              kernel_size=1, stride=1, bias=False))
        
        self.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        self.add_module('cat', _Concatenate(skip_connections))
        self.add_module('norm2', nn.BatchNorm2d(num_output_features * 4))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(num_output_features * 4, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

class DenseUNet(nn.Module):
    def __init__(self, n_classes, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, downsample=False, pretrained_encoder_uri=None, progress=None):
        super(DenseUNet, self).__init__()
        self.skip_connections = []
        self.encoder = _DenseUNetEncoder(self.skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, downsample)
        self.decoder = _DenseUNetDecoder(self.skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, downsample)
        self.classifier = nn.Conv2d(num_init_features, n_classes, kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        
        self.encoder._load_state_dict = self.encoder.load_state_dict
        self.encoder.load_state_dict = lambda state_dict : self.encoder._load_state_dict(state_dict, strict=False)
        if pretrained_encoder_uri:
            _load_state_dict(self.encoder, str(pretrained_encoder_uri), progress)
        self.encoder.load_state_dict = lambda state_dict : self.encoder._load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        y = self.classifier(x)
        return self.softmax(y)
        

import torch
import torch.nn as nn
import torch.nn.functional as FU
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
from scipy.special import expi
from scipy.integrate import quad
from scipy import interpolate

#! -------------------
#! ResNet implementation for feature extraction
#! -------------------
__all__ = ['resnet18', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, low_dim=1000, in_channel=3, width=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.base = int(64 * width)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(self.base * 8 * block.expansion, low_dim)
        self.l2norm = Normalize(2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, layer=7):
        if layer <= 0:
            return x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if layer == 1:
            return x
        x = self.layer1(x)
        if layer == 2:
            return x
        x = self.layer2(x)
        if layer == 3:
            return x
        x = self.layer3(x)
        if layer == 4:
            return x
        x = self.layer4(x)
        if layer == 5:
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if layer == 6:
            return x
        x = self.fc(x)
        x = self.l2norm(x)
        return x

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

class MyResNet(nn.Module):
    def __init__(self, name='resnet18', pretrained=False):
        super(MyResNet, self).__init__()
        if name == 'resnet50':
            self.vision_to_touch = resnet50(in_channel=3, width=1.0, pretrained=pretrained)
            self.touch_to_vision = resnet50(in_channel=3, width=1.0, pretrained=pretrained)
        elif name == 'resnet18':
            self.vision_to_touch = resnet18(in_channel=3, width=1.0, pretrained=pretrained)
            self.touch_to_vision = resnet18(in_channel=3, width=1.0, pretrained=pretrained)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))
        
    def get_device(self):
        # get vision to touch device
        print(next(self.vision_to_touch.parameters()).device)
        # get touch to vision device
        print(next(self.touch_to_vision.parameters()).device)

    def forward(self, vision_input, touch_input, layer=7):
        feat_touch = self.touch_to_vision(touch_input, layer)
        return feat_touch
    

#! -------------------
#! HNF implementation
#! -------------------
class HNF(nn.Module):
    def __init__(self, m_dim, n_freq=10, hidden=128, dropout_rate=0.4):
        super().__init__()
        self.m_dim = m_dim
        self.n_freq = n_freq
        self.hidden = hidden
        self.dropout_rate = dropout_rate
        self.dvf_encoding_dim = (1 + 2 * n_freq)  # direction, velocity, force encoding dimension
        
        bb_path = "resnet_encoder/best_model.pth" # Pretrained backbone path
        self.texture_encoder = MyResNet(name='resnet50', pretrained=False)
        weights = torch.load(bb_path, map_location="cpu", weights_only=True)
        self.texture_encoder.load_state_dict(weights['model_state_dict'])
        
        self.image_encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, m_dim)
        )
        
        self.m_ln = nn.LayerNorm(m_dim)
        self.sig_ln = nn.LayerNorm(4 * self.dvf_encoding_dim)

        self.input_dim = m_dim + 4 * self.dvf_encoding_dim  # material + (2*d,v,F) encodings
        self.sigma_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.sigma_proj = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Softplus()
        )
        self.a_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.a_proj = nn.Linear(hidden, 1)
        
        # initialize weights with except for image encoder
        for m in self.sigma_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.a_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def positional_encoding(x, n_freq=6, include_input=True, log_sampling=True):
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0, n_freq - 1, n_freq)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (n_freq - 1), n_freq)
            
        out = [x] if include_input else []
        for freq in freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)

    def forward(self, I, d, v, F, t):
        """
        m: (B,M), d: (B,2), v: (B,N), F: (B,N), t: (B,N)
        Returns sigma, a: (B,N)
        """
        B, C, N = t.shape
        d = d.reshape(B, C, N, 2)  # (B,C,N,2)
        v = v.reshape(B, C, N, 1)  # (B,C,N,1)
        F = F.reshape(B, C, N, 1)  # (B,C,N,1)
        signals_encoding = self.positional_encoding(torch.cat([d,v,F], dim=-1), n_freq=self.n_freq, include_input=True)  # (B,C,N, 1+2*n_freq)*2
        signals_encoding = self.sig_ln(signals_encoding)
        
        # add dropout to signals_encoding for regularization
        signals_encoding = FU.dropout(signals_encoding, p=0.2, training=self.training)

        # rgb_image_placeholder Ok = torch.zeros_like(I)
        m = self.texture_encoder(I, I, layer=5)     # (B,2048)
        m = FU.adaptive_avg_pool2d(m, (1,1)).squeeze(-1).squeeze(-1)
        m = self.image_encoder(m)                                       # (B,M)
        m = m[:, None, None, :].expand(B, C, N, self.m_dim)             # (B,C,N,M)
        m = self.m_ln(m)

        r_feat = torch.cat([m, signals_encoding], dim=-1)               # (B,C,N,M + 3*dvf_encoding_dim)
        
        # if torch.randint(0, 10, (1,)) % 2 == 0 and self.training:
        #     r_feat = torch.cat([m, signals_encoding], dim=-1)               # (B,C,N,M + 3*dvf_encoding_dim)
        # else:
        #     r_feat = torch.cat([signals_encoding, m], dim=-1)      # (B,C,N,M + 3*dvf_encoding_dim)


        sigma = self.sigma_mlp(r_feat)  # (B, C, N, input_dim)
        sigma = self.sigma_proj(sigma)  # (B, C, N, 1)
        a = self.a_mlp(r_feat)          # (B, C, N, input_dim)
        a = self.a_proj(a)              # (B, C, N, 1)
        
        sigma = sigma.reshape(B, C, N, 1)
        a = a.reshape(B, C, N, 1)
        return a.squeeze(-1), sigma.squeeze(-1)  # (B,C,N), (B,C,N)
    
    def render(self, I, d, v, F, L=0.001, n_samples=5, stratified=True):
        """
        m: (B,M), d: (B,2), v: (B,N), F: (B,N), L: (B,), N: int
        Returns Az: (B,), aux: dict with 't', 'dt', 'sigma', 'a'
        """
        B = d.shape[0]
        C = d.shape[1] // n_samples
        
        # Sample points along the ray
        mids = 0.5 * (torch.linspace(0, 1, n_samples + 1, device=I.device)[:-1] +
                      torch.linspace(0, 1, n_samples + 1, device=I.device)[1:])
        t01 = mids.expand(B, n_samples)  # (B,N)
        t01 = t01.unsqueeze(1).repeat(1, C, 1)  # (B, C, N)
        if stratified:
            t01 = (t01 + (torch.rand_like(t01) - 0.5) / n_samples).clamp(0, 1)
        t = t01 * L  # (B, C, N)
        dt = torch.diff(torch.cat([t, t[..., -1:]], -1), dim=-1).clamp_min(1e-8)  # (B, C, N)

        Az, sigma = self.forward(I, d, v, F, t)  # (B,C,N), (B,C,N)

        alpha = 1 - torch.exp(-sigma * dt)  # (B,C,N)
        accumulated_contribution = torch.cumprod(1-alpha + 1e-10, dim=2)  # (B,C,N)
        accumulated_contribution = torch.cat([torch.ones((B, C, 1), device=I.device), accumulated_contribution[:, :, :-1]], dim=-1)  # (B,C,N)
        weights = alpha * accumulated_contribution  # (B,C,N)
        Az = (weights * Az).sum(dim=-1)  # (B,C)
        
        return Az
    




if __name__ == "__main__":
    torch.manual_seed(0)
    v1 = True
    if v1:
        Bout, Bin, M = 4, 1000, 512
        N = 10
        I  = torch.randn(Bout, 3, 960, 720)         # E(I)
        d  = torch.randn(Bout, Bin, 2)              # in-plane direction
        v  = torch.rand(Bout, Bin, 1)*0.3 + 0.1     # speed
        Fn = torch.rand(Bout, Bin, 1)*2.0           # normal force
        L  = 0.001                                  # path length
    else:
        Bout, M = 1000, 512
        N = 10
        I  = torch.randn(Bout, M)              # E(I)
        d  = torch.randn(Bout, 2)              # in-plane direction
        v  = torch.rand(Bout, N)*0.3 + 0.1     # speed
        Fn = torch.rand(Bout, N)*2.0           # normal force
        L  = 0.001                             # path length

    hnf = HNF(m_dim=M)
    Az = hnf.render(I, d, v, Fn, L, n_samples=N, stratified=True)
    a = torch.cat([Az])
    print("Rendered Az:", Az.detach().cpu().numpy())

import torch
import torch.nn as nn
DIM = 32
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm3d(planes)
            self.norm2 = nn.BatchNorm3d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm3d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm3d(planes)
            self.norm2 = nn.InstanceNorm3d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm3d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
class Encoder3D(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False):
        super(Encoder3D, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm3d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm3d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv3d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM,  stride=1)
        self.layer2 = self._make_layer(2*DIM, stride=2)

        # output convolution
        self.conv2 = nn.Conv3d(2*DIM, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout3d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c1, d1, h1, w1 = x.shape
        x = x.view(b*n, c1, d1, h1, w1).contiguous()

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)

        _, c2, d2, h2, w2 = x.shape
        return x.view(b, n, c2, d2, h2, w2).contiguous()


if __name__ == '__main__':
    input = torch.randn(1,10,3,3,480,640)
    encoder = Encoder3D()
    output = encoder(input)
    output = output.mean(dim=3)
    print(output.shape)
    # ([1, 10, 128, 480, 640])
import torch 
import torch.nn as nn 


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=None)
        return x

#OctConv(self, in_planes, out_planes, kernel, stride ,padding, groups=1, dilation=1)
class OctConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride ,padding, groups=1, dilation=1):
        super(OctConv, self).__init__()
        #print ( [ in_planes, out_planes, kernel, stride, padding] )
        assert all([len(i)==2 for i in [ in_planes, out_planes, kernel, stride, padding]])
        self.in_planes_high, self.in_planes_low = in_planes
        self.out_planes_high, self.out_planes_low = out_planes
        self.kernel_high, self.kernel_low = kernel 
        self.stride_high, self.stride_low = stride 
        self.padding_high, self.padding_low = padding
        self.conv2d_high_to_high = nn.Sequential( nn.Conv2d(self.in_planes_high,self.out_planes_high,
                                                            self.kernel_high, self.stride_high, self.padding_high ), 
                                                ) 
        self.conv2d_high_to_low  = nn.Sequential( nn.AvgPool2d( kernel_size = 2) , 
                                                  nn.Conv2d(self.in_planes_high,self.out_planes_low,
                                                            self.kernel_low, self.stride_low, self.padding_low  ),                                                   
                                                ) 
        self.conv2d_low_to_low   = nn.Sequential( nn.Conv2d(self.in_planes_low,self.out_planes_low,
                                                            self.kernel_low, self.stride_low, self.padding_low  ), 
                                                  
                                                ) 
        self.conv2d_low_to_high  = nn.Sequential( Interpolate(scale_factor=2, mode='nearest'), 
                                                  nn.Conv2d(self.in_planes_low,self.out_planes_high,
                                                            self.kernel_high, self.stride_high, self.padding_high  ),                                                   
                                                ) 

        
    def forward(self, x):
        assert len(x) == 2 
        x_h, x_l = x 
        # high to high
        y_h = self.conv2d_high_to_high(x_h) + self.conv2d_low_to_high(x_l) 
        y_l = self.conv2d_high_to_low(x_h)  + self.conv2d_low_to_low(x_l) 
        #print ('y_h shape: ',y_h.shape, ' , y_l shape: ', y_l.shape)
        return [y_h, y_l]


class OctNorm(nn.Module):
    def __init__(self, in_place):
        super(OctNorm, self).__init__()
        self.in_place = in_place
        self.norm_h = nn.BatchNorm2d(self.in_place[0])
        self.norm_l = nn.BatchNorm2d(self.in_place[1])
        
    def forward(self, x):
        # OctNorm
        assert (len(x) == 2)
        x_h, x_l = x
        x_h = self.norm_h(x_h)
        x_l = self.norm_l(x_l)
        return [x_h, x_l]


class OctRelu(nn.Module):
    def __init__(self, inplace = True):
        super(OctRelu, self).__init__()
        self.inplace = inplace
        self.Relu = nn.ReLU(self.inplace)
        self.Relu = nn.ReLU(self.inplace)

    def forward(self, x):
        # OctRelu
        assert (len(x) == 2)
        x_h, x_l = x
        x_h = self.Relu(x_h)
        x_l = self.Relu(x_l)
        return [x_h, x_l]

    
# self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
class OctMaxPool(nn.Module):
    def __init__(self, kernel =[3,3], stride=[2,2], padding=[1,1]):
        super(OctMaxPool, self).__init__()
        self.MaxPool_h = nn.MaxPool2d(kernel_size=kernel[0], stride=stride[0], padding=padding[0]) 
        self.MaxPool_l = nn.MaxPool2d(kernel_size=kernel[1], stride=stride[1], padding=padding[1]) 
        
    def forward(self, x):
        # OctMaxPool 
        assert (len(x) == 2)
        x_h, x_l = x
        x_h, x_l = self.MaxPool_h(x_h), self.MaxPool_l(x_l)
        return [x_h, x_l]


# nn.AdaptiveAvgPool2d((1, 1))
class OctAvePool(nn.Module):
    def __init__(self, kernel = (1,1)):
        super(OctAvePool, self).__init__()
        self.AvePool_h = nn.AdaptiveAvgPool2d(kernel) 
        self.AvePool_l = nn.AdaptiveAvgPool2d(kernel) 
        
    def forward(self, x):
        # OctAvePool 
        assert (len(x) == 2)
        x_h, x_l = x
        x_h, x_l = self.AvePool_h(x_h), self.AvePool_l(x_l)
        return [x_h, x_l]
 
import torch 
import torch.nn as nn
#from .utils import load_state_dict_from_url




#def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#    """3x3 convolution with padding"""
#    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                     padding=dilation, groups=groups, bias=False, dilation=dilation)


#def conv1x1(in_planes, out_planes, stride=1):
#    """1x1 convolution"""
#    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=[1,1], groups=1, dilation=[1,1]):
    """3x3 convolution with padding"""
    return OctConv(in_planes, out_planes, kernel = [3,3], stride=stride,
                     padding=dilation, groups=groups, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=[1,1]):
    """1x1 convolution"""
    return OctConv(in_planes, out_planes, kernel = [1,1], stride = stride ,padding = [0,0], groups=1, dilation=1)

# BasicBlock(inplanes, planes, stride=[1,1], downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=[1,1], downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = OctNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = OctNorm(planes)
        self.relu = OctRelu(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = OctNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out[0] += identity[0]
        out[1] += identity[1]
        out = self.relu(out)
        return out


# Bottleneck(inplanes, planes, stride=[1,1], downsample=None, groups=1,base_width=64, dilation=[1,1], norm_layer=None)    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=[1,1], downsample=None, groups=1,
                 base_width=64, dilation=[1,1], norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            #norm_layer = nn.BatchNorm2d
            norm_layer = OctNorm
        width = [ int(i) for i in (np.array(planes) * (base_width / 64.)) * groups ]
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = OctNorm(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = OctNorm(width)
        #self.conv3 = conv1x1(width, planes * self.expansion)
        self.conv3 = conv1x1(width, [i* self.expansion for i in planes])
        self.bn3 = OctNorm([i* self.expansion for i in planes])
        self.relu = OctRelu(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        # out += indentity
        out[0] += identity[0]
        out[1] += identity[1]
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            #norm_layer = nn.BatchNorm2d
            norm_layer = OctNorm
        #self._norm_layer = norm_layer
        self._norm_layer = OctNorm 
        #self.inplanes = 64
        self.inplanes = [64, 128]
        self.dilation = [1,1]
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.conv1 = OctConv([3,3], self.inplanes, [7,7], [2,2], [3,3])
        self.bn1 = norm_layer(self.inplanes)
        
        #self.relu = nn.ReLU(inplace=True)
        self.relu = OctRelu(inplace = True)        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = OctMaxPool( kernel =[3,3], stride=[2,2], padding=[1,1] )
        self.layer1 = self._make_layer(block, [64,64], layers[0])
        self.layer2 = self._make_layer(block, [128,128], layers[1], stride=[2,2],
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, [256,256], layers[2], stride=[2,2],
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, [512,512], layers[3], stride=[2,2],
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = OctAvePool()
        self.fc = nn.Linear(512 * block.expansion*2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=[1,1], dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = [1,1]
        if stride != [1,1] or self.inplanes != [i* block.expansion for i in planes ]:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, [i* block.expansion for i in planes ], stride),
                norm_layer([i* block.expansion for i in planes ]),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = [i* block.expansion for i in planes ]
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_h, x_l = x,nn.functional.interpolate(x, scale_factor = 0.5 )
        #x = self.conv1(x)
        x_h, x_l = self.conv1([x_h, x_l])
        x = [x_h, x_l]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x_h, x_l = x[0].view(x[0].size(0), -1), x[1].view(x[1].size(0), -1)
        x = torch.cat([x_h,x_l], 1)
        x = self.fc(x)

        return x


def _resnet(arch, inplanes, planes, pretrained, progress, **kwargs):
    model = ResNet(inplanes, planes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained=False, progress=True, **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained=False, progress=True, **kwargs) 


'''
生成器模型默认来自dcgan94那个模型
'''
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.utils import save_image
import numpy as np
import load_config as config
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from flask import Flask, request, send_file

from gen_mask_data_predict import roberts, prewitt, sobel, laplacian
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

Tensor = torch.cuda.FloatTensor
cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor

IMG_SZ = 256
LATENT_DIM = 100
CHANNELS = 3
BATCH_SZ = 1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = IMG_SZ // 4
        self.l1 = nn.Sequential(nn.Linear(LATENT_DIM, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 9, stride=1, padding=4),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 9, stride=1, padding=4),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, CHANNELS, 9, stride=1, padding=4),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        #return 1.1 * nn.functional.tanh(img)
        return 1.1 * torch.tanh(img)


#鉴别器（Discriminator_norm）简单的Linear模型，返回True或者False 
class Discriminator_norm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator_norm, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.elu = torch.nn.ELU()
 
    def forward(self, x):
        x = self.elu(self.map1(x))
        x = self.elu(self.map2(x))
        return torch.sigmoid( self.map3(x) )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, bn=True, pool=False):
            block = [nn.Conv2d(in_filters, out_filters, 3, stride, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            if pool:
                block.append(nn.MaxPool2d(2, 2))
            return block

        self.generator = Generator()
        self.discriminator_norm = Discriminator_norm()
        '''
        for p in self.parameters():
            p.requires_grad = False
        '''
        self.resnet = resnet50()
        self.resnet.cuda()

        # The height and width of downsampled image
        ds_size = IMG_SZ // 2 ** 4
        #self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, LATENT_DIM))
        self.adv_layer = nn.Sequential(nn.Linear(1000, LATENT_DIM))

    def forward(self, img):
        #out = self.decoder(img)
        out = self.resnet(img)
        out = self.adv_layer(out)
        discriminator_flag = self.discriminator_norm(out)
        #out = Variable(Tensor(np.random.normal(0, 1, (BATCH_SZ, LATENT_DIM))))
        #out = Variable(Tensor(np.random()))
        out = self.generator(out)
        #out = out + 1
        #out = out/2
        #out[out < 0.0] = 0.0
        #out[out > 1.0] = 1.0
        out = out/2.2+0.5

        return (out, discriminator_flag)


net = torch.load('net_318500')

colors = [
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,255,0],
    [255,0,255],
]


def predict_http(image_raw):
    raw_img = image_raw
    shape_y, shape_x  = image_raw.shape[0], image_raw.shape[1]
    image_raw = cv2.resize(image_raw, (256, 256))
    roberts_img = roberts(image_raw)
    prewitt_img = prewitt(image_raw)
    sobel_img = sobel(image_raw)
    laplacian_img = laplacian(image_raw)

    # 将图像转换成灰度图像
    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2BGR)
    trans = transforms.ToTensor()
    imgs_raw = trans(image_raw).type(Tensor).view(1,3,256,256)
    roberts_img = trans(roberts_img).type(Tensor).view(1,1,256,256)
    prewitt_img = trans(prewitt_img).type(Tensor).view(1,1,256,256)
    sobel_img = trans(sobel_img).type(Tensor).view(1,1,256,256)
    laplacian_img = trans(laplacian_img).type(Tensor).view(1,1,256,256)
    # 拼接成七维
    imgs_raw = torch.cat((imgs_raw, roberts_img, roberts_img, sobel_img, laplacian_img), 1)
    segment_imgs, norm_flag = net(imgs_raw)

    save_image(segment_imgs.data.view(-1,1,256,256)[:5], "seg.png", nrow=6, normalize=False)
    img = cv2.imread("seg.png", cv2.IMREAD_GRAYSCALE)

    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    seg_img = cv2.resize(img, (shape_x*5, shape_y), cv2.INTER_NEAREST)
    

    for i in range(5) :
        img = seg_img[:, shape_x*i:shape_x*(i+1)]
        raw_img[img>0] = raw_img[img>0]*0.6 + np.array(colors[i])*0.4
    #cv2.imwrite("seg.png", raw_img)
    return raw_img


# 实例化app
app = Flask(import_name=__name__)

# 通过methods设置POST请求
@app.route('/predict', methods=["POST"])
def predict():

    # 接收post请求上传的文件
    input_file = request.files.get('file')

    if input_file is None:
        return "未上传文件"

    # 从指定的内存缓存中读数据, 并转码成图像格式
    input_file = input_file.read()
    img = cv2.imdecode(np.frombuffer(input_file, np.uint8), cv2.IMREAD_COLOR)

    # 对输入数据进行处理
    img_ret = None
    img_ret = predict_http(img)

    # 将图片格式转码成数据流, 放到内存缓存中
    img_encode = cv2.imencode('.jpg', img_ret)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()
    return str_encode
    
    '''
    # 这个是用来base64编码, 用在浏览器里
    return send_file(
        io.BytesIO(img),
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='result.jpg'
    )
    '''

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)




'''
if __name__ == '__main__':
    predict_http(cv2.imread("raw.png"))
'''

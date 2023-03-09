import torch
import torch.nn as nn
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet
from EARDS_model.SeparableConv2d import SeparableConv2d

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            SeparableConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            SeparableConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.channel_conv(input)
        x = x1 + x2
        return x


# Hook函数获取中间层的值进行交叉连接
hook_values = []


def hook(_, input, output):
    global hook_values
    # 将每个层的值存储在hook_values中
    hook_values.append(output)  # stores values of each layers in hook_values


indices = []
shapes = []


# 初始化
def init_hook(model, device):
    global shapes, indices, hook_values
    # 获取中间结果
    for i in range(len(model._blocks)):
        model._blocks[i].register_forward_hook(hook)
    # 获取中间结果 size
    image = torch.rand([1, 3, 512, 512])
    image = image.to(device)
    out = model(image)

    shape = [i.shape for i in hook_values]

    for i in range(len(shape) - 1):
        if shape[i][2] != shape[i + 1][2]:
            indices.append(i)
    indices.append(len(shape) - 1)

    shapes = [shape[i] for i in indices]
    shapes = shapes[::-1]


encoder_out = []


def epoch_hook(model, image):
    global encoder_out, indices, hook_values
    hook_values = []

    out = model(image)
    encoder_out = [hook_values[i] for i in indices]

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi
class EARDS(nn.Module):

    def __init__(self, model='b0', out_channels=2, dropout=0.1, freeze_backbone=False, pretrained=False, device='cuda',
                 num_gpu=1):
        super(EARDS, self).__init__()
        self.n_classes = out_channels
        global layers, shapes

        if model not in set(['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']):
            raise Exception(f'{model} unavailable.')
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(f'efficientnet-{model}')
        else:
            self.encoder = EfficientNet.from_name(f'efficientnet-{model}')
        # 不使用这些层
        self.encoder._conv_head = torch.nn.Identity()
        self.encoder._bn1 = torch.nn.Identity()
        self.encoder._avg_pooling = torch.nn.Identity()
        self.encoder._dropout = torch.nn.Identity()
        self.encoder._fc = torch.nn.Identity()
        self.encoder._swish = torch.nn.Identity()

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.encoder.to(self.device)
        # 将第一层的步幅从2改为1，以增加o/p大小
        self.encoder._conv_stem.stride = 1
        self.encoder._conv_stem.kernel_size = (1, 1)  # 将 第一层卷积核大小改为1 * 1
        self.encoder._conv_stem.static_padding.padding = (1, 1, 1, 1)  # padding 保持大小

        # freeze encoder 冻结编码器
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        init_hook(self.encoder, self.device)

        self.decoder = torch.nn.modules.container.ModuleList()
        # 构建 AG 模块
        self.decoderAG = torch.nn.modules.container.ModuleList()

        for i in range(len(shapes) - 1):
            self.decoder.append(torch.nn.modules.container.ModuleList())
            self.decoder[i].append(
                nn.ConvTranspose2d(shapes[i][1], shapes[i][1] - shapes[i + 1][1], kernel_size=2, stride=2).to(
                    self.device))
            self.decoder[i].append(DoubleConv(shapes[i][1], shapes[i + 1][1]).to(self.device))
            # 构建AG
            self.decoderAG.append(torch.nn.modules.container.ModuleList())
            self.decoderAG[i].append(
                Attention_block(shapes[i][1] - shapes[i + 1][1], shapes[i+1][1], shapes[i+1][1])
            )

        self.out = nn.Conv2d(shapes[-1][1], out_channels, kernel_size=1).to(self.device)

        if num_gpu > 1 and device == 'cuda':
            self.encoder = nn.DataParallel(self.encoder)

    def forward(self, image):
        global layers

        h = image.shape[2]
        w = image.shape[3]
        if h % 8 != 0 or w % 8 != 0:
            new_h = round(h / 8) * 8
            new_w = round(w / 8) * 8
            image = T.Resize((new_h, new_w))(image)

        # Encoder
        epoch_hook(self.encoder, image)

        # Decoder
        x = encoder_out.pop()
        for i in range(len(self.decoder)):
            x = self.decoder[i][0](x)  # conv transpose
            prev = encoder_out.pop()
            prev = self.decoderAG[i][0](x, prev)
            prev = torch.cat([x, prev], axis=1)  # concatenating
            x = self.decoder[i][1](prev)  # double conv

        # out
        x = self.out(x)
        return x

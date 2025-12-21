import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inp, oup, stride=1, relu=True):
    layers = [
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    ]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def conv_bn_no_relu(inp, oup, stride):
    return conv_bn(inp, oup, stride, relu=False)

def conv_bn1X1(inp, oup, stride, relu=True):
    layers = [
        nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
    ]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class DepthWise(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthWise, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)
        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)
        
        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7x7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7x7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        # Lateral connections (1x1)
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, relu=False)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, relu=False)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, relu=False)

        # Merge / Smoothing layers (3x3) - [FIXED]
        # 가중치 파일의 형태([64, 64, 3, 3])에 맞춰 conv_bn (3x3) 사용
        self.merge1 = conv_bn(out_channels, out_channels, stride=1, relu=False)
        self.merge2 = conv_bn(out_channels, out_channels, stride=1, relu=False)

    def forward(self, inputs):
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return [output1, output2, output3]

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, relu=True),
            DepthWise(8, 16, 1),
            DepthWise(16, 32, 2),
            DepthWise(32, 32, 1),
            DepthWise(32, 64, 2),
            DepthWise(64, 64, 1),
        )
        self.stage2 = nn.Sequential(
            DepthWise(64, 128, 2), 
            DepthWise(128, 128, 1),
            DepthWise(128, 128, 1),
            DepthWise(128, 128, 1),
            DepthWise(128, 128, 1),
            DepthWise(128, 128, 1),
        )
        self.stage3 = nn.Sequential(
            DepthWise(128, 256, 2),
            DepthWise(256, 256, 1),
        )

    def forward(self, x):
        x = self.stage1(x)
        x1 = x 
        x = self.stage2(x)
        x2 = x
        x = self.stage3(x)
        x3 = x
        return x1, x2, x3

class ClassHead(nn.Module):
    def __init__(self, in_channel=64, num_anchor=2):
        super(ClassHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, num_anchor * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self, in_channel=64, num_anchor=2):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, num_anchor * 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, in_channel=64, num_anchor=2):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, num_anchor * 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, phase="test", width_mult=0.25):
        super(RetinaFace, self).__init__()
        self.phase = phase
        self.body = MobileNetV1()
        
        in_channels_list = [64, 128, 256]
        
        self.fpn = FPN(in_channels_list, 64)
        self.ssh1 = SSH(64, 64)
        self.ssh2 = SSH(64, 64)
        self.ssh3 = SSH(64, 64)

        self.ClassHead = self._make_head(ClassHead, 64)
        self.BboxHead = self._make_head(BboxHead, 64)
        self.LandmarkHead = self._make_head(LandmarkHead, 64)

    def _make_head(self, head_cls, in_c):
        return nn.ModuleList([
            head_cls(in_c),
            head_cls(in_c),
            head_cls(in_c)
        ])

    def forward(self, inputs):
        out = self.body(inputs)
        fpn_out = self.fpn(out)
        
        feature1 = self.ssh1(fpn_out[0])
        feature2 = self.ssh2(fpn_out[1])
        feature3 = self.ssh3(fpn_out[2])
        features = [feature1, feature2, feature3]

        bbox_reg = torch.cat([self.BboxHead[i](features[i]) for i in range(len(features))], dim=1)
        cls = torch.cat([self.ClassHead[i](features[i]) for i in range(len(features))], dim=1)
        ldm_reg = torch.cat([self.LandmarkHead[i](features[i]) for i in range(len(features))], dim=1)

        if self.phase == "test":
            cls = F.softmax(cls, dim=-1)
        return bbox_reg, cls, ldm_reg
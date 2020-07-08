import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class transitition_block(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(transitition_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.avg_pool(out)
        return out


class dense_block(nn.Module):
    def __init__(self, in_Channels,growth_rate):
        super(dense_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_Channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_Channels, out_channels=growth_rate*4, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(growth_rate*4)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=growth_rate*4, out_channels=growth_rate, kernel_size=3, padding=1)


    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = torch.cat((x, out), 1)

        return out


class ResNet(nn.Module):
    def __init__(self,arch,growth_rate,reduction,num_of_classes):
        super(ResNet, self).__init__()

        in_Channels=growth_rate*2
        # DenseNet-121 Architecture [6,12,24,16]
        # DenseNet-169 Architecture [6,12,32,32]
        # DenseNet-201 Architecture [6,12,48,32]
        # DenseNet-264 Architecture [6,12,48,32]
        DenseLayers = arch

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=in_Channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(in_Channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)

        ### Densenet Architecture------------------------------------------------------------##

        self.denselayer1 = self.make_dense_layer(dense_block,  in_Channels,growth_rate, DenseLayers[0])
        in_Channels += DenseLayers[0]*growth_rate
        out_Channels = int(math.floor(in_Channels * reduction))
        self.transtition1 = self.make_transtition_layer(transitition_block, in_channels=in_Channels, out_Channels=out_Channels)

        in_Channels = out_Channels
        self.denselayer2 = self.make_dense_layer(dense_block, in_Channels,growth_rate, DenseLayers[1])
        in_Channels += DenseLayers[1]*growth_rate
        out_Channels = int(math.floor(in_Channels * reduction))
        self.transtition2 = self.make_transtition_layer(transitition_block, in_channels=in_Channels, out_Channels=out_Channels)

        in_Channels = out_Channels
        self.denselayer3 = self.make_dense_layer(dense_block, in_Channels,growth_rate, DenseLayers[2])
        in_Channels += DenseLayers[2] * growth_rate
        out_Channels = int(math.floor(in_Channels * reduction))
        self.transtition3 = self.make_transtition_layer(transitition_block, in_channels=in_Channels, out_Channels=out_Channels)

        in_Channels = out_Channels
        self.denselayer4 = self.make_dense_layer(dense_block, in_Channels,growth_rate, DenseLayers[3])
        in_Channels += DenseLayers[3] * growth_rate
        self.avg_pool = nn.AvgPool2d((7, 7), stride=2)

        self.fc = nn.Linear(in_Channels, num_of_classes)

    def make_dense_layer(self, dense_block,  in_Channels, growth_rate,num_of_layers):
        dense_layers = []

        for i in range(num_of_layers):
            dense_layers.append(dense_block(in_Channels=in_Channels, growth_rate=growth_rate))
            in_Channels+=growth_rate
        return nn.Sequential(*dense_layers)

    def make_transtition_layer(self, transitition_block, in_channels, out_Channels):
        transition_layers = []
        transition_layers.append(transitition_block(in_channels, out_Channels))
        return nn.Sequential(*transition_layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.denselayer1(out)
        out = self.transtition1(out)

        out = self.denselayer2(out)
        out = self.transtition2(out)

        out = self.denselayer3(out)
        out = self.transtition3(out)

        out = self.denselayer4(out)
        out = self.avg_pool(out)

        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return F.log_softmax(out,dim=1)

"""
x = torch.randn(4, 3, 224, 224)
model = ResNet(growth_rate=32,reduction=0.5, num_of_classes=10)

print(model.eval())
print(model(x).shape)
"""
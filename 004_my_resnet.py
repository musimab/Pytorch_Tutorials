import torch
import torch.nn as nn

#layers = [3,4,6,3]
class Resnet34(nn.Module):
    def __init__(self,block,layers,img_channels=3,num_classes=1000):
        super(Resnet34,self).__init__()
        self.in_channels =64
        self.conv1 = nn.Conv2d(in_channels=img_channels,out_channels=64,kernel_size=7,stride =2,padding=3)
        self.bn0 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block,layers[0],intermediate_channels =64,stride = 1)
        self.conv1x1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=2,padding=0)

        self.layer2 = self.make_layer(block, layers[1], intermediate_channels=128, stride=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)

        self.layer3 = self.make_layer(block, layers[2], intermediate_channels=256, stride=1)
        self.conv1x1_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0)

        self.layer4 = self.make_layer(block, layers[3], intermediate_channels=512, stride=1)
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,num_classes)
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn0(out)
        out = self.pool1(out)
        out = self.layer1(out)
        out = self.conv1x1(out)
        out = self.layer2(out)
        out = self.conv1x1_2(out)
        out = self.layer3(out)
        out = self.conv1x1_3(out)
        out = self.layer4(out)
        out =self.average_pool(out)
        out = out.view(out.shape[0],-1)
        out =self.fc(out)


        return out

    def make_layer(self,block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None

        layers = []


        for i in range(num_residual_blocks):
            layers.append(block(self.in_channels,intermediate_channels,stride))
        self.in_channels *= 2
        return nn.Sequential(*layers)


class block(nn.Module):
    def __init__(self,in_channels, intermediate_channels, identity_downsample=None,  stride=1):
        super(block,self).__init__()
        self.expansion =2
        self.conv1 = nn.Conv2d(in_channels,intermediate_channels,kernel_size=3, stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels,  kernel_size=3, stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.identity_downsample = identity_downsample

    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.identity_downsample is None:
            identity = self.identity_downsample(identity)
        out+=identity
        return out

net = Resnet34(block,[3,4,6,3],img_channels=3, num_classes=1000)
y = net(torch.randn(4, 3, 224, 224))
print(y.size())
print(net)
parameters = []
for param in net.parameters():
    parameters.append(param.numel())
print(sum(parameters))







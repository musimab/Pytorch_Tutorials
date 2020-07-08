import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self,in_channels,intermediate_channels,identity_downsample = None,stride=1):
        super(block,self).__init__()
        self.expansion =4

        self.conv1 = nn.Conv2d(in_channels,intermediate_channels,kernel_size=1,stride=1,padding=0)
        self.bn1  = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(intermediate_channels,intermediate_channels,kernel_size=3,stride=stride,padding=1)
        self.bn2  = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels,intermediate_channels*self.expansion,kernel_size=1,stride=1,padding=0)
        self.bn3  = nn.BatchNorm2d(intermediate_channels*self.expansion)
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self,out):
        identity = out.clone()
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.identity_downsample is not None:
            identity=self.identity_downsample(identity)

        out +=identity
        out = self.relu(out)
        return out

class Resnet(nn.Module):
    def __init__(self,block,layers,img_channels =3,num_classes =1000):
        super(Resnet,self).__init__()
        self.in_channels =64
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block,layers[0],intermediate_channels=64,stride=1)
        self.layer2 = self.make_layer(block,layers[1],intermediate_channels=128,stride=2)
        self.layer3 = self.make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layer(block, layers[3], intermediate_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x= self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def make_layer(self, block, num_of_residual_block, intermediate_channels, stride):
        layers= []
        identity_downsample = None

        if stride!=1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,intermediate_channels*4,kernel_size=1,stride=stride),nn.BatchNorm2d(intermediate_channels*4))

        layers.append(block(self.in_channels, intermediate_channels,identity_downsample, stride))

        self.in_channels = intermediate_channels * 4

        for i in range(num_of_residual_block-1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


net = Resnet(block,[3,4,6,3],img_channels=3, num_classes=1000)
y = net(torch.randn(4, 3, 224, 224))
print(y.size())
print(net)
parameters = []
for param in net.parameters():
    parameters.append(param.numel())
print(sum(parameters))


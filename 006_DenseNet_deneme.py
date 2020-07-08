import torch
import torch.nn as nn

class transitition_block(nn.Module):
    def __init__(self,in_channels,inter_channels):
        super(transitition_block,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels,inter_channels,kernel_size=1,stride=1,padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out  = self.avg_pool(out)
        return out

class dense_block(nn.Module):
    def __init__(self,in_Channels,inter_Channels):
        super(dense_block,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_Channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_Channels,out_channels=inter_Channels,kernel_size=1,padding=0)
        self.bn2 = nn.BatchNorm2d(inter_Channels)
        self.conv2 = nn.Conv2d(in_channels=inter_Channels,out_channels=32,kernel_size=3,padding=1)

    def forward(self,x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.cat((x,out),1)

        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride =2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1)

        ### Densenet Architecture------------------------------------------------------------##

        self.layer1 = self.make_dense_layer(dense_block, in_Channels=64, constant=0, num_of_layers=6)
        self.transtition1 =self.make_transtition_layer(transitition_block,in_channels=256,inter_channels=128)

        self.layer2 = self.make_dense_layer(dense_block,in_Channels = 128, constant=2, num_of_layers=12)
        self.transtition2 = self.make_transtition_layer(transitition_block,in_channels=512,inter_channels=256)

        self.layer3 = self.make_dense_layer(dense_block,in_Channels =256, constant=6, num_of_layers=24)
        self.transtition3 = self.make_transtition_layer(transitition_block, in_channels=1024, inter_channels=512)

        self.layer4 = self.make_dense_layer(dense_block, in_Channels=512, constant=14, num_of_layers=16)
        self.avg_pool = nn.AvgPool2d((7,7),stride=2)

        self.fc = nn.Linear(1024,1000)


    def make_dense_layer(self,dense_block,in_Channels ,constant,num_of_layers):
        dense_layers = []
        for i in range(num_of_layers):
            dense_layers.append(dense_block(in_Channels=in_Channels,inter_Channels=128))
            in_Channels = 32 * (constant + 3)
            constant+=1
        return nn.Sequential(*dense_layers)

    def make_transtition_layer(self,transitition_block, in_channels, inter_channels):
        transition_layers = []
        transition_layers.append(transitition_block(in_channels, inter_channels))
        return nn.Sequential(*transition_layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.transtition1(out)

        out = self.layer2(out)
        out = self.transtition2(out)

        out = self.layer3(out)
        out = self.transtition3(out)

        out = self.layer4(out)
        out = self.avg_pool(out)

        out = out.view(out.shape[0],-1)
        out = self.fc(out)

        return out


x = torch.randn(4,3,224,224)
model = ResNet()

print(model.eval())
print(model(x).shape)
from platform import architecture
from PIL.ImageFilter import Kernel
import torch
import torch.nn as nn

class transition_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(transition_block, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size =1,stride =1,padding =0)
        self.avgpool = nn.AvgPool2d(2, stride=2)

    def forward(self,x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.avgpool(out)
        return out

class dense_block(nn.Module):
    def __init__(self,in_channels, growth_rate):
        super(dense_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, growth_rate*4,kernel_size = 1, stride = 1, padding =0)
        self.bn2 = nn.BatchNorm2d(growth_rate*4)
        self.conv2 = nn.Conv2d(growth_rate*4, growth_rate, kernel_size = 3, stride = 1,padding =1)

    def forward(self,x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.cat([x,out],dim =1)
        return out

class Dense_Net(nn.Module):
    def __init__(self, growth_rate, num_of_classes):
        super(Dense_Net,self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels =growth_rate*2, kernel_size=7,stride =2,padding =3)
        self.bn = nn.BatchNorm2d(growth_rate*2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        in_channels =64
        self.dense_layer1 = self._make_layer(in_channels, num_l=6)
        self.trans1 = transition_block(256, 128)
        in_channels =128
        
        self.dense_layer2 = self._make_layer(in_channels, num_l=12)
        self.trans2 = transition_block(512, 256)
        in_channels =256
        self.dense_layer3 = self._make_layer(in_channels, num_l=24)
        self.trans3 = transition_block(1024, 512)
        in_channels =512
        self.dense_layer4 = self._make_layer(in_channels, num_l=16) 
        self.avgpool = nn.AvgPool2d(7,stride =2)
        self.fc1 = nn.Linear(1024,num_of_classes)       
        

    def _make_layer(self,in_channels ,num_l):
        
        layers = []
        for i in range(num_l):
            layers.append(dense_block(in_channels,32))
            in_channels +=32

        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dense_layer1(out)
        out = self.trans1(out)
        
        out = self.dense_layer2(out)
        out = self.trans2(out)
        out = self.dense_layer3(out)
        out = self.trans3(out)
        out = self.dense_layer4(out)

        out = self.avgpool(out)
        out = out.view(out.shape[0],-1)
        out = self.fc1(out)
       
        return out

device = 'cuda' if torch.cuda.is_available else 'cpu'

x = torch.randn(2,3,224,224).to(device)
model = Dense_Net(growth_rate =32,num_of_classes =2).to(device)
print(model(x).shape)
print(model)
print(device)
print(model(x).shape)

import my_Densenet
import argparse
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
import os

def main():

    parser = argparse.ArgumentParser(description="Train-Test DenseNet")
    parser.add_argument('--batchSize',type= int,default=10)
    parser.add_argument('--epoch',type = int ,default=1)
    parser.add_argument('--architecture',type=str,choices=('dense121','dense169','dense201','dense264'))
    args = parser.parse_args()

    train_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std =[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std =[0.229, 0.224, 0.225])
    ])

    root ="C:/Users/musta/OneDrive/Desktop/Udemy_Files/data/CATS_DOGS"
    class CustomData():
        def __init__(self,root,test_transform,train_transform):
            self.train_data = datasets.ImageFolder(os.path.join(root,'train'),transform = train_transform)
            self.test_data = datasets.ImageFolder( os.path.join(root,'test'), transform  = test_transform)
        def train_loader(self):
            return DataLoader(self.train_data,batch_size=args.batchSize,shuffle=True)
        def test_loader(self):
            return DataLoader(self.test_data,batch_size=args.batchSize,shuffle=False)
        def class_names(self):
            return self.train_data.classes
    dog_cat = CustomData(root,test_transforms,train_transforms)

    print(f'num of train img:',len(dog_cat.train_data))
    print(f'num of test img:',len(dog_cat.test_data))

    for images,labels in dog_cat.train_loader():
        break
    print("labels:",*labels.numpy())
    print("classes:",*np.array([dog_cat.class_names()[i] for i in labels]))

    im = make_grid(images,nrow=args.batchSize)
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    im_inv = inv_normalize(im)
    plt.figure(figsize=(12,4))
    plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))

    architecture = { 'dense121':[6,12,24,16],'dense169':[6,12,32,32],'dense201':[6,12,48,32],'dense264':[6,12,48,32]}
    if args.architecture == 'dense121':
        arch = architecture['dense121']
    elif args.architecture == 'dense169':
        arch = architecture['dense169']
    elif args.architecture == 'dense201':
        arch = architecture['dense201']
    else:
        arch = architecture['dense264']

    print("current architecture:",args.architecture,arch)

    model = my_Densenet.ResNet(arch,growth_rate=8, reduction=0.5, num_of_classes=2)

    dog_cat =CustomData(root,train_transforms,test_transforms)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

    train_losses = []
    train_correct = []
    test_losses = []
    test_correct = []
    for epoch in range(1,args.epoch+1):
        train(dog_cat.train_loader(), model, criterion,optimizer, train_losses, train_correct)
        test(dog_cat.test_loader(),model,criterion,test_losses,test_correct)
        torch.save(model.state_dict(),'DenseNetModel.pt')

def train(train_loader,model,criterion,optimizer,train_losses,train_correct):

    for b,(X_train,y_train) in enumerate(train_loader):
        trn_corr =0

        b+=1
        y_pred = model(X_train)
        loss = criterion(y_pred,y_train)
        prediction = torch.max(y_pred.data,1)[1]
        trn_corr+= (prediction == y_train).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%100 ==0:
            print(f'batch: {b}, loss:{loss.item():7.3f} accuracy: {trn_corr.item()*100/(len(train_loader)):10.8f}')

    train_correct.append(trn_corr.item())
    train_losses.append(loss.item())
    print(train_losses)
    print(train_correct)




def test(test_loader,model,criterion,test_losses,test_correct):
    tst_corr = 0
    with torch.no_grad():
        for (X_test,y_test) in test_loader:
            y_val =model(X_test)
            predicted = torch.max(y_val.data,1)[1]
            tst_corr+= (predicted==y_test).sum()
        loss = criterion(y_val,y_test)
        test_correct.append(tst_corr.item())
        test_losses.append(loss.item())
    print(test_correct)
    print(test_losses)














if __name__ == '__main__':
    main()

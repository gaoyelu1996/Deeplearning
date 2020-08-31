import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms,models
from torch.optim import SGD
import torchvision
import matplotlib.pylab as plt
import numpy as np
import time
import tqdm
import sys
import copy
from pytorchtools import EarlyStopping


# ImageFolder
def transform_images(istraing=None):
    if istraing == 'train':
        transform=transforms.Compose([transforms.RandomCrop(size=(224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
    elif istraing == 'test':
        transform=transforms.Compose([
            transforms.Resize(size=(256,256)),
            transforms.RandomCrop(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    return transform


root_path = os.path.join(os.path.abspath(os.getcwd()), 'hotdog\\hotdog')
image_folder = {name: ImageFolder(os.path.join(root_path, name+'\\'), transform=transform_images(name)) for name in
                    ['train', 'test']}

datasets_folder = {name: DataLoader(image_folder[name], batch_size=32, shuffle=True) for name in ['train', 'test']}
data_len = {name: len(image_folder[name]) for name in ['train', 'test']}


def train(model,datasets_folder,data_len,root_path,patience,criterion=None,optimizer=None,device=None,num_epochs=10):
    model=model.to(device)
    start_time=time.time()
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]

    # 及时停止,当停止时，会保存模型
    early_stopping=EarlyStopping(patience,verbose=True,path=os.path.join(root_path,'model.pt'))

    for epoch in tqdm.tqdm(range(num_epochs)):
        print('Epoch {} / {} starting!'.format(epoch,num_epochs))
        running_loss=0.0
        running_acc=0.0

        optimizer.zero_grad()

        for mode in ['train','test']:
            if mode == 'train':
                model.train()
            elif mode == 'test':
                model.eval()

            for X,y in datasets_folder[mode]:
                X=X.to(device)
                y=y.to(device)

                with torch.set_grad_enabled(mode == 'train'):
                    outputs=model(X)
                    pred=torch.max(outputs,1)

                    # 对min-batch求了平均之后的
                    loss=criterion(X)

                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()*X.size(0)
                running_acc += torch.sum(pred == y.data)

            epoch_loss=running_loss/data_len[mode]
            epoch_acc=running_acc/data_len[mode]

            if mode == 'train':
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
            else:
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

            print('{} loss is  {:.2f},acc is {:.2f},time sec is {}'.
                  format(mode,epoch_loss,epoch_acc,time.time()-start_time))

        # 及时停止
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('early_stop!')
            break
    print()
    end_time=time.time()-start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(end_time//60,end_time % 60))

    model.load_state_dict(torch.load(os.path.join(root_path,'model.pt')))
    return model,train_loss,train_acc,val_loss,val_acc


model,train_loss,train_acc,val_loss,val_acc=train(model,datasets_folder,data_len,root_path,patience=10,criterion=None,optimizer=None,device=None,num_epochs=10)


# 可视化
def show_loss_curve(train_loss,val_loss):
    fig=plt.figure(figsize=(8,6))
    plt.plot(range(1,len(train_loss)+1),train_loss,label='Training Loss')
    plt.plot(range(1,len(val_loss)+1),val_loss,label='Valid Loss')

    min_loss=val_loss.index(min(val_loss))+1
    plt.axvline(min_loss,linestyle='--',color='r',label='Early Stopping')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xlim(0,len(train_loss)+1)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_curve.png',bbox_inches='tight')
    return plt


# test
def test(model,test_data):







from PIL import Image


def prop_image(img):
    img=img.reshape((1,3,img.size(0),img.size(1)))
    img=map(transform_images(istraing='test'),img)
    return DataLoader(img,batch_size=1)


def test(root_path,image_path,device):
    model=torch.load(root_path)
    model.to(device)

    plt.figure(figsize=(10,6))
    img=Image.open(image_path)








def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root_path = os.path.join(os.path.abspath(os.getcwd()), 'hotdog')

    image_folder = {name: ImageFolder(os.path.join(root_path, name, '/'), transform=transform_images(name)) for name in
                    ['train', 'test']}

    datasets_folder = {name: DataLoader(image_folder[name], batch_size=32, shuffle=True) for name in ['train', 'test']}
    data_len = {name: len(image_folder[name]) for name in ['train', 'test']}

    model=models.resnet34(pretrained=True)
    model.fc=nn.Linear(512,2)

    lr=0.0001
    fc_paramas=list(map(id,model.fc.parameters()))
    features_paramas=filter(lambda p : id(p) not in fc_paramas,model.parameters())

    optimizer=torch.optim.Adam([{'params':features_paramas},
                                {'params':model.fc.parameters(),'lr':lr*10}],
                               lr=lr,weight_decay=0.001)

    criterion=torch.nn.CrossEntropyLoss()
    train(model, datasets_folder, data_len, num_epochs=10, criterion=criterion, optimizer=optimizer, device=device)


if __name__=='__main__':
    main()

model=models.resnet34(pretrained=True)











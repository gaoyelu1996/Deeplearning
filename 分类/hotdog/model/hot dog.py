import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.optim import lr_scheduler
import torchvision
import matplotlib.pylab as plt
import numpy as np
import time
import tqdm
import sys
import copy
from PIL import Image
from pytorchtools import EarlyStopping


# from DataSet import Dataset


def transform_images(istraing=None):
    if istraing == 'train':
        transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                        transforms.RandomCrop(size=(224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.RandomCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    return transform


def train(model, datasets_folder, data_len, lr_scheduler, patience, model_path, criterion, optimizer, device,
          num_epochs=None):
    model = model.to(device)
    start_time = time.time()

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)

    for epoch in range(num_epochs):
        print('Epoch {} / {} starting!'.format(epoch, num_epochs - 1))

        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
            elif mode == 'val':
                model.eval()

            running_loss = 0.0
            running_acc = 0.0

            for X, y in tqdm.tqdm(datasets_folder[mode]):
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model(X)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, y)

                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X.size(0)
                running_acc += torch.sum(pred == y.data)

            epoch_loss = running_loss / data_len[mode]
            epoch_acc = running_acc.double() / data_len[mode]

            if mode == 'train':
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
            elif mode == 'val':
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

            print('{} loss is  {:.2f},acc is {:.2f},time sec is {}'.
                  format(mode, epoch_loss, epoch_acc, time.time() - start_time))

        early_stopping(val_loss[epoch], model)
        if early_stopping.early_stop:
            print('early_stop!')
            break

        lr_scheduler.step()

    print()
    end_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))

    model.load_state_dict(torch.load(model_path))
    return model, train_loss, train_acc, val_loss, val_acc


def show_loss_curve(train_loss, val_loss):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Valid Loss')

    min_loss = val_loss.index(min(val_loss)) + 1
    plt.axvline(min_loss, linestyle='--', color='r', label='Early Stopping')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, len(train_loss) + 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    fig.savefig('loss_curve.png', bbox_inches='tight')
    return plt


def show_acc_curve(train_acc, val_acc):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Acc')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Valid Acc')

    max_acc = val_acc.index(max(val_acc)) + 1
    plt.axvline(max_acc, linestyle='--', color='r', label='Early Stopping')

    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, len(train_acc) + 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    fig.savefig('acc_curve.png', bbox_inches='tight')
    return plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_txt, root, transform=None, target_transform=None, Label_Dic=None):
        super(Dataset, self).__init__()
        files = open(root + "/" + data_txt, 'r')
        self.img = []
        for i in files:
            i = i.rstrip()
            temp = i.split()

            if Label_Dic != None:
                self.img.append((temp[0], Label_Dic[temp[1]]))
            else:
                self.img.append((temp[0], temp[0]))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        files, label = self.img[index]
        img = Image.open(files).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img)


# /home/stu1/gaoyelu/hotdog/hotdog/
root_path = os.path.join(os.path.abspath(os.getcwd()), 'hotdog')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_dic = {'not-hotdog': 0, 'hotdog': 1}

dataset = {name: Dataset(data_txt=name + '_shuffle_Data.txt', root=root_path,
                         transform=transform_images(name), target_transform=None,
                         Label_Dic=label_dic) for name in ['train', 'val', 'test']}

# image_folder = {name: ImageFolder(os.path.join(root_path, name+'/'), transform=transform_images(istraing=name)) for name in ['train', 'val','test']}

datasets_folder = {name: DataLoader(dataset[name], batch_size=32, shuffle=True) for name in ['train', 'val', 'test']}
data_len = {name: len(dataset[name]) for name in ['train', 'val', 'test']}

model = models.resnet34(pretrained=True)
model.fc = nn.Linear(512, 2)

lr = 0.01
fc_paramas = list(map(id, model.fc.parameters()))
features_paramas = filter(lambda p: id(p) not in fc_paramas, model.parameters())

optimizer = torch.optim.Adam([{'params': features_paramas},
                              {'params': model.fc.parameters(), 'lr': lr * 10}],
                             lr=lr, weight_decay=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

criterion = torch.nn.CrossEntropyLoss()
model, train_loss, train_acc, val_loss, val_acc = train(model, datasets_folder, data_len, lr_scheduler=exp_lr_scheduler,
                                                        num_epochs=100, patience=30,
                                                        model_path=os.path.join(os.path.abspath(os.getcwd()),
                                                                                'model.pt'), criterion=criterion,
                                                        optimizer=optimizer, device=device)

show_loss_curve(train_loss, val_loss)
show_acc_curve(train_acc, val_acc)






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











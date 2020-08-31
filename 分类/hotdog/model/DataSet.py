from PIL import Image
import torch.utils.data.dataset
from torch.utils.data import DataLoader
import os


# 代替了ImageFolder
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


label_dic={'not-hotdog':0,'hotdog':1}
root=os.path.join(os.path.abspath(os.getcwd()),'hotdog')
test_dataset=Dataset(data_txt='Data.txt',root=root,transform=None, target_transform=None,Label_Dic=label_dic)
data_loader=DataLoader(test_dataset,batch_size=32,shuffle=True)

files = open(root + "/" + 'Data.txt', 'r')
img = []
for i in files:
    i = i.rstrip()
    temp = i.split()
    img.append((temp[0],temp[1]))

image,label=img[0]
path='D:/pycharm练习/pytorch练习/hotdog/hotdog/test\\hotdog\\1000.png'
img = Image.open(path).convert('RGB')

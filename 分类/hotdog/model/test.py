import os
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root=os.path.abspath(os.getcwd())


model=models.resnet34(pretrained=True)
model.fc=nn.Linear(512,2)
model.load_state_dict(torch.load(os.path.join(root,'model.pt')))


def transform_images(istraing=None):
    if istraing == 'train':
        transform=transforms.Compose([transforms.Resize(size=(256,256)),
                                      transforms.RandomCrop(size=(224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0),
                                      transforms.RandomRotation(degrees=(-10,10)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
    else:
        transform=transforms.Compose([
            transforms.Resize(size=(256,256)),
            transforms.RandomCrop(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    return transform
    
    
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


def test(model,datasets_folder,batch_size,data_len,criteron,device):
    model.to(device)
    model.eval()

    test_loss=0.0
    class_correct=[0,0]
    class_total=[0,0]

    for X,y in datasets_folder:
        if len(y.data) != batch_size:
            break

        X=X.to(device)
        y=y.to(device)

        y_pred=model(X)
        _ , pred=torch.max(y_pred,1)

        loss=criteron(y_pred,y)
        test_loss += loss.item()*X.size(0)

        correct=(y.data == pred)

        for i in range(batch_size):
            label=y.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss=test_loss/data_len
    print('Test Loss :%.3f'%test_loss)

    for ids in range(2):
        if class_total[ids] > 0:
            print('Test Acc of %5s: %2d%%(%2d/%2d)'%(ids,100*class_correct[ids]/class_total[ids],class_correct[ids],class_total[ids]))
        else:
            print('Test Acc of %5s:N/A(no training examples)'%(ids))

    print('Test All Acc is %2d%%(%2d/%2d)'%(100*np.sum(class_correct)/np.sum(class_total),np.sum(class_correct),np.sum(class_total)))


def show_results(model,datasets_folder,batch_size,device,num_to_label):
    dataiter=iter(datasets_folder)
    images,labels=dataiter.next()
    
    images=images.to(device)
    labels=labels.to(device)
    
    output=model(images)
    _,pred=torch.max(output,1)

    print(images.shape)
    print(np.squeeze(images[1]).shape)
    
    fig=plt.figure(figsize=(25,4))
    for idx in range(batch_size):
        img=torch.transpose(images[idx].cpu(),0,2)
        ax=fig.add_subplot(4, batch_size/4, idx+1,xticks=[],yticks=[])
        ax.imshow(img.cpu())
        ax.set_title('{}({})'.format(num_to_label[str(pred[idx].item())],num_to_label[str(labels[idx].item())]),color=('green' if pred[idx] == labels[idx] else 'red'))
    plt.tight_layout()
    fig.savefig('test.png',bbox_inches='tight')
    return plt


label_dic={'not-hotdog':0,'hotdog':1}
test_dataset=Dataset(data_txt='test_shuffle_Data.txt',root=os.path.join(root,'hotdog'),
              transform=transform_images('test'), target_transform=None,
              Label_Dic=label_dic)

datasets_folder =DataLoader(test_dataset, batch_size=32, shuffle=False)
test_data_len =len(test_dataset)

criteron=torch.nn.CrossEntropyLoss()
test(model,datasets_folder,batch_size=32,data_len=test_data_len,criteron=criteron,device=device)

num_to_label={'0':'not-hotdog','1':'hotdog'}
show_results(model,datasets_folder,batch_size=32,device=device,num_to_label=num_to_label)
    
        
    
    



























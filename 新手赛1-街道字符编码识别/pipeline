'''
工作1：对标签进行个数统计，看一下每张图片包含的字符个数
标签顺序不能变

'''

import json
from six.moves import urllib
import glob
from PIL import Image
import numpy as np
from torchvision import transforms,models
import torch.nn as nn
from torch.nn import Module
import torch
from torch.utils.data import Dataset
import pandas as pd


# 处理数据的类
class SVHNDataset(Dataset):
    def __init__(self,img_path,img_label,tranform=None):
        self.img_path=img_path
        self.img_label=img_label
        if tranform is not None:
            self.tranform=tranform
        else:
            self.tranform =None

    def __getitem__(self,index):
        img=self.img_path[index]
        img=Image.open(img).convert('RGB')
        if self.tranform is not None:
            img=self.tranform(img)

        label=np.array(self.img_label[index],dtype=np.int)
        lal=list(label)+(5-len(label))*[10]
        return img,torch.from_numpy(np.array(lal[:5]))

    def __len__(self):
        return len(self.img_path)


# 训练集
train_img_path=glob.glob('datasets/mchar_train/mchar_train/*.png')
train_img_path.sort()
train_json_url='http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json'
train_json_path,_=urllib.request.urlretrieve(train_json_url,'datasets/json/mchar_train.json')
train_json=json.load(open(train_json_path))
train_label=[train_json[x]['label'] for x in train_json]
train_loader=torch.utils.data.DataLoader(SVHNDataset(train_img_path,train_label,
                                                            tranform=transforms.Compose([
                                                                transforms.Resize((64,128)),
                                                                transforms.RandomCrop((60,120)),
                                                                transforms.RandomRotation(10),
                                                                transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.4),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
                                         batch_size=40,
                                         shuffle=True)

# 验证集
val_img_path=glob.glob('datasets/mchar_train/mchar_val/*.png')
val_img_path.sort()
val_json_url='http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json'
val_json_path,_=urllib.request.urlretrieve(val_json_url,'datasets/json/mchar_val.json')
val_json=json.load(open(val_json_path))
val_label=[val_json[x]['label'] for x in val_json]
val_loader=torch.utils.data.DataLoader(SVHNDataset(val_img_path,val_label,
                                                   tranform=transforms.Compose([
                                                       transforms.RandomCrop((120,60)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                   ])),
                                       batch_size=40,
                                       shuffle=False)


# 构建模型
class SVHNModel(Module):
    def __init__(self):
        super(SVHNModel,self).__init__()
        model_conv=models.resnet18(pretrained=True)
        model_conv.avgpool=nn.AdaptiveAvgPool2d(1)

        # 去掉最后一层全连接层
        self.cnn=nn.Sequential(*list(model_conv.children())[:-1])
        self.fc1 = nn.Linear(512,11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)

    def forward(self,inputs):
        x=self.cnn(inputs)
        x=x.view((x.shape[0],-1))
        c1=self.fc1(x)
        c2=self.fc2(x)
        c3 = self.fc3(x)
        c4 = self.fc4(x)
        c5 = self.fc5(x)
        return c1,c2,c3,c4,c5


# 训练
def train(train_loader, model, criterion,optimizer, epoch):
    print('现在开始第{}次训练:'.format(epoch))
    model.train()
    train_loss=[]

    for index,(img,label) in enumerate(train_loader):
        c1, c2, c3, c4, c5=model(img)
        loss=criterion(c1,label[:,:11])+\
                 criterion(c2,label[:,11:22])+\
                 criterion(c3,label[:,22:33])+\
                 criterion(c4,label[:,33:44])+\
                 criterion(c5,label[:,44:55])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return np.mean(train_loss)


# 验证
def val(val_loader, model, criterion, epoch):
    print('现在开始第{}次验证:'.format(epoch))
    model.eval()
    val_loss=[]
    with torch.no_grad():
        for index,(val_img,val_label) in enumerate(val_loader):
            c1, c2, c3, c4, c5=model(val_img)

            loss=criterion(c1,val_label[:,:11])+\
                 criterion(c2,val_label[:,11:22])+\
                 criterion(c3,val_label[:,22:33])+\
                 criterion(c4,val_label[:,33:44])+\
                 criterion(c5,val_label[:,44:55])

            val_loss.append(loss.item())
    return np.mean(val_loss)


# 测试
def predict(test_loader, model, tta=10,use_cuda=False):
    model.eval()
    test_pred_tta = None

    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for index,(img,label) in enumerate(test_loader):
                c1, c2, c3, c4, c5 = model(img)
                out_put=np.concatenate([
                    c1.data.numpy(),
                    c2.data.numpy(),
                    c3.data.numpy(),
                    c4.data.numpy(),
                    c5.data.numpy(),
                ],axis=1)

                test_pred.append(out_put)
        test_pred=np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta=test_pred
        else:
            test_pred_tta += test_pred
    return test_pred_tta


# 训练
model=SVHNModel()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())

for epoch in range(5):
    train_loss=train(train_loader, model, criterion,optimizer, epoch)
    val_loss=val(val_loader,model,criterion,epoch)

    val_label_true=[''.join(map(str,x)) for x in val_loader.dataset.img_label]
    val_label_pred=predict(val_loader,model,tta=1)
    val_label_pred=np.vstack([
        val_label_pred[:,:11].argmax(1),
        val_label_pred[:,11:22].argmax(1),
        val_label_pred[:,22:33].argmax(1),
        val_label_pred[:,33:44].argmax(1),
        val_label_pred[:,44:55].argmax(1),
    ]).T

    val_pred_label=[]
    for x in val_label_pred:
        val_pred_label.append(''.join(map(str,x[x != 10])))

    val_char_acc=np.mean(np.cast(np.equal(val_pred_label,val_label_true),np.int))
    print('Epoch: {0}, Train loss: {1} \t Val loss: {2} \t Val acc: {3}'.format(epoch, train_loss, val_loss,val_char_acc))

    # 保存模型
    best_loss=np.inf
    if best_loss > val_loss:
        best_loss=val_loss
        print('Find best model in {} epoch,saving model.'.format(epoch))
        torch.save(model.state_dict(),'model.pt')

# 测试
test_path = glob.glob('datasets/mchar_test/mchar_test/*.png')
test_path.sort()
test_label = [[1]] * len(test_path)
print(len(test_path), len(test_label))
test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                transforms.Compose([
                    transforms.Resize((70, 140)),
                    # transforms.RandomCrop((60, 120)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=40,
    shuffle=False,
    num_workers=10,
)

model.load_state_dict(torch.load('model.pt'))
test_predict_label=predict(test_loader,model,1)
test_label= [''.join(map(str,x)) for x in test_loader]

test_predict_label=np.vstack([
    test_predict_label[:,:11].argmax(1),
    test_predict_label[:,11:22].argmax(1),
    test_predict_label[:,22:33].argmax(1),
    test_predict_label[:,33:44].argmax(1),
    test_predict_label[:,44:55].argmax(1),
]).T

test_label_pred=[]
for x in test_predict_label:
    test_label_pred.append(''.join(map(str,x[x != 10])))

# 生成csv预测文件
sub_csv_path='http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_sample_submit_A.csv'
sub_csv_path_log,_=urllib.request.urlretrieve(sub_csv_path,'datasets/test_sample_submit.csv')

df_submit=pd.read_csv(sub_csv_path_log)
df_submit['file_code']=test_label_pred
df_submit.to_csv('submit.csv',index=None)


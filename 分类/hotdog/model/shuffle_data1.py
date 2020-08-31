import os
import numpy as np

root_path=os.path.join(os.path.abspath(os.getcwd()),'hotdog\\hotdog')


# 得到原始顺序数据路径[img_path,label_name]
def create(root,mode):
    label={}
    s=[]
    org_file = open(os.path.join(root, mode + "_data.txt"), 'w')

    DataFile=os.path.join(root,mode)
    files = os.listdir(DataFile)
    for labels in files:
        tempdata = os.listdir(DataFile + "\\" + labels)
        label[labels] = len(label)
        for img in tempdata:
            org_file.write(DataFile + "\\" + labels + "\\" + img + " " + labels + "\n")
            s.append([DataFile + "\\" + labels + "\\" + img, labels])
    return s


def shuffle(root,mode,s):
    shuffle_file = open(os.path.join(root, mode + "_shuffle_Data.txt"), 'w')
    np.random.shuffle(s)
    for i in s:
        shuffle_file.write(i[0] + " " + str(i[1]) + "\n")


s=create(root_path,'test')
shuffle(root_path,'test',s)

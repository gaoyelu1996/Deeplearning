'''

'''


# 对标签进行个数统计，看一下每张图片包含的字符个数
import json
import matplotlib.pyplot as plt

train_label=json.load(open('datasets/json/train_json'))
train_label=[train_label[x]['label'] for x in train_label]

val_label=json.load(open('datasets/json/val_json'))
val_label=[val_label[x]['label'] for x in val_label]


# 统计每张图片的字符个数
def count_num(x):
    count_label = [0] * 6
    for i in range(len(x)):
        count = len(x[i])
        count_label[count - 1] += 1
    return count_label


# [4636, 16262, 7813, 1280, 8, 1]
train_count=count_num(train_label)
# [1918, 6393, 1569, 118, 2, 0]
val_count=count_num(val_label)
# 画出柱状图
plt.figure()
x = list(range(1, 7))
train_num=[4636, 16262, 7813, 1280, 8, 1]
plt.subplot(121)
plt.title('train_label')
plt.bar(x,train_count)
plt.subplot(122)
plt.title('val_label')
plt.bar(x,val_count)
plt.show()

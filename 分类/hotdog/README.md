hot dog 的分类问题

本问题是一个二分类问题，使用pytorch编写代码

数据结构：
  train训练集共1798张图片（hotdog 899张，not-hotdog 899张）
  val验证集共800张图片（hotdog 400张，not-hotdog 400张）
  test测试集共202张图片（hotdog 101张，not-hotdog 101张）

代码实现：
  工具：pytorch+gpu
  方法：迁移学习
  输入：(32,3,224,224)
  数据增强：在训练集上用了水平翻转，验证集和测试集上没用
  model：预训练好的resnet34,改最后fc输出层out_features=2
  损失函数：二分类交叉验证
  学习率：使用学习率学习函数lr_scheduler.StepLR() 每隔7个epoch缩小一次学习率，比例为0.1
  优化器：Adam() 因为最后fc层输出改变，这里应用不同的学习率，weight_decay=0.001
  early stop：当验证集损失在30个epoch不下降时，就停止训练
 
共训练100个epoch，最终准确度到达85%左右。




  

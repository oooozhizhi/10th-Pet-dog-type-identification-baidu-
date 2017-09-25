# 10th-Pet-dog-type-identification-baidu
简介：这是百度-西交大联合举办宠物狗识别比赛，训练集一共有2万张图片,测试集A 1万张图片，测试集B 2万张图片。需要对图片中涉及的100类宠物狗进行</br></br>
正确率：0.814 排名：10/1400</br></br>
主要工具：</br>
keras</br>
caffe</br>
torch</br></br>
主要思路：</br>
1 多模型投票。期间我们使用了densenet vgg resnet InceptionV3 InceptionV4等单一模型对图片进行分类，然后用一个投票模型取同一图片的投票众数作为该图片分类结果。</br>
2 我们参考了 http://www.iteye.com/news/32314 里面提到的Mask-CNN 对一只狗分为全图，狗身，狗头三部分作为狗的特征。不过数据有点大，我就不传上去了。</br>
3 至于如何从图中扣出狗参考 https://github.com/rbgirshick/fast-rcnn里面有源码</br>
4 为了防止过拟合，需要确认loss处于什么样的水平比较合理，这个就靠经验了。</br></br>

呃，等我有时间慢慢完善这个分享吧。</br>

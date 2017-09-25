这是本次比赛的源码，主要利用keras进行开发。</br>
以下将会简要介绍各个脚本的功能。</br>
注意：代码里所有的路径,基于某种原因，我都抹去了，如果使用需要修改 your path</br></br>
720-combineTrain.py 主要参照了https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html</br></br>
720-getFea.py 里面包含了很多模型，利用这些训练好的模型抽取图片特征</br>
包含模型有：</br>
1 dense161()</br>
2 dense121()</br>
3 dense169()</br>
4 xcep_2()</br>
5 incepV4()</br>
6 res152_2()</br></br>
densenet121.py densenet121的模型文件，作为720-getFea.py的模块。</br></br>
densenet161.py densenet161的模型文件，作为720-getFea.py的模块。</br></br>
densenet169.py densenet169的模型文件，作为720-getFea.py的模块。</br></br>
inceptionV4.py inceptionV4.py的模型文件，作为720-getFea.py的模块。</br></br>
resnet152.py resnet152.py的模型文件，作为720-getFea.py的模块。</br></br>
split.py 切分数据集</br></br>
torch_feature.py 利用torch库中的densenet resnet vgg等模型抽取数据。torch 和keras的模型使用的网络虽然是同一种网络但是网络权值是不同的，这样抽取出来的图片结果效果也不一致。</br></br>
wh-model-7.7.py 抽取特征（弃用）</br></br>
wh-train-7.8.py 跑出结果（弃用）</br></br>

投票模型：VoteTxT.java</br>
我就写一个简要的使用说明，首先编译VoteTxT.java里面没有跟路径有关的代码，不需要改动。</br>
运行命令 java VoteTxT <votelistFileName.txt> <outputFileName.txt> </br>
votelistFileName.txt 存放的是你之前模型运行出来的结果的文件名，一行一个文件名。</br>
类似如下格式:</br>
vote1.txt</br>
vote2.txt</br>
vote3.txt</br>
...</br></br>
outputFileName.txt 写一个你想要的文件名保存投票结果。</br></br>

训练的过程中主要使用：720-combineTrain.py来得到训练结果。</br>
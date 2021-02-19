一、比赛介绍
本次比赛是图像分类，根据木薯的叶子状态判断其是否健康，或者得的是什么病。
本次比赛难点是，官方给的训练数据有非常大噪音（很多图片的label标注是错误的）

二、我们的银牌解决方案是以下三个模型的Ensemble得到的：
1.EfficientNet-B4 （Public LB 0.902）
2.resnest101e （Public LB 0.899）
3.vit_base_patch16_384 （Public LB 0.897）
融合后Public LB是 0.906，Private LB是0.900。

三、本次比赛的一些trick
1.使用置信度剔除错误标签。
先对模型进行训练，然后对于模型高度自信并且和ground truth不一致的图片进行剔除。
2.使用TaylorCrossEntropyLoss
里面用到了smoothing label，简单的说就是让模型的预测不那么自信（因为数据有噪音）。
3.使用cutmix以及一些轻度的数据增强
4.使用少量且轻度的TTA。

四、代码
1.train.ipynb 是我在colab上的训练文件，把数据放到相应的路径后可以直接跑（详细介绍在“dataset”文件夹里）
修改model_arch和对应的img_size、train_bs、valid_bs即可训练不同的模型

2.inference.ipynb 最终在kaggle上的推理代码，配合“dataset”文件夹里的final_output.zip。

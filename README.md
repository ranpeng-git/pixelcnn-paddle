# PixelCNN

## 飞桨论文复现:Conditional Image Generation with PixelCNN Decoders
Original paper：[Conditional Image Generation with PixelCNN Decoders ]（https://paperswithcode.com/paper/conditional-image-generation-with-pixelcnn ）
<br>Dataset: Cifar-10 ( automatically download the dataset, and readers do not need to download additional.)
<br>验收标准：Cifar-10， NLL loss 3.03
<br>pretrained model：链接：https://pan.baidu.com/s/1oMm6qQzur6PtGLSZmKdWcQ 
提取码：kn3j

## Getting Start
This repository contains a Paddle implementation of PixelCNN in the paper "Conditional Image Generation with PixelCNN Decoders".

### Requirements

1、具有多个GPU的机器  
2、python3  
3、paddle2.1.2、Numpy等包  

### Structure

``` 
├── main                #  training
├── layers
├── model               #  model
├── save_image         
├── utils               # define loss function  
├── train.log           # the training log
├── README.md
```

### Training

```bash
# clone this repo
git clone git@github.com:adreamerof/PixelCNN.git
```

```bash
# start training
python main.py          
```

## Abstract

PixelCNN是一类强大的生成模型，具有易处理的可能性，也很容易从中采样。核心卷积神经网络计算一个像素值的概率分布，条件是它左侧和上方的像素值。  

来自模型的样本（左）和来自以cifar-10类标签为条件的模型样本（右）：  
![image](https://user-images.githubusercontent.com/49580855/138794773-c5520048-b306-4135-990c-d0804e390423.png)

## Results

<br>paddle pretrained model：链接：https://pan.baidu.com/s/1oMm6qQzur6PtGLSZmKdWcQ 
提取码：kn3j

target:NLL 3.03

[train.log](https://github.com/adreamerof/PixelCNN/blob/master/train_paddle.log):test loss : 3.0277796002371877

一个epoch进行一次test，输出loss，无eval文件。

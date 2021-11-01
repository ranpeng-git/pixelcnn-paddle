### PixelCNN

这是PixelCNN 的paddle实现，如下文所述：  
[论文]（https://paperswithcode.com/paper/conditional-image-generation-with-pixelcnn ）
PixelCNN是一类强大的生成模型，具有易处理的可能性，也很容易从中采样。核心卷积神经网络计算一个像素值的概率分布，条件是它左侧和上方的像素值。  

来自模型的样本（左）和来自以cifar-10类标签为条件的模型样本（右）：  
![image](https://user-images.githubusercontent.com/49580855/138794773-c5520048-b306-4135-990c-d0804e390423.png)

### 设置  
1、具有多个GPU的机器  
2、python3  
3、paddle2.2.Orc、Numpy等包  

### 训练模型  
使用main.py脚本来训练模型，想要在cifar-10上训练模型，只需使用：  
```python main.py``` 

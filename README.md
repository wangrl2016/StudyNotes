# StudyNotes

Study notes in computer science

### 机器学习合集

机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。

本教程的目标读者是那些具有Python编程经验，并且想要开始上手机器学习和深度学习的人。另外，熟悉Numpy库也有所帮助，但并不是必须的。
你不需要具有机器学习或深度学习方面的经验，本教程包含从头学习所需的必要基础知识。

[B站视频（倒序）](https://space.bilibili.com/3461566190061988)（视频标题和知识点）

- [ ] 卷积神经网络(4)：最大池化运算

* 最大池化的作用：对特征图进行下采样，与步进卷积类似。
* 最大池化与卷积的最大不同之处在于，最大池化通常使用2×2的窗口和步幅2，其目的是将特征图下采样2倍。
* 使用下采样的原因，一是减少需要处理的特征图的元素个数，二是通过让连续卷积层的观察窗口越来越大(即窗口覆盖原始输入的比例越来越大)，从而引入空间过滤器的层级结构。
* 最大池化代码`layers.MaxPooling2D((2, 2))`
* 参考网址：[Max pooling示意图](https://paperswithcode.com/method/max-pooling)

- [x] 卷积神经网络(3)：卷积计算（下）

* 参考网址：[A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

- [x] 卷积神经网络(2)：卷积计算（上）

- [x] 卷积神经网络(1)：使用CIFAR数据集进行识别


- [x] 神经网络模型中的重要概念介绍
- [x] 神经网络的数据表示：张量(Tensor)
- [x] 科学计算基础包NumPy入门介绍
- [x] 导数基础概念介绍，以及Sigmoid函数求导
- [x] 使用NumPy库进行矩阵计算（点乘、转置）
- [x] 神经网络的“齿轮”：张量运算（上）
- [x] 神经网络的“齿轮”：张量运算（下）
- [x] 神经网络的“引擎”：基于梯度的优化介绍
- [x] 手写神经网络(0)：根据身高和体重预测性别项目介绍
- [x] 手写神经网络(1)：构建神经元(Neuron)
- [x] 手写神经网络(2)：将神经元组合成神经网络
- [x] 手写神经网络(3)：微分知识（链式法则和偏导数）
- [x] 手写神经网络(4)：训练神经网络（上）
- [x] 手写神经网络(5)：训练神经网络（下）
- [x] 手写神经网络(6)：完整代码和总结

- [x] 神经网络模型中的HelloWorld程序介绍



- [x] 详解深度学习MNIST数据集（手写数字）

* [MNIST数据集官网](http://yann.lecun.com/exdb/mnist/)
* Keras加载MNIST数据集代码`(x_train, y_train), (x_test, y_test) = mnist.load_data()`
* 代码：[解析MNIST数据集程序](machine_learning/01_network/mnist_database.py)
* MNIST数据集图像为28x28的大小，使用int32表示，0-255表示单通道颜色，使用byte表示。

- [x] 《使用Python进行深度学习》开场白

* 下载[《使用Python进行深度学习》中文版](resources/deep_learning_with_python.pdf)
* [TensorFlow中文版官网](https://www.tensorflow.org/?hl=zh-cn)
* 机器学习领域的专家需要具备四个方面的知识：编码技能、数学和统计学、机器学习理论、构建自己的项目

#### TensorFlow官网讲解

主要包括：

(1) Keras机器学习基础知识

* 均方误差（MSE）是用于回归问题的常见损失函数（分类问题中使用不同的损失函数）。

* 早期停止是一种防止过度拟合的有效技术。



(2) 加载和预处理数据

(3) 图像

* (卷积基础)[https://betterexplained.com/articles/intuitive-convolution/]

* 
(4) 文本

(5) 生成式

B站视频

- [x] 对服装图像进行分类，了解TensorFlow处理全流程

#### Keras实例


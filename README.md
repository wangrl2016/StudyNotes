# StudyNotes

Study notes in computer science

### 机器学习合集

机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。

本教程的目标读者是那些具有Python编程经验，并且想要开始上手机器学习和深度学习的人。另外，熟悉Numpy库也有所帮助，但并不是必须的。
你不需要具有机器学习或深度学习方面的经验，本教程包含从头学习所需的必要基础知识。

[B站视频（倒序）](https://space.bilibili.com/3461566190061988)（视频标题和知识点）

- [ ] GAN

- [ ] 声音识别

- [ ] 循环神经网络

- [ ] 使用one-hot编码将文本转化为张量

- [ ] 图像分割

- [ ] 数据增强

- [ ] 可视化卷积神经网络的过滤器

* 想要观察卷积神经网络学到的过滤器，一种简单的方法是显示每个过滤器所响应的视觉模式。
* 这可以通过在输入空间中进行梯度上升来实现：从空白输入图像开始，将梯度下降应用于卷积神经网络输入图像的值，其目的是让某个过滤器的响应最大化。得到的输入图像是选定的过滤器具有最大响应的图像。
* 代码：[可视化过滤器](machine_learning/02_tensorflow/visual_conv_filter.py)
* 参考网址：[Visualizing what convnets learn](https://keras.io/examples/vision/visualizing_what_convnets_learn/)

- [x] 卷积神经网络(5)：可视化中间激活

* 可视化中间激活，是指对于给定输入，展示网络中各个卷积层和池化层输出的特征图（层的输出通常被称为该层的激活，即激活函数的输出）。
* 深度神经网络的普遍特征：随着层数的加深，层所提取的特征变得越来越抽象。更高的层激活关于特定输入的信息越来越少，而关于目标的信息越来越多。
* 深度神经网络可以有效地作为信息蒸馏管道，输入原始数据，反复对其进行变换，将无关信息过滤掉，并放大和细化有用的信息。
* 代码：[可视化中间激活](machine_learning/02_tensorflow/visual_conv_layer.py)

- [x] 卷积神经网络(4)：最大池化运算

* 最大池化的作用：对特征图进行下采样，与步进卷积类似。
* 最大池化与卷积的最大不同之处在于，最大池化通常使用2×2的窗口和步幅2，其目的是将特征图下采样2倍。
* 使用下采样的原因，一是减少需要处理的特征图的元素个数，二是通过让连续卷积层的观察窗口越来越大（即窗口覆盖原始输入的比例越来越大），从而引入空间过滤器的层级结构。
* 最大池化代码`layers.MaxPooling2D((2, 2))`
* 参考网址：[Max pooling示意图](https://paperswithcode.com/method/max-pooling)

- [x] 卷积神经网络(3)：卷积计算（下）

* 参考网址：[A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

- [x] 卷积神经网络(2)：卷积计算（上）

- [x] 卷积神经网络(1)：使用CIFAR数据集进行识别

- [ ] 防止过拟合的四种策略：重点介绍添加权重正则化

- [ ] 回归问题(Regression)：通过数据描述输出连续值

- [ ] 对服装图像进行分类，了解TensorFlow处理全流程

- [x] 手写神经网络(6)：完整代码和总结

- [x] 手写神经网络(5)：训练神经网络（下）

- [x] 手写神经网络(4)：训练神经网络（上）

- [x] 手写神经网络(3)：微分知识（链式法则和偏导数）

* 链式法则用于求合成函数的导数。
* 在数学中，偏导数（英语：partial derivative）的定义是：一个多变量的函数（或称多元函数），对其中一个变量（导数）微分，而保持其他变量恒定。
* 参考网址：[链式法则](https://zh.wikipedia.org/wiki/链式法则)
* [偏导数](https://zh.wikipedia.org/zh-hans/偏导数)

- [x] 手写神经网络(2)：将神经元组合成神经网络

* 通过神经元构建三层的神经网络。 
* 将身高和体重两个输入数据传递到神经网络中。 
* 通过隐藏层的计算输入到输出层中。 
* 最后得到男女性别的概率。

- [x] 手写神经网络(1)：构建神经元(Neuron)

* 一个神经元的功能是求得输入向量与权向量的内积后，经一个非线性传递函数得到一个标量结果。
* 代码：[Neurons](machine_learning/01_network/neurons.py)

- [x] 手写神经网络(0)：根据身高和体重预测性别项目介绍

* 代码：[全连接神经网络](machine_learning/01_network/simple_neural_network.py)
* 参考网址：[Machine Learning for Beginners: An Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)

- [x] 神经网络的“引擎”：基于梯度的优化介绍

* 训练过程：
  1. 抽取训练样本x和对应目标y组成的数据批量。
  2. 在x上运行网络[这一步叫作前向传播(forward pass)]，得到预测值y_pred。
  3. 计算网络在这批数据上的损失，用于衡量y_pred和y之间的距离。
  4. 更新网络的所有权重，使网络在这批数据上的损失略微下降。
* 小批量随机梯度下降：
  1. 抽取训练样本x和对应目标y组成的数据批量。
  2. 在x上运行网络，得到预测值y_pred。
  3. 计算网络在这批数据上的损失，用于衡量y_pred和y之间的距离。
  4. 计算损失相对于网络参数的梯度[一次反向传播(backward pass)]。
  5. 将参数沿着梯度的反方向移动一点，比如 W -= step * gradient，从而使这批数据上的损失减小一点。

- [x] 神经网络的“齿轮”：张量运算（下）

* 点积`(a, b, c, d) . (d, e) -> (a, b, c, e)`
* 张量变形是指改变张量的行和列，以得到想要的形状。变形后的张量的元素总个数与初始张量相同。
* 代码：[张量计算](machine_learning/01_network/tensor_operations.py)

- [x] 神经网络的“齿轮”：张量运算（上）

* 代码示例：`output = relu(dot(W, input) + b)`
* 如果将两个形状不同的张量相加，较小的张量会被广播(broadcast)，广播的步骤：
  1. 向较小的张量添加轴(叫作广播轴)，使其ndim与较大的张量相同。
  2. 将较小的张量沿着新轴重复，使其形状与较大的张量相同。
* 代码：[张量计算](machine_learning/01_network/tensor_operations.py)

- [x] 使用NumPy库进行矩阵计算（点积、转置）

* 代码：[矩阵计算](machine_learning/01_network/matrices.py)
* 参考网址：[Matrix introduction](https://www.mathsisfun.com/algebra/matrix-introduction.html)
* [Matrix multiplying](https://www.mathsisfun.com/algebra/matrix-multiplying.html)

- [x] 导数基础概念介绍，以及Sigmoid函数求导

* 参考网址：[Introduction to Derivatives](https://www.mathsisfun.com/calculus/derivatives-introduction.html)
* [e(数学常数)的表示](https://zh.wikipedia.org/zh-cn/E_(数学常数))
* [e^x求导](http://www.intuitive-calculus.com/derivative-of-e-x.html)
* [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)
* [Derivative of the sigmoid function](https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e)

- [x] 科学计算基础包NumPy入门介绍

* 代码示例`np.linspace(0, 10, num=5)`
* 参考网址：[NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)

- [x] 神经网络的数据表示：张量(Tensor)

* 向量数据：2D张量，形状为 (samples, features)。
* 时间序列数据或序列数据：3D张量，形状为 (samples, timestamps, features)。
* 图像：4D张量，形状为 (samples, height, width, channels) 或 (samples, channels, height, width)。
* 视频：5D张量，形状为 (samples, frames, height, width, channels) 或 (samples, frames, channels, height, width)。
* 代码：[张量表示](machine_learning/01_network/tensor_represent.py)

- [x] 神经网络模型中的重要概念介绍

* 深度神经网络通过一系列简单的数据变换(layer)来实现这种输入到目标的映射，而这些数据变换都是通过观察示例学习到的。
* 神经网络中每层对输入数据所做的具体操作保存在该层的权重(weight)中，其本质是一串数字。
* 学习的意思是为神经网络的所有层找到一组权重值，使得该网络能够将每个示例输入与其目标正确地一一对应。
* 损失函数(loss)的输入是网络预测值与真实目标值(即你希望网络输出的结果)，然后计算一个距离值，衡量该网络在这个示例上的效果好坏。
* 深度学习的基本技巧是利用这个距离值作为反馈信号来对权重值进行微调，以降低当前示例对应的损失值。这种调节由优化器(optimizer)来完成，它实现了所谓的反向传播(backpropagation)算法，这是深度学习的核心算法。

- [x] 神经网络模型中的HelloWorld程序介绍

* 神经网络的处理步骤：(1)预处理数据；(2)构建模型；(3)训练模型
* 代码：[QuickStart](machine_learning/01_network/quick_start.py)

- [x] 详解深度学习MNIST数据集（手写数字）

* Keras加载MNIST数据集代码`(x_train, y_train), (x_test, y_test) = mnist.load_data()`
* MNIST数据集图像为28x28的大小，使用int32表示，0-255表示单通道颜色，使用byte表示。
* 代码：[解析MNIST数据集](machine_learning/01_network/mnist_database.py)
* 参考网址：[MNIST数据集官网](http://yann.lecun.com/exdb/mnist/)

- [x] 《使用Python进行深度学习》开场白

* 机器学习领域的专家需要具备四个方面的知识：编码技能、数学和统计学、机器学习理论、构建自己的项目
* 下载链接：[《使用Python进行深度学习》中文版](resources/deep_learning_with_python.pdf)
* 参考网址：[TensorFlow中文版官网](https://www.tensorflow.org/?hl=zh-cn)

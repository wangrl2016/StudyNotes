"""
Machine Learning for Beginners: An Introduction to neural Networks
https://victorzhou.com/blog/intro-to-neural-networks/

解决的问题：根据身高和体重来预测性别

数据集：
Name	Weight (minus 135)	Height (minus 66)	Gender
Alice	        -2	                -1	            1
Bob	            25	                6	            0
Charlie	        17	                4	            0
Diana	        -15	                -6	            1

预测：体重128 pounds, 身高63 inches 性别：男或者女？

手写神经网络(0)：根据身高和体重预测性别项目介绍
手写神经网络(1)：构建神经元
手写神经网络(2)：将神经元组合成神经网络
手写神经网络(3)：微分知识（链式法则和偏导数）
手写神经网络(4)：训练神经网络（上）
手写神经网络(5)：训练神经网络（下）
手写神经网络(6)：完整代码和总结

1. Building Blocks: Neurons（神经元）
2. Combining Neurons into a Neural Network
3. Partial Derivative(维基百科)
4. Training a Neural Network, Part 1
5. Training a Neural Network, Part 2
6. Code: A Complete Neural Network

"""
import numpy as np


def sigmoid(x):
    # sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    """
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)
    """

    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feed_forward(self, x):
        # x is a numpy array with 2 elements.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        """
        learn_rate = 0.1
        epochs = 1000  # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_l_d_pred = -2 * (y_true - y_pred)

                # Neuron o1
                d_pred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_pred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_pred_d_b3 = deriv_sigmoid(sum_o1)

                d_pred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_pred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_l_d_pred * d_pred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_l_d_pred * d_pred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_l_d_pred * d_pred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_l_d_pred * d_pred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_l_d_pred * d_pred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_l_d_pred * d_pred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_l_d_pred * d_pred_d_w5
                self.w6 -= learn_rate * d_l_d_pred * d_pred_d_w6
                self.b3 -= learn_rate * d_l_d_pred * d_pred_d_b3

            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                predicts = np.apply_along_axis(self.feed_forward, 1, data)
                loss = mse_loss(all_y_trues, predicts)
                print("Epoch %d loss: %.3f" % (epoch, loss))


def main():
    # define dataset
    data = np.array([
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ])
    all_y_trues = np.array([
        1,  # Alice
        0,  # Bob
        0,  # Charlie
        1,  # Diana
    ])

    # train our neural network
    network = OurNeuralNetwork()
    network.train(data, all_y_trues)

    # user the network to predict genders
    emily = np.array([-7, -3])  # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches
    print("Emily: %.3f" % network.feed_forward(emily))  # 0.951 - F
    print("Frank: %.3f" % network.feed_forward(frank))  # 0.039 - M


if __name__ == '__main__':
    main()

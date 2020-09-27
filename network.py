#!C:\Users\wb.yexulong\AppData\Local\Programs\Python\Python37\python.exe
# -*- coding: utf-8 -*-
# Sigmoid激活函数类
from functools import reduce

import numpy as np
from fullconnectedlayer import FullConnectedLayer
from loader import *


class SigmoidActivator(object):
    def forward(self, weighted_input):
        if weighted_input.all() < 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
            return np.exp(weighted_input) / (1 + np.exp(weighted_input))
        else:
            return 1.0 / (1 + np.exp(-weighted_input))
        # return 1.0 / (1 + np.exp(-weighted_input))

    def backward(self, output):
        return np.array(output) * (1 - np.array(output))


# 神经网络类
class Network(object):
    def __init__(self, layers):
        """
        构造函数
        """
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i + 1],
                    SigmoidActivator()
                )
            )

    def predict(self, sample):
        """
        使用神经网络实现预测
        sample: 输入样本
        """
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        #     print('output_size:', layer.output_size)
        # print("layer_output:", output)
        return output.sum(1)

    def train(self, labels, data_set, rate, epoch):
        """
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(np.array(labels[d]).reshape(len(labels[d]), 1),
                                      np.array(data_set[d]).reshape(len(data_set[d]), 1), rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()

    def gradient_check(self, sample_feature, sample_label):
        """
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        """

        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i, j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i, j] -= 2 * epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i, j] += epsilon
                    print('weights(%d,%d): expected - actual %.4e - %.4e' % (
                        i, j, expect_grad, fc.W_grad[i, j]))


if __name__ == '__main__':
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network([784, 300, 10])
    # print(network.layers[-1].activator.backward(
    #         network.layers[-1].output
    #     ))
    # print(network.dump())
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.3, 1)
        print('%s epoch %d finished' % (datetime.now(), epoch))

        dirs = 'testModel'
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        if epoch % 10 == 0:
            # 保存模型
            joblib.dump(network, dirs + '/Network.pkl')
            print('第%d轮，模型保存成功' % epoch)
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (datetime.now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

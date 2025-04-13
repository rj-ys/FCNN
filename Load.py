import os
import pickle
import numpy as np
from Model import ThreeLayerNN


def load_cifar10(data_dir):
    """ 数据读入与预处理 """

    train_files = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 6)]
    test_file = os.path.join(data_dir, 'test_batch')

    # 加载训练数据
    X_train, y_train = [], []
    for file in train_files:
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        X_train.append(data[b'data'])
        y_train.extend(data[b'labels'])

    # 加载测试数据
    with open(test_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    # 合并数据
    X_train = np.concatenate(X_train).astype(np.float32) / 255.0
    y_train = np.array(y_train)
    X_test = data[b'data'].astype(np.float32) / 255.0
    y_test = np.array(data[b'labels'])

    # 划分验证集
    mask = np.random.choice([True, False], size=len(X_train), p=[0.9, 0.1])
    X_val = X_train[~mask]
    y_val = y_train[~mask]
    X_train = X_train[mask]
    y_train = y_train[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_model(filename):
    """加载模型参数并初始化网络"""
    loaded = np.load(filename)

    # 从加载的数据中提取参数
    params = {
        'input_size': loaded['input_size'].item(),
        'hidden_size': loaded['hidden_size'].item(),
        'output_size': loaded['output_size'].item(),
        'activation': loaded['activation'].item(),
        'reg': loaded['reg'].item(),
        'W1': loaded['W1'],
        'b1': loaded['b1'],
        'W2': loaded['W2'],
        'b2': loaded['b2']
    }

    # 用加载的参数初始化新模型
    model = ThreeLayerNN(
        input_size=params['input_size'],
        hidden_size=params['hidden_size'],
        output_size=params['output_size'],
        activation=params['activation'],
        reg=params['reg'],
        W1=params['W1'],
        b1=params['b1'],
        W2=params['W2'],
        b2=params['b2']
    )
    return model



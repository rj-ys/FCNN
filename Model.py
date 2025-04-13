import numpy as np
import copy


class ThreeLayerNN:
    """ 三层神经网络类 """

    def __init__(self, input_size, hidden_size, output_size, activation='relu',
                 reg=0.0, W1=None, b1=None, W2=None, b2=None):

        # 超参数载入
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.reg = reg
        self.cache = {}
        if activation not in {'relu', 'sigmoid'}:
            raise ValueError(f"不支持的激活函数: {activation}，请选择 'relu' 或 'sigmoid'")

        # 权重载入/初始化
        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        init_factor = lambda n: np.sqrt(2.0 / n)  # He初始化
        if activation == 'sigmoid':
            init_factor = lambda n: np.sqrt(1.0 / n)  # Xavier初始化
        if W1 is None:
            self.params['W1'] = np.random.randn(input_size, hidden_size) * init_factor(input_size)
        if b1 is None:
            self.params['b1'] = np.zeros(hidden_size)
        if W2 is None:
            self.params['W2'] = np.random.randn(hidden_size, output_size) * init_factor(hidden_size)
        if b2 is None:
            self.params['b2'] = np.zeros(output_size)

        # 形状校验
        self._validate_shape('W1', (input_size, hidden_size))
        self._validate_shape('b1', (hidden_size,))
        self._validate_shape('W2', (hidden_size, output_size))
        self._validate_shape('b2', (output_size,))

    def _validate_shape(self, param_name, expected_shape):
        """ 校验参数形状的辅助方法 """
        actual = self.params[param_name]
        if actual.shape != expected_shape:
            raise ValueError(
                f"参数 {param_name} 形状错误，应为 {expected_shape}，实际为 {actual.shape}"
            )

    def forward(self, X):
        """ 前向计算 """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 第一层
        z1 = X.dot(W1) + b1
        if self.activation == 'relu':
            a1 = np.maximum(0, z1)
        elif self.activation == 'sigmoid':
            a1 = 1 / (1 + np.exp(-z1))

        # 输出层
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.cache = {'z1': z1, 'a1': a1, 'z2': z2, 'probs': probs}
        return probs

    def backward(self, X, y, learning_rate):
        """ 反向传播 """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        a1, z1, probs = self.cache['a1'], self.cache['z1'], self.cache['probs']
        m = X.shape[0]

        # 输出层梯度
        dz2 = probs.copy()
        dz2[range(m), y] -= 1
        dz2 /= m

        dW2 = a1.T.dot(dz2) + self.reg * W2
        db2 = np.sum(dz2, axis=0)

        # 隐藏层梯度
        da1 = dz2.dot(W2.T)
        if self.activation == 'relu':
            dz1 = da1.copy()
            dz1[z1 <= 0] = 0
        elif self.activation == 'sigmoid':
            dz1 = da1 * (a1 * (1 - a1))

        dW1 = X.T.dot(dz1) + self.reg * W1
        db1 = np.sum(dz1, axis=0)

        # 参数更新
        self.params['W1'] -= learning_rate * dW1
        self.params['b1'] -= learning_rate * db1
        self.params['W2'] -= learning_rate * dW2
        self.params['b2'] -= learning_rate * db2

    def compute_loss(self, X, y):
        """ 损失计算 """
        probs = self.forward(X)
        m = X.shape[0]
        corect_logprobs = -np.log(probs[range(m), y])
        data_loss = np.sum(corect_logprobs) / m
        reg_loss = 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))
        return data_loss + reg_loss

    def save(self, filename):
        """保存模型参数到文件"""
        params_all = copy.deepcopy(self.params)
        params_all['input_size'] = self.input_size
        params_all['hidden_size'] = self.hidden_size
        params_all['output_size'] = self.output_size
        params_all['activation'] = self.activation
        params_all['reg'] = self.reg
        np.savez(filename, **params_all)

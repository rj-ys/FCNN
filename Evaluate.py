import numpy as np


# 测试
def evaluate(model, X, y):
    probs = model.forward(X)
    predictions = np.argmax(probs, axis=1)
    return np.mean(predictions == y)

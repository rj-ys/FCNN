from Model import ThreeLayerNN
from Train import train
from Evaluate import evaluate


# 参数搜索函数
def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_acc = 0
    best_params = {}

    # 尝试不同的参数组合
    for hidden_size in [64, 128, 256, 512]:
        for lr in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
            for reg in [0, 0.0001, 0.001, 0.01, 0.1]:
                print(f'Testing hsize={hidden_size}, lr={lr}, reg={reg}:', end=' ')
                model = ThreeLayerNN(3072, hidden_size, 10, activation='relu', reg=reg)
                model = train(model, X_train, y_train, X_val, y_val,
                              epochs=20, learning_rate=lr, info=False)
                val_acc = evaluate(model, X_val, y_val)
                print(f'\n准确率 {val_acc}')
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = {'hidden_size': hidden_size, 'lr': lr, 'reg': reg}

    print(f'Best validation accuracy: {best_acc:.4f}')
    print('Best parameters:', best_params)
    return best_params

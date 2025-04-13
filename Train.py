import numpy as np
from Evaluate import evaluate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", font_scale=1.2, context="talk", color_codes=False)
matplotlib.rcParams['mathtext.fontset'] = 'cm'


# 训练
def train(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64,
          learning_rate=1e-3, lr_decay=0.95, info=True):
    train_ls, val_ls = [], []
    train_ac, val_ac = [], []
    best_val_acc = 0.0
    best_params = model.params.copy()

    for epoch in range(epochs):
        # 学习率衰减
        if epoch % 10 == 0 and epoch > 0:
            learning_rate *= lr_decay

        # 随机打乱数据
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Mini-batch训练
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # 前向传播和反向传播
            model.forward(X_batch)
            model.backward(X_batch, y_batch, learning_rate)

        # 验证集评估
        val_acc = evaluate(model, X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

        # 打印训练信息
        if info:
            train_loss = model.compute_loss(X_train, y_train)
            train_acc = evaluate(model, X_train, y_train)
            train_ls.append(train_loss)
            train_ac.append(train_acc)
            val_loss = model.compute_loss(X_val, y_val)
            val_ls.append(val_loss)
            val_ac.append(val_acc)
            print(
                f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        else:
            print(f'{epoch + 1}', end=' ')

    if info:
        plot(train_ls, val_ls, 'Loss', 'loss.pdf')
        plot(train_ac, val_ac, 'Accuracy', 'acc.pdf')

    # 恢复最佳参数
    model.params = best_params
    return model


def plot(train, val, y, name):
    # 颜色列表
    colors = ["green", (168 / 255, 3 / 255, 38 / 255)]

    # 数据和样式参数
    plot_data = [
        (train, 'Training Set', 'o', colors[0]),
        (val, 'Validation Set', '^', colors[1])
    ]
    t = np.arange(1, len(train) + 1)

    # 绘制曲线
    plt.figure(figsize=(9, 6))
    for ls, label, marker, color in plot_data:
        plt.plot(t, ls, linewidth=5, color=color, label=label, marker=marker,
                 markevery=len(t) // 5, markersize=15)

    # 坐标轴设置
    plt.xlabel('Epoch', fontweight='bold', fontsize=25)
    plt.ylabel(y, fontweight='bold', fontsize=25)
    plt.xlim(1, len(train))
    # plt.xscale('log')

    # 调整刻度样式
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=25, width=2)
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label.set_fontweight('bold')
    ax.xaxis.get_offset_text().set_fontweight('bold')
    ax.xaxis.get_offset_text().set_fontsize(25)

    plt.legend(prop={'weight': 'bold'})
    plt.tight_layout()
    plt.savefig(name, format='pdf', bbox_inches='tight')
    plt.show()

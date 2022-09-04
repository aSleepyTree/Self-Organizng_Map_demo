from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn import datasets
from minisom import MiniSom
import math

iris = datasets.load_iris()
print(">> shape of data:", iris.data.shape)

feature_names = iris.feature_names
class_names = iris.target_names

X = iris.data
y = iris.target

# 划分训练集、测试集  7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


N = X_train.shape[0]  # 样本数量
M = X_train.shape[1]  # 维度/特征数量

"""
设置超参数
"""
size = math.ceil(np.sqrt(5 * np.sqrt(N)))  # 经验公式：决定输出层尺寸
print("训练样本个数:{}  测试样本个数:{}".format(N, X_test.shape[0]))
print("输出网格最佳边长为:", size)

max_iter = 200

# Initialization and training
som = MiniSom(size, size, M, sigma=3, learning_rate=0.5, neighborhood_function="bubble")
"""
初始化权值，有2个API
"""
# som.random_weights_init(X_train)
som.pca_weights_init(X_train)
som.train_batch(X_train, max_iter, verbose=False)
# som.train_random(X_train, max_iter, verbose=False)

winmap = som.labels_map(X_train, y_train)


def classify(som, data, winmap):
    from numpy import sum as npsum

    default_class = npsum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


# 输出混淆矩阵
y_pred = classify(som, X_test, winmap)
print("原标记", y_test)
print("预测标记", y_pred)
print(classification_report(y_test, np.array(y_pred)))

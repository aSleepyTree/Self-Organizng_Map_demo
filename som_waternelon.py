import numpy as np
from minisom import MiniSom
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import math
from numpy import sum as npsum

data = np.array(
    [
        [0.697, 0.460],
        [0.774, 0.376],
        [0.634, 0.264],
        [0.608, 0.318],
        [0.556, 0.215],
        [0.403, 0.237],
        [0.481, 0.149],
        [0.437, 0.211],
        [0.666, 0.091],
        [0.243, 0.267],
        [0.245, 0.057],
        [0.343, 0.099],
        [0.639, 0.161],
        [0.657, 0.198],
        [0.360, 0.370],
        [0.593, 0.042],
        [0.719, 0.103],
    ]
)
print(">> shape of data:", data.shape)

X = data
# y = iris.target
Y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# 留出法划分训练集、测试集  7:3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

N = X_train.shape[0]  # 样本数量
M = X_train.shape[1]  # 维度/特征数量
size = math.ceil(np.sqrt(5 * np.sqrt(N)))  # 经验公式：决定输出层尺寸5*sqrt(N)
print("训练样本个数:{}  测试样本个数:{}".format(N, X_test.shape[0]))
print("输出网格最佳边长为:", size)
# 迭代次数
max_iter = 200

# Initialization and training
som = MiniSom(
    size, size, M, sigma=3, learning_rate=0.5, neighborhood_function="gaussian"
)  # neighborhood_function可使用bubble降低计算负担

# 初始化权重
som.pca_weights_init(X_train)
# som.random_weights_init(X_train)

# 按顺序挑选样本，下一行为随机挑选样本代码
som.train_batch(X_train, max_iter, verbose=False)
# som.train_random(X_train, max_iter, verbose=False)

# 使用标签信息标注训练好的som网络，som本身是无监督的学习方法
winmap = som.labels_map(X_train, y_train)

# for each in winmap:
#     print(each, winmap[each])


def classify(som, data, winmap):
    # default_class的作用:如果不能刚好落于winmap则以训练集中占多数的类为输出即默认类
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for each in data:
        win_position = som.winner(each)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


y_pred = classify(som, X_test, winmap)
print("测试集标记", y_test)
print("预测得标记", y_pred)
# print(list(winmap.values()))
# print(npsum(list(winmap.values())))
print(classification_report(y_test, np.array(y_pred), target_names=["好瓜", "坏瓜"]))

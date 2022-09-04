# Self-Organizng_Map_demo 自组织映射网络(SOM)的学习及简单应用



使用minisom库的som简单应用，som_watermelon数据来自于西瓜书《机器学习》周志华







SOM是一种无监督的神经网络模型，不使用误差单向传播方法而是用竞争策略进行训练，同时使用近邻关系维持网络的拓扑结构。具体方法为每个神经元具有一个初始的权向量，对于每一个输入的测试样本计算神经元向量与其的相似度或者称为“距离”，找到最近的神经元记为“最佳匹配单元”并调整最佳匹配单元及其邻域的神经元的权重，不断迭代上述过程直至收敛。

首先导入所需库

	import numpy as np
	from minisom import MiniSom
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import classification_report
	import math
	from numpy import sum as npsum

获得数据并进行预处理，数据来自于《机器学习》周志华

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
进行迭代训练

	# 迭代次数
	max_iter = 200
	
	# Initialization and training
	som = MiniSom(
	    size, size, M, sigma=3, learning_rate=0.5, neighborhood_function="gaussian"
	)  # neighborhood_function可使用bubble降低计算负担
	
	# 初始化权重，下一行为随机初始化
	som.pca_weights_init(X_train)
	# som.random_weights_init(X_train)
	
	# 按顺序挑选样本，下一行为随机挑选样本代码
	som.train_batch(X_train, max_iter, verbose=False)
	# som.train_random(X_train, max_iter, verbose=False)
	
	# 使用标签信息标注训练好的som网络，som本身是无监督的学习方法
	winmap = som.labels_map(X_train, y_train)

使用训练得到的模型对测试集进行预测并展示结果

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

示例输出为

	>> shape of data: (17, 2)
	训练样本个数:11  测试样本个数:6
	输出网格最佳边长为: 5
	测试集标记 [0, 0, 1, 1, 1, 0]
	预测得标记 [0, 1, 1, 1, 1, 1]
	              precision    recall  f1-score   support
	
	          好瓜       1.00      0.33      0.50         3
	          坏瓜       0.60      1.00      0.75         3
	
	    accuracy                           0.67         6
	   macro avg       0.80      0.67      0.62         6
	weighted avg       0.80      0.67      0.62         6

其中precision为查准率，recall为查全率。网络实现过程中有很多细节在[minisom](https://github.com/JustGlowing/minisom)中体现为不同的参数和实现同一步骤的不同函数，代码中有少量体现。代码可见[于次](https://github.com/aSleepyTree/Self-Organizng_Map_demo)

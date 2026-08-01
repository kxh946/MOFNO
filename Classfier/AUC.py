import numpy as np
from sklearn.datasets import load_iris # 用iris数据集作为示例
from sklearn.neighbors import KNeighborsClassifier # KNN分类器
from sklearn.model_selection import cross_val_score # 交叉验证函数
from sklearn.metrics import roc_auc_score # AUC评估函数

# 加载数据集
X, y = load_iris(return_X_y=True)

# 定义KNN分类器，设置邻居数为5
knn = KNeighborsClassifier(n_neighbors=5)

# 进行十折交叉验证，返回每折的预测概率
y_pred = cross_val_score(knn, X, y, cv=10, scoring='roc_auc_ovr', n_jobs=-1)

# 计算十折的平均AUC
auc = np.mean(y_pred)

# 打印结果
print('The average AUC of 10-fold cross validation is:', auc)
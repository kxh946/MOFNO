import numpy as np


# 适应度函数 10折交叉
from sklearn import neighbors
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier


def fitnessFunction_KNN_CV(findData_x, findData_y , CV):
    if len(findData_x.shape) == 0:
        return 0
    if len(findData_x.shape) == 1:
        findData_x = np.reshape(findData_x,-1)

    knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto", metric='manhattan')
    # # 转化为二维数组
    if len(findData_x.shape) == 1:
        findData_x = findData_x.reshape((len(findData_x), 1))
    # cv参数决定数据集划分比例，这里是按照9:1划分训练集和测试集
    scores = cross_val_score(knn, findData_x, findData_y, cv=CV, scoring='accuracy')
    accuracy = scores.mean()
    return accuracy


# 适应度函数  70% 训练  30%实验
def fitnessFunction_KNN_percent(findData_x, findData_y):
    # 计算2
    # 创建了一个knn分类器的实例，并拟合数据
    x_train, x_test, y_train, y_test = train_test_split(findData_x, findData_y, test_size=0.3, random_state=0)
    clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm="auto", weights="distance")
    # 转化为二维数组
    if len(x_train.shape) == 1:
        x_train = x_train.reshape((len(x_train), 1))

    clf.fit(x_train, y_train)
    if len(x_test.shape) == 1:
        x_test = x_test.reshape((len(x_test), 1))
    predictOfTest = clf.predict(x_test)
    numberOfTrue = 0
    for i in range(0, len(predictOfTest)):
        if predictOfTest[i] == y_test[i]:
            numberOfTrue += 1
    accuracy = numberOfTrue / len(predictOfTest)
    return accuracy

def KNN_AUC(findData_x, findData_y , CV):
    if len(findData_x) == 0:
        return 0

    knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto", metric='manhattan')
    # # 转化为二维数组
    if len(findData_x.shape) == 1:
        findData_x = findData_x.reshape((len(findData_x), 1))
    # cv参数决定数据集划分比例，这里是按照9:1划分训练集和测试集
    scores = cross_val_score(knn, findData_x, findData_y, cv=CV, scoring='roc_auc_ovr')
    auc = scores.mean()
    return auc

def KNN_BACC(findData_x, findData_y , CV):
    if len(findData_x) == 0:
        return 0
    if len(findData_x.shape) == 0:
        return 0

    knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto", metric='manhattan')
    # # 转化为二维数组
    if len(findData_x.shape) == 1:
        findData_x = findData_x.reshape((len(findData_x), 1))
    # cv参数决定数据集划分比例，这里是按照9:1划分训练集和测试集
    scores = cross_val_score(knn, findData_x, findData_y, cv=CV, scoring='balanced_accuracy')
    auc = scores.mean()
    return auc
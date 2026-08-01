import numpy as np
from sklearn import tree

# 适应度函数  70% 训练  30%实验
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score


def fitnessFunction_Tree_percent(findData_x, findData_y):
    # 计算2
    # 创建了一个knn分类器的实例，并拟合数据
    x_train, x_test, y_train, y_test = train_test_split(findData_x, findData_y, test_size=0.3, random_state=0)
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    # 转化为二维数组
    if len(x_train.shape) == 1:
        x_train = x_train.reshape((len(x_train), 1))

    clf.fit(x_train, y_train)
    if len(x_test.shape) == 1:
        x_test = x_test.reshape((len(x_test), 1))
    predictOfTest = clf.predict(x_test)
    acc = np.mean(predictOfTest == y_test)

    return acc

def fitnessFunction_Tree_CV(findData_x, findData_y , CV):

    # 计算1
    # 先采用10折交叉验证的方式计算
    clf = tree.DecisionTreeClassifier()

    # 转化为二维数组
    if len(findData_x.shape) == 1:
        findData_x = findData_x.reshape((len(findData_x), 1))

    # cv参数决定数据集划分比例，这里是按照9:1划分训练集和测试集
    scores = cross_val_score(clf, findData_x, findData_y, cv=CV, scoring='accuracy')
    accuracy = scores.mean()
    return accuracy


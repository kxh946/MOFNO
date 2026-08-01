
from sklearn import svm

# 十折交叉
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC


def fitnessFunction_SVM_CV(findData_x, findData_y , CV):
    # C 越大越好  gamma越小越好
    svm1 = SVC(C = 2,kernel='rbf',gamma= 10,decision_function_shape='ovo')# ovr:一对多策略； ovo一对一：二分类
    # 转化为二维数组
    if len(findData_x.shape) == 1:
        findData_x = findData_x.reshape((len(findData_x), 1))

    # cv参数决定数据集划分比例，这里是按照9:1划分训练集和测试集
    scores = cross_val_score(svm1, findData_x, findData_y, cv=CV, scoring='accuracy')
    accuracy = scores.mean()
    return accuracy


# 70% 训练  30% 测试
def fitnessFunction_SVM_percent(findData_x, findData_y):
    # 计算2
    # 创建了一个svm分类器的实例，并拟合数据
    x_train, x_test, y_train, y_test = train_test_split(findData_x, findData_y, test_size=0.3, random_state=0)
    svm1 = svm.SVC(C=1,kernel='rbf',gamma=10,decision_function_shape='ovo')
    # 转化为二维数组
    if len(x_train.shape) == 1:
        x_train = x_train.reshape((len(x_train), 1))
    svm1.fit(x_train,y_train)

    predictOfTest = svm1.predict(x_test)
    numberOfTrue = 0
    for i in range(0, len(predictOfTest)):
        if predictOfTest[i] == y_test[i]:
            numberOfTrue += 1
    accuracy = numberOfTrue / len(predictOfTest)
    #accuracy = svm1.score(x_test,y_test)

    return accuracy
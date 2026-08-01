# 计算该个体的适应度值
from sklearn.neighbors import KNeighborsClassifier
from Classfier.DicisionTree import fitnessFunction_Tree_CV
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from sklearn.model_selection import train_test_split,StratifiedKFold,StratifiedShuffleSplit
#  使用于二进制



def computeFitnessKNN(chromosome_i,data_x,data_y):
    # 获得该个体的特征组合的 数据集合
    feature_x = data_x[:, chromosome_i.featureOfSelect == 1]
    if len(feature_x) == 0:
        return 0
    # 获得类数据
    feature_y = data_y
    # 进行10折交叉验证
    acc = fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=feature_y, CV=5)
    return acc


# 计算适应度值通过Tree
def computeFitnessTree(chromosome_i,data_x,data_y):
    # 获得该个体的特征组合的 数据集合
    feature_x = data_x[:, chromosome_i.featureOfSelect == 1]
    if len(feature_x) == 0:
        return 0
    # 获得类数据
    feature_y = data_y
    # 进行10折交叉验证
    acc = fitnessFunction_Tree_CV(findData_x=feature_x, CV=10, findData_y=feature_y)
    return acc


# 获得该个体所选择的特征组合
def getSolutionData(dataX, chromosome_i):
    tempFeatureSolutionData = dataX[:, chromosome_i.featureOfSelect == 1]
    return tempFeatureSolutionData


# 用于最后的fit计算
def terminalComputeFitness(trainX,trainY,testX,testY,solution):
    # 获得该个体的特征组合的 数据集合
    feature_x = trainX[:, solution == 1]
    if len(feature_x) == 0:
        return 0
    feature_y = trainY
    knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto", metric='manhattan')
    # 拟合模型
    knn.fit(X=feature_x,y=feature_y)

    test_x = testX[:,solution == 1]
    # 测试
    predictOfTest = knn.predict(test_x)
    numberOfTrue = 0
    for i in range(0, len(predictOfTest)):
        if predictOfTest[i] == testY[i]:
            numberOfTrue += 1
    accuracy = numberOfTrue / len(predictOfTest)
    return accuracy
        
import pandas as pd
import numpy as np
from scipy.io import arff
from dataProcessing.StringDataToNumerical import strToNum

class ReadCSV:
    # 非类标签数据
    dataX = np.asarray([])
    #类标签数据
    dataY = np.asarray([])
    # 特征数据
    dataFeature = np.asarray([])
    # 文件路径
    ducPath = ""

    def __init__(self, path):
        self.ducPath = path

    # 自己的加载数据
    def getData(self):
        data = pd.read_csv(self.ducPath, header=None)  # 获得数组
        self.dataX = data.values[:, 0:-1]
        self.dataY = data.values[:, -1]

        # self.deleteFirstColum(data=self.dataX)

    # 用于删除数据中的第一行列标签,并且写入到csv文件中
    def deleteFirstColum(self,data):
        data = data[1:]
        # data.to_csv("D:/MachineLearningBackUp/dataCSV/nci9.csv",sep=',')
        pd.DataFrame(data).to_csv("D:/MachineLearningBackUp/dataCSV/nci9.csv",header=None)
        data = pd.read_csv("D:/MachineLearningBackUp/dataCSV/nci9.csv", header=None)  # 获得数组
        print()


    # 合并王均可的 data 和 label
    def merger(self,path_data,path_label):
        data1 = pd.read_csv(path_data,header=None)
        data2 = pd.read_csv(path_label,header=None)
        data = pd.concat([data1,data2],axis=1)
        data.to_csv("D:\\MachineLearningBackUp\\dataCSV\\orlraws10P.csv",index=False,header=False)

    # 删除第一行
    def delFirstR(self):
        dataX = self.dataX[1:,]
        dataY = self.dataY[1:]
        data = pd.concat([pd.DataFrame(dataX), pd.DataFrame(dataY)], axis=1)
        data.to_csv("D:\\MachineLearningBackUp\\dataCSV\\Eleven_Tumor.csv",index=False,header=False)
    # 将arrff转csv
    def arffToCsv(self,filepath):
        """
        加载ARFF文件数据并进行处理
        -----------------------
        :param filepath: ARFF文件路径
        :return: 数据,类别和基因名
        """
        file_data, meta = arff.loadarff(filepath)
        x = []

        for row in range(len(file_data)):
            x.append(list(file_data[row]))

        df = pd.DataFrame(x)

        self.dataX = df.values[:,0:-1]
        self.dataY = df.values[:,-1]
        tempY = strToNum(dataY=df.values[:,-1])
        self.dataY = tempY
        dx = pd.DataFrame(self.dataX)
        dy = pd.DataFrame(self.dataY)
        data = pd.concat([dx, dy], axis=1)
        data.to_csv("D:\\MachineLearningBackUp\\dataCSV\\BreastCancer.csv", index=False, header=False)



if __name__ == "__main__":
    #dataCsv = DataCSV("D:\MachineLearningBackUp\\dataCSV\\Breast.csv")
    #dataCsv = ReadCSV("D:\MachineLearningBackUp\data\ionosphere.csv")
    #dataCsv = ReadCSV(path="D:\MachineLearningBackUp\\dataCSV\\Breast.csv")
    dataCsv = ReadCSV(path="F:\\MachineLearningBackUp\\dataCSV\\dataCSV_high\\Eleven_Tumor.csv")#D:\MachineLearningBackUp\\dataCSV\\Breast.csv
    # dataCsv = ReadCSV(path="F:\\MachineLearningBackUp\\dataCSV\\dataCSV_high\\BreastCancer1.csv")#D:\MachineLearningBackUp\\dataCSV\\Breast.csv
    # dataCsv = ReadCSV(path="F:\\MachineLearningBackUp\\dataCSV\\Eleven_Tumor.csv")#D:\MachineLearningBackUp\\dataCSV\\Breast.csv
    dataCsv.getData()
    dataCsv.delFirstR()
    # dataCsv = ReadCSV(path="D:\\MachineLearningBackUp\\dataCSV\\nci9.csv")
    # dataCsv.getData()
    # dataCsv.arffToCsv(filepath='D:\MachineLearningBackUp\数据备份\ARFF\ARFF\\BreastCancer.arff')
    # dataCsv.merger(path_label="D:\\git\\repositories\\dataset\\orlraws10P_100_10304_10\\orlraws10P_label.csv",
    #                path_data="D:\\git\\repositories\\dataset\\orlraws10P_100_10304_10\\orlraws10P_data.csv")
import torch
import random, csv
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing

class MyDataSet(Dataset):
    """
    MyDataSet类继承自torch中的DataSet实现对数据集的读取
    """
    def __init__(self, root, mode):
        """
        :param root: 数据集的路径
        :param mode: train,val,test
        """
        super(MyDataSet, self).__init__()

        self.mode = mode   # 设置读取读取数据集的模式
        self.root = root   # 数据集存放的路径

        label = []

        with open(root, 'r') as f:    # 从csv中读取数据
            reader = csv.reader(f)
            result = list(reader)

            del result[0]             # 删除表头
            random.shuffle(result)

            for i in range(len(result)):
                label.append(float(result[i][-1]))
                del result[i][-1]                   #删除目标值
                del result[i][0]                    #删除首列

        result = np.array(result, dtype=np.float64)

        result = preprocessing.StandardScaler().fit_transform(result).tolist()  # 对数据进行预处理

        # result = preprocessing.MinMaxScaler().fit_transform(result).tolist()

        assert len(result) == len(label)

        self.labels = label
        self.datas = result

        if mode == 'train':  # 划分训练集为60%
            self.datas = self.datas[:int(0.6 * len(self.datas))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':   # 划分验证集为20%
            self.datas = self.datas[int(0.6 * len(self.datas)):int(0.8 * len(self.datas))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:                  # 划分测试集为20%
            self.datas = self.datas[int(0.8 * len(self.datas)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data, label = self.datas[idx], self.labels[idx]
        data = torch.tensor(data)
        label = torch.tensor(label)
        return data, label


# def main():
#     file_path = "dataset/shichen_resource.csv"
#     db = MyDataSet(file_path, 'train')
#     x,y = next(iter(db))
#     print(x.shape)
#     print(y.shape)
#
#
# if __name__ == '__main__':
#     main()

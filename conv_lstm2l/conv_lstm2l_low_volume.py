import copy
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import random_split
import os
from sklearn.model_selection import train_test_split
import math
import random
import pickle
from random import shuffle

# test task ratio
ratio = 0.2

alpha = []


batchsize_intask = 512

csv_file = '../../normalization_processed_rpm_data.csv'
attack_sce = 'rpm'
feature_num = 11

sample_start = [2307273, 1096762, 2485425, 1308299, 1788684, 2364861, 1272307, 2233376, 511579]
sample_end = [2312153, 1099717, 2491025, 1312131, 1794013, 4320093, 1277051, 4381264, 516326]
extract_num = 10

def file_exists(filepath):
    return os.path.exists(filepath)



# 从原始数据集中取样一段流量
def extract_low_volume_dataset(csv_file):


    df = pd.read_csv(csv_file)
    df.loc[:, ['Label']] = df.loc[:, ['Label']].applymap(lambda x: 0 if x == 'R' else 1)


    for i, st, ed in zip(range(2, extract_num+1), sample_start, sample_end):
        xy = df.iloc[:,1:]

        x_data = torch.Tensor(xy.iloc[st:ed+1, :-1].values)
        y_data = torch.Tensor(xy.iloc[st:ed+1,[-1]].values.astype(float))

        torch.save(x_data, '../low_' + attack_sce + f'{i}_x.pt')
        torch.save(y_data, '../low_' + attack_sce + f'{i}_y.pt')


# 为meta-sgd定义sgd优化函数
def sgd_optimize(paralist, lrlist, gradlist):
    for para, lr, grad in zip(paralist, lrlist, gradlist):
        para.data -= lr * grad


def inisitagrad_add(a, b):
    return [x + y for x, y in zip(a, b)]


# 封装每个任务的数据
class Traffic_Task_Dataset(Dataset):

    def __init__(self, attack_sce, i):
        self.x_data = torch.load('../low_' + attack_sce + f'{i}_x.pt')
        self.y_data = torch.load('../low_' + attack_sce + f'{i}_y.pt')
        self.n_samples = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


class LSTM_RNN(nn.Module):
    def __init__(self):
        super(LSTM_RNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, 3, stride=1, padding=1)
        self.lstm = nn.LSTM((feature_num + 2 * 1 - 3) + 1, feature_num * 2, num_layers=2, batch_first=True)
        self.norm = nn.BatchNorm1d(feature_num * 2)
        self.linear1 = nn.Linear(feature_num * 2, feature_num)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(feature_num, math.ceil(feature_num / 2))
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(math.ceil(feature_num / 2), 2)

    def forward(self, x):
        x = x.view(-1, 1, feature_num)
        x = self.conv1(x)
        x = x.view(-1, 1, (feature_num + 2 * 1 - 3) + 1)
        lstm_out, (h_n, h_c) = self.lstm(x)
        b, s, h = lstm_out.shape
        x = lstm_out.view(-1, h)
        x = self.norm(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


if __name__ == '__main__':

    model = torch.load(f'./{attack_sce}.pth')


    # extract_low_volume_dataset(csv_file) # 从原始数据集中取样10段低容量

    for i in range(1, extract_num + 1):
        task_dataset = Traffic_Task_Dataset(attack_sce, i) # 封装任务的数据


        test_loader = torch.utils.data.DataLoader(dataset=task_dataset, batch_size=32768)

        criterion = nn.CrossEntropyLoss()

        # test
        TP = 0.0
        FP = 0.0
        TN = 0.0
        FN = 0.0
        test_loss = 0

        model.eval()
        for inputs, labels in test_loader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            outputs = model(inputs)
            labels = labels.to(torch.int64)
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            test_loss += loss

            res = torch.argmax(outputs, dim=1)

            for pre, truth in zip(res, labels):
                if pre.item() == 1:
                    if truth.item() == 1:
                        TP += 1
                    else:
                        FP += 1

                if pre.item() == 0:
                    if truth.item() == 0:
                        TN += 1
                    else:
                        FN += 1

        print(f"{attack_sce}{i}:")
        print("Test Loss per batch:{}".format(test_loss.item() / len(test_loader)))
        try:
            print('accuracy:{},recall:{},precision:{},f1:{}'.format((TP + TN) / (TP + FP + TN + FN),
                                                                    TP / (TP + FN), TP / (TP + FP),
                                                                    2 * TP / (2 * TP + FP + FN)))
            if FP + TN != 0:
                print("FPR:{},FNR:{}".format(FP / (FP + TN), FN / (TP + FN)))
            else:
                print("FNR:{}".format(FN / (TP + FN)))
        except:
            print('accuracy:{},nrecall:{}'.format((TP + TN) / (TP + FP + TN + FN), TN / (TN + FP)))
            print("FPR:{}".format(FP / (FP + TN)))

        print("TP:{},FP:{},TN:{},FN:{}".format(TP, FP, TN, FN))
        print('-------------------------------------------')
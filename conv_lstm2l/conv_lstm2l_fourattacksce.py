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
feature_num = 11

csv_file = '../../normalization_processed_rpm_data.csv'
attack_sce = 'Dos'
extract_num = 10

# sample_start = 1706
# sample_end = sample_start + 1023
# Dos
# sample_start = [2509721, 627441, 352022, 1764171, 1266109, 288310, 779325, 1956886, 758276]
# sample_end = [2510744, 628464, 353045, 1765194, 1267132, 289333, 780348, 1957909, 759299]
# sample_start = [2333977, 789002, 1921932, 998520, 374848, 1872985, 233230, 334069, 1252684]
# sample_end = [2335000, 790025, 1922955, 999543, 375871, 1874008, 234253, 335092, 1253707]

# gear
# sample_start = [2434043, 1348275, 175965, 1690926, 1700736, 322196, 2333331, 2200056, 428439]
# sample_end = [2435066, 1349298, 176988, 1691949, 1701759, 323219, 2334354, 2201079, 429462]

# rpm
sample_start = [59297, 1179869, 458895, 957188, 1657843, 1271871, 2794791, 1319637, 933717]
sample_end = [60320, 1180892, 459918, 958211, 1658866, 1272894, 2795814, 1320660, 934740]

def file_exists(filepath):
    return os.path.exists(filepath)



# 从原始数据集中取样一段流量
def extract_samples_of_dataset(csv_file):


    df = pd.read_csv(csv_file)
    df.loc[:, ['Label']] = df.loc[:, ['Label']].applymap(lambda x: 0 if x == 'R' else 1)

    for i, st, ed in zip(range(2, extract_num+1), sample_start, sample_end):
        xy = df.iloc[:,1:]

        x_data = torch.Tensor(xy.iloc[st:ed+1, :-1].values)
        y_data = torch.Tensor(xy.iloc[st:ed+1,[-1]].values.astype(float))

        torch.save(x_data, '../' + attack_sce + f'{i}_x.pt')
        torch.save(y_data, '../' + attack_sce + f'{i}_y.pt')



# 为meta-sgd定义sgd优化函数
def sgd_optimize(paralist, lrlist, gradlist):
    for para, lr, grad in zip(paralist, lrlist, gradlist):
        para.data -= lr * grad


def inisitagrad_add(a, b):
    return [x + y for x, y in zip(a, b)]


# 封装每个任务的数据
class Traffic_Task_Dataset(Dataset):

    def __init__(self, attack_sce, i):
        self.x_data = torch.load('../' + attack_sce + f'{i}_x.pt')
        self.y_data = torch.load('../' + attack_sce + f'{i}_y.pt')
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
    # 从原始数据集中取样指定数量段样本作数据集
    # extract_samples_of_dataset(csv_file)


    # 把每个任务的数据集封装成Dataset对象
    # for attack in csv_file:
    #     for i in range(task_num):
    #         tasks_dataset.append(Traffic_Task_Dataset(attack,i))
    #
    # shuffle(tasks_dataset)
    # with open('tasks_dataset1.pkl', 'wb') as f:
    #     pickle.dump(tasks_dataset, f)
    for i in range(1, extract_num+1):

        task_dataset = Traffic_Task_Dataset(attack_sce, i)

        meta_model = torch.load('./convlstm2l2e1861v2.pth')

        # initialize the alpha
        alpha = None
        with open('conv_lstml2_alpha2.pkl', 'rb') as f:
            alpha = pickle.load(f)

        model = copy.deepcopy(meta_model)

        train_dataset, test_dataset = train_test_split(task_dataset, test_size=ratio, shuffle=False)


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

        # train
        criterion = nn.CrossEntropyLoss()
        # task_optimizer = optim.SGD(model.parameters(), lr=alpha)

        model.train()
        train_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            for para in model.parameters(): para.grad = None

            outputs = model(inputs)
            labels = labels.to(torch.int64)
            labels = labels.view(-1)

            loss = criterion(outputs, labels)

            loss.backward()

            sgd_optimize(model.parameters(), alpha, [para.grad for para in model.parameters()])

            train_loss += loss.item()

        # print('Train Loss per batch:{}'.format(train_loss / len(train_loader)))
        torch.save(model, f'{attack_sce}.pth')

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
            print("FPR:{},FNR:{}".format(FP / (FP + TN), FN / (TP + FN)))

        print("TP:{},FP:{},TN:{},FN:{}".format(TP, FP, TN, FN))
        print('-------------------------------------------')



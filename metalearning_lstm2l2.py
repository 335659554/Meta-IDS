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

feature_num = 11
class_num = 2
task_num = 4096
curr_task = 0

# 测试任务的数量
num_test_task = int(4 * task_num * ratio)

alpha = []
# 元学习率
beta = 0.0001
gama = 0.0001

batchsize_intask = 512
meta_batchsize = 1024

csv_file = ['normalization_processed_Dos_data.csv', 'normalization_processed_Fuzzy_data.csv', 'normalization_processed_geardata.csv', 'normalization_processed_rpm_data.csv']

# 存储分割产生的任务集用
tasks_dataset = []

# define the number of the epoch
epoch_num = 1735

def file_exists(filepath):
    return os.path.exists(filepath)


# 根据任务数量分割每个攻击数据集
def split_tasks_datasets(task_num):
    global tasks_dataset
    for attack_dataset_file in csv_file:
        tasks_dataset = []
        for i in range(task_num):
            if file_exists('./splitted_dataset/' + attack_dataset_file + '_x_data' + str(i) + '.pt') and file_exists(
                    './splitted_dataset/' + attack_dataset_file + '_y_data' + str(i) + '.pt'):
                pass
            else:
                # category = 1
                df = pd.read_csv(attack_dataset_file)
                df.loc[:, ['Label']] = df.loc[:, ['Label']].applymap(lambda x: 0 if x == 'R' else 1)
                # category += 1
                groups = df['Label'].unique()
                # 遍历唯一值
                for group in groups:
                    # 过滤出该组数据
                    df_group = df[df['Label'] == group]
                    # 获取组内总行数
                    rows = len(df_group)
                    # 计算每份的行数
                    rows_per_split = rows // task_num

                    # 分割数据集进每个任务
                    if len(tasks_dataset) == 0:
                        for i in range(task_num):
                            if i == 0:
                                tasks_dataset.append(df_group.iloc[:rows_per_split])
                            elif i == task_num - 1:
                                tasks_dataset.append(df_group[rows_per_split * i:])
                            else:
                                tasks_dataset.append(df_group.iloc[rows_per_split * i:rows_per_split * (i + 1)])
                    else:
                        for i in range(task_num):
                            if i == 0:
                                tasks_dataset[i] = pd.concat([tasks_dataset[i], df_group.iloc[:rows_per_split]])
                            elif i == task_num - 1:
                                tasks_dataset[i] = pd.concat([tasks_dataset[i], df_group.iloc[rows_per_split * i:]])
                            else:
                                tasks_dataset[i] = pd.concat(
                                    [tasks_dataset[i], df_group.iloc[rows_per_split * i:rows_per_split * (i + 1)]])

                # 将分割出的每个任务数据集保存
                for i in range(task_num):
                    xy = tasks_dataset[i].iloc[:, 1:]
                    x_data = torch.Tensor(xy.iloc[:xy.shape[0], :-1].values)
                    y_data = torch.Tensor(xy.iloc[:xy.shape[0], [-1]].values.astype(float))

                    torch.save(x_data, './splitted_dataset/' + attack_dataset_file + '_x_data' + str(i) + '.pt')
                    torch.save(y_data, './splitted_dataset/' + attack_dataset_file + '_y_data' + str(i) + '.pt')

                break


# 保留时序的划分
# def split_tasks_datasets(task_num):
#     global tasks_dataset
#     for attack_dataset_file in csv_file:
#         tasks_dataset = []
#         for i in range(task_num):
#             if file_exists('./splitted_dataset/'+attack_dataset_file+'_x_data'+str(i)+'.pt') and file_exists('./splitted_dataset/'+attack_dataset_file+'_y_data'+str(i)+'.pt'):
#                 pass
#             else:
#                 # category = 1
#                 df = pd.read_csv(attack_dataset_file)
#                 df.loc[:, ['Label']] = df.loc[:, ['Label']].applymap(lambda x: 0 if x == 'R' else 1)
#                 # category += 1
#
#                 rows = len(df)
#                 # 计算每份的行数
#                 rows_per_split = rows // task_num
#
#                 # 分割数据集进每个任务
#                 for i in range(task_num):
#                     if i ==0:
#                         tasks_dataset.append(df.iloc[:rows_per_split])
#                     elif i == task_num-1:
#                         tasks_dataset.append(df[rows_per_split*i:])
#                     else:
#                         tasks_dataset.append(df[rows_per_split*i:rows_per_split*(i+1)])
#
#
#                 # 将分割出的每个任务数据集保存
#                 for i in range(task_num):
#                     xy = tasks_dataset[i].iloc[:,1:]
#                     x_data = torch.Tensor(xy.iloc[:xy.shape[0], :-1].values)
#                     y_data = torch.Tensor(xy.iloc[:xy.shape[0],[-1]].values.astype(float))
#
#                     torch.save(x_data, './splitted_dataset/'+attack_dataset_file+'_x_data'+str(i)+'.pt')
#                     torch.save(y_data, './splitted_dataset/'+attack_dataset_file+'_y_data'+str(i)+'.pt')
#
#                 break

# 为meta-sgd定义sgd优化函数
def sgd_optimize(paralist, lrlist, gradlist):
    for para, lr, grad in zip(paralist, lrlist, gradlist):
        para.data -= lr * grad


def inisitagrad_add(a, b):
    return [x + y for x, y in zip(a, b)]


# 封装每个任务的数据
class Traffic_Task_Dataset(Dataset):

    def __init__(self, attack, no_task):
        if file_exists('./splitted_dataset/' + '{}_x_data{}.pt'.format(attack, no_task)) and file_exists(
                './splitted_dataset/' + '{}_y_data{}.pt'.format(attack, no_task)):
            self.x_data = torch.load('./splitted_dataset/' + '{}_x_data{}.pt'.format(attack, no_task))
            self.y_data = torch.load('./splitted_dataset/' + '{}_y_data{}.pt'.format(attack, no_task))
            self.n_samples = len(self.x_data)
        else:
            print("Did not find the dataset files of tasks")

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


class LSTM_RNN(nn.Module):
    def __init__(self):
        super(LSTM_RNN, self).__init__()
        self.lstm = nn.LSTM(feature_num, feature_num * 2, num_layers=2, batch_first=True)
        self.norm = nn.BatchNorm1d(feature_num * 2)
        self.linear1 = nn.Linear(feature_num * 2, feature_num)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(feature_num, math.ceil(feature_num / 2))
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(math.ceil(feature_num / 2), 2)

    def forward(self, x):
        x = x.view(-1, 1, feature_num)
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
    # 分割数据集，形成每个任务的数据集文件
    # split_tasks_datasets(task_num)
    # tasks_dataset = []

    tasks_dataset = None

    # 把每个任务的数据集封装成Dataset对象
    # for attack in csv_file:
    #     for i in range(task_num):
    #         tasks_dataset.append(Traffic_Task_Dataset(attack,i))
    #
    # shuffle(tasks_dataset)
    # with open('tasks_dataset1.pkl', 'wb') as f:
    #     pickle.dump(tasks_dataset, f)

    with open('tasks_dataset1.pkl', 'rb') as f:
        tasks_dataset = pickle.load(f)

    for i in range(len(tasks_dataset)):
        print("the length of task{}'s dataset is {}".format(i, tasks_dataset[i].n_samples))

    # train_dataset,val_dataset = random_split(train_data,[math.floor(len(train_data)*0.7),len(train_data)-math.floor(len(train_data)*0.7)])
    with torch.backends.cudnn.flags(enabled=False):
        meta_model = torch.load('lstm2le20482.pth')
        # meta_model = LSTM_RNN().to(device='cuda')
        # torch.save(meta_model, 'lstm2l.pth')
        meta_optimizer = optim.Adam(meta_model.parameters(), beta, weight_decay=0.01)

        # initialize the alpha
        alpha = None
        with open('lstm2l_alpha2.pkl', 'rb') as f:
            alpha = pickle.load(f)

        # alpha = [torch.rand(para.size()).to('cuda') for para in meta_model.parameters()]
        # with open('lstm2l_alpha1.pkl', 'wb') as f:
        #     pickle.dump(alpha, f)
        alpha_optimizer = optim.Adam(alpha, gama, weight_decay=0.01)

        # train every task
        for epoch in range(epoch_num):
            print("epoch:{}".format(epoch))
            ini_sita_grad = None
            alpha_optimizer.zero_grad()

            # cnt:count the task number
            cnt = 0
            step = 0

            for task_dataset, i in zip(tasks_dataset[0:(len(tasks_dataset) - num_test_task)],
                                       range(len(tasks_dataset) - num_test_task)):
                # the number of positive examples
                trainpn = 0
                testpn = 0
                # the length of the longest consecutive positive examples
                trainlongestpn = 0
                testlongestpn = 0

                model = copy.deepcopy(meta_model)
                try:
                    train_dataset, test_dataset = train_test_split(task_dataset, test_size=ratio, random_state=0,
                                                                   stratify=task_dataset.y_data)
                except:
                    continue
                # the length of the task's train and test dataset
                trainlen = len(train_dataset)
                testlen = len(test_dataset)

                # make the number of classes two:normal,abnormal
                tmplen = 0
                for example in train_dataset:
                    lst = list(example)
                    lst[1][0] = 1.0 if example[1][0] > 0 else 0.0
                    if lst[1][0] > 0:
                        trainpn += 1
                        tmplen += 1
                        if tmplen > trainlongestpn:
                            trainlongestpn = tmplen
                    else:
                        tmplen = 0

                    example = tuple(lst)

                tmplen = 0
                for example in test_dataset:
                    lst = list(example)
                    lst[1][0] = 1.0 if example[1][0] > 0 else 0.0
                    if lst[1][0] > 0:
                        testpn += 1
                        tmplen += 1
                        if tmplen > testlongestpn:
                            testlongestpn = tmplen
                    else:
                        tmplen = 0

                    example = tuple(lst)

                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchsize_intask)
                test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

                # train
                criterion = nn.CrossEntropyLoss()
                # task_optimizer = optim.SGD(model.parameters(), lr=alpha)

                model.train()
                train_loss = 0
                sita_seccond_order_gradlist = []
                sita_first_order_gradlist = []
                post_sita_order = []

                for inputs, labels in train_loader:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                    for para in model.parameters(): para.grad = None

                    outputs = model(inputs)
                    labels = labels.to(torch.int64)
                    labels = labels.view(-1)

                    loss = criterion(outputs, labels)

                    # get the seccond-order gradient and first-order gradient
                    dydx = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
                    d2ydx2 = [torch.autograd.grad(first_grad, para, retain_graph=True,
                                                  grad_outputs=torch.ones_like(first_grad))[0] for first_grad, para in
                              zip(dydx, model.parameters())]
                    sita_seccond_order_gradlist.append(d2ydx2)
                    sita_first_order_gradlist.append(dydx)

                    loss.backward()

                    sgd_optimize(model.parameters(), alpha, [para.grad for para in model.parameters()])

                    train_loss += loss.item()

                print('Train_task:{}, Loss per batch:{}'.format(i, train_loss / len(train_loader)))

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

                    # get the post_sita_order
                    post_sita_order = torch.autograd.grad(loss, model.parameters(), retain_graph=True,
                                                          create_graph=True)
                print("Test_task:{}, Loss per batch:{}".format(i, test_loss.item() / len(test_loader)))
                try:
                    print('accuracy:{},recall:{}'.format((TP + TN) / (TP + FP + TN + FN), TP / (TP + FN)))
                except:
                    print('accuracy:{},nrecall:{}'.format((TP + TN) / (TP + FP + TN + FN), TN / (TN + FP)))
                print("The number of positive examples in train dataset:{},\n"
                      "the number of positive examples in test dataset:{}".format(trainpn, testpn))
                print("The length of the longest consecutive positive examples in train dataset:{},\n"
                      "the length of the longest consecutive positive examples in test dataset:{}".format(
                    trainlongestpn, testlongestpn))
                print("the length of train dataset:{},\n"
                      "the length of test dataset:{}".format(trainlen, testlen))
                print('--------------------------------------------')

                # compute the gradient of initial sita
                sum_seccond_gradient = [torch.zeros_like(x) for x in sita_seccond_order_gradlist[0]]
                for seccond_order_grad in sita_seccond_order_gradlist:
                    for gradsum, grad in zip(sum_seccond_gradient, seccond_order_grad):
                        gradsum.data += grad.data

                factor_rightlist = [1 - al * x for al, x in zip(alpha, sum_seccond_gradient)]

                if ini_sita_grad == None:
                    ini_sita_grad = [fac1 * fac2 for fac1, fac2 in zip(post_sita_order, factor_rightlist)]
                else:
                    ini_sita_grad = inisitagrad_add(ini_sita_grad, [fac1 * fac2 for fac1, fac2 in
                                                                    zip(post_sita_order, factor_rightlist)])

                # compute the gradient of initial alpha
                sum_sita_first_gradient = [torch.zeros_like(x) for x in sita_first_order_gradlist[0]]
                for first_order_grad in sita_first_order_gradlist:
                    for gradsum, grad in zip(sum_sita_first_gradient, first_order_grad):
                        gradsum.data += grad.data

                for al, po, firstg in zip(alpha, post_sita_order, sum_sita_first_gradient):
                    if al.grad == None:
                        al.grad = po * (-firstg)
                    else:
                        al.grad += po * (-firstg)

                cnt += 1
                if cnt % meta_batchsize == 0 or i == len(tasks_dataset) - num_test_task - 1:
                    # update the initial sita
                    for ini_sita, ini_sita_grad in zip(meta_model.parameters(), ini_sita_grad):
                        ini_sita.grad = ini_sita_grad
                    meta_optimizer.step()
                    # update the initial alpha
                    alpha_optimizer.step()
                    # recover the meta count and grad
                    cnt = 0
                    meta_optimizer.zero_grad()
                    ini_sita_grad = None
                    alpha_optimizer.zero_grad()
                    step += 1
                    print('-----------------step{}----------------------'.format(step))
            print('--------------------------------------------')
            print('test:')
            total_test_loss = 0.0
            for task_dataset, i in zip(tasks_dataset[len(tasks_dataset) - num_test_task:len(tasks_dataset)],
                                       range(len(tasks_dataset) - num_test_task, len(tasks_dataset))):
                # the number of positive examples
                trainpn = 0
                testpn = 0
                # the length of the longest consecutive positive examples
                trainlongestpn = 0
                testlongestpn = 0

                model = copy.deepcopy(meta_model)

                try:
                    train_dataset, test_dataset = train_test_split(task_dataset, test_size=ratio, random_state=0,
                                                                   stratify=task_dataset.y_data)
                except:
                    continue
                # the length of the task's train and test dataset
                trainlen = len(train_dataset)
                testlen = len(test_dataset)

                # make the number of classes two:normal,abnormal
                for example in train_dataset:
                    lst = list(example)
                    lst[1][0] = 1.0 if example[1][0] > 0 else 0.0
                    if lst[1][0] > 0:
                        trainpn += 1
                        tmplen += 1
                        if tmplen > trainlongestpn:
                            trainlongestpn = tmplen
                    else:
                        tmplen = 0
                    example = tuple(lst)

                tmplen = 0
                for example in test_dataset:
                    lst = list(example)
                    lst[1][0] = 1.0 if example[1][0] > 0 else 0.0
                    if lst[1][0] > 0:
                        testpn += 1
                        tmplen += 1
                        if tmplen > testlongestpn:
                            testlongestpn = tmplen
                    else:
                        tmplen = 0
                    example = tuple(lst)

                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchsize_intask)
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

                print('Train_task:{}, Loss per batch:{}'.format(i, train_loss / len(train_loader)))

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

                total_test_loss += test_loss.item() / len(test_loader)
                print("Test_task:{}, Loss per batch:{}".format(i, test_loss.item() / len(test_loader)))
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
                print("The number of positive examples in train dataset:{},\n"
                      "the number of positive examples in test dataset:{}".format(trainpn, testpn))
                print("The length of the longest consecutive positive examples in train dataset:{},\n"
                      "the length of the longest consecutive positive examples in test dataset:{}".format(
                    trainlongestpn, testlongestpn))
                print("the length of train dataset:{},\n"
                      "the length of test dataset:{}".format(trainlen, testlen))
                print('--------------------------------------------')

            print("epoch:{},the loss per test task:{}".format(epoch,total_test_loss / num_test_task))
            print('--------------------------------------------')
            print('--------------------------------------------')
            print('--------------------------------------------')
            torch.save(meta_model, 'lstm2le' + str(epoch_num) + '2.pth')
            with open('lstm2l_alpha2.pkl', 'wb') as f:
                pickle.dump(alpha, f)



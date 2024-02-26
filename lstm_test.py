import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import random_split
import os
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter

import math

ratio = 0.8
feature_num = 11
class_num = 2


def file_exists(filename):
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)

    if filename in files_in_dir:
        return True
    else:
        return False

# 创建数据集类
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


# class Traffic_Test_Dataset(Dataset):
#
#     def __init__(self):
#         df_normal = pd.read_csv('processed_normal_run_data.csv')
#         df_dos = pd.read_csv('processed_Dos_data.csv')
#         xy = pd.concat([df_normal, df_dos])
#
#         xy = xy.iloc[:, 1:]
#
#         xy.loc[:, ['Label']] = xy.loc[:, ['Label']].applymap(lambda x: 0 if x == 'R' else 1)
#
#         self.train_samples = math.floor(xy.shape[0] * 0.8)
#
#
#         self.x_data = torch.Tensor(xy.iloc[self.train_samples:,:-1].values)
#
#         self.y_data = torch.Tensor(xy.iloc[self.train_samples:,[-1]].values.astype(int))
#
#         self.n_samples = xy.shape[0] - math.floor(xy.shape[0] * 0.8)
#
#
#     def __getitem__(self, index):
#         return self.x_data[index],self.y_data[index]
#
#
#     def __len__(self):
#         return self.n_samples




class LSTM_RNN(nn.Module):
    def __init__(self):
        super(LSTM_RNN,self).__init__()
        self.lstm = nn.LSTM(feature_num,feature_num*2,num_layers=1,batch_first=True)
        self.norm = nn.BatchNorm1d(feature_num*2)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(feature_num*2, 2)


    def forward(self,x):
        x = x.view(-1,1,feature_num)
        lstm_out, (h_n, h_c) = self.lstm(x)
        b,s,h = lstm_out.shape
        x = lstm_out.view(-1,h)
        x = self.norm(x)
        x = self.relu1(x)
        x = self.linear1(x)
        return x


def calculate_and_print_ratio(dataset):
    # 将tensor标签转换为整数
    labels = [label.item() if hasattr(label, 'item') else label for _, label in dataset]
    label_counts = Counter(labels)

    # 确保我们有两个标签
    if len(label_counts) != 2:
        raise ValueError("Dataset does not contain exactly two labels.")

    # 获取两个标签
    label1, label2 = label_counts.keys()
    ratio = f"{label_counts[label1]}:{label_counts[label2]}"
    print(f"Label {label1} to Label {label2} Ratio: {ratio}")


def reduce_label_samples(dataset, label_to_reduce, step=1):
    # 减少特定标签的样本
    reduced_dataset = []
    removal_count = 0

    for data, label in dataset:
        if removal_count < step and label.item() == label_to_reduce:
            removal_count += 1
            continue
        reduced_dataset.append((data, label))

    return reduced_dataset

if __name__ == '__main__':

    tasks_dataset = None
    with open('tasks_dataset1.pkl', 'rb') as f:
        tasks_dataset = pickle.load(f)

    train_data = tasks_dataset[0]
    print(train_data.n_samples)
    # train_dataset,val_dataset = random_split(train_data,[math.floor(len(train_data)*0.7),len(train_data)-math.floor(len(train_data)*0.7)])

    train_dataset, test_dataset = train_test_split(train_data,test_size=0.3,random_state=0,stratify=train_data.y_data)
    val_dataset, test_dataset = train_test_split(test_dataset,test_size=1.0/3,random_state=0,stratify=torch.Tensor([[l[1]] for l in test_dataset]))

    for _ in range(100):
        # calculate_and_print_ratio(train_dataset)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=256)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=256)

        print(len(train_dataset))
        print(len(val_dataset))
        print(len(test_dataset))
        # train
        # model = LSTM_RNN().to(device='cuda')
        # torch.save(model, 'lstm_test.pth')
        model = torch.load('lstm_test.pth')
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
        optimizer = optim.Adam(model.parameters(), 0.001, weight_decay=0.01)

        num_epochs = 50

        for epoch in range(num_epochs):

            epoch_loss = 0.0
            # train
            model.train()
            for inputs,labels in train_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                optimizer.zero_grad()

                outputs = model(inputs)
                labels = labels.to(torch.int64)
                labels = labels.view(-1)

                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print('Epoch:{}, training Loss:{}'.format(epoch,epoch_loss/len(train_loader)))
            print('------------------------------------------------------')

            # validate
            TP = 0.0
            FP = 0.0
            TN = 0.0
            FN = 0.0
            model.eval()
            for inputs,labels in val_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                outputs = model(inputs)
                res = torch.argmax(outputs,dim=1)

                for pre,truth in zip(res,labels):
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

            print('validating accuracy:{},recall:{}'.format((TP+TN)/(TP+FP+TN+FN),TP/(TP+FN)))
            print('------------------------------------------------------')
        # test
        for inputs, labels in test_loader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            outputs = model(inputs)
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

        calculate_and_print_ratio(train_dataset)
        print('testing accuracy:{},recall:{}'.format((TP + TN) / (TP + FP + TN + FN), TP / (TP + FN)))
        print('------------------------------------------------------')
        print('------------------------------------------------------')
        train_dataset = reduce_label_samples(train_dataset, 1, step=5)

    for _ in range(50):

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=256)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=256)

        print(len(train_dataset))
        print(len(val_dataset))
        print(len(test_dataset))
        # train
        # model = LSTM_RNN().to(device='cuda')
        # torch.save(model, 'lstm_test.pth')
        model = torch.load('lstm_test.pth')
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
        optimizer = optim.Adam(model.parameters(), 0.001, weight_decay=0.01)

        num_epochs = 50

        for epoch in range(num_epochs):

            epoch_loss = 0.0
            # train
            model.train()
            for inputs,labels in train_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                optimizer.zero_grad()

                outputs = model(inputs)
                labels = labels.to(torch.int64)
                labels = labels.view(-1)

                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print('Epoch:{}, training Loss:{}'.format(epoch,epoch_loss/len(train_loader)))
            print('------------------------------------------------------')

            # validate
            TP = 0.0
            FP = 0.0
            TN = 0.0
            FN = 0.0
            model.eval()
            for inputs,labels in val_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                outputs = model(inputs)
                res = torch.argmax(outputs,dim=1)

                for pre,truth in zip(res,labels):
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

            print('validating accuracy:{},recall:{}'.format((TP+TN)/(TP+FP+TN+FN),TP/(TP+FN)))
            print('------------------------------------------------------')
        # test
        for inputs, labels in test_loader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            outputs = model(inputs)
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

        calculate_and_print_ratio(train_dataset)
        print('testing accuracy:{},recall:{}'.format((TP + TN) / (TP + FP + TN + FN), TP / (TP + FN)))
        print('------------------------------------------------------')
        print('------------------------------------------------------')
        train_dataset = reduce_label_samples(train_dataset, 1, step=1)
    # print("???????{}????????{}???".format(total_samples, n_iterations))
    # for epoch in range(num_epochs):
    #     for i, (inputs, labels) in enumerate(train_loader):
    #
    #         if (i + 1) % 5 == 0:
    #             print(f'Epoch: {epoch + 1}/{num_epochs}, Step {i + 1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')
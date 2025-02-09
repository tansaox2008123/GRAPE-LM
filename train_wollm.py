# -*- coding: utf-8 -*-
import os
import re
import ast
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import numpy as np
import time
import math
import torch.optim as optim
import pandas as pd
from model_wollm import *
import sys

sys.path.append(os.path.abspath(''))

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def stable_sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig


class NewDataset(Dataset):
    def __init__(self, file_path):
        super(NewDataset, self).__init__()
        fh = open(file_path, 'r')
        rnas = []
        input_seqs = []
        true_seqs = []
        bd_scores = []
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:
            words = line.split()
            b = words[1:]
            a = ''.join(b)
            match = re.search(r'\[.*\]', a)
            if match:
                # 获取匹配到的部分
                substring = match.group(0)
                # 将字符串表示的列表转换为实际的列表
                values_list = ast.literal_eval(substring)
                # 将列表转换为张量
                values_list.insert(0, 0)

                input_seq = values_list[:-1]
                true_seq = values_list[1:]

                label = stable_sigmoid(math.log(float(words[-1])) - 7.5)

                rna_fm = np.load(words[0])
                rna_fm = np.mean(rna_fm, axis=0)

                rna_fm = torch.tensor(rna_fm)
                input_seq = torch.tensor(input_seq)
                true_seq = torch.tensor(true_seq)
                label = torch.tensor(label)

                rnas.append((rna_fm, input_seq, true_seq, label))
            else:
                pass
        self.rnas = rnas

    def __len__(self):
        return len(self.rnas)

    def __getitem__(self, index):
        rna_fm, input_seq, true_seq, label = self.rnas[index]

        return rna_fm, input_seq, true_seq, label


def sigmoid(x, k=0.05):
    return 1 / (1 + np.exp(-k * x))


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def read_data_evo(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    rnas = []
    input_seqs = []
    true_seqs = []
    bd_scores = []
    # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
    for line in lines:
        words = line.split()
        b = words[1:]
        a = ''.join(b)
        match = re.search(r'\[.*\]', a)
        if match:
            # 获取匹配到的部分
            substring = match.group(0)
            # 将字符串表示的列表转换为实际的列表
            values_list = ast.literal_eval(substring)

            values_list.insert(0, 0)
            # print(values_list)

            input_seq = values_list[:-1]
            true_seq = values_list[1:]

            # input_seq = torch.tensor(input_seq)
            # true_seq = torch.tensor(true_seq)

            decimal_part = float(words[-1])
            decimal_part = (sigmoid(decimal_part, 0.05) - 0.5) * 2.0
            # decimal_part = int((int(words[-1]) - 100) >= 0)

            # 不使用rna_fm直接使用one-hot编码
            rna_fm = values_list[1:]

            # rna-fm的表征
            # rna_fm = np.load(words[0])
            # 下面这个是展开FM表征
            # rna_fm = rna_fm.reshape(-1)
            # rna_fm = standardization(rna_fm)

            # 下面这个是平均FM表征
            # rna_fm = rna_fm.squeeze(0)
            # rna_fm = np.mean(rna_fm, axis=0)
            # rna_fm = standardization(rna_fm)

            # rna_fm = torch.load(words[0])
            # rna_fm = torch.mean(rna_fm, axis=0)
            # rna_fm = torch.mean(rna_fm, dim=0)

            # rna_fm = torch.load(words[0])
            # print(rna_fm.shape)
            rnas.append(rna_fm)
            input_seqs.append(input_seq)
            true_seqs.append(true_seq)
            bd_scores.append(decimal_part)
        else:
            pass
    return rnas, input_seqs, true_seqs, bd_scores


# 处理等长序列
def read_data(file_path, is_batch=False):
    # rnas, input_seqs, true_seqs, bd_scores = multiprocess_read_line_2.get_data(file_path)
    rnas, input_seqs, true_seqs, bd_scores = read_data_evo(file_path)

    if is_batch:
        rnas1 = torch.tensor(rnas, dtype=torch.float32)
        # rnas1 = torch.tensor(rnas)
        print(rnas1.shape)
        input_seqs1 = torch.tensor(np.asarray(input_seqs))
        true_seqs1 = torch.tensor(np.asarray(true_seqs))
        bd_scores1 = torch.tensor(np.asarray(bd_scores))
    else:
        rnas1 = torch.tensor(rnas).to(device)  # .to(torch.long)
        input_seqs1 = torch.tensor(np.asarray(input_seqs)).to(device)
        true_seqs1 = torch.tensor(np.asarray(true_seqs)).to(device)
        bd_scores1 = torch.tensor(np.asarray(bd_scores)).to(device)

    return rnas1, input_seqs1, true_seqs1, bd_scores1
    # rnas.append((rna_fm, input_seq, true_seq, decimal_part))


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, pred_scores, true_labels):
        mask = true_labels != -1
        mask = mask.float()
        _loss = F.binary_cross_entropy(pred_scores, true_labels, reduction='none')
        loss = torch.mean(_loss * mask)
        return loss


if __name__ == '__main__':
    # 设置随机数种子
    # setup_seed(123)
    train_file = '/home2/public/data/RNA_aptamer/All_data/2_ex_apt/train_bdscore.txt'
    test_file = '/home2/public/data/RNA_aptamer/All_data/2_ex_apt/test_bdscore.txt'
    batch_size = 5000

    tr_feats, tr_input_seqs, tr_true_seqs, tr_bd_scores = read_data(train_file, is_batch=True)
    te_feats, te_input_seqs, te_true_seqs, te_bd_scores = read_data(test_file, is_batch=True)

    train_data = torch_data.TensorDataset(tr_feats, tr_input_seqs, tr_true_seqs, tr_bd_scores)
    test_data = torch_data.TensorDataset(te_feats, te_input_seqs, te_true_seqs, te_bd_scores)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False)

    model = FullModel(input_dim=20,  # 输入特征的维度
                      model_dim=128,  # LLM适配器（Encoder）隐含层的大小, Transformer模型维度
                      tgt_size=5,  # 碱基种类数
                      n_declayers=2,  # Transformer解码器层数
                      d_ff=128,  # Transformer前馈网络隐含层维度
                      d_k_v=64,
                      n_heads=2,  # Transformer注意力头数
                      dropout=0.05)
    #
    # model = SimpleModel(input_dim=640,
    #                     model_dim=128,
    #                     dropout=0.0)

    model = model.to(device)

    loss_func1 = nn.MSELoss()  # nn.BCEWithLogitsLoss()#
    loss_func2 = nn.CrossEntropyLoss(ignore_index=0)  #
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#0.0001
    w = 0.85

    model_name = 'CD3E-loss8515-wollm'
    fw = open('log/' + model_name + '_training_log.txt', 'w')
    """分批次训练"""
    for epoch in range(250):
        start_t = time.time()
        loss1_value = 0.0
        loss2_value = 0.0
        acc2 = 0.0
        b_num = 0.0
        # optimizer = torch.optim.Adam(model.parameters(), lr=(250.0 - epoch) * 0.00001)
        # optimizer = torch.optim.Adam(model.parameters(), lr=max((250.0 - epoch) * 0.00001, 0.0001))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0006)
        model.train()
        for i, data in enumerate(train_loader):
            inputs, input_seqs, true_seqs, labels = data
            inputs = inputs.to(device)
            input_seqs = input_seqs.to(device)
            true_seqs = true_seqs.to(device)
            labels = labels.to(device)

            labels = labels.float().view(-1, 1)
            bind_socres, pred_seqs = model(inputs, input_seqs)  # model(inputs)#

            pred_seqs = torch.softmax(pred_seqs, -1)
            true_seqs = true_seqs.view(-1)
            pred_seqs = pred_seqs.view(true_seqs.shape[0], 5)

            loss1 = loss_func1(bind_socres, labels)
            loss2 = loss_func2(pred_seqs, true_seqs)
            pred_seqs = torch.argmax(pred_seqs, -1)
            # acc1 = torchmetrics.functional.accuracy(bind_socres, labels, task="binary", num_classes=2)
            acc2 += torchmetrics.functional.accuracy(pred_seqs,
                                                     true_seqs,
                                                     task="multiclass",
                                                     num_classes=5,
                                                     ignore_index=0,
                                                     average="micro")

            loss = w * loss1 + (1.0 - w) * loss2
            # loss = loss2
            loss1_value += loss1.item()
            loss2_value += loss2.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b_num += 1.

        test_loss1_value = 0.0
        test_loss2_value = 0.0

        te_acc2 = 0.0
        test_b_num = 0.

        model.eval()
        for i, data in enumerate(test_loader):
            inputs, input_seqs, true_seqs, labels = data
            inputs = inputs.to(device)
            input_seqs = input_seqs.to(device)
            true_seqs = true_seqs.to(device)
            labels = labels.to(device)

            bind_socres, pred_seqs = model(inputs, input_seqs)  # model(inputs)#
            pred_seqs = torch.softmax(pred_seqs, -1)
            labels = labels.float().view(-1, 1)

            true_seqs = true_seqs.view(-1)
            pred_seqs = pred_seqs.view(true_seqs.shape[0], 5)

            loss1 = loss_func1(bind_socres, labels)
            loss2 = loss_func2(pred_seqs, true_seqs)

            # te_acc1 = torchmetrics.functional.accuracy(bind_socres, labels, task="binary", num_classes=2)
            pred_seqs = torch.argmax(pred_seqs, -1)
            te_acc2 += torchmetrics.functional.accuracy(pred_seqs,
                                                        true_seqs,
                                                        task="multiclass",
                                                        num_classes=5,
                                                        ignore_index=0,
                                                        average="micro")

            test_loss1_value += loss1.item()
            test_loss2_value += loss2.item()
            test_b_num += 1.
        end_t = time.time()
        fw.write('{:4d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(epoch,
                                                                  loss1_value / b_num,
                                                                  loss2_value / b_num,
                                                                  test_loss1_value / test_b_num,
                                                                  test_loss2_value / test_b_num,
                                                                  acc2 / b_num,
                                                                  te_acc2 / test_b_num), )
        print('Epoch:', '%04d' % (epoch + 1),
              '| tr_loss1 =', '{:.4f}'.format(loss1_value / b_num),
              '| tr_loss2 =', '{:.4f}'.format(loss2_value / b_num),
              '| tr_acc =', '{:.2f}'.format(acc2 / b_num),
              '| te_loss1 =', '{:.4f}'.format(test_loss1_value / test_b_num),
              '| te_loss2 =', '{:.4f}'.format(test_loss2_value / test_b_num),
              '| te_acc =', '{:.2f}'.format(te_acc2 / test_b_num),
              '| time =', '{:.2f}'.format(end_t - start_t)
              )
    """全部数据训练"""
    torch.save(model, 'model/' + model_name + '.model')
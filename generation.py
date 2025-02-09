# -*- coding: utf-8 -*-
import os
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
import torch.optim as optim
import pandas as pd
from model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_sample(low, high, num):
    fp = '/home2/public/data/RNA_aptamer/All_embedding/2-ex-apt/representations/'
    rnas = []
    for _ in range(num):
        i = random.randint(low, high)
        j = random.randint(low, high)
        # rna_fm1 = np.load(fp + str(i) + '.npy')
        # rna_fm2 = np.load(fp + str(j) + '.npy')
        rna_fm1 = np.load(fp + str(i) + '.npy')
        rna_fm2 = np.load(fp + str(j) + '.npy')

        rna_fm = (rna_fm1 + rna_fm2) / 2

        # rna_fm = rna_fm.reshape(-1)

        # 下面这个是展开FM表征
        rna_fm = rna_fm.reshape(-1)
        rna_fm = standardization(rna_fm)
        # rna_fm = np.mean(rna_fm, axis=0)

        rnas.append(rna_fm)
    # rnas = np.asarray(rnas)
    # rnas = np.mean(rnas, axis=0)
    rnas = torch.tensor(rnas).to(device)
    return rnas


def get_onehot(low, high, num):
    fp = '/home2/public/data/RNA_aptamer/All_data/4_ex_apt/4-her2-all-sort.txt'
    with open(fp, 'r') as file:
        lines = file.readlines()

    i = random.randint(low, high)
    j = random.randint(low, high)

    words1 = lines[i].split()
    b1 = words1[1:]
    a1 = ''.join(b1)
    match1 = re.search(r'\[.*\]', a1)

    words2 = lines[j].split()
    b2 = words2[1:]
    a2 = ''.join(b2)
    match2 = re.search(r'\[.*\]', a2)

    # 获取匹配到的部分
    substring1 = match1.group(0)
    substring2 = match2.group(0)
    list1 = substring1
    list2 = substring2

    list1 = [try_convert_to_int(x) for x in list1 if try_convert_to_int(x) is not None]
    list2 = [try_convert_to_int(y) for y in list2 if try_convert_to_int(y) is not None]
    # 使用列表推导式来计算每一对值的平均值
    average_list = [(x + y) / 2 for x, y in zip(list1, list2)]
    rnas1 = torch.tensor(average_list, dtype=torch.float32).to(device)

    return rnas1


def try_convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return None


# womlp
def get_sample2():
    input_file = '/home2/public/data/RNA_aptamer/RNA_generation/same_sample/round1-sample1-ex_sample_num_10.txt'
    fp = '/home2/public/data/RNA_aptamer/All_embedding/2-ex-apt/representations/'
    rnas = []

    with open(input_file, 'r') as file1:
        for line in file1:
            i = int(line.split()[0])
            j = int(line.split()[1])

            rna_fm1 = np.load(fp + str(i) + '.npy')
            rna_fm2 = np.load(fp + str(j) + '.npy')

            rna_fm = (rna_fm1 + rna_fm2) / 2

            # 下面这个是展开FM表征
            # rna_fm = rna_fm.reshape(-1)
            rna_fm = np.mean(rna_fm, axis=0)
            rna_fm = standardization(rna_fm)

            rnas.append(rna_fm)
    # rnas = np.asarray(rnas)
    # rnas = np.mean(rnas, axis=0)
    rnas = torch.tensor(rnas).to(device)
    return rnas


# without rna-fm
def get_sample3():
    input_file = '/home2/public/data/RNA_aptamer/RNA_generation/same-sample-CD3E/CD3E-8515_sample_num_1.txt'
    fp = '/home2/public/data/RNA_aptamer/All_embedding/round1-sample1-ex-cluster/'
    rnas = []
    input_file2 = '/home2/public/data/RNA_aptamer/All_data/round1-sample1-ex-apt/All_sort_bdscores.txt'

    with open(input_file2, 'r') as file2:
        lines = file2.readlines()

    with open(input_file, 'r') as file1:
        for line in file1:
            i = int(line.split()[0])
            j = int(line.split()[1])

            words1 = lines[i].split()
            b1 = words1[1:]
            a1 = ''.join(b1)
            match1 = re.search(r'\[.*\]', a1)

            words2 = lines[j].split()
            b2 = words2[1:]
            a2 = ''.join(b2)
            match2 = re.search(r'\[.*\]', a2)

            # 获取匹配到的部分
            substring1 = match1.group(0)
            substring2 = match2.group(0)
            list1 = substring1
            list2 = substring2

            list1 = [try_convert_to_int(x) for x in list1 if try_convert_to_int(x) is not None]
            list2 = [try_convert_to_int(y) for y in list2 if try_convert_to_int(y) is not None]
            # 使用列表推导式来计算每一对值的平均值
            average_list = [(x + y) / 2 for x, y in zip(list1, list2)]

            rnas.append(average_list)
    # rnas = np.asarray(rnas)
    # rnas = np.mean(rnas, axis=0)

    rnas = torch.tensor(rnas).to(device)

    return rnas


# 无MLP
def get_sample4():
    input_file = '/home2/public/data/RNA_aptamer/RNA_generation/same_sample/round1-sample1-ex_sample_num_10.txt'
    fp = '/home2/public/data/RNA_aptamer/All_embedding/round1-sample1-ex-cluster/'
    rnas = []

    with open(input_file, 'r') as file1:
        for line in file1:
            i = int(line.split()[0])
            j = int(line.split()[1])

            rna_fm1 = np.load(fp + str(i) + '.npy')
            rna_fm2 = np.load(fp + str(j) + '.npy')

            rna_fm = (rna_fm1 + rna_fm2) / 2

            # rna_fm = np.mean(rna_fm, axis=0)

            # 下面这个是展开FM表征
            rna_fm = rna_fm.reshape(-1)
            rna_fm = standardization(rna_fm)

            rnas.append(rna_fm)
    # rnas = np.asarray(rnas)
    # rnas = np.mean(rnas, axis=0)
    rnas = torch.tensor(rnas).to(device)
    return rnas


# evo&rna-fm
def get_sample5():
    input_file = '/home2/public/data/RNA_aptamer/RNA_generation/same_sample/round1-sample1-ex_sample_num_10.txt'
    fp = '/home2/public/data/RNA_aptamer/All_embedding/round1-sample1-ex-apt-evo/representations/'
    rnas = []

    with open(input_file, 'r') as file1:
        for line in file1:
            i = int(line.split()[0])
            j = int(line.split()[1])

            rna_fm1 = np.load(fp + str(i) + '.npy')
            rna_fm2 = np.load(fp + str(j) + '.npy')

            rna_fm = (rna_fm1 + rna_fm2) / 2

            # rna_fm = np.mean(rna_fm, axis=0)

            # 下面这个是展开FM表征
            rna_fm = rna_fm.reshape(-1)
            rna_fm = standardization(rna_fm)

            rnas.append(rna_fm)
    # rnas = np.asarray(rnas)
    # rnas = np.mean(rnas, axis=0)
    rnas = torch.tensor(rnas).to(device)
    return rnas




# evo&rna-fm
def get_sample6(input_file):
    # input_file = '/home2/public/data/RNA_aptamer/RNA_generation/same_sample/round1-sample1-ex_sample_num_10.txt'
    fp = '/home2/public/data/RNA_aptamer/All_embedding/2-ex-apt-evo/representations/'
    rnas = []

    with open(input_file, 'r') as file1:
        for line in file1:
            i = int(line.split()[0])
            j = int(line.split()[1])

            rna_fm1 = np.load(fp + str(i) + '.npy')
            rna_fm2 = np.load(fp + str(j) + '.npy')

            rna_fm = (rna_fm1 + rna_fm2) / 2

            # rna_fm = np.mean(rna_fm, axis=0)

            # 下面这个是展开FM表征
            rna_fm = rna_fm.reshape(-1)
            rna_fm = standardization(rna_fm)

            rnas.append(rna_fm)
    # rnas = np.asarray(rnas)
    # rnas = np.mean(rnas, axis=0)
    rnas = torch.tensor(rnas).to(device)
    return rnas


# 使用平均值来进行样本提取
def get_sample7(input_file):
    rnas = []
    input_file2 = '/home2/public/data/RNA_aptamer/All_data/2_ex_apt/All_sort_train_data.txt'

    with open(input_file2, 'r') as file2:
        lines = file2.readlines()

    with open(input_file, 'r') as file1:
        for line in file1:
            i = int(line.split()[0])
            j = int(line.split()[1])

            words1 = lines[i].split()
            b1 = words1[1:]
            a1 = ''.join(b1)
            match1 = re.search(r'\[.*\]', a1)

            words2 = lines[j].split()
            b2 = words2[1:]
            a2 = ''.join(b2)
            match2 = re.search(r'\[.*\]', a2)

            # 获取匹配到的部分
            substring1 = match1.group(0)
            substring2 = match2.group(0)
            list1 = substring1
            list2 = substring2

            list1 = [try_convert_to_int(x) for x in list1 if try_convert_to_int(x) is not None]
            list2 = [try_convert_to_int(y) for y in list2 if try_convert_to_int(y) is not None]
            # 使用列表推导式来计算每一对值的平均值
            average_list = [(x + y) / 2 for x, y in zip(list1, list2)]

            rnas.append(average_list)
    # rnas = np.asarray(rnas)
    # rnas = np.mean(rnas, axis=0)

    rnas = torch.tensor(rnas).to(device)

    return rnas


def greedy_decode_wollm(model, input_src, max_len, start_symbol, is_noise=False):
    if is_noise:
        input_src = add_gaussian_noise(input_src, mean=0.0, std=0.1)
    # input_src = input_src.unsqueeze(0)
    memory = model.encoder(input_src)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(input_src.data).long()
    for i in range(max_len):
        out = model.decoder(ys, memory)
        selected_tensor = out[0]
        prob = model.generator(selected_tensor[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(input_src.data).fill_(next_word)], dim=1).long()
    return ys


def greedy_decode_ex_MLP(model, input_src, max_len, start_symbol, is_noise=True):
    if is_noise:
        input_src = add_gaussian_noise(input_src, mean=0.0, std=0.1)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(input_src.data).long()
    for i in range(max_len):
        out = model.decoder(ys, input_src)
        selected_tensor = out[0]
        prob = model.generator(selected_tensor[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(input_src.data).fill_(next_word)], dim=1).long()
    return ys


def greedy_decode_rna_fm(model, input_src, max_len, start_symbol, is_noise=True):
    if is_noise:
        input_src = add_gaussian_noise(input_src, mean=0.0, std=0.1)
    memory = model.encoder(input_src)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(input_src.data).long()
    for i in range(max_len):
        out = model.decoder(ys, memory)
        selected_tensor = out[0]
        prob = model.generator(selected_tensor[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(input_src.data).fill_(next_word)], dim=1).long()
    return ys


def greedy_decode2(model, input_src, max_len, start_symbol, is_noise=True):
    if is_noise:
        input_src = add_gaussian_noise(input_src, mean=0.0, std=0.1)
    memory = model.encoder(input_src)
    ys = torch.ones(1).fill_(start_symbol).type_as(input_src.data).long()

    pro_matrix = torch.empty((1, 5), device=device)
    for i in range(max_len):
        out = model.decoder(ys, memory)
        selected_tensor = out[0]
        selected_tensor = torch.squeeze(selected_tensor, dim=0)
        prob = model.generator(selected_tensor)
        pro_matrix = torch.cat((pro_matrix, prob))

        _, next_word_idx = torch.max(prob, dim=1)
        next_word = next_word_idx[-1]

        ys = torch.cat([ys,
                        torch.ones(1).type_as(input_src.data).fill_(next_word)], dim=0).long()
    return ys, pro_matrix


def greedy_decode3(model, input_src, max_len, start_symbol, is_noise=True):
    if is_noise:
        input_src = add_gaussian_noise(input_src, mean=0.0, std=0.1)
    memory = model.encoder(input_src)
    ys = torch.ones(1, 5).fill_(0).type_as(input_src.data).long()

    pro_matrix = torch.empty((1, 5), device=device)
    for i in range(max_len):
        out = model.decoder(ys, memory)
        selected_tensor = out[0]
        selected_tensor = torch.squeeze(selected_tensor, dim=0)
        prob = model.generator(selected_tensor)
        pro_matrix = torch.cat((pro_matrix, prob))

        ys = pro_matrix
    return ys, pro_matrix


def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    noise = torch.randn(tensor.size()).to(device) * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


if __name__ == '__main__':
    model = torch.load('model/2-CD3E_rnafm-250_loss1_loss2_015-085.model')
    model.eval()
    model = model.to(device)

    random_rnas1 = []
    random_rnas2 = []

    # rna-fm random
    rnas = get_sample(0,2000,10000)
    for rna_input in rnas:
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
        random_seq2 = greedy_decode_rna_fm(model, random_rna_inputs, 20, 0, True)
        random_rnas2.append(random_seq2)
        print("使用greedy_decode生成的随机采用生成序列：" + str(random_seq2))


    # 无MLP
    # rnas = get_sample2()
    # for rna_input in rnas:
    #     random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
    #     random_seq2 = greedy_decode_ex_MLP(model, random_rna_inputs, 20, 0, True)
    #     random_rnas2.append(random_seq2)
    #     print("使用greedy_decode生成的随机采用生成序列：" + str(random_seq2))



    # evo&rna-fm
    # for i in range(10):
    #     input_file = f'/home2/public/data/RNA_aptamer/RNA_generation/same_sample/round1-sample1-ex_sample_num_{i+1}.txt'
    #     rnas = []
    #     rnas = get_sample6(input_file)
    #     for rna_input in rnas:
    #         random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
    #         random_seq2 = greedy_decode_rna_fm(model, random_rna_inputs, 20, 0, True)
    #         random_rnas2.append(random_seq2)
    #         print("使用greedy_decode生成的随机采用生成序列：" + str(random_seq2))
    #
    #     input_file2 = f"/home2/public/data/RNA_aptamer/RNA_generation/2-ex-apt-loss8515/evo/CD3E-highbd-8515-evo-generation_{i+1}.txt"
    #     with open(input_file2, 'w') as file2:
    #         for line in random_rnas2:
    #             file2.write(str(line) + '\n')


    # wollm
    # for i in range(10):
    #     input_file = f'/home2/public/data/RNA_aptamer/RNA_generation/same_sample/round1-sample1-ex_sample_num_{i+1}.txt'
    #     rnas = []
    #     rnas = get_sample7(input_file)
    #     for rna_input in rnas:
    #         random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
    #         random_seq2 = greedy_decode_wollm(model, random_rna_inputs, 20, 0, True)
    #         random_rnas2.append(random_seq2)
    #         print("使用greedy_decode生成的随机采用生成序列：" + str(random_seq2))


    input_file2 = "/home2/public/data/RNA_aptamer/RNA_generation/2-CD3E_tsne/CD3E-lowbd-015085-rnafm-generation_low200-222.txt"
    with open(input_file2, 'w') as file2:
        for line in random_rnas2:
            file2.write(str(line) + '\n')





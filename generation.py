# -*- coding: utf-8 -*-
import os
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import fm
import argparse
from evo import Evo

from model_AE import *

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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


def greedy_decode_llm(model, input_src, max_len, start_symbol, is_noise=True):
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


def greedy_VAE(model):
    z = torch.randn(1, 128).to(device)  # 生成64个随机的潜在向量
    ys = torch.ones(1, 1).fill_(0).type_as(z.data).long().to(device)
    # generated_sequences = model.decoder(ys, z)

    for i in range(20):

        out = model.decoder(ys, z)
        selected_tensor = out[0]
        prob = model.generator(selected_tensor[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(z).fill_(next_word)], dim=1).long()

    return ys,


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


def convert_to_rna_sequence_rna_fm(data):
    # 创建映射字典
    rna_to_num = {'A': 1, 'C': 2, 'G': 3, 'U': 4}

    # 将RNA序列转换为数字
    numbers = [rna_to_num.get(base, -1) for base in data.upper()]

    return numbers

def get_rna_fm_model():
    torch.cuda.empty_cache()

    EmbbingModel, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    EmbbingModel.to(device)
    EmbbingModel.eval()

    return EmbbingModel, batch_converter


def get_evo_model():
    evo_model = Evo('evo-1-8k-base')
    model, tokenizer = evo_model.model, evo_model.tokenizer

    return model,tokenizer


def rna_seq_embbding(OriginSeq, batch_converter, EmbeddingModel):
    EmbeddingModel = EmbeddingModel.to(device)
    batch_labels, batch_strs, batch_tokens = batch_converter(OriginSeq)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = EmbeddingModel(batch_tokens, repr_layers=[12])
    token_embeddings = results["representations"][12][0]

    return token_embeddings




def get_rna_fm_embedding(seq, batch_converter, EmbbingModel):
    rna_seq = ("undefined", seq)
    seq_unused = ('UNUSE', 'ACGU')
    all_rna = []
    all_rna.append(rna_seq)
    all_rna.append(seq_unused)

    rna_fm = rna_seq_embbding(all_rna, batch_converter, EmbbingModel)
    rna_fm = rna_fm[1:-1, :]
    rna_fm = rna_fm.cpu().numpy()

    return rna_fm


def get_evo_embedding(seq, model, tokenizer):
    sequence = seq

    input_ids = torch.tensor(
        tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).to(device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(input_ids)

    logits = logits.detach()
    logits = logits.float()
    cpu_logits = logits.cpu()

    rna_evo = cpu_logits.numpy()

    return rna_evo


def get_sample_AE_rna_fm(low, high ,num, input_file):
    EmbbingModel, batch_converter = get_rna_fm_model()
    with open(input_file,'r') as file:
        lines = file.readlines()


    rnas = []
    for _ in range(num):
        i = random.randint(low, high)
        j = random.randint(low, high)

        line_1 =lines[i].split()[-1]
        line_2 =lines[j].split()[-1]

        rna_fm1 = get_rna_fm_embedding(line_1, batch_converter, EmbbingModel)
        rna_fm2 = get_rna_fm_embedding(line_2, batch_converter, EmbbingModel)

        rna_fm = (rna_fm1 + rna_fm2) / 2

        rna_fm = rna_fm.reshape(-1)
        rna_fm = standardization(rna_fm)

        rnas.append(rna_fm)
    # rnas = np.asarray(rnas)
    # rnas = np.mean(rnas, axis=0)
    rnas = torch.tensor(rnas).to(device)
    return rnas



def get_sample_AE_evo(low, high ,num, input_file):
    model, tokenizer = get_evo_model()
    model.to(device)
    model.eval()


    with open(input_file,'r') as file:
        lines = file.readlines()

    rnas = []
    for _ in range(num):
        i = random.randint(low, high)
        j = random.randint(low, high)

        line_1 =lines[i].split()[-1]
        line_2 =lines[j].split()[-1]

        rna_fm1 = get_evo_embedding(line_1, model, tokenizer)
        rna_fm2 = get_evo_embedding(line_2, model, tokenizer)

        rna_fm = (rna_fm1 + rna_fm2) / 2

        rna_fm = rna_fm.reshape(-1)
        rna_fm = standardization(rna_fm)

        rnas.append(rna_fm)
    rnas = torch.tensor(rnas).to(device)
    return rnas


def get_sample_AE_wollm(low, high ,num, input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    rnas = []

    for _ in range(num):
        rna_1 = []
        rna_2 = []

        i = random.randint(low, high)
        j = random.randint(low, high)

        words1 = lines[i].split()
        seq_1 = words1[-1]
        words2 = lines[j].split()
        seq_2 = words2[-1]

        rna_1 = convert_to_rna_sequence_rna_fm(seq_1)
        rna_2 = convert_to_rna_sequence_rna_fm(seq_2)
        average_list = [(x + y) / 2 for x, y in zip(rna_1, rna_2)]

        rnas1 = torch.tensor(average_list, dtype=torch.float32).to(device)

        rnas.append(rnas1)
    return rnas



def get_sample_CAE_womlp(low, high ,num, input_file):
    EmbbingModel, batch_converter = get_rna_fm_model()

    with open(input_file, 'r') as file:
        lines = file.readlines()
    rnas = []


    for _ in range(num):
        i = random.randint(low, high)
        j = random.randint(low, high)

        line_1 =lines[i].split()[-1]
        line_2 =lines[j].split()[-1]

        rna_fm1 = get_rna_fm_embedding(line_1, batch_converter, EmbbingModel)
        rna_fm2 = get_rna_fm_embedding(line_2, batch_converter, EmbbingModel)

        rna_fm = (rna_fm1 + rna_fm2) / 2

        rna_fm = np.mean(rna_fm, axis=0)
        rna_fm = standardization(rna_fm)

        rnas.append(rna_fm)
    # rnas = np.asarray(rnas)
    # rnas = np.mean(rnas, axis=0)
    rnas = torch.tensor(rnas).to(device)
    return rnas


def generation_AE_rna_fm(input_file, output_file, model_name, num):
    model_name_2=f'model/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(device)

    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)

    random_rnas = []

    # rna-fm random
    rnas = get_sample_AE_rna_fm(0, num_lines-1, num, input_file)
    for rna_input in rnas:
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
        random_seq = greedy_decode_llm(model, random_rna_inputs, 20, 0, True)
        random_rnas.append(random_seq)
        print("使用greedy_decode生成的随机采用生成序列：" + str(random_seq))

    with open(output_file, 'w') as file2:
        for line in random_rnas:
            file2.write(str(line) + '\n')


def generation_AE_evo(input_file, output_file, model_name, num):
    model_name_2=f'model/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(device)

    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)

    random_rnas = []

    # rna-fm random
    rnas = get_sample_AE_evo(0, num_lines - 1, num, input_file)
    for rna_input in rnas:
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
        random_seq = greedy_decode_llm(model, random_rna_inputs, 20, 0, True)
        random_rnas.append(random_seq)
        print("使用greedy_decode生成的随机采用生成序列：" + str(random_seq))

    with open(output_file, 'w') as file2:
        for line in random_rnas:
            file2.write(str(line) + '\n')



def generation_AE_wollm(input_file, output_file, model_name, num):
    model_name_2=f'model/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(device)

    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)

    random_rnas = []

    rnas = get_sample_AE_wollm(0, num_lines-1, num, input_file)

    for rna_input in rnas:
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
        random_seq2 = greedy_decode_wollm(model, random_rna_inputs, 20, 0, False)
        random_rnas.append(random_seq2)
        print("使用greedy_decode生成的随机采用生成序列：" + str(random_seq2))

    with open(output_file, 'w') as file2:
        for line in random_rnas:
            file2.write(str(line) + '\n')


def generation_CAE_womlp(input_file, output_file, model_name, num):
    model_name_2=f'model/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(device)

    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)

    random_rnas = []

    rnas = get_sample_CAE_womlp(0, num_lines - 1, num, input_file)

    for rna_input in rnas:
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
        random_seq = greedy_decode_ex_MLP(model, random_rna_inputs, 20, 0, True)
        random_rnas.append(random_seq)
        print("使用greedy_decode生成的随机采用生成序列：" + str(random_seq))


    with open(output_file, 'w') as file2:
        for line in random_rnas:
            file2.write(str(line) + '\n')


def generation_VAE(input_file, output_file, model_name, num):
    model_name_2=f'model/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(device)


    random_rnas = []

    for i in range(num):
        random_seq2 = greedy_VAE(model)
        print(random_seq2)
        random_rnas.append(random_seq2)


    with open(output_file, 'w') as file2:
        for line in random_rnas:
            file2.write(str(line) + '\n')



def main():
    parser = argparse.ArgumentParser(description="Choose which function to run.")
    parser.add_argument('function', choices=['1', '2', '3', '4', '5'], help="Function to run")
    parser.add_argument('--cuda', type=str, default="0", help="CUDA device ID (e.g., '0', '1', '2')")
    parser.add_argument('--input_file', type=str, help="-----")
    parser.add_argument('--output_file', type=str, help="-----")
    parser.add_argument('--model_name', type=str, help="-----")
    parser.add_argument('--num', type=int, help="-----")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    input_file = args.input_file
    output_file = args.output_file
    model_name = args.model_name
    num = args.num

    # 根据参数选择函数
    if args.function == '1':
        generation_AE_rna_fm(input_file,
                             output_file,
                             model_name,
                             num)
    elif args.function == '2':
        generation_AE_evo(input_file,
                          output_file,
                          model_name,
                          num)
    elif args.function == '3':
        generation_AE_wollm(input_file,
                            output_file,
                            model_name,
                            num)
    elif args.function == '4':
        generation_CAE_womlp(input_file,
                             output_file,
                             model_name,
                             num)
    elif args.function == '5':
        generation_VAE(input_file,
                       output_file,
                       model_name,
                       num)

if __name__ == '__main__':
    main()

    # 无MLP
    # rnas = get_sample2()
    # for rna_input in rnas:
    #     random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
    #     random_seq2 = greedy_decode_ex_MLP(model, random_rna_inputs, 20, 0, True)
    #     random_rnas2.append(random_seq2)
    #     print("使用greedy_decode生成的随机采用生成序列：" + str(random_seq2))









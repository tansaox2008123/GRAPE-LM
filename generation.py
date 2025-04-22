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

from model import *

#  If you have any internet error please try this code with your proxy setting.
#  os.environ["http_proxy"] = "http://...:8888"
#  os.environ["https_proxy"] = "http://...:8888"



# GRAPE generation method
def greedy_decode_guidance(model, input_src, max_len, start_symbol, is_noise, device):
    if is_noise:
        input_src = add_gaussian_noise(input_src, device, mean=0.0, std=0.1)
    memory = model.adapter(input_src)
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


def add_gaussian_noise(tensor, device, mean=0.0, std=1.0):
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


def get_rna_fm_model(device):
    torch.cuda.empty_cache()

    EmbbingModel, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    EmbbingModel.to(device)
    EmbbingModel.eval()

    return EmbbingModel, batch_converter


def get_evo_model():
    evo_model = Evo('evo-1-8k-base')
    model, tokenizer = evo_model.model, evo_model.tokenizer

    return model, tokenizer


def rna_seq_embbding(OriginSeq, batch_converter, EmbeddingModel, device):
    EmbeddingModel = EmbeddingModel.to(device)
    batch_labels, batch_strs, batch_tokens = batch_converter(OriginSeq)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = EmbeddingModel(batch_tokens, repr_layers=[12])
    token_embeddings = results["representations"][12][0]

    return token_embeddings


def get_rna_fm_embedding(seq, batch_converter, EmbbingModel, device):
    rna_seq = ("undefined", seq)
    seq_unused = ('UNUSE', 'ACGU')
    all_rna = []
    all_rna.append(rna_seq)
    all_rna.append(seq_unused)

    rna_fm = rna_seq_embbding(all_rna, batch_converter, EmbbingModel, device)
    rna_fm = rna_fm[1:-1, :]
    rna_fm = rna_fm.cpu().numpy()

    return rna_fm


def get_evo_embedding(seq, model, tokenizer, device):
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


def get_sample_AE_rna_fm(low, high, num, input_file, device):
    EmbbingModel, batch_converter = get_rna_fm_model(device)
    with open(input_file, 'r') as file:
        lines = file.readlines()

    rnas = []
    for _ in range(num):
        i = random.randint(low, high)
        j = random.randint(low, high)

        line_1 = lines[i].split()[-1]
        line_2 = lines[j].split()[-1]

        rna_fm1 = get_rna_fm_embedding(line_1, batch_converter, EmbbingModel, device)
        rna_fm2 = get_rna_fm_embedding(line_2, batch_converter, EmbbingModel, device)

        rna_fm = (rna_fm1 + rna_fm2) / 2

        rna_fm = rna_fm.reshape(-1)
        rna_fm = standardization(rna_fm)

        rnas.append(rna_fm)
    rnas = torch.tensor(rnas).to(device)
    return rnas


def get_sample_AE_evo(low, high, num, input_file, device):
    model, tokenizer = get_evo_model()
    model.to(device)
    model.eval()

    with open(input_file, 'r') as file:
        lines = file.readlines()

    rnas = []
    for _ in range(num):
        i = random.randint(low, high)
        j = random.randint(low, high)

        line_1 = lines[i].split()[-1]
        line_2 = lines[j].split()[-1]

        rna_fm1 = get_evo_embedding(line_1, model, tokenizer, device)
        rna_fm2 = get_evo_embedding(line_2, model, tokenizer, device)

        rna_fm = (rna_fm1 + rna_fm2) / 2

        rna_fm = rna_fm.reshape(-1)
        rna_fm = standardization(rna_fm)

        rnas.append(rna_fm)
    rnas = torch.tensor(rnas).to(device)
    return rnas



def generation_guidance_rna_fm(input_file, output_file, model_name, num, device):
    model_name_2 = f'model/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(device)

    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)

    random_rnas = []

    rnas = get_sample_AE_rna_fm(0, num_lines - 1, num, input_file, device)
    for rna_input in rnas:
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
        random_seq = greedy_decode_guidance(model, random_rna_inputs, 20, 0, True, device)
        random_rnas.append(random_seq)
        print("Using greedy_decode generate random RNA aptamers seqs：" + str(random_seq))

    with open(output_file, 'w') as file2:
        for line in random_rnas:
            file2.write(str(line) + '\n')


def generation_guidance_evo(input_file, output_file, model_name, num, device):
    model_name_2 = f'model/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(device)

    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)

    random_rnas = []

    # rnas = get_sample_AE_evo(0, num_lines - 1, num, input_file, device)
    rnas = get_sample_AE_evo(0, num_lines, num, input_file, device)
    for rna_input in rnas:
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
        random_seq = greedy_decode_guidance(model, random_rna_inputs, 20, 0, True, device)
        random_rnas.append(random_seq)
        print("Using greedy_decode generate random RNA aptamers seqs：" + str(random_seq))

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
    CUDA = args.cuda
    input_file = args.input_file
    output_file = args.output_file
    model_name = args.model_name
    num = args.num

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{CUDA}'

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 根据参数选择函数
    if args.function == '1':
        generation_guidance_rna_fm(input_file,
                                   output_file,
                                   model_name,
                                   num,
                                   device)
    elif args.function == '2':
        generation_guidance_evo(input_file,
                                output_file,
                                model_name,
                                num,
                                device)



if __name__ == '__main__':
    main()

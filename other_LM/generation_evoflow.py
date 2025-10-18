# -*- coding: utf-8 -*-
import random
import argparse

from model import *
from src.ncrna.tasks.lm.drnafm import EvoFlow

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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
    noise = torch.randn(tensor.size()).to(f'cuda:{device}') * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def get_Evoflow_model(device):
    model = EvoFlow.load_from_pretrained('weights/mini-v1.ckpt').to(f'cuda:0')
    tokenizer = model.alphabet

    return model, tokenizer


def get_Evoflow_embedding(seq, model, tokenizer, device):
    b_list = [seq]
    embedding = torch.tensor(tokenizer.batch_tokenize(b_list), dtype=torch.int64, device=f'cuda:{device}')
    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits, hidden_states = model(embedding, return_last_hidden_state=True)

    hidden_states = hidden_states[:, 1:-1, :]

    return hidden_states



def get_sample_AE_Evoflow(low, high, num, input_file, device):
    EmbbingModel, tokenizer = get_Evoflow_model(device)
    with open(input_file, 'r') as file:
        lines = file.readlines()

    rnas = []
    for _ in range(num):
        i = random.randint(low, high)
        j = random.randint(low, high)

        line_1 = lines[i].split()[-1]
        line_2 = lines[j].split()[-1]

        rna_fm1 = get_Evoflow_embedding(line_1, EmbbingModel, tokenizer, device)
        rna_fm2 = get_Evoflow_embedding(line_2, EmbbingModel, tokenizer, device)

        rna_fm = (rna_fm1 + rna_fm2) / 2

        rna_fm = rna_fm.detach().cpu().numpy()
        rna_fm = rna_fm.reshape(-1)
        rna_fm = standardization(rna_fm)

        rnas.append(rna_fm)
    # rnas = torch.tensor(rnas).to(f'cuda:{device}')
    return rnas



def generation_Evoflow(input_file, output_file, model_name, num, device):
    model_name_2 = f'model/zh_generation/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(f'cuda:{device}')

    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)

    random_rnas = []

    # rnas = get_sample_AE_rna_fm(0, num_lines - 1, num, input_file, device)
    rnas = get_sample_AE_Evoflow(0, 2000, num, input_file, device)
    for rna_input in rnas:
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(f'cuda:{device}')
        random_seq = greedy_decode_guidance(model, random_rna_inputs, 20, 0, True, device)

        id_to_base = {1: 'A', 2: 'C', 3: 'G', 4: 'U'}

        sequence_ids = random_seq[0].tolist()
        rna_sequence = ''.join([id_to_base.get(i, '') for i in sequence_ids])
        print(rna_sequence)
        random_rnas.append(rna_sequence)

    with open(output_file, 'w') as file2:
        for line in random_rnas:
            file2.write(str(line) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Choose which function to run.")
    parser.add_argument('function', choices=['1', '2', '3', '4', '5', '6', '7', '8'], help="Function to run")
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

    if args.function == '1':
        generation_Evoflow(input_file,
                           output_file,
                           model_name,
                           num,
                           CUDA)


if __name__ == '__main__':
    main()

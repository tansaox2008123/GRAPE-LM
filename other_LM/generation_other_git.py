# -*- coding: utf-8 -*-
import os
import random
import argparse
from multimolecule import RnaTokenizer, RnaBertModel
from multimolecule import RnaTokenizer, RnaErnieModel
from multimolecule import RnaTokenizer, RiNALMoModel
from model import *

from tqdm import tqdm
from rinalmo.pretrained import get_pretrained_model



def greedy_decode_guidance(model, input_src, max_len, start_symbol, is_noise, device):
    noise_tensor = None
    memory = model.adapter(input_src)

    if is_noise:
        memory, noise_tensor = add_gaussian_noise(memory, device, mean=0.0, std=0.1)

    ys = torch.ones(1, 1).fill_(start_symbol).type_as(input_src.data).long()

    for i in range(max_len):
        out = model.decoder(ys, memory)
        selected_tensor = out[0]
        prob = model.generator(selected_tensor[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(input_src.data).fill_(next_word)], dim=1).long()

    if is_noise:
        return ys, noise_tensor
    else:
        return ys, None


def add_gaussian_noise(tensor, device, mean=0.0, std=1.0):
    noise = torch.randn(tensor.size()).to(device) * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor, noise


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma




def get_rna_bert_model(device):
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnabert")
    model = RnaBertModel.from_pretrained("multimolecule/rnabert").to(device)

    return model, tokenizer


def get_RiNALMo_model(device):
    model, alphabet = get_pretrained_model(model_name="giga-v1")
    model = model.to(device)

    return model, alphabet


def get_Ernie_model(device):
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnaernie")
    model = RnaErnieModel.from_pretrained("multimolecule/rnaernie").to(device)

    return model, tokenizer



def get_RNABERT_embedding(seq, model, tokenizer, device):
    input = tokenizer(seq, return_tensors="pt")
    input = {k: v.to(device) for k, v in input.items()}
    encoder_outputs = model(**input)  # output is BaseModelOutput...
    encoder_hidden_states = encoder_outputs.last_hidden_state
    encoder_hidden_states = encoder_hidden_states[:, 1:-1, :]
    hidden_states_numpy = encoder_hidden_states.detach().cpu().numpy()

    return hidden_states_numpy

def get_RiNALMo_embedding(seq, model, alphabet, device):
    sequence = [seq]
    tokens = torch.tensor(alphabet.batch_tokenize(sequence), dtype=torch.int64, device=device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model(tokens)

    input = outputs["representation"]

    encoder_hidden_states = input[:, 1:-1, :]
    hidden_states_numpy = encoder_hidden_states.detach().cpu().numpy()

    return hidden_states_numpy


def get_Ernie_embedding(seq, model, tokenizer, device):
    input = tokenizer(seq, return_tensors="pt")
    input = {k: v.to(device) for k, v in input.items()}
    encoder_outputs = model(**input)  # output is BaseModelOutput...
    encoder_hidden_states = encoder_outputs.last_hidden_state
    encoder_hidden_states = encoder_hidden_states[:, 1:-1, :]
    hidden_states_numpy = encoder_hidden_states.detach().cpu().numpy()

    return hidden_states_numpy



def get_sample_AE_RNABERT_2(low, high, num, input_file, device):
    EmbbingModel, tokenizer = get_rna_bert_model(device)
    with open(input_file, 'r') as file:
        lines = file.readlines()

    rnas = []
    line_1 = ''
    line_2 = ''
    for _ in range(num):
        i = random.randint(low, high)
        j = random.randint(low, high)

        line_1 = lines[i].split()[-1]
        line_2 = lines[j].split()[-1]

        rna_fm1 = get_RNABERT_embedding(line_1, EmbbingModel, tokenizer, device)
        rna_fm2 = get_RNABERT_embedding(line_2, EmbbingModel, tokenizer, device)

        rna_fm = (rna_fm1 + rna_fm2) / 2

        rna_fm = rna_fm.reshape(-1)
        rna_fm = standardization(rna_fm)

        rnas.append(rna_fm)
    rnas = torch.tensor(rnas).to(device)
    return rnas, line_1, line_2


def get_sample_AE_RiNALMo_2(low, high, num, input_file, device):
    model, alphabet = get_RiNALMo_model(device)
    with open(input_file, 'r') as file:
        lines = file.readlines()

    rnas = []
    line_1 = ''
    line_2 = ''
    for _ in range(num):
        i = random.randint(low, high)
        j = random.randint(low, high)

        line_1 = lines[i].split()[-1]
        line_2 = lines[j].split()[-1]

        rna_fm1 = get_RiNALMo_embedding(line_1, model, alphabet, device)
        rna_fm2 = get_RiNALMo_embedding(line_2, model, alphabet, device)

        rna_fm = (rna_fm1 + rna_fm2) / 2

        rna_fm = rna_fm.reshape(-1)
        rna_fm = standardization(rna_fm)

        rnas.append(rna_fm)
    rnas = torch.tensor(rnas).to(device)
    return rnas, line_1, line_2


def get_sample_AE_Ernie_2(low, high, num, input_file, device):
    EmbbingModel, tokenizer = get_Ernie_model(device)
    with open(input_file, 'r') as file:
        lines = file.readlines()

    rnas = []
    line_1 = ''
    line_2 = ''
    for _ in range(num):
        i = random.randint(low, high)
        j = random.randint(low, high)

        line_1 = lines[i].split()[-1]
        line_2 = lines[j].split()[-1]

        rna_fm1 = get_Ernie_embedding(line_1, EmbbingModel, tokenizer, device)
        rna_fm2 = get_Ernie_embedding(line_2, EmbbingModel, tokenizer, device)

        rna_fm = (rna_fm1 + rna_fm2) / 2

        rna_fm = rna_fm.reshape(-1)
        rna_fm = standardization(rna_fm)

        rnas.append(rna_fm)
    rnas = torch.tensor(rnas).to(device)
    return rnas, line_1, line_2


def generation_RNABERT(input_file, output_file, model_name, num, device):
    model_name_2 = f'model/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(device)

    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)

    random_rnas = []
    all_noise = []
    all_line_1 = []
    all_line_2 = []

    # rnas = get_sample_AE_rna_fm(0, num_lines - 1, num, input_file, device)
    # rnas = get_sample_AE_RNABERT(0, 2000, num, input_file, device)
    rnas, line_1, line_2 = get_sample_AE_RNABERT_2(0, 2000, num, input_file, device)
    all_line_1.append(line_1)
    all_line_2.append(line_2)

    for idx, rna_input in tqdm(enumerate(rnas), desc="Generating RNA Sequences"):
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
        random_seq, noise = greedy_decode_guidance(model, random_rna_inputs, 20, 0, True, device)

        if noise is not None:
            all_noise.append(noise.cpu())

        id_to_base = {1: 'A', 2: 'C', 3: 'G', 4: 'U'}

        sequence_ids = random_seq[0].tolist()
        rna_sequence = ''.join([id_to_base.get(i, '') for i in sequence_ids])
        random_rnas.append(rna_sequence)

    with open(output_file, 'w') as file2:
        for line in random_rnas:
            file2.write(str(line) + '\n')


def generation_RiNALMo(input_file, output_file, model_name, num, device):
    model_name_2 = f'model/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(device)

    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)

    random_rnas = []
    all_noise = []
    all_line_1 = []
    all_line_2 = []

    # rnas = get_sample_AE_rna_fm(0, num_lines - 1, num, input_file, device)
    # rnas = get_sample_AE_RNABERT(0, 2000, num, input_file, device)
    rnas, line_1, line_2 = get_sample_AE_RiNALMo_2(0, 9999, num, input_file, device)
    all_line_1.append(line_1)
    all_line_2.append(line_2)

    for idx, rna_input in tqdm(enumerate(rnas), desc="Generating RNA Sequences"):
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
        random_seq, noise = greedy_decode_guidance(model, random_rna_inputs, 20, 0, True, device)

        if noise is not None:
            all_noise.append(noise.cpu())

        id_to_base = {1: 'A', 2: 'C', 3: 'G', 4: 'U'}

        sequence_ids = random_seq[0].tolist()
        rna_sequence = ''.join([id_to_base.get(i, '') for i in sequence_ids])
        random_rnas.append(rna_sequence)

    with open(output_file, 'w') as file2:
        for line in random_rnas:
            file2.write(str(line) + '\n')



def generation_Ernie(input_file, output_file, model_name, num, device):
    model_name_2 = f'model/{model_name}'

    model = torch.load(model_name_2)
    model.eval()
    model = model.to(device)

    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)

    random_rnas = []
    all_noise = []
    all_line_1 = []
    all_line_2 = []

    # rnas = get_sample_AE_rna_fm(0, num_lines - 1, num, input_file, device)
    # rnas = get_sample_AE_Ernie(0, 2000, num, input_file, device)
    rnas, line_1, line_2 = get_sample_AE_Ernie_2(0, 2000, num, input_file, device)
    all_line_1.append(line_1)
    all_line_2.append(line_2)
    for idx, rna_input in tqdm(enumerate(rnas), desc="Generating RNA Sequences"):
        random_rna_inputs = torch.tensor(rna_input).unsqueeze(0).to(device)
        random_seq, noise = greedy_decode_guidance(model, random_rna_inputs, 20, 0, True, device)

        if noise is not None:
            all_noise.append(noise.cpu())

        id_to_base = {1: 'A', 2: 'C', 3: 'G', 4: 'U'}

        sequence_ids = random_seq[0].tolist()
        rna_sequence = ''.join([id_to_base.get(i, '') for i in sequence_ids])
        random_rnas.append(rna_sequence)

    with open(output_file, 'w') as file2:
        for line in random_rnas:
            file2.write(str(line) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Choose which function to run.")
    parser.add_argument('function', choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
                        help="Function to run")
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

    if args.function == '1':
        generation_RNABERT(input_file,
                           output_file,
                           model_name,
                           num,
                           device)

    elif args.function == '2':
        generation_Ernie(input_file,
                         output_file,
                         model_name,
                         num,
                         device)

    elif args.function == '3':
        generation_RiNALMo(input_file,
                           output_file,
                           model_name,
                           num,
                           device)


if __name__ == '__main__':
    main()


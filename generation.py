# -*- coding: utf-8 -*-
import os
import random
import torch
import fm
import argparse
from evo import Evo
import numpy as np
from model_zh import FullModel_guidance, FullModel_guidance_LSTM, FullModelGru, FullModelCNN
from tqdm import tqdm
from pathlib import Path


# GRAPE generation method
def greedy_decode_guidance(model, input_src, max_len, start_symbol, is_noise, noise, device):
    if args.feature == "one-hot":
        input_src = input_src.squeeze(0)
        rna1, rna2 = input_src
        rna1, rna2 = rna1.unsqueeze(0), rna2.unsqueeze(0)

        if args.arch == "cnn":
            rna1 = rna1.permute(0, 2, 1)
            rna2 = rna2.permute(0, 2, 1)

        memory1 = model.adapter(rna1)
        memory2 = model.adapter(rna2)
        # memory1 = model.lstm(rna1)
        # memory2 = model.lstm(rna2)

        memory = ((memory1 + memory2) / 2).squeeze(-1)
        if is_noise:
            memory += noise

    else:
        if is_noise:
            input_src += noise
        memory = model.adapter(input_src)

    ys = torch.ones(1, 1).fill_(start_symbol).type_as(input_src.data).long()
    for i in range(max_len):
        out = model.decoder(ys, memory)
        selected_tensor = out[0]
        prob = model.generator(selected_tensor[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(input_src.data).fill_(next_word)], dim=1).long()

    return ys


# GRU verison
def gru_predict_sequence(model, input_seq, max_len, start_symbol, is_noise, noise, device):
    if is_noise:
        input_seq += noise

    h = model.adapter(input_seq)
    _, hidden = model.gru(
        h,
        torch.zeros(model.gru.num_layers, 1, model.gru.hidden_size, device=device),
    )

    current = torch.empty(1, 1).fill_(start_symbol).long().to(device)

    generated_ids = []
    for i in range(max_len):
        current = model.embed(current)
        gru_out, hidden = model.gru(current, hidden)
        pred = model.generator(gru_out)  # [1, 1, vocab_size]
        pred = torch.softmax(pred, -1)
        next_token = torch.argmax(pred, -1)  # [1, 1]
        generated_ids.append(next_token.item())
        current = next_token

    return generated_ids


def gaussian_noise(size, mean=0.0, std=1.0):
    noise = torch.randn(size) * std + mean
    return noise


def standardization(data):
    mu = torch.mean(data, dim=-1, keepdim=True)
    sigma = torch.std(data, dim=-1, keepdim=True)
    return (data - mu) / sigma


def rna_to_numbers(seq):
    base_to_num = {"A": 1, "C": 2, "G": 3, "U": 4}

    numbers = [base_to_num.get(base, -1) for base in seq.upper()]

    return numbers


def rna_to_onehot(seq):
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3}
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            onehot[i, mapping[base]] = 1.0
    if args.arch == "base":
        onehot = onehot.reshape(-1)
    return onehot


def run_rna_fm(rna_fm_inputs, device, batch_size=5000):
    rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    rna_fm_model.to(device)
    rna_fm_model.eval()

    all_reps = []
    with tqdm(total=len(rna_fm_inputs), desc="RNA-FM") as pbar:
        for i in range(0, len(rna_fm_inputs), batch_size):
            batch_inputs = rna_fm_inputs[i : i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_inputs)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = rna_fm_model(batch_tokens, repr_layers=[12])

            reps = results["representations"][12]
            reps = reps[:, 1:-1, :].detach().cpu().float()
            reps = reps.reshape(reps.shape[0], -1)
            all_reps.append(reps)

            pbar.update(len(batch_inputs))

    del rna_fm_model
    torch.cuda.empty_cache()

    return torch.cat(all_reps, dim=0)


def run_evo(evo_inputs, device, batch_size=1000):
    evo_model = Evo("evo-1-8k-base")
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(device)
    model.eval()

    all_reps = []
    with tqdm(total=len(evo_inputs), desc="EVO") as pbar:
        for i in range(0, len(evo_inputs), batch_size):
            batch_inputs = evo_inputs[i : i + batch_size]
            input_ids_tensor = torch.stack([torch.tensor(tokenizer.tokenize(seq), dtype=torch.int) for seq in batch_inputs]).to(device)
            with torch.no_grad():
                logits, _ = model(input_ids_tensor)

            logits = logits.detach().cpu().float()
            logits = logits.reshape(logits.shape[0], -1)
            all_reps.append(logits)

            pbar.update(len(batch_inputs))

    del evo_model
    torch.cuda.empty_cache()

    return torch.cat(all_reps, dim=0)


def get_samples(feature, low, high, num, input_file, use_saved_samples, sample_seqs_file, device):
    seqs1 = []
    seqs2 = []
    if not use_saved_samples:
        with open(input_file, "r") as file:
            lines = file.readlines()
        for _ in range(num):
            i = random.randint(low, high)
            j = random.randint(low, high)
            seqs1.append(lines[i].split()[-1])
            seqs2.append(lines[j].split()[-1])

        sample_seqs = [(seq1, seq2) for seq1, seq2 in zip(seqs1, seqs2)]
        with open(sample_seqs_file, "w") as f:
            for seq1, seq2 in sample_seqs:
                f.write(f"{seq1}\t{seq2}\n")
    else:
        with open(sample_seqs_file, "r") as file:
            for line in file:
                seq1, seq2 = line.strip().split("\t")
                seqs1.append(seq1)
                seqs2.append(seq2)

    if feature == "rna-fm":
        reps = run_rna_fm([(i, seq) for i, seq in enumerate(seqs1 + seqs2)], device)
        reps = standardization(reps)
        reps = (reps[:num] + reps[num:]) / 2
    elif feature == "evo":
        reps = run_evo(seqs1 + seqs2, device)
        reps = standardization(reps)
        reps = (reps[:num] + reps[num:]) / 2
    elif feature == "one-hot":
        reps = []
        for seq1, seq2 in zip(seqs1, seqs2):
            one_hot1 = rna_to_onehot(seq1)
            one_hot2 = rna_to_onehot(seq2)
            reps.append((one_hot1, one_hot2))
        reps = torch.from_numpy(np.asarray(reps))

    return reps


def generation(
    model_name,
    input_file,
    output_file,
    low,
    high,
    gen_num,
    device,
    use_saved_samples=False,
):
    arch, feature, *_ = model_name.split("_")
    input_dim = {"rna-fm": 12800, "evo": 10240, "one-hot": 80}
    if arch == "base":
        model = FullModel_guidance(
            input_dim=input_dim[feature],
            model_dim=128,
            tgt_size=5,
            n_declayers=2,
            d_ff=128,
            d_k_v=64,
            n_heads=2,
            dropout=0.05,
        )
    elif arch == "gru":
        model = FullModelGru(
            input_dim=input_dim[feature] // 20,
            model_dim=128,
            vocab_size=5,
            num_gru_layers=2,
            dropout=0.05,
        )
    elif arch == "cnn":
        model = FullModelCNN(
            input_dim=4,
            model_dim=128,
            tgt_size=5,
            n_declayers=2,
            d_ff=128,
            d_k_v=64,
            n_heads=2,
            dropout=0.05,
        )
    elif arch == "lstm":
        model = FullModel_guidance_LSTM(
            input_dim=4,
            model_dim=128,
            tgt_size=5,
            n_declayers=2,
            d_ff=128,
            d_k_v=64,
            n_heads=2,
            dropout=0.05,
        )
    model.load_state_dict(torch.load(f"./model/{model_name}"))
    model.to(device)
    model.eval()

    os.makedirs("samples", exist_ok=True)

    sample_seqs_file = f"samples/{Path(output_file).stem}_samples.txt"
    noises_file = f"samples/{Path(output_file).stem}_noises.pt"

    samples = get_samples(feature, low, high, gen_num, input_file, use_saved_samples, sample_seqs_file, device)

    generated_seqs = []

    if not use_saved_samples:
        if args.feature == "one-hot":
            noises = [gaussian_noise((128,), mean=0.0, std=0.1) for _ in range(gen_num)]
        else:
            noises = [gaussian_noise(samples[0].size(), mean=0.0, std=0.1) for _ in range(gen_num)]
        torch.save(noises, f"samples/{Path(output_file).stem}_noises.pt")
    else:
        noises = torch.load(noises_file, map_location="cpu")

    assert len(samples) == len(noises), "Samples and noises must have the same length."

    for sample, noise in tqdm(zip(samples, noises), total=len(samples), desc="Generating sequences"):
        sample = sample.unsqueeze(0).to(device)
        noise = noise.unsqueeze(0).to(device)
        if arch == "gru":
            generated_seq = gru_predict_sequence(model, sample, 20, 0, True, noise, device)
        else:
            generated_seq = greedy_decode_guidance(model, sample, 20, 0, True, noise, device)
            generated_seq = generated_seq.squeeze().tolist()

        id_to_base = {1: "A", 2: "C", 3: "G", 4: "U"}

        rna_sequence = "".join([id_to_base.get(i, "") for i in generated_seq])
        generated_seqs.append(rna_sequence)

    with open(output_file, "w", buffering=1) as file2:
        for line in generated_seqs:
            file2.write(str(line) + "\n")


def main():
    global args
    parser = argparse.ArgumentParser(description="Choose which function to run.")
    parser.add_argument("model_name", type=str)
    parser.add_argument("input_file", type=str, default="")
    parser.add_argument("output_file", type=str, help="-----")
    parser.add_argument("low", type=int)
    parser.add_argument("high", type=int)
    parser.add_argument("gen_num", type=int)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--arch", type=str)
    parser.add_argument("--feature", type=str)
    parser.add_argument("--use_saved_samples", action="store_true", default=False)

    args = parser.parse_args()
    CUDA = args.cuda
    input_file = args.input_file
    output_file = args.output_file
    model_name = args.model_name
    low, high = args.low, args.high
    gen_num = args.gen_num
    use_saved_samples = args.use_saved_samples
    arch, feature, *_ = model_name.split("_")
    args.arch = arch
    args.feature = feature

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{CUDA}"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if use_saved_samples:
        print(f"Using saved samples from ./samples/{Path(output_file).stem}_samples.txt and noises from ./samples/{Path(output_file).stem}_samples.pt.")

    generation(
        model_name,
        input_file,
        output_file,
        low,
        high,
        gen_num,
        device,
        use_saved_samples=use_saved_samples,
    )


if __name__ == "__main__":
    main()

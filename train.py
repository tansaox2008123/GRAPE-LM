import os
import gc
import types
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import time
import numpy as np
from model import FullModel_guidance, FullModel_guidance_LSTM, FullModelGru, FullModelCNN
import fm
from evo import Evo
import argparse
from torchmetrics.regression import R2Score
from torchmetrics.functional import pearson_corrcoef
from tqdm import tqdm
import glob
from multimolecule import RnaTokenizer, RnaBertModel

def sigmoid(x, k):
    return 1 / (1 + np.exp(-k * x))


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


def get_act_score(cluster_reads, seq_reads):
    k = args.k

    cluster_socre = (sigmoid(cluster_reads, k) - 0.5) * 2.0
    seq_socre = (sigmoid(seq_reads, k) - 0.5) * 2.0

    return (cluster_socre + seq_socre) / 2


def load_cache(save_name, method):
    cache_files = glob.glob(f"./datasets/{args.dataset}/{save_name}_{method}_*-*.pt")
    cache_files.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("-")[0]))
    for file in cache_files:
        print(f"Loading {args.dataset} {method} {save_name} data from file {file}...")
        rna_reps = torch.load(file, map_location="cpu")
        yield standardization(rna_reps)


def run_rna_fm(rna_fm_inputs, device, save_name, batch_size=1000):
    cache_files = glob.glob(f"./datasets/{args.dataset}/{save_name}_RNA-FM_*-*.pt")
    if cache_files:
        expected_chunks_num = int(os.path.splitext(cache_files[0])[0].split("_")[-1].split("-")[-1]) + 1
        if expected_chunks_num != len(cache_files):
            for file in cache_files:
                print(f"Removing incomplete cache file: {file}")
                os.remove(file)
        else:
            return load_cache(save_name, "RNA-FM")

    print(f"Generating {args.dataset} RNA-FM {save_name} data...")

    rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    rna_fm_model.to(device)
    rna_fm_model.eval()

    all_reps = []
    batch_per_chunk: int = 50
    chunk = 0
    batch_num = (len(rna_fm_inputs) + batch_size - 1) // batch_size
    chunk_num = (batch_num + batch_per_chunk - 1) // batch_per_chunk
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
            if len(all_reps) == batch_per_chunk:
                rna_reps = torch.cat(all_reps, dim=0)
                del all_reps
                gc.collect()
                torch.save(
                    rna_reps,
                    f"./datasets/{args.dataset}/{save_name}_RNA-FM_{chunk}-{chunk_num - 1}.pt",
                )
                chunk += 1
                all_reps = []

            pbar.update(len(batch_inputs))

    if len(all_reps) > 0:
        rna_reps = torch.cat(all_reps, dim=0)
        del all_reps
        gc.collect()
        torch.save(
            rna_reps,
            f"./datasets/{args.dataset}/{save_name}_RNA-FM_{chunk}-{chunk_num - 1}.pt",
        )

    del rna_fm_model
    gc.collect()

    return load_cache(save_name, "RNA-FM")


def run_evo(evo_inputs, device, save_name, batch_size=1000):
    cache_files = glob.glob(f"./datasets/{args.dataset}/{save_name}_EVO_*-*.pt")
    if cache_files:
        expected_chunks_num = int(os.path.splitext(cache_files[0])[0].split("_")[-1].split("-")[-1]) + 1
        if expected_chunks_num != len(cache_files):
            for file in cache_files:
                print(f"Removing incomplete cache file: {file}")
                os.remove(file)
        else:
            return load_cache(save_name, "EVO")

    print(f"Generating {args.dataset} EVO {save_name} data...")

    evo_model = Evo("evo-1-8k-base")
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(device)
    model.eval()

    all_reps = []
    batch_per_chunk: int = 250
    chunk = 0
    batch_num = (len(evo_inputs) + batch_size - 1) // batch_size
    chunk_num = (batch_num + batch_per_chunk - 1) // batch_per_chunk
    with tqdm(total=len(evo_inputs), desc="EVO") as pbar:
        for i in range(0, len(evo_inputs), batch_size):
            batch_inputs = evo_inputs[i : i + batch_size]
            input_ids_tensor = torch.stack([torch.tensor(tokenizer.tokenize(seq), dtype=torch.int) for seq in batch_inputs]).to(device)
            with torch.no_grad():
                logits, _ = model(input_ids_tensor)

            logits = logits.detach().cpu().float()
            logits = logits.reshape(logits.shape[0], -1)
            all_reps.append(logits)

            if len(all_reps) == batch_per_chunk:
                rna_reps = torch.cat(all_reps, dim=0)
                del all_reps
                gc.collect()
                torch.save(
                    rna_reps,
                    f"./datasets/{args.dataset}/{save_name}_EVO_{chunk}-{chunk_num - 1}.pt",
                )
                chunk += 1
                all_reps = []

            pbar.update(len(batch_inputs))

    if len(all_reps) > 0:
        rna_reps = torch.cat(all_reps, dim=0)
        del all_reps
        gc.collect()
        torch.save(
            rna_reps,
            f"./datasets/{args.dataset}/{save_name}_EVO_{chunk}-{chunk_num - 1}.pt",
        )

    del evo_model
    gc.collect()

    return load_cache(save_name, "EVO")


def run_evo_2_7b(rna_fm_inputs, device, save_name, batch_size=20000):
    cache_files = glob.glob(f"./datasets/{args.dataset}/{save_name}_RNA-FM_*-*.pt")
    if cache_files:
        expected_chunks_num = int(os.path.splitext(cache_files[0])[0].split("_")[-1].split("-")[-1]) + 1
        if expected_chunks_num != len(cache_files):
            for file in cache_files:
                print(f"Removing incomplete cache file: {file}")
                os.remove(file)
        else:
            return load_cache(save_name, "RNA-FM")

    print(f"Generating {args.dataset} RNA-FM {save_name} data...")

    rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    rna_fm_model.to(device)
    rna_fm_model.eval()

    all_reps = []
    batch_per_chunk: int = 50
    chunk = 0
    batch_num = (len(rna_fm_inputs) + batch_size - 1) // batch_size
    chunk_num = (batch_num + batch_per_chunk - 1) // batch_per_chunk
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
            if len(all_reps) == batch_per_chunk:
                rna_reps = torch.cat(all_reps, dim=0)
                del all_reps
                gc.collect()
                torch.save(
                    rna_reps,
                    f"./datasets/{args.dataset}/{save_name}_RNA-FM_{chunk}-{chunk_num - 1}.pt",
                )
                chunk += 1
                all_reps = []

            pbar.update(len(batch_inputs))

    if len(all_reps) > 0:
        rna_reps = torch.cat(all_reps, dim=0)
        del all_reps
        gc.collect()
        torch.save(
            rna_reps,
            f"./datasets/{args.dataset}/{save_name}_RNA-FM_{chunk}-{chunk_num - 1}.pt",
        )

    del rna_fm_model
    gc.collect()

    return load_cache(save_name, "RNA-FM")



def run_rnabert(rna_bert_inputs, device, save_name, batch_size=5000):
    cache_files = glob.glob(f"./datasets/{args.dataset}/{save_name}_RNABERT_*-*.pt")
    if cache_files:
        expected_chunks_num = int(os.path.splitext(cache_files[0])[0].split("_")[-1].split("-")[-1]) + 1
        if expected_chunks_num != len(cache_files):
            for file in cache_files:
                print(f"Removing incomplete cache file: {file}")
                os.remove(file)
        else:
            return load_cache(save_name, "RNABERT")

    print(f"Generating {args.dataset} RNABERT {save_name} data...")

    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnabert")
    rna_bert_model = RnaBertModel.from_pretrained("multimolecule/rnabert").to(device)
    rna_bert_model.to(device)
    rna_bert_model.eval()

    all_reps = []
    batch_per_chunk: int = 50
    chunk = 0
    batch_num = (len(rna_bert_inputs) + batch_size - 1) // batch_size
    chunk_num = (batch_num + batch_per_chunk - 1) // batch_per_chunk
    with tqdm(total=len(rna_bert_inputs), desc="RNABERT") as pbar:
        for i in range(0, len(rna_bert_inputs), batch_size):
            batch_inputs = rna_bert_inputs[i : i + batch_size]

            input = tokenizer(batch_inputs, return_tensors="pt")
            input = {k: v.to(device) for k, v in input.items()}
            with torch.no_grad():
                encoder_outputs = rna_bert_model(**input)  # output is BaseModelOutput...
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_hidden_states = encoder_hidden_states[:, 1:-1, :]
            # hidden_states_numpy = encoder_hidden_states.detach().cpu().numpy()

            reps = encoder_hidden_states.reshape(encoder_hidden_states.shape[0], -1)
            all_reps.append(reps)
            if len(all_reps) == batch_per_chunk:
                rna_reps = torch.cat(all_reps, dim=0)
                del all_reps
                gc.collect()
                torch.save(
                    rna_reps,
                    f"./datasets/{args.dataset}/{save_name}_RNABERT_{chunk}-{chunk_num - 1}.pt",
                )
                chunk += 1
                all_reps = []

            pbar.update(len(batch_inputs))

    if len(all_reps) > 0:
        rna_reps = torch.cat(all_reps, dim=0)
        del all_reps
        gc.collect()
        torch.save(
            rna_reps,
            f"./datasets/{args.dataset}/{save_name}_RNABERT_{chunk}-{chunk_num - 1}.pt",
        )

    del rna_bert_model
    gc.collect()

    return load_cache(save_name, "RNABERT")

def run_ernie_rna(rna_ernie_inputs, device, save_name, batch_size=20000):
    cache_files = glob.glob(f"./datasets/{args.dataset}/{save_name}_Ernie-RNA_*-*.pt")
    if cache_files:
        expected_chunks_num = int(os.path.splitext(cache_files[0])[0].split("_")[-1].split("-")[-1]) + 1
        if expected_chunks_num != len(cache_files):
            for file in cache_files:
                print(f"Removing incomplete cache file: {file}")
                os.remove(file)
        else:
            return load_cache(save_name, "Ernie-RNA")

    print(f"Generating {args.dataset} Ernie-RNA {save_name} data...")

    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnaernie")
    rna_bert_model = RnaBertModel.from_pretrained("multimolecule/rnaernie").to(device)
    rna_bert_model.to(device)
    rna_bert_model.eval()

    all_reps = []
    batch_per_chunk: int = 50
    chunk = 0
    batch_num = (len(rna_ernie_inputs) + batch_size - 1) // batch_size
    chunk_num = (batch_num + batch_per_chunk - 1) // batch_per_chunk
    with tqdm(total=len(rna_ernie_inputs), desc="Ernie-RNA") as pbar:
        for i in range(0, len(rna_ernie_inputs), batch_size):
            batch_inputs = rna_ernie_inputs[i: i + batch_size]

            input = tokenizer(batch_inputs, return_tensors="pt")
            input = {k: v.to(device) for k, v in input.items()}
            with torch.no_grad():
                encoder_outputs = rna_bert_model(**input)  # output is BaseModelOutput...
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_hidden_states = encoder_hidden_states[:, 1:-1, :]

            reps = encoder_hidden_states.reshape(encoder_hidden_states.shape[0], -1)
            all_reps.append(reps)
            if len(all_reps) == batch_per_chunk:
                rna_reps = torch.cat(all_reps, dim=0)
                del all_reps
                gc.collect()
                torch.save(
                    rna_reps,
                    f"./datasets/{args.dataset}/{save_name}_Ernie-RNA_{chunk}-{chunk_num - 1}.pt",
                )
                chunk += 1
                all_reps = []

            pbar.update(len(batch_inputs))

    if len(all_reps) > 0:
        rna_reps = torch.cat(all_reps, dim=0)
        del all_reps
        gc.collect()
        torch.save(
            rna_reps,
            f"./datasets/{args.dataset}/{save_name}_Ernie-RNA_{chunk}-{chunk_num - 1}.pt",
        )

    del rna_bert_model
    gc.collect()

    return load_cache(save_name, "RNABERT")



def get_seqs_and_labels(file_path):
    true_seqs = []
    input_seqs = []
    act_scores = []
    with open(file_path, "r") as file:
        for line in file:
            true_numbers = rna_to_numbers(line.split()[-1])
            input_numbers = [0] + true_numbers[:-1]
            act_score = get_act_score(int(line.split()[0]), int(line.split()[1]))
            true_seqs.append(true_numbers)
            input_seqs.append(input_numbers)
            act_scores.append(act_score)

    input_seqs_np = torch.from_numpy(np.asarray(input_seqs)).to(torch.int64)
    true_seqs_np = torch.from_numpy(np.asarray(true_seqs)).to(torch.int64)
    act_scores_np = torch.from_numpy(np.asarray(act_scores)).to(torch.float32)

    return input_seqs_np, true_seqs_np, act_scores_np


def get_rna_reps(file_path, device):
    fname = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
    if args.feature == "rna-fm":
        rna_fm_inputs = [(i, line.split()[-1]) for i, line in enumerate(lines)]
        rna_reps = run_rna_fm(rna_fm_inputs, device, fname)
    elif args.feature == "evo":
        evo_inputs = [line.split()[-1] for line in lines]
        rna_reps = run_evo(evo_inputs, device, fname)
    elif args.feature == "evo-2-7b-base":
        evo_inputs = [line.split()[-1] for line in lines]
        rna_reps = run_evo_2_7b_base(evo_inputs, device, fname)
    elif args.feature == "evo-2-7b":
        evo_inputs = [line.split()[-1] for line in lines]
        rna_reps = run_evo_2_7b(evo_inputs, device, fname)
    elif args.feature == "RNABERT":
        evo_inputs = [line.split()[-1] for line in lines]
        rna_reps = run_rnabert(evo_inputs, device, fname)
    elif args.feature == "Ernie-RNA":
        evo_inputs = [line.split()[-1] for line in lines]
        rna_reps = run_ernie_rna(evo_inputs, device, fname)
    elif args.feature == "Evoflow":
        rna_reps = torch.stack([torch.from_numpy(rna_to_onehot(line.split()[-1])) for line in lines]).to(torch.float32)
    else:
        raise ValueError(f"Unsupported feature type: {args.feature}")
    return rna_reps


class MyDataset(Dataset):
    def __init__(self, file_path, device):
        self.reps = get_rna_reps(file_path, device)
        self.input_seqs, self.true_seqs, self.act_scores = get_seqs_and_labels(file_path)

        if isinstance(self.reps, types.GeneratorType):
            self.feats = list(self.reps)
            self.chunk_size = len(self.feats[0])
        else:
            self.feats = self.reps  # one-hot
            self.chunk_size = None

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        if self.chunk_size:
            feats = self.feats[idx // self.chunk_size][idx % self.chunk_size]
        else:
            feats = self.feats[idx]

        return feats, self.input_seqs[idx], self.true_seqs[idx], self.act_scores[idx]


def train_guidance_LLM(device):
    arch = args.arch
    feature = args.feature
    act_weight = args.act_weight
    batch_size = args.batch_size
    model_name = args.model_name
    dataset = args.dataset
    train_file = f"./datasets/{dataset}/train.txt"
    test_file = f"./datasets/{dataset}/test.txt"

    input_dim = {"rna-fm": 12800, "evo": 10240, "RNABERT": 2400, "Ernie-RNA" :15360, "one-hot": 80}

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

    tr_dataset = MyDataset(train_file, device)
    te_dataset = MyDataset(test_file, device)

    tr_dataloader = DataLoader(
        dataset=tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    te_dataloader = DataLoader(
        dataset=te_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0006)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        # lr = 0.0006,
        weight_decay=0.05,
        betas=(0.9,0.999),
        eps=1e-8
    )
    loss_func1 = nn.MSELoss()
    loss_func2 = nn.CrossEntropyLoss(ignore_index=0)
    r2score = R2Score()

    lowest_loss = float("inf")
    best_epoch = 0
    best_results = ""
    patience = 30
    epochs_no_improvement = 0
    best_weights = model.state_dict()
    model_saved = False

    fw = open("log/" + model_name + "_training_log.txt", "w", buffering=1)
    for epoch in range(400):
        start_t = time.time()
        tr_loss1_value = 0.0
        tr_loss2_value = 0.0
        tr_acc = 0.0
        tr_b_num = 0.0
        tr_r2 = 0.0
        tr_pearson = 0.0
        model.train()

        for data in tr_dataloader:
            inputs, input_seqs, true_seqs, labels = data
            inputs = inputs.to(device)
            input_seqs = input_seqs.to(device)
            true_seqs = true_seqs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, 1)
            bind_socres, pred_seqs = model(inputs, input_seqs)
            pred_seqs = torch.softmax(pred_seqs, -1)
            true_seqs = true_seqs.view(-1)
            pred_seqs = pred_seqs.view(true_seqs.shape[0], 5)

            loss1 = loss_func1(bind_socres, labels)
            loss2 = loss_func2(pred_seqs, true_seqs)
            pred_seqs = torch.argmax(pred_seqs, -1)

            tr_acc += torchmetrics.functional.accuracy(
                pred_seqs,
                true_seqs,
                task="multiclass",
                num_classes=5,
                ignore_index=0,
                average="micro",
            )

            loss = act_weight * loss1 + (1.0 - act_weight) * loss2
            tr_r2 += r2score(bind_socres, labels)
            tr_pearson += pearson_corrcoef(bind_socres, labels)
            tr_loss1_value += loss1.item()
            tr_loss2_value += loss2.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_b_num += 1.0

        te_loss1_value = 0.0
        te_loss2_value = 0.0

        te_acc = 0.0
        te_r2 = 0.0
        te_pearson = 0.0
        te_b_num = 0.0

        model.eval()

        with torch.no_grad():
            for data in te_dataloader:
                inputs, input_seqs, true_seqs, labels = data
                inputs = inputs.to(device)
                input_seqs = input_seqs.to(device)
                true_seqs = true_seqs.to(device)
                labels = labels.to(device)

                bind_socres, pred_seqs = model(inputs, input_seqs)
                pred_seqs = torch.softmax(pred_seqs, -1)
                labels = labels.float().view(-1, 1)

                true_seqs = true_seqs.view(-1)
                pred_seqs = pred_seqs.view(true_seqs.shape[0], 5)

                loss1 = loss_func1(bind_socres, labels)
                loss2 = loss_func2(pred_seqs, true_seqs)

                pred_seqs = torch.argmax(pred_seqs, -1)
                te_acc += torchmetrics.functional.accuracy(
                    pred_seqs,
                    true_seqs,
                    task="multiclass",
                    num_classes=5,
                    ignore_index=0,
                    average="micro",
                )
                te_r2 += r2score(bind_socres, labels)
                te_pearson += pearson_corrcoef(bind_socres, labels)

                te_loss1_value += loss1.item()
                te_loss2_value += loss2.item()
                te_b_num += 1.0

        end_t = time.time()
        results = f"Epoch: {epoch} | tr_loss1 = {(tr_loss1_value / tr_b_num):.4f} | tr_loss2 = {(tr_loss2_value / tr_b_num):.4f} | tr_r2 = {(tr_r2 / tr_b_num):.4f} | \
tr_pearson={(tr_pearson / tr_b_num):.4f} | tr_acc = {(tr_acc / tr_b_num):.4f} | \
te_loss1 = {(te_loss1_value / te_b_num):.4f} | te_loss2 = {(te_loss2_value / te_b_num):.4f} | \
te_r2 = {(te_r2 / te_b_num):.4f} | te_pearson = {(te_pearson / te_b_num):.4f} | te_acc = {(te_acc / te_b_num):.4f} | time = {(end_t - start_t):.2f}"
        fw.write(results + "\n")
        print(results)

        if ((te_loss1_value / te_b_num) * act_weight + (te_loss2_value / te_b_num) * (1 - act_weight)) < lowest_loss:
            lowest_loss = (te_loss1_value / te_b_num) * act_weight + (te_loss2_value / te_b_num) * (1 - act_weight)
            best_weights = deepcopy(model.state_dict())
            best_epoch = epoch
            best_results = results
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= patience:
            print(f"Early stop. Best epoch is {best_epoch}.Best results is :\n")
            print(best_results)
            torch.save(best_weights, "model/" + model_name)
            model_saved = True
            break

    if not model_saved:
        print(f"Best epoch is {best_epoch}. Best results is :\n")
        print(best_results)
        torch.save(best_weights, "model/" + model_name)

    fw.close()


def main():
    global args
    parser = argparse.ArgumentParser(description="Choose which function to run.")
    parser.add_argument("arch", type=str)
    parser.add_argument("feature", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--act_weight", type=float, default=0.5)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--k", type=float, default=0.01)

    args = parser.parse_args()

    if not args.model_name:
        args.model_name = f"{args.arch}_{args.feature}_{args.dataset}_{args.k}.model"
    else:
        args.model_name = f"{args.arch}_{args.feature}_{args.dataset}_{args.k}_{args.model_name}.model"

    CUDA = args.cuda

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{CUDA}"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_guidance_LLM(device)


if __name__ == "__main__":
    main()

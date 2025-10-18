# -*- coding: utf-8 -*-
import torch.utils.data as torch_data
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import time
import argparse
from model import *
from multimolecule import RnaTokenizer, RnaErnieModel
from torchmetrics.functional import pearson_corrcoef
import os
import copy

def sigmoid(x, k=0.05):
    return 1 / (1 + np.exp(-k * x))


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def convert_to_rna_sequence(data):
    rna_to_num = {'A': 1, 'C': 2, 'G': 3, 'U': 4}

    numbers = [rna_to_num.get(base, -1) for base in data.upper()]

    return numbers


def read_data_ernie(file_path, model, tokenizer, device):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    rnas = []
    input_seqs = []
    true_seqs = []
    bd_scores = []

    for line in lines:
        words = line.split()
        b = words[-1]

        rna_one_hot = convert_to_rna_sequence(b)

        values_list = list(rna_one_hot)

        values_list.insert(0, 0)

        input_seq = values_list[:-1]
        true_seq = values_list[1:]

        decimal_part_cluster = float(words[0])
        decimal_part_self = float(words[1])
        decimal_part = ((sigmoid(decimal_part_cluster, 0.001) - 0.5) * 2.0 * 0.95 +
                        (sigmoid(decimal_part_self, 0.001) - 0.5) * 2.0 * 0.05)

        sequence = b

        input = tokenizer(sequence, return_tensors="pt")
        input = {k: v.to(device) for k, v in input.items()}
        encoder_outputs = model(**input)  # output is BaseModelOutput...
        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_hidden_states = encoder_hidden_states[:, 1:-1, :]
        hidden_states_numpy = encoder_hidden_states.detach().cpu().numpy()


        rna_embedding = hidden_states_numpy.reshape(-1)
        rna_embedding = standardization(rna_embedding)

        rnas.append(rna_embedding)
        input_seqs.append(input_seq)
        true_seqs.append(true_seq)
        bd_scores.append(decimal_part)
    return rnas, input_seqs, true_seqs, bd_scores


def get_data_ernie(file_path, is_batch, device):
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnaernie")
    model = RnaErnieModel.from_pretrained("multimolecule/rnaernie").to(device)

    rnas, input_seqs, true_seqs, bd_scores = read_data_ernie(file_path, model, tokenizer, device)

    if is_batch:
        rnas1 = torch.tensor(rnas)
        input_seqs1 = torch.tensor(np.asarray(input_seqs))
        true_seqs1 = torch.tensor(np.asarray(true_seqs))
        bd_scores1 = torch.tensor(np.asarray(bd_scores))
    else:
        rnas1 = torch.tensor(rnas).to(device)
        input_seqs1 = torch.tensor(np.asarray(input_seqs)).to(device)
        true_seqs1 = torch.tensor(np.asarray(true_seqs)).to(device)
        bd_scores1 = torch.tensor(np.asarray(bd_scores)).to(device)

    return rnas1, input_seqs1, true_seqs1, bd_scores1


def train_guidance_Ernie(train_file, test_file, batch_size, model_name, device):
    tr_feats, tr_input_seqs, tr_true_seqs, tr_bd_scores = get_data_ernie(train_file, is_batch=True, device=device)
    te_feats, te_input_seqs, te_true_seqs, te_bd_scores = get_data_ernie(test_file, is_batch=True, device=device)

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

    model = FullModel_guidance(input_dim=15360,
                               model_dim=128,
                               tgt_size=5,
                               n_declayers=2,
                               d_ff=128,
                               d_k_v=64,
                               n_heads=2,
                               dropout=0.05)

    model = model.to(device)

    loss_func1 = nn.MSELoss()
    loss_func2 = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    w = 0.50
    best_loss1_loss2 = 10000.0
    temp = 10000.0
    patience = 10
    # no_improve_count = 0
    best_model = None

    best_temp = float('inf')
    best_pearson = -float('inf')
    no_improve_count = 0

    fw = open('log/' + model_name + '_training_log.txt', 'w')
    for epoch in range(400):
        start_t = time.time()
        loss1_value = 0.0
        loss2_value = 0.0
        acc2 = 0.0
        b_num = 0.0
        tr_total_pearson = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            inputs, input_seqs, true_seqs, labels = data
            inputs = inputs.to(device)
            input_seqs = input_seqs.to(device)
            true_seqs = true_seqs.to(device)
            labels = labels.to(device)

            labels = labels.float().view(-1, 1)
            bind_socres, pred_seqs = model(inputs, input_seqs)

            pred_seqs = torch.softmax(pred_seqs, -1)
            true_seqs = true_seqs.view(-1)
            pred_seqs = pred_seqs.view(true_seqs.shape[0], 5)

            loss1 = loss_func1(bind_socres, labels)
            loss2 = loss_func2(pred_seqs, true_seqs)
            pred_seqs = torch.argmax(pred_seqs, -1)

            pearson_corr = pearson_corrcoef(bind_socres.view(-1), labels.view(-1))
            tr_total_pearson += pearson_corr.item()

            acc2 += torchmetrics.functional.accuracy(pred_seqs,
                                                     true_seqs,
                                                     task="multiclass",
                                                     num_classes=5,
                                                     ignore_index=0,
                                                     average="micro")

            loss = w * loss1 + (1.0 - w) * loss2

            loss1_value += loss1.item()
            loss2_value += loss2.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b_num += 1.

        test_loss1_value = 0.0
        test_loss2_value = 0.0

        te_acc2 = 0.0
        test_b_num = 0.0
        te_pearson_corr = 0.0
        te_total_pearson = 0.0

        model.eval()
        for i, data in enumerate(test_loader):
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

            te_pearson_corr = pearson_corrcoef(bind_socres.view(-1), labels.view(-1))
            te_total_pearson += te_pearson_corr.item()

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
                                                                  te_acc2 / test_b_num,
                                                                  te_total_pearson / test_b_num), )
        print('Epoch:', '%04d' % (epoch + 1),
              '| tr_loss1 =', '{:.4f}'.format(loss1_value / b_num),
              '| tr_loss2 =', '{:.4f}'.format(loss2_value / b_num),
              '| tr_acc =', '{:.4f}'.format(acc2 / b_num),
              '| tr_pearson =', '{:.4f}'.format(tr_total_pearson / b_num),
              '| te_loss1 =', '{:.4f}'.format(test_loss1_value / test_b_num),
              '| te_loss2 =', '{:.4f}'.format(test_loss2_value / test_b_num),
              '| te_acc =', '{:.4f}'.format(te_acc2 / test_b_num),
              '| te_pearson =', '{:.4f}'.format(te_total_pearson / test_b_num),
              '| time =', '{:.2f}'.format(end_t - start_t)
              )
        now_temp = loss1_value / b_num + loss2_value / b_num
        now_pearson = te_total_pearson / test_b_num

        if now_temp < best_temp:
            no_improve_count = 0
            best_temp = now_temp
            best_model = copy.deepcopy(model)
            torch.save(best_model, f'model/{model_name}_best_loss.model')
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping triggered.")
                break

        if now_pearson > best_pearson:
            best_pearson = now_pearson
            best_pearson_model = copy.deepcopy(model)
            torch.save(best_pearson_model, f'model/{model_name}_best_pearson.model')


        torch.save(model, f'model/{model_name}_latest.model')

def main():
    parser = argparse.ArgumentParser(description="Choose which function to run.")
    parser.add_argument('function', choices=['1', '2', '3'], help="Function to run")
    parser.add_argument('--cuda', type=str, default="0", help="CUDA device ID (e.g., '0', '1', '2')")
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default="1000")

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda}'

    CUDA = args.cuda
    train_file = args.train_file
    test_file = args.test_file
    batch_size = args.batch_size
    model_name = args.model_name

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{CUDA}'

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.function == '1':
        train_guidance_Ernie(train_file, test_file, batch_size, model_name, device)


if __name__ == '__main__':
    main()


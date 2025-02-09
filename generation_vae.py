# -*- coding: utf-8 -*-
import os
from model_vae import *

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



# vae generation
def greedy_decode3(model):
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



if __name__ == '__main__':
    model = torch.load('model/2-ex-apt-VAE.model')
    model.eval()
    model = model.to(device)

    """model1的序列生成"""
    num1 = 10000
    random_rnas1 = []
    random_rnas2 = []

    for i in range(10000):
        random_seq2 = greedy_decode3(model)
        print(random_seq2)
        random_rnas2.append(random_seq2)

    output_file = '/home/sxtang/Graptor/vae_cd3e_generated_sequences.txt'

    with open(output_file, 'w') as file2:
        for line in random_rnas2:
            file2.write(str(line) + '\n')

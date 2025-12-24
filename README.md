# GRAPE-LM

# Introduction
GRAPE-LM (Generator of RNA Aptamers Powered by activity-guided Evolution and Language Model) is an AI-driven framework 
designed to accelerate RNA aptamer discovery and optimization. Aptamers are short nucleic acid sequences capable of 
binding specific targets with high affinity, making them valuable for applications in therapeutics, diagnostics, 
and synthetic biology.

Traditional aptamer selection methods, such as SELEX, often require multiple rounds of enrichment and are limited by in 
vitro conditions that do not fully capture intracellular interactions. GRAPE-LM overcomes these limitations by 
integrating a Transformer-based conditional autoencoder with nucleic acid language models (LMs). It is uniquely guided by 
Next-Generation Sequencing (NGS) enrichment data obtained from CRISPR/Cas-based intracellular screening (CRISmers), 
enabling biologically relevant and highly functional aptamer generation.

GRAPE-LM demonstrates superior performance compared to existing generative models by producing diverse, rational, and 
high-affinity aptamers with just a single round of intracellular screening. This has been validated across multiple 
targets, including human and viral proteins, highlighting its potential as a transformative tool in RNA evolution.

## Install dependencies
First, download the repository and install dependencies.

Python: 3.8.18

System: Ubuntu 22.04.4

```bash
   git clone https://github.com/tansaox2008123/GRAPE-LM.git
   pip install torch==2.0.1
   pip install flash-attn==2.5.6
   pip install -r requirements.txt
```

If you have any problem with install evo-model try this code 
```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Because of the complexity of the environment these dependencies only support to rna-fm and evo, other LMs need to go to 
their own github to deploy. And you can get more details in these websites：
RNA-FM https://github.com/ml4bio/RNA-FM;
Evo https://github.com/evo-design/evo;
RiNALMo https://github.com/lbcb-sci/RiNALMo;
RNAErnie https://github.com/CatIIIIIIII/RNAErnie;
RNABERT https://github.com/mana438/RNABERT;
Evoflow-RNA https://github.com/AtomBio/evoflow-rna.
RNABERT, RNAErnie, RiNALMo need to deploy multimolecule (https://huggingface.co/multimolecule) to run the training and generation code.

## Quickstart
Train your own model should follow this code
```bash
python train.py <arch> <feature> <dataset> \
    --cuda 0 \
    --act_weight 0.5 \
    --model_name mymodel \
    --batch_size 5000 \
    --k 0.001
```

For example
```bash
   python train.py base rna-fm RBD --cuda 0 --act_weight 0.5 --model_name RBD_test_model --batch_size 5000 --k 0.001
```

**Arguments:**

* `arch`: Model architecture
  * `base` – LM-based model
  * `cnn` – CNN-based model
  * `lstm` – LSTM-based model
* `feature`: Input representation type
  * `rna-fm` – RNA foundation model embeddings
  * `evo` – Genomic model embeddings
  * `one-hot` – One-hot encoding of RNA sequences
* `dataset`: Dataset name (folder under `datasets/`)

**Options:**

* `--cuda <id>`: GPU ID to use (default: `0`)
* `--act_weight <float>`: Weight of activity loss relative to sequence loss. For RBD, 0.85 is recommended, while for CD3ε and c-Myc, 0.5 is recommended.
* `--model_name <str>`: Name of the model checkpoint to save under `./model/`
* `--batch_size <int>`: Training batch size (default: `5000`)
* `--k <float>`: The regulatory factor for pseudo activity calculation needs to be optimized based on different target data. For RBD, 0.001 is recommended, while for CD3ε and c-Myc, 0.01 is recommended.

---

Generate RNA aptamers should follow this code
```bash
   python generation.py <model> <input_file> <output_file> <low> <high> <gen_num> --cuda 0

```

For example
```bash
   python generation.py base_rna-fm_RBD.model datasets/RBD/train.txt generated_sequences.txt 0 1000 50 --cuda 0 
```


**Arguments:**

* `model <str>`: Name of the trained model checkpoint (located in `./model/`, e.g., `base_rna-fm_RBD.model`)
* `input_file <str>`: Input seed file (e.g., `datasets/mydata/train.txt`)
* `output_file <str>`: Path to save the generated RNA sequences (e.g., `generated_sequences.txt`)
* `low <int>`: Lower bound index for sampling sequences from the seed file (e.g., `0`)
* `high <int>`: Upper bound index for sampling sequences  (e.g., `1000`)
* `gen_num <int>`: Number of sequences to generate (e.g., `50`)

**Options:**

* `--cuda <id>`: GPU ID to use (default: `0`)


---

Train with other language models should follow this code

```base

python train_<LM name>.py --cuda <cuda_id> --train_file <train_file> \
--test_file <test_file> --model_name <model_name> --batch_size <number> \
--weight <weight> --k <k>

```

For example
```base

python train_rna_bert.py --cuda 0 --train_file datasets/RBD/train.txt \
--test_file datasets/RBD/test.txt --model_name test_RBD_rna_bert --batch_size 1000 \
--weight 0.5 --k 0.001

```

**Arguments:**
* `--train_file <str>`: Training dataset file (e.g., `datasets/RBD/train.txt`)
* `--test_file <str>`: Test dataset file (e.g., `datasets/RBD/test.txt`)
* `--model_name <str>`: Name of the stored training model (e.g., `test_RBD_rna_bert`)
* `--batch_size <int>`: Number of sequences to batch_size (default: `1000`)
* `--weight <float>`: Weight of activity loss relative to sequence loss. For RBD, 0.85 is recommended, while for CD3ε and c-Myc, 0.5 is recommended.
* `--k <float>`: The regulatory factor for pseudo activity calculation needs to be optimized based on different target data. For RBD, 0.001 is recommended, while for CD3ε and c-Myc, 0.01 is recommended.
---

Generate RNA aptamers with other language models should follow this code

```base

python generation_other.py <function> --model_name <model_name> \
--cuda <cuda_id> --input_file <your_input_file> \
--output_file <your_output_file> --num <gen_num>

```
For example
```base

python generation_other.py rna-bert --model_name base_rna-bert_RBD.model \
--cuda 0 --input_file datasets/RBD/train.txt \
--output_file generated_sequences.txt --num 50

```

**Arguments:**
* `function <str>`: Which language model to chose to generation (e.g. `rna-bert`, `rna-ernie`, `rinalmo`) 
* `model_name <str>`: Name of the trained model checkpoint (e.g. `base_rna-bert_RBD.model`)
* `input_file <str>`: Input seed file (e.g., `datasets/RBD/train.txt`)
* `output_file <str>`: Path to save the generated RNA sequences (e.g., `generated_sequences.txt`)
* `gen_num <int>`: Number of sequences to generate  (e.g., `50`)

The original datasets and the trained checkpoints with other language models are stored on the following website:

```bash
https://drive.google.com/drive/folders/1cTFhEZJrLScKX-mEqJxUOp_MIEUc9dc1?usp=sharing
```
## Refernces
```bash
RNA-FM
@article{
  author={Shen, Tao and Hu, Zhihang and Sun, Siqi and Liu, Di and Wong, Felix and Wang, Jiuming and Chen, Jiayang and Wang, Yixuan and Hong, Liang and Xiao, Jin and others},
  title={Accurate RNA 3D structure prediction using a language model-based deep learning approach},
  journal={Nature Methods},
  year={2024}
}
Evo
@article{
   author = {Eric Nguyen and Michael Poli and Matthew G. Durrant and Brian Kang and Dhruva Katrekar and David B. Li and Liam J. Bartie and Armin W. Thomas and Samuel H. King and Garyk Brixi and Jeremy Sullivan and Madelena Y. Ng and Ashley Lewis and Aaron Lou and Stefano Ermon and Stephen A. Baccus and Tina Hernandez-Boussard and Christopher Ré and Patrick D. Hsu and Brian L. Hie },
   title = {Sequence modeling and design from molecular to genome scale with Evo},
   journal = {Science},
   year = {2024}
}
RiNALMo
@article{
  author={Penić, Rafael Josip and Vlašić, Tin and Huber, Roland G. and Wan, Yue and Šikić, Mile},
  title={RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks},
  journal={Nature Communications},
  year={2025}
}
RNAErnie
@Article{
  author={Wang, Ning and Bian, Jiang and Li, Yuchen and Li, Xuhong and Mumtaz, Shahid and Kong, Linghe and Xiong, Haoyi},
  title={Multi-purpose RNA language modelling with motif-aware pretraining and type-guided fine-tuning},
  journal={Nature Machine Intelligence},
  year={2024}
}
RNA-BERT
@Article{
  author={Akiyama, Manato, and Yasubumi Sakakibara},
  title={Informative RNA base embedding for RNA structural alignment and clustering by deep representation learning},
  journal={NAR genomics and bioinformatics},
  year={2022}
}
Evoflow-RNA
@Article{
  author={Patel S, Peng F Z, Fraser K, et al.},
  title={EvoFlow-RNA: Generating and Representing non-coding RNA with a Language Model},
  journal={bioRxiv},
  year={2025}
}
Zhang, J. et al. CRISmers NGS data used and generated by GRAPE-LM (1.0) [Data set]. Zenodo (2025). https://doi.org/10.5281/zenodo.18005327
```


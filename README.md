# GRAPE-LM

# Introduction
GRAPE-LM GRAPE-LM (Generator of RNA Aptamers Powered by activity-guided Evolution and Language Model) is an AI-driven framework designed to accelerate RNA aptamer discovery and optimization. Aptamers are short nucleic acid sequences capable of binding specific targets with high affinity, making them valuable for applications in therapeutics, diagnostics, and synthetic biology.

Traditional aptamer selection methods, such as SELEX, often require multiple rounds of enrichment and are limited by in vitro conditions that do not fully capture intracellular interactions. GRAPE-LM overcomes these limitations by integrating a Transformer-based conditional autoencoder with nucleic acid language models. It is uniquely guided by Next-Generation Sequencing (NGS) enrichment data obtained from CRISPR/Cas-based intracellular screening (CRISmers), enabling biologically relevant and highly functional aptamer generation.

GRAPE-LM demonstrates superior performance compared to existing generative models by producing diverse, rational, and high-affinity aptamers with just a single round of intracellular screening. This has been validated across multiple targets, including human and viral proteins, highlighting its potential as a transformative tool in RNA evolution.

## Install dependencies
First, download the repository and install dependencies.

Python: 3.8.18

System: Ubuntu 22.04.4

```bash
   git clone https://github.com/tansaox2008123/Grape.git
   pip install torch==2.0.1
   pip install flash-attn==2.5.6
   pip install -r requirements.txt
```

If you have any problem with install evo-model try this code 
```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Because of the complexity of the environment these dependencies only support to rna-fm and evo, other LM need to go to their own github to deploy. And you can get more details in these websites
RNA-FM https://github.com/ml4bio/RNA-FM.
Evo https://github.com/evo-design/evo
RiNALMo https://github.com/lbcb-sci/RiNALMo
RNAErnie https://github.com/CatIIIIIIII/RNAErnie
RNABERT https://github.com/mana438/RNABERT
Evoflow-RNA https://github.com/AtomBio/evoflow-rna
## Quickstart
Train your own model should follow this code
```bash
python train.py <arch> <feature> <dataset> \
    --cuda 0 \
    --act_weight 0.5 \
    --model_name myrun \
    --n 2
```

**Arguments:**

* `arch`: Model architecture
  * `base` – Transformer-based model
  * `cnn` – CNN-based model
  * `lstm` – LSTM-based model
* `feature`: Input representation type
  * `rna-fm` – Pre-trained RNA-FM embeddings
  * `evo` – Evolutionary model embeddings
  * `one-hot` – One-hot encoding of RNA sequences
* `dataset`: Dataset name (folder under `datasets/`)

**Options:**

* `--cuda <id>`: GPU ID to use (default: `0`)
* `--act_weight <float>`: Weight of activity loss relative to sequence loss
* `--model_name <str>`: Name of the model checkpoint to save under `./model/`
* `--batch_size <int>`: Training batch size
* `--n <int>`: Number of layers (only used for `base` architecture)

---

generation RNA aptamers should follow this code
```bash
   python generation.py base_rna-fm_RBD.model \
    datasets/mydata/rna_seq.txt \
    outputs/generated.txt \
    0 1000 50 \
    --cuda 0

```

**Arguments:**

* `model_name`: Name of the trained model checkpoint (located in `./model/`)
* `input_file`: Input dataset file (e.g., `datasets/mydata/train.txt`)
* `output_file`: Path to save the generated RNA sequences
* `low`: Lower bound index for sampling sequences from the input file
* `high`: Upper bound index for sampling sequences
* `gen_num`: Number of sequences to generate

**Options:**

* `--cuda <id>`: GPU ID to use (default: `0`)
* `--arch <str>`: Model architecture (`base`, `gru`, `cnn`, `lstm`)
* `--feature <str>`: Input feature type (`rna-fm`, `evo`, `one-hot`)

---

Trian with ohther LLM should follow this code

```base

python generation_other.py 1 --cuda <cuda_id> --train_file <input_file> \
--test_file <output_file> --model_name <model_name.model> --batch_size <nums>


* `train_file`: Input dataset file (e.g., `datasets/mydata/train.txt`)
* `test_file`: Path to save the generated RNA sequences
* `batch_size`: Number of sequences to batch_size
```

---

generation other LLM RNA aptamers should follow this code

```base

python generation_other.py <id> --cuda <cuda_id> --input_file <your_input_file>  \
--output_file <your_output_file> \
--model_name <model_name>  --num <gen_num>

* `id`: Which LLM to chose to generation, 1: RNA-BERT, 2: Ernie, 3: RiNALMo
* `model_name`: Name of the trained model checkpoint (located in `./model/`)
* `input_file`: Input dataset file (e.g., `datasets/mydata/train.txt`)
* `output_file`: Path to save the generated RNA sequences
* `gen_num`: Number of sequences to generate

```


The RBD and CD3e original dataset is stored on the following website.
```bash
https://drive.google.com/drive/folders/1cTFhEZJrLScKX-mEqJxUOp_MIEUc9dc1?usp=sharing
```
## Refernce
```bash
   RNA-FM
   @article{shen2024accurate,
  title={Accurate RNA 3D structure prediction using a language model-based deep learning approach},
  author={Shen, Tao and Hu, Zhihang and Sun, Siqi and Liu, Di and Wong, Felix and Wang, Jiuming and Chen, Jiayang and Wang, Yixuan and Hong, Liang and Xiao, Jin and others},
  journal={Nature Methods},
  pages={1--12},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
   Evo
   @article{nguyen2024sequence,
   author = {Eric Nguyen and Michael Poli and Matthew G. Durrant and Brian Kang and Dhruva Katrekar and David B. Li and Liam J. Bartie and Armin W. Thomas and Samuel H. King and Garyk Brixi and Jeremy Sullivan and Madelena Y. Ng and Ashley Lewis and Aaron Lou and Stefano Ermon and Stephen A. Baccus and Tina Hernandez-Boussard and Christopher Ré and Patrick D. Hsu and Brian L. Hie },
   title = {Sequence modeling and design from molecular to genome scale with Evo},
   journal = {Science},
   volume = {386},
   number = {6723},
   pages = {eado9336},
   year = {2024},
   doi = {10.1126/science.ado9336},
   URL = {https://www.science.org/doi/abs/10.1126/science.ado9336},
}
@article{penic2024_rinalmo,
  title={RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks},
  author={Penić, Rafael Josip and Vlašić, Tin and Huber, Roland G. and Wan, Yue and Šikić, Mile},
  journal={arXiv preprint arXiv:2403.00043},
  year={2024}
}
@Article{Wang2024,
author={Wang, Ning
and Bian, Jiang
and Li, Yuchen
and Li, Xuhong
and Mumtaz, Shahid
and Kong, Linghe
and Xiong, Haoyi},
title={Multi-purpose RNA language modelling with motif-aware pretraining and type-guided fine-tuning},
journal={Nature Machine Intelligence},
year={2024},
month={May},
day={13},
issn={2522-5839},
doi={10.1038/s42256-024-00836-4},
url={https://doi.org/10.1038/s42256-024-00836-4}
}
RNA-BERT
@Article{
Akiyama, Manato, and Yasubumi Sakakibara.
"Informative RNA base embedding for RNA structural alignment and clustering by deep representation learning."
NAR genomics and bioinformatics 4.1 (2022): lqac012.
}
Evoflow-RNA
@Article{
Patel S, Peng F Z, Fraser K, et al. EvoFlow-RNA: Generating and Representing non-coding RNA
with a Language Model[J]. bioRxiv, 2025: 2025.02. 25.639942.
}
```


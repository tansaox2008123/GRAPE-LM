# Grape

# Introduction
GRAPE (Generator of RNA Aptamers Powered by AI-Assisted Evolution) is an AI-driven framework designed to accelerate RNA aptamer discovery and optimization. Aptamers are short nucleic acid sequences capable of binding specific targets with high affinity, making them valuable for applications in therapeutics, diagnostics, and synthetic biology.

Traditional aptamer selection methods, such as SELEX, often require multiple rounds of enrichment and are limited by in vitro conditions that do not fully capture intracellular interactions. GRAPE overcomes these limitations by integrating a Transformer-based conditional autoencoder with nucleic acid language models. It is uniquely guided by Next-Generation Sequencing (NGS) enrichment data obtained from CRISPR/Cas-based intracellular screening (CRISmers), enabling biologically relevant and highly functional aptamer generation.

GRAPE demonstrates superior performance compared to existing generative models by producing diverse, rational, and high-affinity aptamers with just a single round of intracellular screening. This has been validated across multiple targets, including human and viral proteins, highlighting its potential as a transformative tool in RNA evolution.

## Create Environment with yml
First,download the repository and create the environment.
```bash
   git clone https://github.com/tansaox2008123/Grape.git
   conda env create -f environment.yml
```
And we need another environment with RNA-FM to get the RNA embedding to training new model you need go to 
https://github.com/ml4bio/RNA-FM, and follow the details to get the RNA-fm environment.

## Quickstart

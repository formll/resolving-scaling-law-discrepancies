# Resolving Discrepancies in Compute-Optimal Scaling of Language Models: Data and code

This repository contains the data and code to reproduce the figures and tables in the paper ["Resolving Discrepancies in Compute-Optimal Scaling of Language Models"](https://arxiv.org/abs/2406.19146), by Tomer Porian, Mithcell Wortsman, Jenia Jitsev, Ludwig Schmidt, and Yair Carmon.

## Folder structure
- `data/experiment_results.pickle.xz`: Contains the a pandas dataframe with the experiment results - see "data structure" below.
- `data/summary*`: Files containing the data analysis results (can be reproduced by running `make_paper.ipynb`).
- `data.py`: Functions to load and enrich the analysis data.
- `analysis.py`: Includes the main functions to analyze the results, such as fitting compute-optimal power laws.
- `plotting.py`: Functions to plot each sub figure in main plots will be stored here.
- `configs.py`: Contains the configurations for the experiments and plotting.
- `paper_figures.py`: Includes the code to generate the figures in the paper.
- `paper_tables.py`: Includes the code to generate the tables in the paper.
- `make_paper.ipynb`: Notebook to generate the figures and tables in the paper.
- `requirements.txt`: Required packages to run the code.

## Data structure
We provide a pandas dataframe with the results of the experiments. For compactness, we extract the relavant information from the raw logs. Each row represents a separate training run.

### Important columns
- `hparams`: Whether we used tuned hyperparameters or not.
- `warmup`: Whether we used a long learning rate warmup or not.
- `decay`: Decay settings applied during training - either constant, Chinchilla decay or cosine decay with long total budget.
- `dataset`: Either RefinedWeb or OpenWebText2 - the two datasets used in the experiments.
- `train/loss`: Each entry in this column is the logged training loss.
- `val/loss`: Similarly, each entry in this column is the logged validation loss. 

### Additional configurations
- `width`
- `depth`
- `val/loss_std`
- `train/batch_time`
- `train/data_time`
- `train/lr`
- `warmup_tokens`
- `beta1`
- `beta2`
- `world_size`
- `lr`
- `seed`
- `seq_len`
- `vocab_size`
- `grad_clip_norm`
- `optimizer`
- `precision`
- `qk_norm`
- `z_loss_coefficient`
- `bs`
- `independent_wd`


## Getting started
Run `make_paper.ipynb` to reproduce the experiments.

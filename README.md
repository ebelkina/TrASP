# TrASP: Transformer for Activity Suffix Prediction

**Investigating out-of-domain generalization in predictive process monitoring using a GPT-2-style Transformer.**

TrASP is a decoder-only Transformer developed for **activity suffix prediction (ASP)**: predicting the remaining sequence of activities in a business process from the activities observed so far. This research project studies how well such a model generalizes when trained on a single process, multiple processes, or processes different from the one used for evaluation.

The repository accompanies the bachelor's project report **_Towards Out-of-Domain Generalization in Predictive Process Monitoring Using Transformers_** by **Elena A. Belkina**, Vrije Universiteit Amsterdam.

![TrASP Research Poster](poster.png)

## Overview

Given an observed activity prefix, the goal is to predict the rest of the process trace:

```text
Observed prefix:   [apply, check]
Predicted suffix:  [approve, close]
Complete trace:    [apply, check, approve, close]
```

TrASP treats activities as discrete tokens and generates the suffix autoregressively. The central research question is how its predictive performance changes across three settings:

| Setting | Training | Evaluation |
| --- | --- | --- |
| **In-single-domain (ISD)** | One event log | Held-out cases from the same log |
| **In-multi-domain (IMD)** | Multiple event logs | Held-out cases from the training domains |
| **Out-of-domain (OOD)** | One or more source domains | An unseen target domain |

The experiments also examine the effect of **overlapping versus non-overlapping activity vocabularies**. Real-world logs have largely distinct activity labels across domains, whereas the synthetic logs share activity labels despite representing different process behavior.

## Model and approach

TrASP is implemented from scratch in PyTorch. Its architecture includes:

- Learned activity-token and positional embeddings.
- Stacked, GPT-2-style decoder blocks with causal multi-head self-attention, pre-layer normalization, feed-forward networks, residual connections, and dropout.
- Autoregressive next-activity prediction using a shared activity vocabulary.
- Two suffix-generation modes: generate until the end-of-sequence (`<EOS>`) token or generate using a supplied ground-truth suffix length for analysis.

```mermaid
flowchart LR
    A["Event log<br/>Ordered activity traces"] --> B["Preprocessing<br/>Activity IDs + EOS"]
    B --> C["TrASP<br/>Causal Transformer"]
    C --> D["Autoregressive<br/>Suffix prediction"]
    D --> E["Sequence<br/>Similarity evaluation"]
```

Training traces are flattened into a token stream with `<EOS>` between cases. The model learns by predicting the next token in randomly sampled fixed-length windows. At evaluation time, an individual test trace is split into a prefix and a ground-truth suffix, and the model generates its predicted suffix.

For the main study, activity labels are collected into a fixed vocabulary across all experimental domains, including target domains. This is a **research assumption**, not a solution for genuinely unknown activity labels encountered after deployment.

## Datasets

The project uses two groups of event logs:

| Data | Event logs | Why they are included |
| --- | --- | --- |
| Real-world, non-overlapping labels | BPIC17 Offer (`BPIC17Of`), BPIC19 (`BPIC19f`), BPIC20 Request for Payment (`BPIC20R`), Helpdesk | Evaluate within-domain and cross-domain behavior when processes have different activity vocabularies. |
| Synthetic, overlapping labels | 100 synthetic process logs | Evaluate the effect of shared activity names across distinct process behaviors. |

For the principal real-data OOD experiment, TrASP is trained on **BPIC17Of + BPIC19f + BPIC20R** and tested on **Helpdesk**, which is excluded from training. The processed training, validation, and test files are stored as CSVs in [`data_processed/`](data_processed/). The repository also contains the corresponding vocabulary JSON files. The notebook [`experiments.ipynb`](experiments.ipynb) contains exploratory analysis and experimental work.

Original real-world datasets: [BPIC17 Offer Log](https://doi.org/10.4121/12705737.v2), [BPIC19](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1), [BPIC20 Request for Payment](https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030), and [Helpdesk](https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb). Consult the report for dataset selection and preprocessing decisions, including filtering and downsampling.

## Key findings

The following are **selected results as reported in the accompanying study**. Scores are percentages, with higher values indicating greater similarity between predicted and true suffixes.

| Experimental setting | Test data | Reported similarity (%) |
| --- | --- | ---: |
| ISD | BPIC17Of | 60.58 |
| ISD | BPIC19f | 74.95 |
| ISD | BPIC20R | 88.22 |
| ISD | Helpdesk | 85.91 |
| IMD, trained on three real-world logs | Combined in-domain test data | 77.23 |
| OOD, trained on three real-world logs | Unseen Helpdesk, generation stops at EOS | 0.00 |
| OOD, trained on three real-world logs | Unseen Helpdesk, true suffix length provided | 12.51 |

The results highlight three observations:

1. **Within a single domain**, TrASP obtained substantial suffix-prediction performance on the selected real-world logs.
2. **Across known domains**, combining logs with distinct activity vocabularies had little impact on performance, while the synthetic experiments suggested that reusing the same activity names across different processes can introduce ambiguity.
3. **In an unseen domain**, the model struggled. Providing the true suffix length improved its score but did not solve cross-domain generalization.

Exploratory experiments also examined zero-shot predictions with GPT-4o (reported mean score: **23.75** on 100 Helpdesk prefixes with the true suffix lengths supplied) and semantic clustering of activity names using sentence embeddings and K-means. These are analyses of possible research directions, not integrated solutions to the OOD problem.

## Repository structure

```text
TrASP/
├── train_model.py          # Transformer definition and training CLI
├── evaluate_model.py      # Evaluate one saved checkpoint
├── evaluate_models.py     # Run evaluations specified in a CSV plan
├── experiments.ipynb      # Notebook-based experiments and analysis
├── requirements.txt       # Python dependencies
├── data_processed/        # Processed datasets and vocabulary files
├── evaluation/            # Evaluation plan and evaluation artifacts
├── outputs/               # Training logs and saved checkpoints
└── README.md
```

## Getting started

### 1. Install dependencies

The study used **Python 3.12.3**, **PyTorch 2.6.0**, and **CUDA 12.6**. A CUDA-capable GPU is recommended for research-sized training, although the training script also detects and supports a CPU.

```bash
git clone https://github.com/ebelkina/TrASP.git
cd TrASP

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The supplied `requirements.txt` contains the project's recorded dependencies. For GPU execution, ensure that your installed PyTorch build and CUDA environment are compatible. The research runs were performed on the DAS-5 cluster; runtime and results may vary with hardware and software versions.

### 2. Train a model

The training entry point is [`train_model.py`](train_model.py), which accepts command-line options through [Python Fire](https://github.com/google/python-fire).

For example, this command runs a **small demonstration** on the processed BPIC20R data:

```bash
python train_model.py \
  --name=BPIC20R \
  --vocab=vocab_real \
  --idx2label=idx2label_real \
  --total_num_steps=200 \
  --validate_x_times=4 \
  --batch_size=8 \
  --max_prefix_len=32 \
  --max_suffix_len=32 \
  --emb_dim=32 \
  --num_heads=4 \
  --depth=2 \
  --lr_warmup=100
```

This is an illustrative quick run, **not a reproduction of the published experiments**. The script reads `data_processed/BPIC20R_train.csv`, `data_processed/BPIC20R_val.csv`, and the selected vocabulary JSON files. It writes a run log and model checkpoints into `outputs/`. Checkpoints are named according to the run timestamp, dataset, and saved step, for example:

```text
outputs/<run_timestamp>_BPIC20R_model_step_200.pt
```

For the main experiments, the report specifies 60,000 training steps, batch size 32, embedding dimension 128, eight attention heads, four Transformer blocks, dropout 0.1, and a maximum sequence length of 512. See the report's experimental setup and check the script's parameter semantics when configuring a reproduction: its `hidden_size` argument is a **multiplier** of the embedding dimension for the feed-forward layer, rather than the layer width itself.

Weights & Biases logging is optional through `--wb_log=True`. **Before enabling it on your own machine**, update the cluster-specific W&B cache and data paths hard-coded in `train_model.py`.

### 3. Evaluate a checkpoint

Use [`evaluate_model.py`](evaluate_model.py) to evaluate a trained model on a processed test CSV. Replace the checkpoint placeholder with an actual path from `outputs/`:

```bash
python evaluate_model.py \
  --checkpoint_path=outputs/YOUR_CHECKPOINT.pt \
  --test_dataset_path=data_processed/BPIC20R_test.csv \
  --device=cpu \
  --n_samples=100 \
  --verbose_generated=5 \
  --stop_at_eos=True
```

When `stop_at_eos=True`, generation terminates at `<EOS>` or the configured maximum suffix length. Setting it to `False` runs the research-only condition that supplies the true suffix length. Evaluation prints the mean normalized edit similarity, standard deviation, and an approximate 95% confidence interval.

### 4. Run the evaluation plan

[`evaluate_models.py`](evaluate_models.py) reads the experiment specification in [`evaluation/evaluation_plan.csv`](evaluation/evaluation_plan.csv) and writes aggregate results and a log. It expects the **specific named checkpoints and processed test datasets in that plan** to be available locally.

```bash
python evaluate_models.py \
  --input_csv=evaluation/evaluation_plan.csv \
  --results_csv=evaluation/evaluation_results.csv \
  --log_path=evaluation/evaluation_log.txt \
  --device=cpu \
  --n_samples=1000
```

The CSV plan determines the `<EOS>` setting for individual runs. You can create a smaller plan with your own checkpoint names and test datasets. The required columns are `model`, `test`, and, when needed, `stop at EOS`; the optional descriptive columns used for the output summary are `generalization type`, `data type`, and `test data name`. Values in `model` and `test` are file basenames **without** the `.pt` or `.csv` extension.

import copy
import json
import math
import os
import sys
import time
import random
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import tqdm
import fire
from datetime import datetime
import torch.distributions as distrib
import wandb
import ast
from transformers import get_scheduler

from collections import Counter

"""
Transformer-based sequence model for activity suffix prediction in process mining.
Trains a character-level language model using masked self-attention with configurable depth and logging to Weights & Biases.
"""

# Set random seeds
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def encode_column_to_tensor(df, column_name, vocab):
    """
    Converts a column of lists of strings to lists of vocab indices and flattens into a tensor.
    :param df (pd.DataFrame): The dataframe containing the column.
    :param column_name (str): The name of the column to encode (e.g., "activities").
    :param vocab (dict): Mapping from string to integer index.
    :return df (pd.DataFrame): Updated dataframe with new column `<column_name>_idx`.
    :return flat_tensor (torch.Tensor): Flattened tensor of all encoded sequences.
    """
    idx_col = f'{column_name}_idx'
    df[idx_col] = df[column_name].apply(
        lambda seq: [vocab.get(token, vocab.get('<UNK>', 1)) for token in seq]
    )
    flat_tensor = torch.tensor(
        [idx for sublist in df[idx_col] for idx in sublist], dtype=torch.long
    )
    return df, flat_tensor


def sample_from_log_probab(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    from https://codeberg.org/pbm/former/src/commit/7177a350d0ad16d219670a0ac04e205f5d6e5bff/former/util/util.py#L10

    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()
    p = F.softmax(lnprobs / temperature, dim=0)
    cd = distrib.Categorical(p)
    return cd.sample()

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation
    from https://codeberg.org/pbm/former/src/commit/7177a350d0ad16d219670a0ac04e205f5d6e5bff/former/util/util.py#L10

    :param tns:
    :return:
    :patram matrices (Tensor): A tensor representing a batch of square matrices,
                           where the last two dimensions are the matrix dimensions.
    :param maskval (float): The value to assign to masked elements. Default is 0.0.
    :param mask_diagonal (bool): Whether to include the diagonal in the mask (i.e., mask i == j).
                              If False, only mask strictly upper triangle (i < j).
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval


class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.

    Adapted from https://codeberg.org/pbm/former/src/commit/7177a350d0ad16d219670a0ac04e205f5d6e5bff/former/modules.py
    """

    def __init__(self, emb: int, heads: int=8, mask: bool=True, scalefactor=None):
        """
        Initializes the multi-head self-attention module.
        :param emb: The dimension of the input and output vectors.
        :param heads: The number of heads (parallel executions of the self-attention)
        :param mask: Whether to apply an autoregressive mask.
        :param scalefactor (float or None): Multiplier for the attention weights. If none, the default `1/sqrt(emb/heads)` is used,
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads # Size of each head's subspace

        # Linear projections for keys, queries, and values
        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        # Final linear layer to unify heads
        self.unifyheads = nn.Linear(emb, emb)

        # Scaling factor for dot-product attention
        self.scalefactor = 1/math.sqrt(emb // heads) if scalefactor is None else scalefactor

    def forward(self, x):
        """
        Computes self-attention over input tensor `x`.

        :param x (Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
        :return Tensor: Output tensor of the same shape as input.
               """
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        # Compute key, query, and value projections on the whole embedding vectors
        keys    = self.tokeys(x)      # (b, t, e)
        queries = self.toqueries(x)   # (b, t, e)
        values  = self.tovalues(x)    # (b, t, e)

        # Break the embedding into `heads` chunks to feed each to a different attention head
        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        # Transpose to (batch, heads, time, subdim) and merge batch & head
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries
        keys    = keys

        # Compute scaled dot-product attention
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot * self.scalefactor

        assert dot.size() == (b*h, t, t)

        # Apply autoregressive mask (prevent attending to future tokens)
        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # Convert to attention weights (to row-wise self-attention probabilities)
        dot = F.softmax(dot, dim=2)

        # Apply attention weights to values and reshape: (batch, heads, time, subdim)
        out = torch.bmm(dot, values).view(b, h, t, s)

        # Transpose and combine heads back: (batch, time, embedding)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, seq_length, hidden_size=4, dropout=0.0,
                 pos_embedding=None):
        super().__init__()
        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, hidden_size * emb),
            nn.ReLU(),
            nn.Linear(hidden_size * emb, emb)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # GPT-style: Pre-LayerNorm
        # print('tbloc input', x.shape)

        normed_x1 = self.norm1(x)
        # print('after norm1 (before attention)', normed_x1.shape)
        attended = self.attention(normed_x1)
        # print('attention output (attended)', attended.shape)
        # print('attention output (x)', x.shape)
        attended = self.dropout(attended)
        # print('after attended = self.dropout(attended)', attended.shape)
        x = x + attended
        # print('after x = x + attended', x.shape)

        normed_x2 = self.norm2(x)
        # print('after norm2 (before ff)', normed_x2.shape)
        fforward = self.ff(normed_x2)
        # print('after fforward = self.ff(normed_x2)', fforward.shape)
        fforward = self.dropout(fforward)
        # print('after fforward = self.dropout(fforward)', fforward.shape)
        x = x + fforward
        # print('after x = x + fforward', x.shape)

        return x

class GenTransformer(nn.Module):
    """
    Transformer model for Activity Suffix Prediction (ASP) or sequence generation.
    It predicts the most likely sequence of future activities (suffix) based on a given prefix.
    """

    def __init__(self, emb_dim, num_heads, depth, max_prefix_len, vocab, hidden_size, dropout, device):
        """
        Initializes the Transformer for activity suffix prediction.

        :param emb_dim (int): Dimension of token and position embeddings.
        :param num_heads (int): Number of attention heads in each Transformer block.
        :param depth (int): Number of stacked Transformer blocks.
        :param max_prefix_len (int): Maximum length of the prefix sequence.
        :param vocab (int): Number of unique activities (vocabulary size).
        :param hidden_size (int): Hidden dimension in the feedforward network of each block.
        :param dropout (float): Dropout probability in attention and feedforward layers.
        :param device (torch.device): Device to use for computations.
        """
        super().__init__()
        self.vocab = vocab  # Total number of tokens in the vocabulary
        self.device = device

        # Embeds discrete activity tokens into dense vectors
        self.token_embedding = nn.Embedding(embedding_dim=emb_dim, num_embeddings=vocab)  # token indices to dense vectors

        # Positional embeddings to preserve order of activities in prefix
        self.pos_embedding = nn.Embedding(embedding_dim=emb_dim, num_embeddings=max_prefix_len)

        # Stack of autoregressive Transformer blocks
        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=emb_dim,
                    heads=num_heads,
                    seq_length=max_prefix_len,
                    mask=True,  # Prevent attention to future positions (causal masking)
                    pos_embedding=self.pos_embedding,
                    hidden_size=hidden_size,
                    dropout=dropout
                )
            )

        self.tblocks = nn.Sequential(*tblocks)  # Combine blocks into a single module

        # Final linear projection to map each hidden state to a vocabulary-sized vector (logits)
        self.toprobs = nn.Linear(emb_dim, vocab)

    def forward(self, x):
        """
        Forward pass of the model.

        :param x (Tensor): Tensor of shape (batch_size, sequence_length) with activity indices (prefixes).
        :return (Tensor): Log-probabilities over the activity vocabulary for each position in the sequence.
                    Shape: (batch_size, sequence_length, vocab_size)
        """
        # Embed activity tokens
        tokens = self.token_embedding(x)  # (b, t, e)
        b, t, e = tokens.size()

        # Add positional embeddings to token embeddings (b, t, e)
        positions = self.pos_embedding(torch.arange(t, device=x.device))[None, :, :].expand(b, t, e)
        x = tokens + positions  # Combined input embeddings

        # Stacked Transformer blocks
        x = self.tblocks(x)   # (b, t, e)

        # Reshape to (b*t,e) >>
        # project to vocabulary size (b*t,vocab) >>
        # reshape back to vocabulary logits (b,t,vocab)
        x = self.toprobs(x.view(b*t, e)).view(b, t, self.vocab) # (b, t, vocab)

        # Convert logits to log-probabilities
        return F.log_softmax(x, dim=2)  # (b,t,vocab)


class WandbLogger:
    """
    Custom logger that duplicates stdout to Weights & Biases (wandb).
    Captures standard output and logs it to wandb in real-time.
    """
    def __init__(self):
        """
        :param self.terminal (IO): Stores the original sys.stdout.
        :param self.log_buffer (str): Buffer for accumulating log lines before flushing to wandb.
        """
        self.terminal = sys.stdout
        self.log_buffer = ""

    def write(self, message):
        """
        Writes a message to both the terminal and wandb log.
        :param message (str): The message to write (usually a line from print
        """
        self.terminal.write(message)
        self.terminal.flush()
        self.log_buffer += message
        if "\n" in message:
            wandb.log({"stdout": self.log_buffer})
            self.log_buffer = ""

    def flush(self):
        """
        Ensures any buffered output is written to the terminal (required for compatibility).
        """
        self.terminal.flush()


# ---------- Experiment ----------
def run_experiment(
    name='BPIC20R',
    vocab = "vocab_real", # "vocab_real", "vocab_synthetic"
    idx2label = 'idx2label_real', # 'idx2label_real', 'idx2label_synthetic'
    batch_size=32,  # 32, 64, 128
    total_num_steps=1000, #100_000
    validate_x_times=10,  # 1_000,
    max_prefix_len=10,  # 512
    max_suffix_len=10,  # 512
    lr_max=0.0001,
    lr_warmup=1000,# 5000,
    gradient_clipping=1.0,
    emb_dim=12, #128,  # 12,#128, # 10
    num_heads=1,  # 3,#8,
    depth=1,  # 8, 12,          # nr transformer blocks
    hidden_size=4,
    dropout=0.1,
    seed=42,
    verbose_generated=10,
    verbose_shapes=False,
    wb_log=False,  # todo
):
    """
    Run a full training loop for activity suffix prediction using a Transformer model.

    :param name (str): Name of the dataset to load (used to locate train/val CSVs).
    :param vocab (str): File name (without extension) of the vocabulary JSON file.
    :param idx2label (str): File name (without extension) of the idx2label JSON file.
    :param batch_size (int): Number of sequences per training batch.
    :param total_num_steps (int): Total number of training iterations.
    :param validate_x_times (int): Number of validation points throughout training.
    :param max_prefix_len (int): Maximum length of input (prefix) sequences.
    :param max_suffix_len (int): Maximum length of predicted output (suffix) sequences.
    :param lr_max (float): Maximum learning rate.
    :param lr_warmup (int): Number of steps for learning rate warm-up.
    :param gradient_clipping (float): Max norm value for gradient clipping.
    :param emb_dim (int): Dimensionality of the token embeddings.
    :param num_heads (int): Number of attention heads in each Transformer block.
    :param depth (int): Number of stacked Transformer blocks.
    :param hidden_size (int): Hidden size of the feed-forward layer in each block.
    :param dropout (float): Dropout rate used in Transformer layers.
    :param seed (int): Random seed for reproducibility.
    :param verbose_generated (int): How many generated suffixes to show per validation.
    :param verbose_shapes (bool): Whether to print shape of tensors for debugging.
    :param wb_log (bool): Whether to enable logging to Weights & Biases (wandb).
    """

    # Set up experiment metadata and logging
    timestamp = datetime.now().strftime("%m%d_%H%M")  # Unique run timestamp
    script_name = os.path.basename(__file__).replace('.py', '')  # Get the script name (without path)
    os.makedirs('outputs', exist_ok=True) # Create output directory if not exists

    set_seed(42) # Set random seeds for reproducibility

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = locals() | {"device": DEVICE} # Store all hyperparameters and device config in a dict for logging

    # Construct run name using timestamp and key settings
    RUN_NAME = f"{timestamp}_{name}_bs{batch_size}_lr{lr_max}_emb{emb_dim}_h{num_heads}_dep{depth}" \
               f"_pref{max_prefix_len}_suf{max_suffix_len}_dr{dropout}"

    # Initialize wandb logging if enabled
    if wb_log:
        os.environ["WANDB_CACHE_DIR"] = "/var/scratch/ebe268/wandb_cache"  #todo
        os.environ["WANDB_DIR"] = "/var/scratch/ebe268/wandb_local"
        os.environ["WANDB_DATA_DIR "] = "/var/scratch/ebe268/wandb_data"
        wandb.init(project=f"TrASP",
                   name=RUN_NAME,
                   config=config)

    # Redirect stdout to a log file
    sys.stdout = open(f'outputs/{RUN_NAME}@{script_name}.txt', 'w', encoding='utf-8')
    if wb_log:
        sys.stdout = WandbLogger()  # Custom stdout handler that also sends output to wandb
    # sys.stderr = sys.stdout     # todo (prints to WandbLogger)

    # Decide how often to validate based on number of steps
    validate_every = max(1, total_num_steps // validate_x_times)  # avoid division by zero

    print('================== HYPERPARAMETERS ==================')
    for key, value in locals().items():
        print(f"{key.ljust(20)} : {value}")

    start_time = time.time()

    # ---------- Load data ----------
    print('\n================== LOAD AND SPLIT DATA ==================')

    # # Load preprocessed training and validation data with column 'activities'
    df_train_cases = pd.read_csv(f'data_processed/{name}_train.csv')
    df_val_cases = pd.read_csv(f'data_processed/{name}_val.csv')

    # Decode activity sequences stored as strings back to lists
    df_train_cases['activities'] = df_train_cases['activities'].apply(ast.literal_eval) # todo use .pkl
    df_val_cases['activities'] = df_val_cases['activities'].apply(ast.literal_eval)
    print(df_train_cases['activities'][:3])  # Show first few examples

    # Load vocabulary: maps activity labels to token indices
    with open(f'data_processed/{vocab}.json', 'r') as f:
        vocab = json.load(f)
        # Ensure keys are strings, and values are integers
        vocab = {str(k): int(v) for k, v in vocab.items()}

    # Load reverse mapping: maps token indices to labels (for decoding) # todo not necessary
    with open(f'data_processed/{idx2label}.json', 'r') as f:
        idx2label = json.load(f)
        # Convert keys to integers, values stay as strings
        idx2label = {int(k): v for k, v in idx2label.items()}

    # Convert activity sequences into tensors of indices
    df_train_cases, data_train = encode_column_to_tensor(df_train_cases, 'activities', vocab)
    df_val_cases, data_val = encode_column_to_tensor(df_val_cases, 'activities', vocab)

    print('\nSHAPES of data_train, data_val:', data_train.shape, data_val.shape)
    print('\ndata_train[:20]\n', data_train[:20].tolist())

    # ========== Model Setup ==========

    # Initialize the Transformer model for activity suffix prediction
    model = GenTransformer(
        emb_dim=emb_dim,
        num_heads=num_heads,
        depth=depth,
        max_prefix_len=max_prefix_len,
        vocab=len(vocab) + 5, # Add extra tokens for padding or special tokens#
        hidden_size=hidden_size,
        dropout=dropout,
        device=DEVICE,
    ).to(DEVICE)

    # Enable multi-GPU training if available
    if DEVICE == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)  # todo

    # Watch the model in wandb for gradient and parameter tracking
    if wb_log: wandb.watch(model)

    # Set up the optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)
    warmup_steps = int(lr_warmup / batch_size)
    scheduler = get_scheduler(
        name="cosine",
        # name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_num_steps
    )

    print('\n================== START TRAINING ==================')
    start_training_time = time.time()
    losses = []

    for step in tqdm.trange(1, total_num_steps+1, desc="Training        "):
        model.train()
        optimizer.zero_grad()

        # Randomly sample a batch of random sequence slices from training data
        starts = torch.randint(0, data_train.size(0) - max_prefix_len - 1, (batch_size,))       # (batch_size,) random start indices from the training corpus
        seqs = torch.stack([data_train[start:start + max_prefix_len + 1] for start in starts])  # (batch_size, max_prefix_len + 1) input + target sequence

        # Split into prefix and next-token targets
        source, target = seqs[:, :-1].long(), seqs[:, 1:].long()   # both: (batch_size, max_prefix_len)

        if verbose_shapes:
            print(f'\n---- training for step={step} ----')
            print(f'\nSTARTS\t{starts.shape}\n{starts}')
            print(f'\nSEQS\t{seqs.shape}\n{seqs}')
            print(f'\nSOURCE\t{source.shape}\n{source}')
            print(f'\nTARGET\t{target.shape}\n{target}')

        # Move data to GPU if available
        if DEVICE == 'cuda' and torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        # Forward pass
        output = model(source)

        # Loss computation (need to transpose so shape becomes (batch_size, vocab_size, max_prefix_len))
        loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')   # average NLL loss over all predicted tokens
        losses.append(loss.item())

        if verbose_shapes: print(f'>TRAIN LOSS for step {step}:\t{loss.item():.4f}')
        if wb_log:
            wandb.log({
                "train/loss": loss.item(),  # todo avg
                "train/learning_rate": scheduler.get_last_lr()[0],  # todo
                # "train/learning_rate": optimizer.param_groups[0]['lr'],
                "step": step
            })

        # Backpropagation
        loss.backward()

        # Clip gradients to avoid exploding gradients: if the total gradient vector length > x, clip it back down to x.
        if gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()  # stochastic gradient descent step
        scheduler.step()

        # ========== Validate every X steps ==========

        # Run validation if it's time to validate or we're at the last step
        is_last_step = (step == total_num_steps)
        should_validate = (step % validate_every == 0)
        steps_to_save = total_num_steps / 2
        should_save = (step % steps_to_save == 0)  # todo as arg ?

        if step > 0 and (should_validate or is_last_step):
            model.eval()  # Disable dropout, etc.
            val_losses = []

            with torch.no_grad(): # Disable gradient computation
                for _ in range(10):  # todo num_val_batches as args
                    # Sample random validation subsequences (same logic as training)
                    val_starts = torch.randint(0, data_val.size(0) - max_prefix_len - 1, (batch_size,))
                    val_seqs = torch.stack([data_val[start:start + max_prefix_len + 1] for start in val_starts])

                    # Split into input and target (next tokens)
                    val_src, val_target = val_seqs[:, :-1].long(), val_seqs[:, 1:].long()

                    if DEVICE == 'cuda' and torch.cuda.is_available():
                        val_src, val_target = val_src.cuda(), val_target.cuda()

                    # Run model forward pass on validation data
                    val_output = model(val_src)

                    # Compute loss for this batch and store it
                    val_loss = F.nll_loss(val_output.transpose(2, 1), val_target, reduction='mean')
                    val_losses.append(val_loss.item())

                # Average validation loss across all sampled batches
                avg_val_loss = sum(val_losses) / len(val_losses)

            # Print and log results
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step\t{step}\tAvg val loss:\t{avg_val_loss:.4f}\tLR:\t{current_lr:.6f}")
            if wb_log:
                wandb.log({
                    "val/loss": avg_val_loss,
                    "step": step,
                })

            # Save model checkpoint if needed
            if should_save:
                model_save_path = f'outputs/{timestamp}_{name}_model_step_{step}.pt'
                torch.save({
                    "state_dict": model.state_dict(),
                    "model_args": config,  # config contains all necessary args
                    "vocab": vocab,
                    "idx2label": idx2label
                }, model_save_path)

    # ========== Final Timing Summary ==========
    finish_training_time = time.time()

    print(f"\n"
          f"Total time:\t{(finish_training_time - start_time) / 60:.2f} minutes\t"
          f"Preprocessing time:\t{(start_training_time - start_time) / 60:.2f}\t"
          f"Training time:\t{(finish_training_time - start_training_time) / 60:.2f}\t")

    # Log to W&B
    if wb_log:
        wandb.log({
            "total_time_min": (finish_training_time - start_time) / 60,
            "preprocessing_time_min": (start_training_time - start_time) / 60,
            "training_time_min": (finish_training_time - start_training_time) / 60,
        })


# ---------- Main ----------
if __name__ == "__main__":

    # Run experiment with CLI arguments (if any were passed)
    if len(sys.argv) > 1:
        fire.Fire(run_experiment)  # Allow command-line overrides via `python script.py --batch_size=64`
    else:
        run_experiment()  # Run with default parameters

    # Print entire script source at the end (useful for reproducibility or debugging logs)
    print("\n\n\n================== SCRIPT SOURCE CODE ==================\n")
    with open(__file__, 'r') as f:
        print(f.read())

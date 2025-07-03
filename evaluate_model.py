import ast
import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import tqdm
import fire
import Levenshtein
import editdistance
import wandb
import torch.distributions as distrib
from transformers import get_scheduler

import train_model

# Set random seeds
def set_seed(seed):
    """
    Set random seeds for reproducibility.
    :param seed (int): Seed value to use for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_bounds_from_eos(flat_tensor, eos_token_id):
    """
    Scan a flat tensor for <EOS> tokens and return sequence bounds.
    :param flat_tensor (torch.Tensor): Flattened tensor of token IDs, containing sequences ending with EOS.
    :param eos_token_id (int): ID representing the <EOS> token.
    :return: List of (start_idx, end_idx) tuples for each sequence.
    """
    bounds = []
    start_idx = 0

    for i, token_id in enumerate(flat_tensor):
        if token_id == eos_token_id:
            bounds.append((start_idx, i))
            start_idx = i + 1

    return bounds

def evaluate_final(model, data_test: torch.Tensor, bounds_test: List[Tuple[int, int]], vocab: dict, idx2label: dict,
    max_prefix_len: int, max_suffix_len: int, device: str, n_samples: int,
    temperature: float, verbose_generated: int, stop_at_eos: bool, seed: int) -> Dict[str, Any]:
    """
    Evaluate a model on random suffix splits from test cases and return average normalized edit similarity.

    :param model: Trained Transformer model.
    :param data_test (torch.Tensor): Flattened tensor of token indices.
    :param bounds_test (List): List(start_idx, end_idx) bounds per test case.
    :param vocab (dict): Vocabulary with token to index mapping.
    :param idx2label (dict): Reverse vocabulary.
    :param max_prefix_len (int): Maximum input prefix length.
    :param max_suffix_len (int): Maximum output suffix length.
    :param device (str): "cpu" or "cuda".
    :param n_samples (int): Number of samples to evaluate.
    :param temperature (float): Sampling temperature.
    :param verbose_generated (int): Number of examples to print for debugging.
    :param stop_at_eos (bool): Whether to stop generation upon reaching <EOS>.
    :param seed (int): Random seed for reproducibility.
    :return: Dictionary with mean, std, ci95, and list of DLSimilarities.
    """
    set_seed(seed)
    model.eval()
    edit_similarities_all = []
    verbose_count = 0
    eos_token_id = vocab['<EOS>']

    with torch.no_grad():
        for sample in range(n_samples):
            random_bounds = random.choice(bounds_test)
            prefix_start, suffix_end = random_bounds

            # Make sure case is long enough
            if suffix_end - prefix_start < 2:
                continue

            suffix_start_random = random.randint(prefix_start + 1, suffix_end - 1)
            prefix_indices = data_test[prefix_start:suffix_start_random].to(torch.long).to(device)
            true_suffix_indices = data_test[suffix_start_random:suffix_end + 1].tolist()

            # Check sampling
            # if True:
            #     print(f'\n---- FINAL TESTING for instance={instance} ----')
            #     print(f'all sequence with margins +1:  {data_test[prefix_start - 1:suffix_end + 1]}')
            #     print(f'random_case, prefix_start, suffix_start_random, suffix_end: {random_bounds}, {prefix_start}, {suffix_start_random}, {suffix_end}\n')
            #     print(f'suffix_start_random = random.randint(prefix_start+1, suffix_end-1)\n{suffix_start_random} = random.randint({prefix_start}+1, {suffix_end}-1)\n')
            #     print(f'prefix_indices = data_test[prefix_start:suffix_start_random]\n{prefix_indices} = data_test[{prefix_start}:{suffix_start_random}]\n')
            #     print(f'true_suffix_indices = data_test[suffix_start_random:suffix_end+1]\n{true_suffix_indices} = data_test[{suffix_start_random}:{suffix_end}+1]\n')


            prefix_indices_tmp = prefix_indices.clone()
            pred_suffix_indices = []

            if stop_at_eos:
                for _ in range(max_suffix_len):
                    prefix_indices_tmp = prefix_indices_tmp[-max_prefix_len:]
                    logits = model(prefix_indices_tmp[None, :]) #  (1, seq_len, vocab_size)
                    next_token = train_model.sample_from_log_probab(logits[0, -1, :], temperature)
                    pred_suffix_indices.append(next_token.item())
                    if next_token.item() == eos_token_id:
                        break
                    prefix_indices_tmp = torch.cat([prefix_indices_tmp, next_token.unsqueeze(0)])
            else:  # assume length of suffix known
                for _ in range(len(true_suffix_indices)):
                    prefix_indices_tmp = prefix_indices_tmp[-max_prefix_len:]
                    logits = model(prefix_indices_tmp[None, :]) #  (1, seq_len, vocab_size)
                    next_token = train_model.sample_from_log_probab(logits[0, -1, :], temperature)
                    pred_suffix_indices.append(next_token.item())
                    prefix_indices_tmp = torch.cat([prefix_indices_tmp, next_token.unsqueeze(0)])

            if verbose_count < verbose_generated:
                print(f'\nPREFIX             :\t{len(prefix_indices)} tokens\t{prefix_indices.tolist()}')
                print(f'TRUE_SUFFIX          :\t{len(true_suffix_indices)} tokens\t{true_suffix_indices}')
                print(f'PREDICTED_SUFFIX     :\t{len(pred_suffix_indices)} tokens\t{pred_suffix_indices}')

            if stop_at_eos:
                # Remove <EOS> and everything after it
                if eos_token_id in pred_suffix_indices:
                    eos_idx = pred_suffix_indices.index(eos_token_id)
                    pred_suffix_indices = pred_suffix_indices[:eos_idx]
                if eos_token_id in true_suffix_indices:
                    eos_idx = true_suffix_indices.index(eos_token_id)
                    true_suffix_indices = true_suffix_indices[:eos_idx]

            # Debug print
            if verbose_count < verbose_generated:
                # print(f'\nTRUE SUFFIX before EOS:\t{len(true_suffix_indices)} tokens\t{true_suffix_indices}')
                # print(f'SUFFIX before EOS     :\t{len(pred_suffix_indices)} tokens\t{pred_suffix_indices}')

                # Convert indices to labels (only for debugging)
                prefix = [idx2label.get(token, "<UNK>") for token in prefix_indices.tolist()]
                pred_suffix = [idx2label.get(token, f"<UNK>") for token in pred_suffix_indices]
                true_suffix = [idx2label.get(token, f"<UNK>") for token in true_suffix_indices]

                print(f'PREFIX               :\t{len(prefix)} tokens\t{prefix}')
                print(f'TRUE_SUFFIX          :\t{len(true_suffix)} tokens\t{true_suffix}')
                print(f'PREDICTED_SUFFIX     :\t{len(pred_suffix)} tokens\t{pred_suffix}')
                verbose_count += 1

            # Compute normalized edit similarity
            edit_dist = editdistance.eval(pred_suffix_indices, true_suffix_indices)
            norm_dist = edit_dist / max(len(pred_suffix_indices), len(true_suffix_indices))
            edit_similarity = 1 - norm_dist

            # Check calculations
            # if verbose_count < 10:
            #     print('edit_dist = editdistance.eval(pred_suffix_indices, true_suffix_indices)')
            #     print(f'{edit_dist} = editdistance.eval({pred_suffix_indices}, {true_suffix_indices})')
            #     print('norm_dist = edit_dist / max(len(pred_suffix_indices), len(true_suffix_indices))')
            #     print(f'{norm_dist} = {edit_dist} / max({len(pred_suffix_indices)}, {len(true_suffix_indices)})')
            #     print('edit_similarity = 1 - norm_dist')
            #     print(f'{edit_similarity} = 1 - {norm_dist}')

            print(f'\tSample:\t{sample}\tDL:\t{edit_dist}\tDLS:\t{edit_similarity*100:.2f}%')

            edit_similarities_all.append(edit_similarity)

    edit_similarities_all = np.array(edit_similarities_all)

    mean_score = edit_similarities_all.mean()
    std_score = edit_similarities_all.std(ddof=1)  # sample standard deviation
    ci95 = 1.96 * std_score / np.sqrt(len(edit_similarities_all))

    return {
        "mean": mean_score,
        "std": std_score,
        "ci95": ci95,
        "edit_similarities_all": edit_similarities_all.tolist(),
        "stop_at_eos": stop_at_eos
    }


def load_model_and_encode_test(checkpoint_path, test_dataset_path, device):
    """
    Load model from checkpoint and encode the test dataset.
    :param checkpoint_path (str): Path to the PyTorch model checkpoint.
    :param test_dataset_path (str): CSV file path for the test dataset.
    :param device (str): Device identifier, e.g., "cpu" or "cuda".
    :return: Tuple of (model, data_tensor, bounds, vocab, idx2label, max_prefix_len, max_suffix_len).
    """

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load dataset and vocabulary from checkpoint
    df_test_cases = pd.read_csv(test_dataset_path)
    df_test_cases['activities'] = df_test_cases['activities'].apply(ast.literal_eval)
    vocab = checkpoint['vocab']
    idx2label = checkpoint['idx2label']

    # Encode the test dataset and save bounds of traces
    df_test_cases, data_test = train_model.encode_column_to_tensor(df_test_cases, 'activities', vocab)
    # print('\nSHAPES of data_test:', data_test.shape)
    # print('\ndata_train[:20]\n', data_test[:20].tolist())

    bounds_test = get_bounds_from_eos(data_test, vocab['<EOS>'])

    # Re-initialize the model with the same architecture used during training.
    model = train_model.GenTransformer(
        emb_dim=checkpoint["model_args"]['emb_dim'],
        num_heads=checkpoint["model_args"]['num_heads'],
        depth=checkpoint["model_args"]['depth'],
        max_prefix_len=checkpoint["model_args"]['max_prefix_len'],
        vocab=len(vocab) + 5,  # bcs we used it in train
        hidden_size=checkpoint["model_args"]['hidden_size'],
        dropout=checkpoint["model_args"]['dropout'],
        device=device,
    ).to(device)
    # print(model)

    # Load trained weights
    model.load_state_dict(checkpoint["state_dict"])

    max_prefix_len=checkpoint["model_args"]['max_prefix_len']
    max_suffix_len=checkpoint["model_args"]['max_suffix_len']

    return model, data_test, bounds_test, vocab, idx2label, max_prefix_len, max_suffix_len


def evaluate_saved_model(
        checkpoint_path="outputs/0617_1208_Helpdesk_model_step_60000.pt",
        test_dataset_path="data_processed/Helpdesk_test.csv",
        device="cpu",
        n_samples=1000,
        verbose_generated=20,
        seed=42,
        stop_at_eos=True):
    """
    Load a saved model, run evaluation, and print summary.

    :param checkpoint_path (str): Path to the model checkpoint.
    :param test_dataset_path (str): CSV file path for test data.
    :param device (str): Device identifier ("cpu" or "gpu").
    :param n_samples (int): Number of samples to evaluate.
    :param verbose_generated (int): Number of examples to print.
    :param seed (int): Random seed for reproducibility.
    :param stop_at_eos (bool): Stop when prdicted <EOS>; if False, generated suffix matches true suffix length.
    :return: Dictionary of evaluation results: Mean DLS, Std, CI.
    """

    set_seed(seed)

    model, data_test, bounds_test, vocab, idx2label, max_prefix_len, max_suffix_len = load_model_and_encode_test(checkpoint_path, test_dataset_path, device)


    results = evaluate_final(
        model=model,
        data_test=data_test,
        bounds_test=bounds_test,
        vocab=vocab,
        idx2label=idx2label,
        max_prefix_len=max_prefix_len,
        max_suffix_len=max_suffix_len,
        device=device,
        n_samples=n_samples,
        temperature=0.5,
        verbose_generated=verbose_generated,
        stop_at_eos=stop_at_eos,
        seed=42
    )

    print(f"Mean DLS: {results['mean']:.4f}")
    print(f"Std Dev: {results['std']:.4f}")
    print(f"95% Confidence Interval: ±{results['ci95']:.4f}")
    return results

def evaluate_models_from_csv(input_csv, results_csv, device, n_samples, verbose_generated, seed):
    """
    Evaluate multiple models specified in a CSV file and save summary.
    :param input_csv (str): CSV listing models and test datasets.
    :param results_csv (str): Output path for results summary CSV.
    :param device (str): Device identifier ("cpu" or "gpu").
    :param n_samples (int): Number of samples per model.
    :param verbose_generated (int): Number of printed examples.
    :param seed (int): Random seed for reproducibility.
    :return: DataFrame summarizing all model evaluations.
    """
    df = pd.read_csv(input_csv)
    print(df)

    summary_rows = []

    for i, row in df.iterrows():
        model = row["model"]
        test = row["test"]
        stop_at_eos = row["stop at EOS"] == 'yes'
        print(f'\n{i}\tmodel:\t{model}\t\t\ttest:\t{test}')


        results = evaluate_saved_model(
            checkpoint_path=f"outputs/{model}.pt",
            test_dataset_path=f"data_2_processed_/{test}.csv",
            device=device,
            n_samples=n_samples,
            verbose_generated=verbose_generated,
            seed=seed,
            stop_at_eos=stop_at_eos
        )
        # print(results)
        summary_rows.append({
            "generalization type": row["generalization type"],
            "data type": row["data type"],
            "test data name": row["test data name"],
            "model": model,
            "test data": test,
            "DL similarity": round(results["mean"], 4),
            "std": round(results["std"], 4),
            "ci": round(results["ci95"], 4),
            "stop at EOS": stop_at_eos
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_csv, index=False)
    return summary

# ---------- Main ----------
if __name__ == "__main__":


    if len(sys.argv) > 1:
        fire.Fire(evaluate_saved_model)
    else:
        evaluate_saved_model()

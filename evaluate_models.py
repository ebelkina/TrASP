import sys
import pandas as pd
import fire
from evaluate_model import evaluate_saved_model

def evaluate_models_from_csv(
    input_csv: str = "evaluation/evaluation_plan.csv",
    results_csv: str = "evaluation/evaluation_results.csv",
    device: str = "cpu",
    n_samples: int = 1000,
    verbose_generated: int = 10,
    seed: int = 42,
    stop_at_eos: bool = False,
    log_path: str = "evaluation/evaluation_log.txt",
):
    """
    Evaluate multiple models specified in a CSV file, log output, and save summary.

    :param input_csv:       CSV listing models and test datasets.
    :param results_csv:     Output path for summary CSV.
    :param device:          Device identifier ("cpu" or "gpu").
    :param n_samples:       Number of samples per model.
    :param verbose_generated: Number of printed examples.
    :param seed:            Random seed for reproducibility.
    :param stop_at_eos:     Whether to stop at `<EOS>` during generation.
    :param log_path:        File to write full stdout log.
    :return:                pd.DataFrame summarizing all model evaluations.
    """
    # Read plan
    df = pd.read_csv(input_csv)

    # Redirect stdout to log
    orig_stdout = sys.stdout
    log_file = open(log_path, "w")
    sys.stdout = log_file

    print(df)

    # Process each model
    summary_rows = []
    for i, row in df.iterrows():
        model = row.get("model")
        test = row.get("test")
        stop_flag = str(row.get("stop at EOS", "no")).lower() == 'yes' or stop_at_eos
        print(f"\n[{i}] model: {model} | test: {test} | stop_at_eos: {stop_flag}")
        results = evaluate_saved_model(
            checkpoint_path=f"outputs/{model}.pt",
            test_dataset_path=f"data_processed/{test}.csv",
            device=device,
            n_samples=n_samples,
            verbose_generated=verbose_generated,
            seed=seed,
            stop_at_eos=stop_flag,
        )
        summary_rows.append({
            "generalization type": row.get("generalization type"),
            "data type": row.get("data type"),
            "test data name": row.get("test data name"),
            "model": model,
            "test data": test,
            "DL similarity": round(results.get("mean", 0), 4),
            "std": round(results.get("std", 0), 4),
            "ci": round(results.get("ci95", 0), 4),
            "stop at EOS": stop_flag,
        })

    # Save summary
    summary = pd.DataFrame(summary_rows)
    print(summary)
    summary.to_csv(results_csv, index=False)

    # Restore stdout and close log
    sys.stdout = orig_stdout
    log_file.close()

    print(f"Evaluation complete. Results saved to '{results_csv}'; log at '{log_path}'.")
    return summary

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(evaluate_models_from_csv)
    else:
        evaluate_models_from_csv()

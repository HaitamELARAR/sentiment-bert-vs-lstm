"""Build the comparison table from the saved metrics files.

Usage (from the repository root):
    python scripts/summarize_results.py
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def wilson_interval(correct, n, z=1.96):
    """95 % Wilson confidence interval for a proportion."""
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def main():
    rnn = json.load(open(RESULTS / "lstm_gru_metrics.json", encoding="utf-8"))
    bert = json.load(open(RESULTS / "bert_metrics.json", encoding="utf-8"))

    if rnn["split"]["test_fingerprint"] != bert["split"]["test_fingerprint"]:
        raise SystemExit("The two runs used different test sets: the comparison is not valid.")

    n = rnn["split"]["test"]
    rows = [
        ("BiLSTM (GloVe 100d)", rnn["models"]["bilstm"], rnn["models"]["bilstm"]["train_time_s"]),
        ("BiGRU (GloVe 100d)", rnn["models"]["bigru"], rnn["models"]["bigru"]["train_time_s"]),
        ("BERT-base fine-tuned", {**bert["test"], "n_parameters": bert["n_parameters"]}, bert["train_time_s"]),
    ]

    lines = [
        f"Test set: {n} reviews (fingerprint {rnn['split']['test_fingerprint']}), "
        f"GPU: {bert['environment']['gpu']}",
        "",
        "| Model | Accuracy | 95 % CI | Precision | Recall | F1 | Train time | Parameters |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m, seconds in rows:
        lo, hi = wilson_interval(round(m["accuracy"] * n), n)
        lines.append(
            f"| {name} | **{m['accuracy']:.2%}** | {lo:.1%} to {hi:.1%} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} | {seconds / 60:.1f} min | {m['n_parameters'] / 1e6:.1f} M |"
        )
    text = "\n".join(lines)
    print(text)
    (RESULTS / "comparison.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

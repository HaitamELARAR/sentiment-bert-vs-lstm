Test set: 5000 reviews (fingerprint f158b26f6f35), GPU: Tesla T4

| Model | Accuracy | 95 % CI | Precision | Recall | F1 | Train time | Parameters |
|---|---|---|---|---|---|---|---|
| BiLSTM (GloVe 100d) | **88.50%** | 87.6% to 89.4% | 0.890 | 0.879 | 0.884 | 1.7 min | 6.8 M |
| BiGRU (GloVe 100d) | **89.50%** | 88.6% to 90.3% | 0.883 | 0.911 | 0.897 | 1.3 min | 6.7 M |
| BERT-base fine-tuned | **93.08%** | 92.3% to 93.8% | 0.928 | 0.934 | 0.931 | 17.7 min | 109.5 M |

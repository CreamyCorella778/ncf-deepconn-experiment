# NCF and DeepCoNN baseline

## Install

```powershell
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches the local CUDA setup
before installing the other requirements.

## Prepare data

The generated datasets and raw review files are intentionally excluded from
version control because they can be very large. Place the Yelp review JSONL
file at `data/raw/yelp_academic_dataset_review.json`, or let the Amazon
preprocessor download its source dataset.

```powershell
python data/preprocess.py --dataset yelp
python data/preprocess.py --dataset amazon --amazon-category Movies_and_TV
```

Use `--max-records N` for a smaller preprocessing smoke test. Outputs are
written under `data/processed/yelp` or `data/processed/amazon`, including the
rating splits and review documents required by DeepCoNN.

Before training and evaluating, precompute fixed test negatives from all three
interaction splits:

```powershell
python export_negatives.py --data-dir data/processed/amazon
```

The script writes 99 unseen item IDs per test user to
`data/processed/test_negatives.pkl` using seed 42. For Yelp, pass
`--data-dir data/processed/yelp`; this regenerates the same output file for
that dataset. If the shared file is absent, evaluation also checks for
`test_negatives.pkl` inside the selected dataset directory.

## Dataset statistics

After preparing both datasets, print their user/item/review counts, word
counts, average reviews per user, average words per review, and sparsity:

```powershell
python data/statistics.py
```

Use `--dataset amazon` or `--dataset yelp` to report one dataset. Word counts
are calculated from the cleaned, length-limited training review documents in
`user_reviews.pkl`; words per review is divided by the number of training
reviews. The other interaction statistics use all three rating splits.

## Train and evaluate

Use the config for the model/dataset pair you want to run:

```powershell
python run_compare.py --config configs/ncf.yaml
python run_compare.py --config configs/ncf_yelp.yaml
python run_compare.py --config configs/deepconn_amazon.yaml
python run_compare.py --config configs/deepconn.yaml
```

The four configs cover NCF/Amazon, NCF/Yelp, DeepCoNN/Amazon, and DeepCoNN/Yelp.
`--dataset` can still override the dataset without editing a config.
Relative data paths are resolved from the repository root, and the runner
prints the selected dataset and absolute data directory before training.
DeepCoNN loads 300-dimensional GloVe vectors from
`data/embeddings/glove.6B.300d.txt` and caches the vocabulary matrix as
`glove_matrix_<vocab-size>x300.npy` beside the source file. Cache metadata
checks the vocabulary and GloVe source before reuse. Words without a matching
vector use seeded random initialization, and the padding vector remains zero.
The hybrid DeepCoNN also adds learned user and item ID embeddings to the
corresponding CNN text representations; GloVe and ID embeddings are fine-tuned
during training.
Training stops when validation RMSE fails to improve for three consecutive
epochs and restores the best validation checkpoint before test evaluation.

To log training and test metrics to Weights & Biases, set `wandb.enabled: true`
in the selected config and optionally change `wandb.project`. Authenticate with
`wandb login` or the `WANDB_API_KEY` environment variable before running.
Training loss and validation RMSE/MAE are logged each epoch; test RMSE, MAE,
HR@10, and NDCG@10 are written to the run summary.

Start a W&B Sweep by selecting both the model and dataset from the command
line; no config file edits are needed:

```powershell
python run_sweep.py --model deepconn --dataset yelp --count 30
```

Use `--model ncf` to tune NCF, or choose `amazon` instead of `yelp`. The
optional `--project` and `--entity` arguments select the W&B destination.
Sweep runs minimize validation RMSE from the best epoch without evaluating the
held-out test set. After tuning, run the regular training command with the same
`--dataset` to evaluate the selected model on test. Ensure all rating splits
for that dataset are prepared before starting the sweep; the fixed test
negative pickle is only needed for final test evaluation.

To resume an existing sweep, pass its ID. If using a short sweep ID, also pass
the project (and entity, if applicable):

```powershell
python run_sweep.py --model deepconn --dataset yelp --sweep_id abc123 --project ncf-deepconn-experiment
```

Pass both config paths with repeated `--config` options to run and compare
both models. Test output includes RMSE, MAE, HR@10, and NDCG@10. Ranking
metrics use one held-out positive and the 99 fixed items in
`data/processed/test_negatives.pkl`; these items are not rated by the user in
any split. The fixed K and sample count are in each config. The test split is
evaluated only after model selection on validation RMSE.

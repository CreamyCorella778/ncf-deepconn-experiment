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

## Train and evaluate

Set `data.dir` in the selected model config to the generated dataset directory
(`data/processed/yelp` or `data/processed/amazon`), then run:

```powershell
python run_compare.py --config configs/ncf.yaml
python run_compare.py --config configs/deepconn.yaml
```

The `data.dataset` value must match the name of the directory in `data.dir`.
Relative data paths are resolved from the repository root, and the runner
prints the selected dataset and absolute data directory before training.

Pass both config paths with repeated `--config` options to run and compare
both models. Test output includes RMSE, MAE, HR@10, and NDCG@10. Ranking
metrics use one held-out positive and 99 reproducibly sampled items that the
user has not rated in any split; the fixed K and sample count are in each
config. The test split is evaluated only after model selection on validation
RMSE.

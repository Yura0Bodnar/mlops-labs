optimize.py:

./venv/bin/python src/optimize.py \
  --train_path data/prepared/train.csv \
  --test_path data/prepared/test.csv \
  --model_type XGBoost \
  --n_trials 5
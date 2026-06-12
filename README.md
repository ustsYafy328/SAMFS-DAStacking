# SAM-FS + DA-Stacking

This repository contains the paper-aligned implementation of **Stability-aware Multi-view Feature Selection (SAM-FS)** and **DDPG-driven Adaptive Stacking (DA-Stacking)** for aluminum extrusion speed prediction.

## Structure

```text
samfs_dastacking/
  config.py             # Paper hyperparameters and feature lists
  data.py               # CSV loading and column validation
  feature_selection.py  # SAM-FS: multi-view scores, entropy weights, TOPSIS
  experts.py            # TabPFN, Random Forest, XGBoost, Extra Trees
  models.py             # Attention block, Actor, Critic, replay buffer
  train.py              # DA-Stacking training and evaluation
train_da_stacking.py    # Command-line training entry
```

## Run

Place `train.csv` and `test.csv` in one directory, then run:

```bash
python train_da_stacking.py --split-dir ./split --feature-mode paper --verbose
```

`--feature-mode paper` uses the 17-feature SAM-FS subset reported in the article.  
`--feature-mode samfs` recomputes SAM-FS ranking from the training data.  
`--feature-mode all` uses all available configured features.

## Paper Alignment

The default configuration follows the article:

- Selected features: `M=17`
- Experts: `TabPFN`, `RandomForest`, `XGBoost`, `ExtraTrees`
- Attention: feature-wise calibration using `softmax(W_att x + b)`
- DDPG: Actor `(256, 160, 96)`, Critic `(512, 320, 192)`, `gamma=0.9`, `tau=1e-3`
- Tree experts: RF/ET `n_estimators=200`, `max_depth=12`; XGBoost `n_estimators=300`, `max_depth=8`, `learning_rate=0.03`


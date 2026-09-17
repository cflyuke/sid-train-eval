# Semantic ID : rqvae and rqkmeans implementation

Generate **Semantic IDs** for recommender systems. Two approaches are provided: **RQ-VAE** and **RQ-KMeans**.

## Directory Layout

```
config/          # configs for 100-dim and 1024-dim inputs, each with rqvae.yaml and rqkmeans.yaml
modules/
  encoder.py     # MLP encoder / decoder
  quantize.py    # single-layer quantization module (quantization core)
  rqvae.py       # RQ-VAE model (Encoder + multi-layer Quantize + Decoder)
  rqkmeans.py    # RQ-KMeans model (pure KMeans residual quantization, no NN training)
  loss.py        # loss functions and RqVaeCriterion
  evaluate.py    # SID quality evaluation
  utils.py       # kmeans and utility functions
train_rqvae.py   # RQ-VAE training entry
train_rqkmeans.py# RQ-KMeans fitting entry
train.sh         # accelerate launch script
scripts.ipynb
```

## Losses (`loss.py`)

- Reconstruction loss, quantization / commitment loss
- `AuxiliaryLoss`: codebook usage balancing (entropy regularization)
- `PrefixInfoNceLoss`: per-layer prefix contrastive learning
- `ContrastiveLoss` (encoder CL), `PrefixCollisionLoss` (prefix collision penalty)

## Evaluation Metrics (`evaluate.py`)

- Per-layer codebook **utilization (CUR)**, **entropy**, **Gini coefficient**
- Global number of unique SIDs, **ID collision rate**, global Gini coefficient
- Cosine similarity between embeddings of colliding items

## Training

> To see the training detail, you can refer to the train.sh and config/

Multi-GPU data parallelism via HuggingFace `accelerate`. Data is in parquet format (with `embedding`, and `embedding_a/b` for contrastive learning). Example launches are shown in `train.sh`, and logs are written to TensorBoard.


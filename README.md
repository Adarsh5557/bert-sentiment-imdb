# BERT Fine-Tuning for Sentiment Classification

Fine-tuning `bert-base-uncased` on the IMDB movie review dataset for binary sentiment classification (positive / negative). Built with HuggingFace Transformers and PyTorch.

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~93-94% |
| Training Time (T4 GPU) | ~50-55 min |
| Epochs | 3 |
| Model Parameters | 109M |

---

## Dataset

[Stanford IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) — 50,000 movie reviews split equally into 25,000 train and 25,000 test examples. Each review is labeled as positive (1) or negative (0).

---

## Model Architecture

```
bert-base-uncased
    ↓
[CLS] token representation  (768-dim)
    ↓
Linear classification head  (768 → 2)
    ↓
softmax → positive / negative
```

---

## Project Structure

```
├── lec_1_fine_tuning.py   # main training script
├── README.md
└── bert_imdb_finetuned/    # saved model (after training)
    ├── config.json
    ├── tokenizer_config.json
    └── model.safetensors
```

---

## Setup

```bash
pip install torch transformers datasets scikit-learn tqdm accelerate
```

> **Note:** If you encounter a `torchvision VideoReader` import error, uninstall torchvision — it is not needed for this project:
> ```bash
> pip uninstall torchvision -y
> ```

---

## Usage

**Train the model:**

```bash
python bert_finetune_imdb.py
```

**Run inference on new text:**

```python
from bert_finetune_imdb import predict

results = predict([
    "This movie was absolutely fantastic!",
    "Terrible film, complete waste of time.",
])

for r in results:
    print(r)
# {'label': 'positive', 'confidence': 0.9823, 'text': 'This movie was absolutely fantastic!'}
# {'label': 'negative', 'confidence': 0.9711, 'text': 'Terrible film, complete waste of time.'}
```

---

## Configuration

All hyperparameters are in the `CONFIG` dict at the top of the script:

```python
CONFIG = {
    "model_name"     : "bert-base-uncased",
    "num_labels"     : 2,
    "max_length"     : 256,
    "batch_size"     : 16,
    "num_epochs"     : 3,
    "lr"             : 2e-5,
    "weight_decay"   : 0.01,
    "warmup_ratio"   : 0.1,
    "train_subset"   : None,   # set to e.g. 2000 for a quick debug run
    "test_subset"    : None,
    "save_path"      : "./bert_imdb_finetuned",
}
```

**Want faster training?** Two easy knobs:

```python
"max_length" : 128,   # cuts time ~50%, accuracy drops ~0.5%
"batch_size" : 32,    # safe on T4 16GB VRAM
```

---

## Training Details

**Optimizer:** AdamW with decoupled weight decay. Bias and LayerNorm weights are excluded from weight decay following the standard BERT fine-tuning recipe.

**Scheduler:** Linear warmup (10% of steps) followed by linear decay to zero. Warmup prevents catastrophic forgetting of pretrained weights in early steps.

**Tokenizer:** WordPiece tokenizer from `bert-base-uncased`. Sequences are truncated or padded to `max_length=256`. Attention mask marks real tokens (1) vs padding tokens (0).

---

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
scikit-learn>=1.0.0
tqdm>=4.64.0
accelerate>=0.20.0
```

---

## References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)

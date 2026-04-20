# VocabAudio

Anonymous code release for the VocabAudio method for Audio Moment Retrieval
(AMR).

This repository is a clean reconstruction of the main VocabAudio pipeline. It
keeps the paper's core two-stage design and omits exploratory branches that
are not required for understanding or reproducing the method.

## What is implemented

The code follows the paper's main structure:

1. `Stage A`: offline feature extraction and discrete acoustic vocabulary
   learning on temporally contextualized AF3 features.
2. `Stage B`: hybrid discrete-continuous token packing and LoRA fine-tuning
   of the AF3 language model for AMR generation.
3. Sliding-window inference with coarse scoring and optional fine ranking.
4. Built-in AMR metrics so the repo does not depend on external private paths.

Implemented components:

- AF3 feature extraction from raw audio
- residual TCN temporal module `G`
- task-aware RVQ training with:
  - reconstruction loss
  - retrieval alignment loss
  - codebook usage regularization
  - temporal consistency loss
- hybrid token packing:
  - global token
  - continuous tokens
  - summed RVQ discrete tokens
- AF3 LoRA fine-tuning with:
  - generation loss
  - coarse relevance score loss
- AMR metrics:
  - `R1@0.5`
  - `R1@0.7`
  - `mAP@0.5`
  - `mAP@0.75`
  - `mAP@avg`

## Repository layout

```text
VocabAudio/
  configs/
  scripts/
  src/vocabaudio/
    data/
    metrics/
    models/
    training/
    utils/
```

## Expected data format

This repo assumes JSONL rows with fields like:

```json
{
  "qid": "123",
  "vid": "clip_001",
  "dataset": "castella",
  "query": "dog barking",
  "audio_path": "/abs/path/to/audio.wav",
  "duration": 183.5,
  "relevant_windows": [[12.1, 16.8], [88.0, 90.2]]
}
```

## Feature cache format

Extracted features are stored as:

```text
FEATURE_ROOT/
  castella/
    audio_1hz/<vid>.npz
    audio_5hz/<vid>.npz
    text/qid<qid>.npz
```

with:

- audio npz: `features`, `clip_length`, `duration`
- text npz: `last_hidden_state`

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

If `peft` is unavailable, Stage A still works. Stage B requires `peft`.

## Quick start

### 1. Extract AF3 features

```bash
python scripts/extract_af3_features.py \
  --model-path /path/to/af3_checkpoint \
  --input-jsonl /path/to/train.jsonl /path/to/val.jsonl \
  --output-root /path/to/features
```

### 2. Train tokenizer and RVQ

```bash
python scripts/train_tokenizer_stage.py \
  --train-jsonl /path/to/train.jsonl \
  --feature-root /path/to/features \
  --output-dir /path/to/stage_a_run
```

### 3. Fine-tune AF3 with hybrid tokens

```bash
python scripts/train_amr_stage.py \
  --model-path /path/to/af3_checkpoint \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --feature-root /path/to/features \
  --stage-a-dir /path/to/stage_a_run \
  --output-dir /path/to/stage_b_run
```

### 4. Evaluate

```bash
python scripts/eval_amr.py \
  --model-path /path/to/af3_checkpoint \
  --model-ckpt /path/to/stage_b_run/best.pt \
  --input-jsonl /path/to/test.jsonl \
  --feature-root /path/to/features \
  --stage-a-dir /path/to/stage_a_run \
  --output-dir /path/to/eval_run
```

## Notes

- This implementation prioritizes the paper's core method over reproducing
  every exploratory branch from the original research workspace.
- The code uses AF3 cached features by default because that is the most stable
  interface for the paper's two-stage design.
- The paper mentions EMA-updated RVQ codebooks. This repo keeps the Stage A
  logic practical and compact by training the temporal module with task-aware
  objectives and fitting RVQ codebooks offline on the resulting contextualized
  features. This is sufficient as a clean core implementation.

## Reproducibility Scope

This release intentionally includes:

- the model code
- training and evaluation entrypoints
- configuration files
- metric implementation

This release intentionally excludes:

- dataset copies
- feature caches
- trained checkpoints
- exploratory branches not needed for the paper method

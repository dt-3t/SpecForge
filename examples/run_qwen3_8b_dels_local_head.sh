#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export TORCHINDUCTOR_CACHE_DIR
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-32}

NUM_GPUS=${1:-1}
PYTHON=${PYTHON:-python}
NUM_EPOCHS=${NUM_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
SEED=${SEED:-42}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen/Qwen3-8B}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$ROOT_DIR/cache/dataset/sharegpt_train_regen_temperature0_no_think.jsonl}
OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/outputs/qwen3-8b-dels}
LOCAL_HEAD_TYPE=${LOCAL_HEAD_TYPE:-rnn}
LOCAL_HEAD_PARAMETERIZATION=${LOCAL_HEAD_PARAMETERIZATION:-target_factorized}

"$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    "$ROOT_DIR/scripts/train_local_head.py" \
    --local-head-type "$LOCAL_HEAD_TYPE" \
    --local-head-parameterization "$LOCAL_HEAD_PARAMETERIZATION" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "$OUT_ROOT/local_head" \
    --unigram-output-dir "$OUT_ROOT/loss_mask_unigram" \
    --data-format conversation \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --accumulation-steps 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --block-size 16 \
    --num-anchors 512 \
    --local-gru-hidden-dim 1024 \
    --local-input-rank 256 \
    --local-rank 256 \
    --markov-rank 256 \
    --local-up-proj-init zero \
    --local-rank-activation silu \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 2000 \
    --save-merged-rnn-weights \
    --save-merged-markov-weights \
    --save-unigram-prior \
    --report-to none

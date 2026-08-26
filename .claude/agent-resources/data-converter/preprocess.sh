#!/bin/bash
# EveNet preprocessing: converts NPZ files to parquet shards
# Run this INSIDE the EveNet container from the run/<project_name>/ directory
#
# Usage: Copy this script to EveNet-Full/run/<project_name>/, replace <project_name>, then:
#   cd <path-to-EveNet-Full>
#   export PYTHONPATH=$(pwd):$PYTHONPATH
#   cd run/<project_name>
#   bash preprocess.sh
#
# For 2-fold mode: remove the --val line and change symlink loop to "train test"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Standard mode (with val):
python3 ../../preprocessing/preprocess.py \
  --config ../../share/<project_name>.yaml \
  --train data_processed/npz/train.npz \
  --test data_processed/npz/test.npz \
  --store_dir data_processed/parquet \
  --val data_processed/npz/val.npz

# 2-fold mode (no val): uncomment below and comment out above
# python3 ../../preprocessing/preprocess.py \
#   --config ../../share/<project_name>.yaml \
#   --train data_processed/npz/train.npz \
#   --test data_processed/npz/test.npz \
#   --store_dir data_processed/parquet

BASE_DIR="data_processed/parquet"

# Standard mode:
mkdir -p "$BASE_DIR"/{train,val,test}
for split in train val test; do
  ln -sfn "../${split}.parquet" "$BASE_DIR/$split/${split}.parquet"
  ln -sfn "../shape_metadata.json" "$BASE_DIR/$split/shape_metadata.json"
done

# 2-fold mode: uncomment below and comment out above
# mkdir -p "$BASE_DIR"/{train,test}
# for split in train test; do
#   ln -sfn "../${split}.parquet" "$BASE_DIR/$split/${split}.parquet"
#   ln -sfn "../shape_metadata.json" "$BASE_DIR/$split/shape_metadata.json"
# done

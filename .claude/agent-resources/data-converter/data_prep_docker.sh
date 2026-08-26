#!/bin/bash
# EveNet — data preparation (Docker)
# No GPU needed.
#
# Placeholders to replace:
#   <evenet_full>       absolute path to EveNet-Full repo on host
#   <input_data_dir>    directory containing input .root or .pt files on host
#   <tree_name>         TTree name, e.g. BPHY25 — ROOT input only. If the plan's input
#                        format is .pt, delete the "--tree_name ${TREE_NAME}" line below
#                        (and the TREE_NAME= line) rather than leaving a placeholder value.
#   <project_name>      project name — this run's artifacts live under run/<project_name>/,
#                        not directly under run/, so a second analysis doesn't overwrite this one

EVENET_FULL=<evenet_full>
CONTAINER_IMAGE=docker.io/avencast1994/evenet:1.5
INPUT_DIR=<input_data_dir>
TREE_NAME=<tree_name>
PROJECT_NAME=<project_name>

echo "=== Input -> NPZ ==="
docker run --rm \
  -v ${INPUT_DIR}:/input_data \
  -v ${EVENET_FULL}:/workspace/EveNet_Full \
  ${CONTAINER_IMAGE} bash -c \
  "cd /workspace/EveNet_Full/run/${PROJECT_NAME} && \
   export PYTHONPATH=/workspace/EveNet_Full:\$PYTHONPATH && \
   python3 data_processed/npz/to_npz.py \
     --input_dir /input_data \
     --output_dir data_processed/npz \
     --tree_name ${TREE_NAME}"

echo "=== NPZ -> Parquet ==="
docker run --rm \
  -v ${EVENET_FULL}:/workspace/EveNet_Full \
  ${CONTAINER_IMAGE} bash -c \
  "cd /workspace/EveNet_Full/run/${PROJECT_NAME} && \
   export PYTHONPATH=/workspace/EveNet_Full:\$PYTHONPATH && \
   bash preprocess.sh"

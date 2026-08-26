#!/bin/bash
# EveNet — data preparation (NERSC / Shifter)
# No GPU needed; run from the login node.
#
# Placeholders to replace:
#   <evenet_full>       absolute path to EveNet-Full repo
#   <container_image>   Shifter image, e.g. docker:avencast1994/evenet:1.5
#   <input_data_dir>    directory containing input .root or .pt files
#   <tree_name>         TTree name, e.g. BPHY25 — ROOT input only. If the plan's input
#                        format is .pt, delete the "--tree_name ${TREE_NAME}" line below
#                        (and the TREE_NAME= line) rather than leaving a placeholder value.
#   <project_name>      project name — this run's artifacts live under run/<project_name>/,
#                        not directly under run/, so a second analysis doesn't overwrite this one

EVENET_FULL=<evenet_full>
CONTAINER_IMAGE=<container_image>
INPUT_DIR=<input_data_dir>
TREE_NAME=<tree_name>
PROJECT_NAME=<project_name>

echo "=== Input -> NPZ ==="
shifter --image=${CONTAINER_IMAGE} bash -c \
  "cd ${EVENET_FULL}/run/${PROJECT_NAME} && \
   export PYTHONPATH=${EVENET_FULL}:\$PYTHONPATH && \
   python3 data_processed/npz/to_npz.py \
     --input_dir ${INPUT_DIR} \
     --output_dir data_processed/npz \
     --tree_name ${TREE_NAME}"

echo "=== NPZ -> Parquet ==="
shifter --image=${CONTAINER_IMAGE} bash -c \
  "cd ${EVENET_FULL}/run/${PROJECT_NAME} && \
   export PYTHONPATH=${EVENET_FULL}:\$PYTHONPATH && \
   bash preprocess.sh"

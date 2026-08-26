#!/bin/bash
# EveNet — export predictions back into the original data format (NERSC / Shifter)
# No GPU needed; run from the login node.
#
# Placeholders to replace:
#   <evenet_full>       absolute path to EveNet-Full repo
#   <container_image>   Shifter image, e.g. docker:avencast1994/evenet:1.5
#   <run_dir>           absolute path to EveNet-Full/run/<project_name> (this analysis's own subdirectory)
#   <input_data_dir>    directory containing original input .root or .pt files
#   <run_name>          run name used in prediction filename
#   <output_path>        output file path (standard mode) — extension (.root or .pt) must
#                        match the plan's input format; for 2-fold, <output_path_fold0> /
#                        <output_path_fold1> / <merged_output_path> instead (fill in the
#                        actual final paths yourself, don't derive them via string suffixing)
#   <tree_name>         TTree name, e.g. BPHY25 — ROOT input only; delete the
#                        "--tree_name ${TREE_NAME}" line (and TREE_NAME= line) if input is .pt
#   <branch_prefix>     prefix for predicted branches/fields, e.g. pred_gam

EVENET_FULL=<evenet_full>
CONTAINER_IMAGE=<container_image>
RUN_DIR=<run_dir>
INPUT_DIR=<input_data_dir>
RUN_NAME=<run_name>
OUTPUT_PATH=<output_path>
TREE_NAME=<tree_name>
BRANCH_PREFIX=<branch_prefix>

# --- Standard mode ---
shifter --image=${CONTAINER_IMAGE} bash -c \
  "cd ${RUN_DIR} && \
   export PYTHONPATH=${EVENET_FULL}:\$PYTHONPATH && \
   python3 merge_predictions.py \
     --pt_file predict/test_${RUN_NAME}.pt \
     --input_dir ${INPUT_DIR} \
     --output ${OUTPUT_PATH} \
     --tree_name ${TREE_NAME} \
     --branch_prefix ${BRANCH_PREFIX}"

# --- 2-fold mode (uncomment and comment out standard mode above) ---
# Fold 0
# shifter --image=${CONTAINER_IMAGE} bash -c \
#   "cd ${RUN_DIR} && \
#    export PYTHONPATH=${EVENET_FULL}:\$PYTHONPATH && \
#    python3 merge_predictions.py \
#      --pt_file predict_0/test_${RUN_NAME}.pt \
#      --input_dir ${INPUT_DIR} \
#      --output <output_path_fold0> \
#      --tree_name ${TREE_NAME} \
#      --branch_prefix ${BRANCH_PREFIX}"
#
# Fold 1
# shifter --image=${CONTAINER_IMAGE} bash -c \
#   "cd ${RUN_DIR} && \
#    export PYTHONPATH=${EVENET_FULL}:\$PYTHONPATH && \
#    python3 merge_predictions.py \
#      --pt_file predict_1/test_${RUN_NAME}.pt \
#      --input_dir ${INPUT_DIR} \
#      --output <output_path_fold1> \
#      --tree_name ${TREE_NAME} \
#      --branch_prefix ${BRANCH_PREFIX}"
#
# Merge folds
# shifter --image=${CONTAINER_IMAGE} bash -c \
#   "cd ${RUN_DIR} && \
#    export PYTHONPATH=${EVENET_FULL}:\$PYTHONPATH && \
#    python3 merge_folds.py \
#      --fold0 <output_path_fold0> \
#      --fold1 <output_path_fold1> \
#      --output <merged_output_path> \
#      --tree_name ${TREE_NAME}"

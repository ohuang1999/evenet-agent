#!/bin/bash
# EveNet — export predictions back into the original data format (Docker)
# No GPU needed.
#
# Placeholders to replace:
#   <evenet_full>       absolute path to EveNet-Full repo on host
#   <input_data_dir>    directory containing original input .root or .pt files on host
#   <project_name>      project name — artifacts live under run/<project_name>/
#   <run_name>          run name used in prediction filename
#   <output_path>        output file path (standard) — extension (.root or .pt) must match
#                        the plan's input format; for 2-fold, fill actual per-fold/merged
#                        paths in directly below rather than deriving them via suffixing
#   <tree_name>         TTree name, e.g. BPHY25 — ROOT input only; delete the
#                        "--tree_name ${TREE_NAME}" line (and TREE_NAME= line) if input is .pt
#   <branch_prefix>     prefix for predicted branches/fields, e.g. pred_gam

EVENET_FULL=<evenet_full>
CONTAINER_IMAGE=docker.io/avencast1994/evenet:1.5
INPUT_DIR=<input_data_dir>
PROJECT_NAME=<project_name>
RUN_NAME=<run_name>
OUTPUT_PATH=<output_path>
TREE_NAME=<tree_name>
BRANCH_PREFIX=<branch_prefix>

# --- Standard mode ---
docker run --rm \
  -v ${EVENET_FULL}:/workspace/EveNet_Full \
  -v ${INPUT_DIR}:/input_data \
  -v $(dirname ${OUTPUT_PATH}):/output_dir \
  ${CONTAINER_IMAGE} bash -c \
  "cd /workspace/EveNet_Full/run/${PROJECT_NAME} && \
   export PYTHONPATH=/workspace/EveNet_Full:\$PYTHONPATH && \
   python3 merge_predictions.py \
     --pt_file predict/test_${RUN_NAME}.pt \
     --input_dir /input_data \
     --output /output_dir/$(basename ${OUTPUT_PATH}) \
     --tree_name ${TREE_NAME} \
     --branch_prefix ${BRANCH_PREFIX}"

# --- 2-fold mode (uncomment and comment out standard mode above) ---
# Fill in <output_path_fold0> / <output_path_fold1> / <merged_output_path> with the
# actual final paths (same extension as OUTPUT_PATH above) before uncommenting.
# Fold 0
# docker run --rm \
#   -v ${EVENET_FULL}:/workspace/EveNet_Full \
#   -v ${INPUT_DIR}:/input_data \
#   -v $(dirname <output_path_fold0>):/output_dir \
#   ${CONTAINER_IMAGE} bash -c \
#   "cd /workspace/EveNet_Full/run/${PROJECT_NAME} && \
#    export PYTHONPATH=/workspace/EveNet_Full:\$PYTHONPATH && \
#    python3 merge_predictions.py \
#      --pt_file predict_0/test_${RUN_NAME}.pt \
#      --input_dir /input_data \
#      --output /output_dir/$(basename <output_path_fold0>) \
#      --tree_name ${TREE_NAME} \
#      --branch_prefix ${BRANCH_PREFIX}"
#
# Fold 1
# docker run --rm \
#   -v ${EVENET_FULL}:/workspace/EveNet_Full \
#   -v ${INPUT_DIR}:/input_data \
#   -v $(dirname <output_path_fold1>):/output_dir \
#   ${CONTAINER_IMAGE} bash -c \
#   "cd /workspace/EveNet_Full/run/${PROJECT_NAME} && \
#    export PYTHONPATH=/workspace/EveNet_Full:\$PYTHONPATH && \
#    python3 merge_predictions.py \
#      --pt_file predict_1/test_${RUN_NAME}.pt \
#      --input_dir /input_data \
#      --output /output_dir/$(basename <output_path_fold1>) \
#      --tree_name ${TREE_NAME} \
#      --branch_prefix ${BRANCH_PREFIX}"
#
# Merge folds
# docker run --rm \
#   -v ${EVENET_FULL}:/workspace/EveNet_Full \
#   -v $(dirname <merged_output_path>):/output_dir \
#   ${CONTAINER_IMAGE} bash -c \
#   "cd /workspace/EveNet_Full/run/${PROJECT_NAME} && \
#    export PYTHONPATH=/workspace/EveNet_Full:\$PYTHONPATH && \
#    python3 merge_folds.py \
#      --fold0 <output_path_fold0> \
#      --fold1 <output_path_fold1> \
#      --output /output_dir/$(basename <merged_output_path>) \
#      --tree_name ${TREE_NAME}"

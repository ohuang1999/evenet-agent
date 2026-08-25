#!/bin/bash
# EveNet — Docker job (fine-tuning + prediction)
# Run in background via the Bash tool with run_in_background=True.
#
# Placeholders to replace:
#   <project_name>      user's project name
#   <evenet_full>       absolute path to EveNet-Full repo on host
#   <wandb_project>     W&B project name
#   <run_dir>           absolute path to EveNet-Full/run/<project_name> (project-scoped, not bare run/; for log output)
#
# Does NOT take a literal <wandb_api_key> placeholder -- never write the key
# itself into this file. Export it in the shell that runs this script;
# `-e WANDB_API_KEY` below (no `=value`) tells docker to pass through
# whatever is already in the calling shell's environment, e.g.:
#   export WANDB_API_KEY="<the actual value>" && bash <this script>

EVENET_FULL=<evenet_full>
CONTAINER_IMAGE=docker.io/avencast1994/evenet:1.5

: "${WANDB_API_KEY:?WANDB_API_KEY not set -- export it in the calling shell before running this script}"

docker run --rm --gpus all \
  -v ${EVENET_FULL}:/workspace/EveNet_Full \
  -e WANDB_API_KEY \
  -e WANDB_PROJECT=<wandb_project> \
  ${CONTAINER_IMAGE} bash -c \
  "cd /workspace/EveNet_Full && \
   export PYTHONPATH=/workspace/EveNet_Full:\$PYTHONPATH && \
   echo '=== Fine-tuning ===' && \
   python scripts/train.py share/<project_name>.yaml --ray_dir ~/ray_results && \
   echo '=== Prediction ===' && \
   python scripts/predict.py share/predict_<project_name>.yaml" \
  > <run_dir>/logs/run.out 2>&1

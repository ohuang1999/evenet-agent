#!/bin/bash
# EveNet — NERSC batch job (fine-tuning + prediction)
#
# Placeholders to replace:
#   <project_name>      user's project name
#   <account>           NERSC account without _g suffix, e.g. m2616
#   <wall_time>         e.g. 04:00:00
#   <total_gpus>        number_of_workers * resources_per_worker["GPU"] from finetune YAML
#   <container_image>   e.g. registry.nersc.gov/<project>/avencast/evenet:1.5
#   <run_dir>           absolute path to EveNet-Full/run/<project_name> (project-scoped, not bare run/)
#   <evenet_full>       absolute path to EveNet-Full repo
#   <wandb_project>     W&B project name
#
# NOTE: only for jobs that fit on a single node (<total_gpus> <= 4 on
# Perlmutter). For anything larger, use job_submit_nersc_multinode_template.sh
# instead -- a naive single-node invocation silently under-schedules rather
# than erroring when total_gpus exceeds one node's worth (see that
# template's header for why).
#
# Does NOT take a literal <wandb_api_key> placeholder -- never write the key
# itself into this file. Export it in the submitting shell and sbatch in the
# same command instead, e.g.:
#   export WANDB_API_KEY="<the actual value>" && sbatch <this script>
# The line below fails loudly if that wasn't done, rather than silently
# running without W&B logging.

#SBATCH --job-name=evenet_<project_name>
#SBATCH --account=<account>_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=<wall_time>
#SBATCH --nodes=1
#SBATCH --gpus=<total_gpus>
#SBATCH --image=<container_image>
#SBATCH --output=<run_dir>/logs/run_%j.out
#SBATCH --error=<run_dir>/logs/run_%j.err

cd <evenet_full>
export PYTHONPATH=<evenet_full>:$PYTHONPATH
export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY not set -- export it in the submitting shell before calling sbatch}
export WANDB_PROJECT=<wandb_project>

echo "=== Fine-tuning ==="
shifter python3 scripts/train.py <run_dir>/config/<project_name>.yaml --load_all
TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
  echo "=== Fine-tuning failed (exit $TRAIN_EXIT) -- skipping prediction ==="
  exit $TRAIN_EXIT
fi

echo "=== Prediction ==="
shifter python3 scripts/predict.py <run_dir>/config/predict_<project_name>.yaml

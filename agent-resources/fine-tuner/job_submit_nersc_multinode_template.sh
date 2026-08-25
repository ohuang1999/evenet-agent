#!/bin/bash
# EveNet — NERSC batch job (fine-tuning + prediction), MULTI-NODE variant.
#
# Use this instead of job_submit_nersc_template.sh whenever the finetune
# YAML's platform.number_of_workers * resources_per_worker["GPU"] exceeds
# what a single node provides (4 GPUs/node on Perlmutter) -- i.e. whenever
# <num_nodes> below would be >1. Below that threshold, the single-node
# template is simpler and sufficient; don't use this one for a 1-4 GPU job.
#
# Why this is needed at all: scripts/train.py's ray.init() is called with
# no `address=` -- by itself that only ever sees the LOCAL node's resources.
# Ray does not auto-span multiple Slurm-allocated nodes into one cluster
# just because more nodes were requested under a naive single `shifter
# python3 scripts/train.py` invocation (the single-node template's
# approach) -- confirmed by hitting exactly this as a silent
# under-scheduling/hang, not a clean error, in production. This script
# instead explicitly bootstraps a multi-node Ray cluster across all
# allocated nodes via srun (head on one node, workers on the rest),
# following EveNet-Full's own reference recipe at
# example/slurm/{start-head,start-worker,submit-ray-cluster}.sh.
# train.py/predict.py are then run on the head node, where ray.init()
# auto-discovers and joins the already-running local Ray head (which by
# then has every node's GPUs registered cluster-wide).
#
# The predict YAML should reuse this same cluster (same
# number_of_workers/resources_per_worker as the finetune YAML) for the
# prediction phase that follows train.py in this same job -- see
# fine-tuner.md's design note on why train+predict run as one combined
# submitted job, and predictor.md for the standalone-predict-only case
# (which needs this same multi-node bootstrap when re-predicting at >4 GPUs
# without an active finetune job's cluster to reuse).
#
# Placeholders to replace:
#   <project_name>      user's project name
#   <account>           NERSC account without _g suffix, e.g. m2616
#   <wall_time>         e.g. 12:00:00
#   <num_nodes>         ceil(total_gpus / <gpus_per_node>)
#   <gpus_per_node>      GPUs per node on this system, 4 on Perlmutter
#   <cpus_per_node>      CPUs per node on this system, 128 on Perlmutter
#   <container_image>   e.g. registry.nersc.gov/<project>/avencast/evenet:1.5
#   <run_dir>           absolute path to EveNet-Full/run/<project_name>
#   <evenet_full>       absolute path to EveNet-Full repo
#   <wandb_project>     W&B project name
#   <finetune_yaml>     e.g. <project_name>.yaml or <project_name>_finetune.yaml
#   <predict_yaml>      e.g. predict_<project_name>.yaml
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
#SBATCH --nodes=<num_nodes>
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=<gpus_per_node>
#SBATCH --cpus-per-task=<cpus_per_node>
#SBATCH --image=<container_image>
#SBATCH --output=<run_dir>/logs/run_%j.out
#SBATCH --error=<run_dir>/logs/run_%j.err

cd <evenet_full>
export PYTHONPATH=<evenet_full>:$PYTHONPATH
export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY not set -- export it in the submitting shell before calling sbatch}
export WANDB_PROJECT=<wandb_project>

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
# handle possible multi-address hostname --ip-address output (see EveNet-Full's
# own example/slurm/submit-ray-cluster.sbatch, same normalization logic)
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
fi
port=6379

echo "=== Starting Ray head on $head_node ($head_node_ip:$port) ==="
srun --nodes=1 --ntasks=1 --gpus-per-task=<gpus_per_node> --cpus-per-task=<cpus_per_node> -w "$head_node" \
  shifter bash -c "ray start --head --dashboard-host 0.0.0.0 --port=$port; sleep infinity" &
sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ "$worker_num" -gt 0 ]; then
  echo "=== Starting $worker_num Ray worker node(s) ==="
  srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 --gpus-per-task=<gpus_per_node> --cpus-per-task=<cpus_per_node> \
    --exclude="$head_node" \
    shifter bash -c "ray start --address=$head_node_ip:$port; sleep infinity" &
  sleep 30
fi

echo "=== Waiting for all $SLURM_JOB_NUM_NODES Ray node(s) to join ==="
JOINED=0
for i in $(seq 1 20); do
  JOINED=$(shifter python3 -c "
import ray
ray.init(address='auto')
print(len(ray.nodes()))
ray.shutdown()
" 2>/dev/null | tail -1)
  echo "  attempt $i: ${JOINED:-0} / $SLURM_JOB_NUM_NODES Ray node(s) joined"
  if [ "$JOINED" == "$SLURM_JOB_NUM_NODES" ]; then
    break
  fi
  sleep 15
done
if [ "$JOINED" != "$SLURM_JOB_NUM_NODES" ]; then
  echo "=== WARNING: only $JOINED / $SLURM_JOB_NUM_NODES Ray node(s) joined after waiting -- proceeding anyway, train.py may hang or under-schedule ==="
fi

echo "=== Fine-tuning ==="
shifter python3 scripts/train.py share/<finetune_yaml> --load_all
TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
  echo "=== Fine-tuning failed (exit $TRAIN_EXIT) -- skipping prediction ==="
  exit $TRAIN_EXIT
fi

echo "=== Prediction ==="
shifter python3 scripts/predict.py share/<predict_yaml>

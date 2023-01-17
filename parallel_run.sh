#!/bin/bash

cuda_device=0
dataset=census
devtest=False
mode=train

while getopts m:d:c:t: flag
do
    case "${flag}" in
        s) mode=${OPTARG};;
        d) dataset=${OPTARG};;
        c) cuda_device=${OPTARG};;
        t) devtest=${OPTARG};;
    esac
done

CONFIG="configs/${dataset}_config.py"
WORKDIR="results/$dataset/seed"

echo "Training dataset:$dataset on gpu:$cuda_device devtest_enabled:$devtest"

# Start first two runs in the background
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$cuda_device python main.py --mode $mode --config $CONFIG \
--workdir "${WORKDIR}_0" --config.seed=42 --config.devtest=$devtest &

# store the process ID
run_1_pid=$!

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$cuda_device python main.py --mode $mode --config $CONFIG \
--workdir "${WORKDIR}_1" --config.seed=43 --config.devtest=$devtest &

run_2_pid=$!

# trap ctrl-c so that we can kill
# the first two programs
trap onexit INT
function onexit() {
  kill -9 $run_1_pid
  kill -9 $run_2_pid
}

# start the third run, don't fork it
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$cuda_device python main.py --mode $mode --config $CONFIG \
--workdir "${WORKDIR}_2" --config.seed=44 --config.devtest=$devtest

# Two more runs in parallel
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$cuda_device python main.py --mode $mode --config $CONFIG \
--workdir "${WORKDIR}_3" --config.seed=45 --config.devtest=$devtest &
run_1_pid=$!

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$cuda_device python main.py --mode $mode --config $CONFIG \
--workdir "${WORKDIR}_4" --config.seed=46 --config.devtest=$devtest

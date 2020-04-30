#!/bin/bash

NUM_USERS=100
SHARDS=2
while getopts s:n: option; do
  case "${option}" in
    s) SHARDS=${OPTARG};;
    n) NUM_USERS=${OPTARG};;
  esac
done

for FED_MODEL in 400 500 600 700 800; do
  for RUN in 1 2 3 4 5; do
    python3 main_lg.py --dataset mnist --model mlp --num_classes 10 --epochs 500 --lr 0.05 \
    --num_users ${NUM_USERS} --shard_per_user ${SHARDS} --frac 0.1 --local_ep 1 --local_bs 10 --num_layers_keep 3 \
    --results_save run${RUN} --load_fed best_${FED_MODEL}.pt
  done
done


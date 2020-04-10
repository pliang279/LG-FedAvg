#!/bin/bash

SHARDS=2
while getopts s:n: option; do
  case "${option}" in
    s) SHARDS=${OPTARG};;
  esac
done

for RUN in 1 2 3 4 5; do
  for NUM_USERS in 20 50 200; do
    python3 main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 \
    --num_users ${NUM_USERS} --shard_per_user ${SHARDS} --frac 0.1 --local_ep 1 --local_bs 50 --results_save run${RUN}

    python3 main_local.py --dataset cifar10 --model cnn --num_classes 10 --epochs 200 --lr 0.1 \
    --num_users ${NUM_USERS} --shard_per_user ${SHARDS} --frac 0.1 --local_ep 1 --local_bs 50 --results_save run${RUN}

    for FED_MODEL in 1000 1200 1400 1600 1800; do
      python3 main_lg.py --dataset cifar10 --model cnn --num_classes 10 --epochs 200 --lr 0.1 \
      --num_users ${NUM_USERS} --shard_per_user ${SHARDS} --frac 0.1 --local_ep 1 --local_bs 50 --num_layers_keep 2 \
      --results_save run${RUN} --load_fed best_${FED_MODEL}.pt
    done
  done
done


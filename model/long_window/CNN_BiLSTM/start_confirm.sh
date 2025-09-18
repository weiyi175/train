#!/usr/bin/env bash
cd "$(dirname "$0")"
source /home/user/projects/train/.venv/bin/activate
mkdir -p result_confirm
nohup python3 train_cnn_bilstm.py --result_dir result_confirm --no_early_stop > result_confirm/sweep.nohup.out 2>&1 &
pid=$!
echo $pid > result_confirm/sweep.pid
echo "Started confirm run PID: $pid"

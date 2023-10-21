#!/bin/bash

command="$1"  # The command to run

if [ "$command" = "train_baseline" ]; then
    # Baseline Model Configuration
    python train_model.py --type 'point' --max_iter 2000 --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-4
    wait
    python train_model.py --type 'point' --max_iter 4000 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-5
    wait
    python train_model.py --type 'point' --max_iter 6000 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 16 --lr 1e-6
    wait
    python train_model.py --type 'point' --max_iter 8000 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 16 --lr 5e-7
    wait
    python train_model.py --type 'point' --max_iter 10002 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 30 --lr 5e-7
    wait
elif [ "$command" = "train_n_points_10000" ]; then
    # N_points 10000 points Configuration
    python train_model.py --type 'point' --max_iter 2000 --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-4 --n_points 10000
    wait
    python train_model.py --type 'point' --max_iter 4000 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-5 --n_points 10000
    wait
    python train_model.py --type 'point' --max_iter 6000 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 16 --lr 1e-6 --n_points 10000
    wait
    python train_model.py --type 'point' --max_iter 8000 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 16 --lr 5e-7 --n_points 10000
    wait
    python train_model.py --type 'point' --max_iter 10002 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 30 --lr 5e-7 --n_points 10000
    wait
elif [ "$command" = "train_n_points_25000" ]; then
    # N_points 25000 points Configuration
    python train_model.py --type 'point' --max_iter 2000 --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-4 --n_points 25000
    wait
    python train_model.py --type 'point' --max_iter 4000 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-5 --n_points 25000
    wait
    python train_model.py --type 'point' --max_iter 6000 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 16 --lr 1e-6 --n_points 25000
    wait
    python train_model.py --type 'point' --max_iter 8000 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 16 --lr 5e-7 --n_points 25000
    wait
    python train_model.py --type 'point' --max_iter 10002 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 30 --lr 5e-7 --n_points 25000
    wait
else
    echo "Usage: $0 [train_baseline|train_n_points_10000|train_n_points_25000]"
fi

#!/bin/bash

command="$1"  # The command to run

if [ "$command" = "train_baseline" ]; then
    # Baseline Model Configuration
    echo "Training Baseline Model Configuration"
    python train_model.py --type 'mesh' --max_iter 4000 --batch_size 16 --num_workers 4 --save_freq 500 --lr 4e-5 --w_smooth 0.500
    wait
    python train_model.py --type 'mesh' --max_iter 6000 --load_checkpoint --batch_size 16 --num_workers 4 --save_freq 500 --lr 1e-6 --w_smooth 0.500
    wait
    python train_model.py --type 'mesh' --max_iter 8000 --load_checkpoint --batch_size 30 --num_workers 4 --save_freq 500 --lr 5e-7 --w_smooth 0.500
    wait
    python train_model.py --type 'mesh' --max_iter 10002 --load_checkpoint --batch_size 30 --num_workers 4 --save_freq 500 --lr 5e-7 --w_smooth 0.500
    wait
elif [ "$command" = "train_smooth_5" ]; then
    # Change Hyperparameter w_smooth=5 Configuration
    echo "Training Change Hyperparameter w_smooth=5 Configuration"
    python train_model.py --type 'mesh' --max_iter 6000 --batch_size 16 --num_workers 4 --save_freq 500 --lr 1e-2 --w_smooth 5
    wait
    python train_model.py --type 'mesh' --max_iter 8000 --load_checkpoint --batch_size 16 --num_workers 4 --save_freq 500 --lr 1e-3 --w_smooth 5
    wait
    python train_model.py --type 'mesh' --max_iter 10002 --load_checkpoint --batch_size 30 --num_workers 4 --save_freq 500 --lr 4e-5 --w_smooth 5
    wait
    python train_model.py --type 'mesh' --max_iter 15002 --load_checkpoint --batch_size 30 --num_workers 4 --save_freq 500 --lr 1e-6 --w_smooth 5
    wait
else
    echo "Usage: $0 [train_baseline|train_smooth_5]"
fi

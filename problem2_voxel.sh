#!/bin/bash
command="$1"  # The command to run

if [ "$command" = "train_baseline" ]; then
    # Baseline Model (Linear Layers only)
    python train_model.py --type 'vox' --max_iter 2000 --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-4 --model_type 'baseline'
    wait
    python train_model.py --type 'vox' --max_iter 4000 --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-5 --load_checkpoint --model_type 'baseline'
    wait
    python train_model.py --type 'vox' --max_iter 6000 --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint --model_type 'baseline'
    wait
    python train_model.py --type 'vox' --max_iter 8000 --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint --model_type 'baseline'
    wait
    python train_model.py --type 'vox' --max_iter 10002 --num_workers 4 --save_freq 200 --batch_size 30 --lr 5e-7 --load_checkpoint --model_type 'baseline'
    wait
elif [ "$command" = "train_conv" ]; then
    # Baseline Model (Conv layers Conv Model)  Avg F1@0.05: 82.663
    python train_model.py --type 'vox' --max_iter 2000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-4 --model_type 'conv'
    wait
    python train_model.py --type 'vox' --max_iter 4000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-5 --load_checkpoint --model_type 'conv'
    wait
    python train_model.py --type 'vox' --max_iter 6000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint --model_type 'conv'
    wait
    python train_model.py --type 'vox' --max_iter 8000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint --model_type 'conv'
    wait
    python train_model.py --type 'vox' --max_iter 10002  --num_workers 4 --save_freq 200 --batch_size 30 --lr 5e-7 --load_checkpoint --model_type 'conv'
    wait

elif [ "$command" = "train_mlp" ]; then
    # Baseline Model (ImplicitMLPDecoder)  Avg F1@0.05: 0.0 
    python train_model.py --type 'vox' --max_iter 2000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-4 --model_type 'mlp'
    wait
    python train_model.py --type 'vox' --max_iter 4000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-5 --load_checkpoint --model_type 'mlp'
    wait
    python train_model.py --type 'vox' --max_iter 6000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint --model_type 'mlp'
    wait
    python train_model.py --type 'vox' --max_iter 8000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint --model_type 'mlp'
    wait
    python train_model.py --type 'vox' --max_iter 10002  --num_workers 4 --save_freq 200 --batch_size 24 --lr 5e-7 --load_checkpoint --model_type 'mlp'
    wait
else
    echo "Usage: $0 [train_baseline|train_conv|train_implicitMLP]"
fi

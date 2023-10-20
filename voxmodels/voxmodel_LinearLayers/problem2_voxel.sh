#!/bin/bash
# Baseline Model (Linear Layers only)  Avg F1@0.05: 81.916
python train_model.py --type 'vox' --max_iter 2000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-4
wait
python train_model.py --type 'vox' --max_iter 4000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-5 --load_checkpoint
wait
python train_model.py --type 'vox' --max_iter 6000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint
wait
python train_model.py --type 'vox' --max_iter 8000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint
wait
python train_model.py --type 'vox' --max_iter 10002  --num_workers 4 --save_freq 200 --batch_size 30 --lr 5e-7 --load_checkpoint
wait
# Baseline Model (Conv layers Pix2Vox Model)  Avg F1@0.05: 
# python train_model.py --type 'vox' --max_iter 2000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-4
# wait
# python train_model.py --type 'vox' --max_iter 4000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-5 --load_checkpoint
# wait
# python train_model.py --type 'vox' --max_iter 6000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint
# wait
# python train_model.py --type 'vox' --max_iter 8000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint
# wait
# python train_model.py --type 'vox' --max_iter 10002  --num_workers 4 --save_freq 200 --batch_size 30 --lr 5e-7 --load_checkpoint
# wait

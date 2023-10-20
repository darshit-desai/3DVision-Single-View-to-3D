# Baseline Model
# python train_model.py --type 'point' --max_iter 2000  --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-4
# wait
# python train_model.py --type 'point' --max_iter 4000 --load_checkpoint  --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-5
# wait
# python train_model.py --type 'point' --max_iter 6000 --load_checkpoint  --num_workers 4 --save_freq 500 --batch_size 16 --lr 1e-6
# wait
# python train_model.py --type 'point' --max_iter 8000 --load_checkpoint  --num_workers 4 --save_freq 500 --batch_size 16 --lr 5e-7
# wait
# python train_model.py --type 'point' --max_iter 10002 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 30 --lr 5e-7
# wait


# N_points 10000 points AvgF1@0.05 = 96.891
# python train_model.py --type 'point' --max_iter 2000  --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-4 --n_points 10000
# wait
# python train_model.py --type 'point' --max_iter 4000 --load_checkpoint  --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-5 --n_points 10000
# wait
# python train_model.py --type 'point' --max_iter 6000 --load_checkpoint  --num_workers 4 --save_freq 500 --batch_size 16 --lr 1e-6 --n_points 10000
# wait
# python train_model.py --type 'point' --max_iter 8000 --load_checkpoint  --num_workers 4 --save_freq 500 --batch_size 16 --lr 5e-7 --n_points 10000
# wait
# python train_model.py --type 'point' --max_iter 10002 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 30 --lr 5e-7 --n_points 10000
# wait

# N_points 25000 points AvgF1@0.05 = 97.009
python train_model.py --type 'point' --max_iter 2000  --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-4 --n_points 25000
wait
python train_model.py --type 'point' --max_iter 4000 --load_checkpoint  --num_workers 4 --save_freq 500 --batch_size 16 --lr 4e-5 --n_points 25000
wait
python train_model.py --type 'point' --max_iter 6000 --load_checkpoint  --num_workers 4 --save_freq 500 --batch_size 16 --lr 1e-6 --n_points 25000
wait
python train_model.py --type 'point' --max_iter 8000 --load_checkpoint  --num_workers 4 --save_freq 500 --batch_size 16 --lr 5e-7 --n_points 25000
wait
python train_model.py --type 'point' --max_iter 10002 --load_checkpoint --num_workers 4 --save_freq 500 --batch_size 30 --lr 5e-7 --n_points 25000
wait
# Baseline model Avg F1@0.05: 92.034
python train_model.py --type 'mesh' --max_iter 4000 --batch_size 16 --num_workers 4 --save_freq 500 --lr 4e-5 --w_smooth 0.500
wait
python train_model.py --type 'mesh' --max_iter 6000 --load_checkpoint --batch_size 16 --num_workers 4 --save_freq 500 --lr 1e-6 --w_smooth 0.500
wait
python train_model.py --type 'mesh' --max_iter 8000 --load_checkpoint --batch_size 30 --num_workers 4 --save_freq 500 --lr 5e-7 --w_smooth 0.500
wait
python train_model.py --type 'mesh' --max_iter 10002 --load_checkpoint --batch_size 30 --num_workers 4 --save_freq 500 --lr 5e-7 --w_smooth 0.500
wait

# Change Hyperparameter w_smooth=5, AvgF1@0.05 = 66.825
# python train_model.py --type 'mesh' --max_iter 6000 --batch_size 16 --num_workers 4 --save_freq 500 --lr 1e-2 --w_smooth 5
# wait
# python train_model.py --type 'mesh' --max_iter 8000 --load_checkpoint --batch_size 16 --num_workers 4 --save_freq 500 --lr 1e-3 --w_smooth 5
# wait
# python train_model.py --type 'mesh' --max_iter 10002 --load_checkpoint --batch_size 30 --num_workers 4 --save_freq 500 --lr 4e-5 --w_smooth 5
# wait
# python train_model.py --type 'mesh' --max_iter 15002 --load_checkpoint --batch_size 30 --num_workers 4 --save_freq 500 --lr 1e-6 --w_smooth 5
# wait

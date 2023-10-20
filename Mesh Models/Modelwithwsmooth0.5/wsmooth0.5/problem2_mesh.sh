python train_model.py --type 'mesh' --max_iter 4000 --batch_size 16 --num_workers 4 --save_freq 500 --lr 4e-5 --w_smooth 0.05
wait
python train_model.py --type 'mesh' --max_iter 6000 --load_checkpoint --batch_size 16 --num_workers 4 --save_freq 500 --lr 1e-6 --w_smooth 0.05
wait
python train_model.py --type 'mesh' --max_iter 8000 --load_checkpoint --batch_size 30 --num_workers 4 --save_freq 500 --lr 5e-7 --w_smooth 0.05
wait
python train_model.py --type 'mesh' --max_iter 10002 --load_checkpoint --batch_size 30 --num_workers 4 --save_freq 500 --lr 5e-7 --w_smooth 0.05
wait
#python train_model.py --type 'mesh' --max_iter 8127 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 10836 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 13545 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 16254 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 18963 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 21672 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 24381 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 27090 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 29799 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 32508 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 35217 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 37926 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 40635 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 43344 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 46053 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 48762 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 51471 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 54180 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000
#wait
#python train_model.py --type 'mesh' --max_iter 56889 --load_checkpoint --batch_size 16 --num_workers 1 --save_freq 1000

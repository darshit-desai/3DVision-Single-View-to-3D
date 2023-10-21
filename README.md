# 3DVision-Single-View-to-3D

**Access the final report in the root/starter.md.html file or on darshit-desai.github.io/3DVision-Single-View-to-3D**

**All the baseline models are already in root folder to run evaluation skip the steps to check how to run eval_model.py**

This readme is for replicating the results primarily shown in Problem 1.1-1.3 abd Problem 2.1-2.3, Rest of the problems the models are already generated and can be accessed based on below instructions. The model.py, train_model.py have been configured to train the baseline models shown in Problem 2.1-2.3, for training the models for hyperparameters uncomment the bash commands as shown in the following bash file representation and also uncomment the appropriate model in the model.py file for voxels:

```BASH
#!/bin/bash

###Comment this out
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

###Uncomment the below block of 5 commands
# Baseline Model (Conv layers Pix2Vox Model)  Avg F1@0.05: 82.663
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

# Baseline Model (ImplicitMLPDecoder)  Avg F1@0.05: 0.0 
# python train_model.py --type 'vox' --max_iter 2000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-4
# wait
# python train_model.py --type 'vox' --max_iter 4000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 4e-5 --load_checkpoint
# wait
# python train_model.py --type 'vox' --max_iter 6000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint
# wait
# python train_model.py --type 'vox' --max_iter 8000  --num_workers 4 --save_freq 200 --batch_size 16 --lr 1e-6 --load_checkpoint
# wait
# python train_model.py --type 'vox' --max_iter 10002  --num_workers 4 --save_freq 200 --batch_size 24 --lr 5e-7 --load_checkpoint
# wait
```
Note for voxel evaluation this is mandatorily required to be done, uncomment and comment the model you are using accordingly
```python
# Model 1
# self.decoder =  ImplicitMLPDecoder()
  
# Model 2 
self.decoder = nn.Sequential(
    nn.Linear(512, 1024), 
    nn.ReLU(),
    nn.Linear(1024, 2048),
    nn.ReLU(), 
    nn.Linear(2048, 32*32*32)) 

# Model 3 Upconvolutions
# self.decoder = nn.Sequential(
# nn.Linear(512, 1024),
# nn.Unflatten(1, (128, 2, 2, 2)),
# torch.nn.ConvTranspose3d(128, 256, kernel_size=3, stride=1),
# torch.nn.ReLU(), # shape [16, 256, 4, 4, 4]

# torch.nn.ConvTranspose3d(256, 384, kernel_size=3, stride=3, padding=1),
# torch.nn.ReLU(), # shape  [16, 384, 10, 10, 10]

# torch.nn.ConvTranspose3d(384, 256, kernel_size=3, stride=3, padding=1),
# torch.nn.ReLU(), # [16, 256, 28, 28, 28]

# torch.nn.ConvTranspose3d(256, 96, kernel_size=5, stride=1),
# torch.nn.ReLU(), # shape [16, 96, 32, 32, 32]

# torch.nn.Conv3d(96, 1, kernel_size=1, stride=1), # shape [16, 1, 32, 32, 32]
# )      
```

For point cloud and meshes this type of operation is not required.

## How can you locate the results?

### For Problem 1
Go to the root of the directory and cd into `Results_problem1/` directory

### For Problem 2.1-2.5
```tree
.
├── Mesh Models
│   ├── Modelwithwsmooth0.5
│   │   ├── eval_mesh.png
│   │   ├── Notimportant
│   │   ├── problem2_mesh.sh
│   │   └── Resultswsmooth0.5
│   └── Model_withwsmooth5
│       ├── eval_mesh.png
│       ├── problem2_mesh.sh
│       └── Results
├── PointModels
│   ├── Model2_npoints10000
│   │   ├── eval_point.png
│   │   ├── problem2_point.sh
│   │   └── Results
│   ├── Model2_pointsbaseline
│   │   ├── eval_point.png
│   │   ├── problem2_point.sh
│   │   └── Results
│   └── Model3_npoints25000
│       ├── eval_point.png
│       ├── problem2_point.sh
│       └── Results
└── voxmodels
    ├── voxmodel_ConvLayers
    │   ├── eval_vox.png
    │   ├── problem2_voxel.sh
    │   └── Results
    ├── voxmodel_implicitMLPDecoder
    │   ├── eval_vox.png
    │   ├── problem2_voxel.sh
    │   └── Results
    └── voxmodel_LinearLayers
        ├── eval_vox.png
        ├── problem2_voxel.sh
        └── results
```

The result directory is bifurcated model wise:
* The Point cloud model has three result folders:
    * `PointModels/Model2_pointsbaseline`, This model is trained with 5000 points as the hyperparameter
    * `PointModels/Model2_npoints10000`, This model is trained with 10000 points as the hyperparameter
    * `PointModels/Model2_npoints25000`, This model is trained with 25000 points as the hyperparameter
For running each of the models you need to shift the bash file inside the above folders to the root folder.
Also, The results are inside each of this folders and the models are also saved in those folders.

* The Voxel model has three result folders:
    * `Results_Problem2.1-2.5\voxmodels\voxmodel_ConvLayers`, This model is trained with ConvLayers as mentioned in the report
    * `Results_Problem2.1-2.5\voxmodels\voxmodel_implicitMLPDecoder`, This model is trained with implicitMLPDecoder
    * `Results_Problem2.1-2.5\voxmodels\voxmodel_LinearLayers`, This model is trained with Linear Layers
For running each of the models you need to shift the bash file inside the above folders to the root folder.
Also, The results are inside each of this folders and the models are also saved in those folders.

* The Mesh model has two result folders:
    * `Results_Problem2.1-2.5\Mesh Models\Model_withwsmooth5`, This model is trained with wsmooth=5
    * `Results_Problem2.1-2.5\Mesh Models\Model_withwsmooth0.5`, This model is trained with wsmooth=0.5
For running each of the models you need to shift the bash file inside the above folders to the root folder.
Also, The results are inside each of this folders and the models are also saved in those folders.

**Note:** *For evaluating the hyperparameter models you would need to copy the models from the above folders to the root folder*

## Run command for Q1

### For Voxels:

```BASH
python fit_data.py --type 'vox' --max_iter 100000
```
### For Meshes:

```BASH
python fit_data.py --type 'mesh' --max_iter 100000
```
### For Point Cloud:

```BASH
python fit_data.py --type 'point' --max_iter 100000
```



## Training for Q2:

### For Voxels:

```BASH
./problem2_voxel.sh
```

### For Point Clouds:

```BASH
./problem2_point.sh
```

### For Mesh:

```BASH
./problem2_mesh.sh
```

## Evaluation

### For Voxels, Point Clouds and Meshes:

For point clouds with n_points 5000, 10000 or 25000, Mention n_points below
```BASH
python eval_model.py --vis_freq 100 --load_checkpoint --type 'point' --n_points $n_points
```

For mesh and vox the command remains the same
```BASH
python eval_model.py --vis_freq 100 --load_checkpoint --type 'mesh|point'
```

## Interpretation
The run command for the gaussian, white and black image outputs are mentioned below:
```BASH
python eval_model_interpret.py --type 'vox|point|mesh'
```

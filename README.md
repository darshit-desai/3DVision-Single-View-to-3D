# 3DVision-Single-View-to-3D

**Download the models from here**

https://umd.box.com/s/v2c6pdm1nzrvgtszi2veyy2dmja4u5mt

**Access the final report in the root/report.md.html file or on https://darshit-desai.github.io/3DVision-Single-View-to-3D (highly recommended to refer online)**

**All the baseline models results are already in root folder to run evaluation skip the steps to check how to run eval_model.py**

This readme is for replicating the results primarily shown in Problem 1.1-1.3 abd Problem 2.1-2.3, Rest of the problems the models are already generated and can be accessed based on below instructions. The model.py, train_model.py have been configured to train the baseline models shown in Problem 2.1-2.3, for training the models for hyperparameters uncomment the bash commands as shown in the following bash file representation and also uncomment the appropriate model in the model.py file for voxels:

### For training problem 2 models

#### For Problem 2.1

    #"Usage: $0 [train_baseline|train_conv|train_implicitMLP]"
    ./problem2_vox.sh train_baseline

#### For Problem 2.2

    #"Usage: $0 [train_baseline|train_n_points_10000|train_n_points_25000]"
    ./problem2_point.sh train_baseline


#### For Problem 2.2

"Usage: $0 [train_baseline|train_smooth_5]"
./problem2_mesh.sh train_baseline


#### For Problem 2.3

    #"Usage: $0 [train_baseline|train_smooth_5]"
    ./problem2_mesh.sh train_baseline

Note for every model you train the checkpoints are saved as defaults `checkpoint_vox|point|mesh.pth` so if you train a hyperparameter model, then you might be overwritting the same files.


## How can you locate the models along with the results?

### For Problem 1
Go to the root of the directory of the ****box link**** and cd into `Results_problem1/` directory

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
## Evaluation

### For Voxels, Point Clouds and Meshes:

For point clouds with n_points 5000, 10000 or 25000, Mention n_points below
```BASH
python eval_model.py --vis_freq 100 --load_checkpoint --type 'point' --n_points $n_points
```

For mesh the command remains the same
```BASH
python eval_model.py --vis_freq 100 --load_checkpoint --type 'mesh' 
```

For voxel the command is changed a bit
```BASH
python eval_model.py --vis_freq 100 --load_checkpoint --type 'vox' #--model_type 'baseline|mlp|conv' if using vox in --type
```

## Interpretation
The run command for the gaussian, white and black image outputs are mentioned below:
```BASH
python eval_model_interpret.py --type 'vox|point|mesh' #--model_type 'baseline|mlp|conv' if using vox in --type
```

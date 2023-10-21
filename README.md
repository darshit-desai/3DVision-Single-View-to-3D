# 3DVision-Single-View-to-3D

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

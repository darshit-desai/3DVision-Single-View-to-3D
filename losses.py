import torch
import pytorch3d
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# voxel_src_clamping = torch.clamp(voxel_src, 0.0, 1.0)
	# voxel_tgt_clamping = torch.clamp(voxel_tgt, 0.0, 1.0)
	# print("Voxel src shape", voxel_src.shape)
	# print("Voxel target shape", voxel_tgt.shape)
	bceloss = torch.nn.BCEWithLogitsLoss()
	loss = bceloss(voxel_src, voxel_tgt)
	# implement some loss for binary voxel grids
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# implement chamfer loss from scratch
	# Verify input shapes
	# Calculate nearest neighbors using PyTorch3D's knn_points
	source2target_knn = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, K=1, norm=2)

	# Calculate chamfer loss
	source2target_dist = source2target_knn.dists[..., 0]  # (B, N)
	target2source_knn = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, K=1, norm=2)
	target2source_dist = target2source_knn.dists[..., 0]  # (B, M)
	loss_chamfer = source2target_dist.mean() + target2source_dist.mean()
	# loss_chamfer, _ = pytorch3d.loss.chamfer_distance(point_cloud_src, point_cloud_tgt)
	return loss_chamfer

def smoothness_loss(mesh_src):
	# implement laplacian smoothening loss
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")
	return loss_laplacian

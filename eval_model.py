import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from utils import get_mesh_renderer, get_points_renderer
from PIL import Image
import numpy as np
from tqdm import tqdm

import imageio

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)
    # print(gt_to_pred_dists)
    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        # voxels_src = torch.nn.Sigmoid(voxels_src)
        # predictions = predictions.squeeze(0)
        print("Prediction information:")
        print("Number of vertices:", len(predictions))
        print("Number of faces:", len(predictions))
        H,W,D = voxels_src.shape[2:]
        print("Shape of voxels_src:", voxels_src.shape)
        # vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.3)
        # USE CUBIFY
        # cubes = pytorch3d.ops.cubify(voxels_src.squeeze(0), thresh=0.0)
        # vertices_src = cubes.verts_packed()
        # faces_src = cubes.faces_packed()
        #####
        vertices_src = torch.tensor(vertices_src).float()
        print("Length of vertices_src:", len(vertices_src))
        print("Length of faces_src:", len(faces_src))
        ###############
        # USE CUBIFY
        # faces_src = torch.tensor(faces_src.detach().cpu().numpy().astype(int))
        # mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src.to(args.device)])
        ############
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])

        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points.detach().cpu(), H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics

def render_voxels(optimized_voxel, output_path):
    voxels_src = optimized_voxel
    # voxels_src = torch.nn.Sigmoid(voxels_src)
    voxel_size = 32
    max_value = 1.1
    min_value = -1.1
    #make vertices and faces for symmetric 360 degree rotation
    print(voxels_src.shape)
    vertices, faces = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), 0.3)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    color1 = [0.7, 0.0, 0.4]
    color2 = [0.6, 1.0, 1.0]
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value    
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)
    print("Shape of vertices:", vertices.shape)
    z_min = vertices[:,:,2].min()
    z_max = vertices[:,:,2].max()
    alpha = (vertices[:, :, 2] - z_min) / (z_max - z_min)
    new_colors = alpha[:, :, None] * torch.tensor(color2) + (1 - alpha[:, :, None]) * torch.tensor(color1)
    textures = pytorch3d.renderer.TexturesVertex(new_colors)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=args.device)
    voxel_chair_mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures).to(
        args.device
    )
    renderer = get_mesh_renderer(image_size=512, device=args.device)
    num_frames = 36
    render_full = []
    camera_positions = []
    azim = torch.linspace(0, 360, num_frames)
    for azi in azim:
        azimuth = azi
        distance = 3.0
        elevation = 30.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=args.device, degrees=True)
        camera_positions.append((R,T))
    for R,T in tqdm(camera_positions):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=args.device)
        rend = renderer(voxel_chair_mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        render_full.append(rend)
    images = []
    for i, r in enumerate(render_full):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave(output_path, images, duration=12.0, loop=0)

def render_points(optimized_points, output_path, type_data):
    image_size= 512
    background_color=(1, 1, 1)
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    verts = optimized_points
    rgb = (verts - verts.min()) / (verts.max() - verts.min())
    # color1 = [1.0, 0.0, 0.0]
    # color2 = [0.0, 0.0, 1.0]
    # print(rgb.shape)
    device = torch.device("cuda:0")
    # rgb = rgb.to(device)
    # color=torch.tensor([1.0, 0.0, 0.0])
    # rgb = torch.ones_like(verts) * color.to(device)
    color1 = torch.tensor([1.0, 0.0, 0.0])
    color2 = torch.tensor([0.0,0.0, 1.0])
    # print("Color 1 value: ", color1)
    # print("Color 2 value: ", color2)
    color1 = color1.to(device)
    color2 = color2.to(device)
    color = rgb[:, :, None] * color2 + (1 - rgb[:, :, None]) * color1
    color=color.squeeze(0).permute(1,0,2)
    
    if (type_data != "gt"):
        # verts = verts.unsqueeze(0)
        print("Points shape: ", verts.shape)
        print("RGB shape: ", color.shape)
        # rgb = rgb.unsqueeze(0)
    # print("Color shape: ", color.shape)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=color)
    num_frames = 36
    render_full = []
    camera_positions = []
    azim = torch.linspace(0, 360, num_frames)
    for azi in azim:
        azimuth = azi
        distance = 1.0
        elevation = 30.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=args.device, degrees=True)
        camera_positions.append((R,T))
    for R,T in tqdm(camera_positions):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=args.device)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].detach().cpu().numpy()  # (N, H, W, 3)
        render_full.append(rend)
    images = []
    for i, r in enumerate(render_full):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave(output_path, images, duration=12.0, loop=0)    
    print('Done!')

def render_mesh(mesh_src, args, output_file):
    vertices = mesh_src.verts_packed().to(args.device)
    faces = mesh_src.faces_packed().to(args.device)
    color1 = [0.7, 0.0, 0.4]
    color2 = [0.6, 1.0, 1.0]
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    z_min = vertices[:,:,2].min()
    z_max = vertices[:,:,2].max()
    alpha = (vertices[:, :, 2] - z_min) / (z_max - z_min)
    new_colors = alpha[:, :, None] * torch.tensor(color2).to(args.device) + (1 - alpha[:, :, None]) * torch.tensor(color1).to(args.device)
    textures = pytorch3d.renderer.TexturesVertex(new_colors)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=textures
    )
    renderer = get_mesh_renderer(image_size=512, device=args.device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=args.device)
    num_frames = 36
    camera_positions = []
    # output_file = "Results/mesh.gif"
    azim = torch.linspace(0, 360, num_frames)
    for azi in azim:
        azimuth = azi
        distance = 2.0
        elevation = 30.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=args.device, degrees=True)
        camera_positions.append((R,T))
    render_full = []
    for R,T in tqdm(camera_positions):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=args.device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.detach().cpu().numpy()[0, ..., :3]  # (N, H, W, 3)
        render_full.append(rend)
    images = []
    for i, r in enumerate(render_full):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=12.0, loop=0)
    print('Done!')
    

def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_iter1 = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter1}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)
        if args.type == "vox":
            predictions = torch.sigmoid(predictions)

        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)
        print("Mesh_gt information:")
        print("Number of vertices:", len(mesh_gt.verts_list()))
        print("Number of faces:", len(mesh_gt.faces_list()))
        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        # TODO:
        if (step % args.vis_freq) == 0:
            img_step = step % args.vis_freq
            num_imgs = step // args.vis_freq
            if args.type == "vox":
                
                render_voxels(predictions, output_path=f'Results/Q2_1_{num_imgs}_{args.type}.gif')
                voxel_ground_truth = feed_dict['voxels'].to(args.device)
                render_voxels(voxel_ground_truth, output_path=f'Results/Q2_1_{num_imgs}_gt_{args.type}.gif')
            elif args.type == "point":
                render_points(predictions, output_path=f'Results/Q2_2_{num_imgs}_{args.type}.gif', type_data="pred")
                gt_points = sample_points_from_meshes(mesh_gt, args.n_points).to(args.device)
                render_points(gt_points, output_path=f'Results/Q2_2_{num_imgs}_gt_{args.type}.gif', type_data="gt")
            elif args.type == "mesh":
                render_mesh(predictions, args, output_file=f'Results/Q2_3_{num_imgs}_{args.type}.gif')
                render_mesh(mesh_gt, args, output_file=f'Results/Q2_3_{num_imgs}_gt_{args.type}.gif')    
            plt.imsave(f'Results/{step}_{args.type}.png', images_gt.squeeze().detach().cpu().numpy())
        if(step == max_iter-1 ):
            plt.imsave(f'Results/{step}_{args.type}.png', images_gt.squeeze().detach().cpu().numpy())
            if args.type == "vox":
                render_voxels(predictions, output_path=f'Results/Q2_1_final_{args.type}.gif')
                voxel_ground_truth = feed_dict['voxels'].to(args.device)
                render_voxels(voxel_ground_truth[0], output_path=f'Results/Q2_1_final_gt_{args.type}.gif')
            elif args.type == "point":
                render_points(predictions, output_path=f'Results/Q2_2_final_{args.type}.gif', type_data="pred")
                gt_points = sample_points_from_meshes(mesh_gt, args.n_points).to(args.device)
                render_points(gt_points, output_path=f'Results/Q2_2_final_gt_{args.type}.gif', type_data="gt")        
            elif args.type == "mesh":
                render_mesh(predictions, args, output_file=f'Results/Q2_3_final_{args.type}.gif')
                render_mesh(mesh_gt, args, output_file=f'Results/Q2_3_final_gt_{args.type}.gif')

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)

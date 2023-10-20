import argparse
import os
import time
import pytorch3d
import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch
import numpy as np
from tqdm import tqdm
import mcubes
from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
)
from matplotlib import pyplot as plt
from utils import get_mesh_renderer, get_points_renderer
from PIL import Image

import imageio





def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    return parser
def render_mesh(mesh_source,deformed_mesh,args,iters,data_type):
    if data_type=="preds":
        mesh_source.offset_verts_(deformed_mesh)
        
    vertices = mesh_source.verts_packed().to(args.device)
    faces = mesh_source.faces_packed().to(args.device)
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
    if data_type=="preds":
        output_file = "Results_problem1/Mesh/mesh_fitdata"+str(iters)+".gif"
    if data_type=="gt":
        output_file = "Results_problem1/Mesh/mesh_fitdata"+data_type+str(iters)+".gif"
    azim = torch.linspace(0, 360, num_frames)
    for azi in azim:
        azimuth = azi
        if iters>=5000:
            distance = 1.0
        else:
            distance=3
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

def fit_mesh(mesh_src, mesh_tgt, args, device):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis)) 
        if step == args.max_iter - 1:
            optimized_mesh = mesh_src.detach().clone()
            render_mesh(optimized_mesh, deform_vertices_src,args, step, data_type="preds")
            render_mesh(mesh_tgt, deform_vertices_src,args, step, data_type="gt")
        if step==0:
            optimized_mesh = mesh_src.detach().clone()
            render_mesh(optimized_mesh, deform_vertices_src,args, step, data_type="preds")
        if step==500:
            optimized_mesh = mesh_src.detach().clone()
            render_mesh(optimized_mesh, deform_vertices_src,args, step, data_type="preds")
        if step==5000:
            optimized_mesh = mesh_src.detach().clone()
            render_mesh(optimized_mesh, deform_vertices_src,args, step, data_type="preds")

    
    

def render_pointcloud(optimized_pc,args, iters, data_type):
    image_size= 512
    background_color=(1, 1, 1)
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    print("Shape of optimized point cloud of optimized_pc[verts] and optimized[rgb]: ", optimized_pc.shape)
    verts = optimized_pc
    rgb = (optimized_pc - optimized_pc.min()) / (optimized_pc.max() - optimized_pc.min())
    # color1 = [1.0, 0.0, 0.0]
    # color2 = [0.0, 0.0, 1.0]
    # print(rgb.shape)
    device = torch.device("cuda:0")
    rgb = rgb.to(device)
    
    color1 = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)
    color2 = torch.tensor([0.0, 0.0, 1.0]).unsqueeze(0)
    print("Color 1 value: ", color1)
    print("Color 2 value: ", color2)
    color1 = color1.to(device)
    color2 = color2.to(device)
    color = rgb[:, :, None] * color2 + (1 - rgb[:, :, None]) * color1
    color=color.squeeze(0).permute(1,0,2)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=color)
    num_frames = 36
    render_full = []
    camera_positions = []
    if data_type=="preds":
        output_file = "Results_problem1/Pointcloud/pointcloud_fitdata"+str(iters)+".gif"
    if data_type == "gt":
        output_file = "Results_problem1/Pointcloud/pointcloud_fitdata"+data_type+str(iters)+".gif"
    azim = torch.linspace(0, 360, num_frames)
    for azi in azim:
        azimuth = azi
        distance = 1.0
        elevation = 15.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=args.device, degrees=True)
        camera_positions.append((R,T))
    for R,T in tqdm(camera_positions):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=args.device)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        render_full.append(rend)
    images = []
    for i, r in enumerate(render_full):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=12.0, loop=0)    
    print('Done!')


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
        if step == args.max_iter - 1:
            optimized_pc = pointclouds_src.detach().clone()
            render_pointcloud(optimized_pc, args, step, data_type="preds")
            render_pointcloud(pointclouds_tgt, args, step, data_type="gt")
        if step==0:
            optimized_pc = pointclouds_src.detach().clone()
            render_pointcloud(optimized_pc, args, step, data_type="preds")
        if step==5000:
            optimized_pc = pointclouds_src.detach().clone()
            render_pointcloud(optimized_pc, args, step, data_type="preds")
        if step==10000:
            optimized_pc = pointclouds_src.detach().clone()
            render_pointcloud(optimized_pc, args, step, data_type="preds")
    
def render_voxel(optimized_voxel,args,iters,data_type):
    #cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    binarray = optimized_voxel.cpu().numpy()
    binarray = np.squeeze(binarray, axis=0)
    if data_type == "preds":
        output_file = "Results_problem1/Voxels/voxel_fitdata"+str(iters)+".gif"
    if data_type == "gt":
        output_file = "Results_problem1/Voxels/voxel_fitdata"+data_type+str(iters)+".gif"
    voxel_size = 32
    max_value = 1.1
    min_value = -1.1
    #make vertices and faces for symmetric 360 degree rotation
    vertices, faces = mcubes.marching_cubes(binarray, 0.5)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    color1 = [0.7, 0.0, 0.4]
    color2 = [0.6, 1.0, 1.0]
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value    
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)
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
    imageio.mimsave(output_file, images, duration=12.0, loop=0)
    # USE CUBIFY
    # cubes = pytorch3d.ops.cubify(optimized_voxel, thresh=0.5, device = torch.device(args.device))
    # vertices = cubes.verts_packed()
    # color2 = torch.tensor([0.0, 0.0, 1.0], device=args.device)
    # color1 = torch.tensor([1.0, 0.0, 0.0], device=args.device)
    # vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    # # faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    # z_min = vertices[:,:,2].min()
    # z_max = vertices[:,:,2].max()
    # alpha = (vertices[:, :, 2] - z_min) / (z_max - z_min)
    # new_colors = alpha[:, :, None] * torch.tensor(color2) + (1 - alpha[:, :, None]) * torch.tensor(color1)
    # textures = pytorch3d.renderer.TexturesVertex(new_colors)
    # cubes.textures = textures
    # renderer = get_mesh_renderer(image_size=512, device=args.device)
    # lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=args.device)
    # num_frames = 36
    # camera_positions = []
    # for frame_idx in range(num_frames):
    #     azimuth = 360 * frame_idx / num_frames
    #     distance = 3.0
    #     elevation = 30.0
    #     R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=args.device)
    #     camera_positions.append((R,T))

    # renders = []
    # for R,T in tqdm(camera_positions):
    #     cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=args.device)
    #     rend = renderer(cubes, cameras=cameras, lights=lights)
    #     rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
    #     renders.append(rend)

    # images = []
    # duration  = 10
    # for i, r in enumerate(renders):
    #     image = Image.fromarray((r * 255).astype(np.uint8))
    #     images.append(np.array(image))
    # imageio.mimsave(output_file, images, duration=duration, loop=0)
    print('Done!')

def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    progress_bar = tqdm(total=args.max_iter, leave=False, dynamic_ncols=True, desc='Training')
    optimized_voxel = None
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()


        progress_bar.update(1)  # Update the progress bar
        progress_bar.set_postfix(step=f"{step}/{args.max_iter}", ttime=f"{total_time:.0f} ({iter_time:.2f})", loss=f"{loss_vis:.3f}")
        if step == args.max_iter - 1:
            optimized_voxel = voxels_src.detach().clone()
            render_voxel(optimized_voxel, args, step, data_type="preds")
            render_voxel(voxels_tgt, args, step, data_type="gt")
        if step==0:
            optimized_voxel = voxels_src.detach().clone()
            render_voxel(optimized_voxel, args, step, data_type="preds")
        if step==1000:
            optimized_voxel = voxels_src.detach().clone()
            render_voxel(optimized_voxel, args, step, data_type="preds")
        if step==5000:
            optimized_voxel =voxels_src.detach().clone()
            render_voxel(optimized_voxel, args, step, data_type="preds")
    print("Loss calculation done")
    


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)        
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args, device=args.device)        


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
